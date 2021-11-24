import logging
import os
import sys
import sklearn
import numpy as np
import time
import re
import joblib

import torch
import librosa
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from tqdm import tqdm
from data_func.view_generator import ViewGenerator
from utils import accuracy, save_checkpoint, get_machine_id_list, create_test_file_list, log_mel_spect_to_vector, \
    calculate_anomaly_score, file_to_wav_vector

import utils


# torch.manual_seed(666)


class wave_Mel_MFN_trainer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.classifier = kwargs['classifier'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        visdom_folder = f'./log/'
        os.makedirs(visdom_folder, exist_ok=True)
        visdom_path = os.path.join(visdom_folder, f'{self.args.version}_visdom_ft.log')
        self.writer = Visdom(env=self.args.version, log_to_filename=visdom_path)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.recon_criterion = torch.nn.L1Loss().to(self.args.device)

        self.csv_lines = []

    def train(self, train_loader):
        # self.eval()
        n_iter = 0

        # create model dir for saving
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, 'fine-tune'), exist_ok=True)

        print(f"Start classifier training for {self.args.epochs} epochs.")
        print(f"Training with gpu: {self.args.disable_cuda}.")

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        for epoch_counter in range(self.args.epochs):
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for waveform, melspec, labels in pbar:
                waveform = waveform.float().unsqueeze(1).to(self.args.device)
                melspec = melspec.float().to(self.args.device)
                labels = labels.long().squeeze().to(self.args.device)
                self.classifier.train()
                predict_ids, _ = self.classifier(waveform, melspec, labels)
                loss = self.criterion(predict_ids, labels)
                pbar.set_description(f'Epoch:{epoch_counter}'
                                     f'\tLclf:{loss.item():.5f}\t')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.line([loss.item()], [n_iter],
                                     win='Classifier Loss',
                                     update='append',
                                     opts=dict(
                                         title='Classifier Loss',
                                         legend=['loss']
                                     ))
                    if self.scheduler is not None:
                        self.writer.line([self.scheduler.get_last_lr()[0]], [n_iter],
                                         win='Classifier LR',
                                         update='append',
                                         opts=dict(
                                             title='AE Learning Rate',
                                             legend=['lr']
                                         ))

                n_iter += 1

            if self.scheduler is not None and epoch_counter >= 20:
                self.scheduler.step()
            print(f"Epoch: {epoch_counter}\tLoss: {loss}")
            if epoch_counter % 2 == 0:
                # save model checkpoints
                auc, pauc = self.eval()
                self.writer.line([[auc, pauc]], [epoch_counter], win=self.machine_type,
                                 update='append',
                                 opts=dict(
                                     title=self.machine_type,
                                     legend=['AUC_clf', 'pAUC_clf']
                                 ))
                print(f'{self.machine_type}\t[{epoch_counter}/{self.args.epochs}]\tAUC: {auc:3.3f}\tpAUC: {pauc:3.3f}')
                if (auc + pauc) > best_auc:
                    no_better = 0
                    best_auc = pauc + auc
                    p = pauc
                    a = auc
                    e = epoch_counter
                    checkpoint_name = 'checkpoint_best.pth.tar'
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'clf_state_dict': self.classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False,
                        filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))
            else:
                no_better += 1
            # if no_better > self.args.early_stop:
            #     break

            # if epoch_counter % 10 == 0:
            #     checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            #     save_checkpoint({
            #         'epoch': epoch_counter,
            #         'clf_state_dict': self.classifier.state_dict(),
            #         'optimizer': self.optimizer.state_dict(),
            #     }, is_best=False,
            #         filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))

        print(f'Traing {self.machine_type} completed!\tBest Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')

    def eval(self):
        sum_auc, sum_pauc, num, total_time = 0, 0, 0, 0
        #
        # sum_auc_r, sum_pauc_r = 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            start = time.perf_counter()
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            performance = []
            performance_recon = []
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                y_pred = [0. for _ in test_files]
                # y_pred_recon = [0. for _ in test_files]
                # print(111, len(test_files), target_dir)
                for file_idx, file_path in enumerate(test_files):
                    id_str = re.findall('id_[0-9][0-9]', file_path)
                    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                        id = int(id_str[0][-1]) - 1
                    else:
                        id = int(id_str[0][-1])
                    label = int(self.id_factor[machine_type] * 7 + id)
                    labels = torch.from_numpy(np.array(label)).long().to(self.args.device)
                    (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                    x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                    x_wav = torch.from_numpy(x_wav)
                    x_wav = x_wav.float().to(self.args.device)

                    x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                    x_mel = torch.from_numpy(x_mel)
                    x_mel = ViewGenerator(self.args.sr,
                                          n_fft=self.args.n_fft,
                                          n_mels=self.args.n_mels,
                                          win_length=self.args.win_length,
                                          hop_length=self.args.hop_length,
                                          power=self.args.power,
                                          )(x_mel).unsqueeze(0).unsqueeze(0).to(self.args.device)

                    with torch.no_grad():
                        self.classifier.eval()
                        predict_ids, _ = self.classifier.module(x_wav, x_mel, labels)
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = probs[label]

                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
            sum_auc += mean_auc
            sum_pauc += mean_p_auc

            time_nedded = time.perf_counter() - start
            total_time += time_nedded
            print(f'Test {machine_type} cost {time_nedded} secs')
        print(f'Total test time: {total_time} secs!')
        return sum_auc / num, sum_pauc / num

    def test(self, save=True):
        recore_dict = {}
        if not save:
            self.csv_lines = []

        sum_auc, sum_pauc, num = 0, 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # result csv
            self.csv_lines.append([machine_type])
            self.csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'{machine_type}_anomaly_score_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                        id = int(id_str[-1]) - 1
                    else:
                        id = int(id_str[-1])
                    label = int(self.id_factor[machine_type] * 7 + id)
                    labels = torch.from_numpy(np.array(label)).long().to(self.args.device)
                    (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                    x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                    x_wav = torch.from_numpy(x_wav)
                    x_wav = x_wav.float().to(self.args.device)

                    x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                    x_mel = torch.from_numpy(x_mel)
                    x_mel = ViewGenerator(self.args.sr,
                                          n_fft=self.args.n_fft,
                                          n_mels=self.args.n_mels,
                                          win_length=self.args.win_length,
                                          hop_length=self.args.hop_length,
                                          power=self.args.power,
                                          )(x_mel).unsqueeze(0).unsqueeze(0).to(self.args.device)

                    with torch.no_grad():
                        self.classifier.eval()
                        predict_ids, feature = self.classifier.module(x_wav, x_mel, labels)
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                #
                self.csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
            recore_dict[machine_type] = mean_auc + mean_p_auc
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            self.csv_lines.append(['Average'] + list(averaged_performance))
        self.csv_lines.append(['Total Average', sum_auc / num, sum_pauc / num])
        print('Total average:', sum_auc / num, sum_pauc / num)
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, self.csv_lines)
        return recore_dict

    def TFE_visual(self, wav_path):
        (x, _) = librosa.core.load(wav_path, sr=self.args.sr, mono=True)
        x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x_wav)
        x_wav = x_wav.float()

        x_mel = x[:self.args.sr * 10]  # (1, audio_length)
        x_mel = torch.from_numpy(x_mel)
        x_mel = ViewGenerator(self.args.sr,
                              n_fft=self.args.n_fft,
                              n_mels=self.args.n_mels,
                              win_length=self.args.win_length,
                              hop_length=self.args.hop_length,
                              power=self.args.power,
                              )(x_mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.classifier.eval()
            wav_enco, mel = self.classifier(x_wav, x_mel)
        wav_enco = wav_enco.cpu().squeeze().flip(dims=(0,)).numpy()
        mel = mel.cpu().squeeze().flip(dims=(0,)).numpy()
        print(mel.shape)
        cmap = ['magma', 'inferno', 'plasma', 'hot']
        index = 1
        plt.imshow(mel, cmap=cmap[index])
        plt.axis('off')
        plt.show()
        plt.close()
        plt.imshow(wav_enco, cmap=cmap[index])
        plt.axis('off')
        plt.show()
        plt.close()
