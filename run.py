import argparse
import torch.backends.cudnn as cudnn
import yaml

from dataset import *
from model import STgramMFN, ArcMarginProduct
from trainer import *

config_path = './config.yaml'
with open(config_path) as f:
    param = yaml.safe_load(f)

parser = argparse.ArgumentParser(description='STgram-MFN')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-j', '--workers', default=param['workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=param['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=param['batch_size'], type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=param['lr'], type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=param['wd'], type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('--log-every-n-steps', default=20, type=int,
                    help='Log every n steps')

parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model saved dir')
parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result saved dir')
parser.add_argument('--data-dir', default=param['data_dir'], type=str, help='data dir')
parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='processing data saved dir')
parser.add_argument('--data-type', default=param['data_type'], type=str, help='data type, dev_data or evl_data')
parser.add_argument('--split-flag', default=param['split_flag'], type=bool, help='pre-processing split or not')
parser.add_argument('--process-machines', default=param['process_machines'], type=list,
                    help='allowed processing machines')
parser.add_argument('--thread-num', default=param['thread_num'], type=int,
                    help='number of threading workers for data processing')

parser.add_argument('--sr', default=param['sr'], type=int, help='sample rate of wav files')
parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT param: n_fft')
parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT param: n_mels')
parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT param: hop_length')
parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT param: win_length')
parser.add_argument('--power', default=param['power'], type=float, help='STFT param: power')
parser.add_argument('--frames', default=param['frames'], type=int, help='split frames')
parser.add_argument('--skip-frames', default=param['skip_frames'], type=int, help='skip frames in spliting')

parser.add_argument('--save-every-n-epochs', default=param['save_every_n_epochs'], type=int, help='save encoder and decoder model every n epochs')
parser.add_argument('--early-stop', default=param['early_stop'], type=int, help='number of epochs for early stopping')

parser.add_argument('--t', default=param['t'], type=float)

parser.add_argument('--version', default='STgram_MFN_ArcFace(m=0.7,s=30)', type=str,
                    help='trail version')
parser.add_argument('--arcface', default=True, type=bool, help='using arcface or not')
parser.add_argument('--m', type=float, default=param['margin'], help='margin for arcface')
parser.add_argument('--s', type=float, default=param['scale'], help='scale for arcface')
parser.add_argument('--num-class', default=param['num_class'])
parser.add_argument('--device-ids', default=param['device_ids'])
parser.add_argument('--ID-factor', default=param['ID_factor'])


def preprocess():
    args = parser.parse_args()
    root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
    if not os.path.exists(root_folder):
        utils.path_to_dict(process_machines=args.process_machines,
                           data_dir=args.data_dir,
                           root_folder=root_folder,
                           ID_factor=args.ID_factor)


def test(args):
    if args.arcface:
        arcface = ArcMarginProduct(128, args.num_class, m=args.m, s=args.s)
    else:
        arcface = None
    model = STgramMFN(num_class=args.num_class,
                      c_dim=args.n_mels,
                      win_len=args.win_length,
                      hop_len=args.hop_length,
                      arcface=arcface)
    if torch.cuda.is_available() and len(args.device_ids) > 0:
        args.device = torch.device(f'cuda:{args.device_ids[0]}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    with torch.cuda.device(args.gpu_index):
        model_path = os.path.join(args.model_dir, args.version, f'checkpoint_best.pth.tar')
        if len(args.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        model.load_state_dict(torch.load(model_path)['clf_state_dict'])

        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=args.ID_factor,
                                       classifier=model,
                                       arcface=arcface,
                                       optimizer=None,
                                       scheduler=None,
                                       args=args)
        trainer.test(save=True)


def train(args):
    if args.arcface:
        arcface = ArcMarginProduct(128, args.num_class, m=args.m, s=args.s)
    else:
        arcface = None
    model = STgramMFN(num_class=args.num_class,
                      c_dim=args.n_mels,
                      win_len=args.win_length,
                      hop_len=args.hop_length,
                      arcface=arcface)
    if torch.cuda.is_available() and len(args.device_ids) > 0:
        args.device = torch.device(f'cuda:{args.device_ids[0]}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
    clf_dataset = WavMelClassifierDataset(root_folder, args.sr, param['ID_factor'])
    train_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
    train_clf_loader = torch.utils.data.DataLoader(
        train_clf_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_clf_loader), eta_min=0,
                                                           last_epoch=-1)
    #
    with torch.cuda.device(args.gpu_index):
        if len(args.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=args.ID_factor,
                                       classifier=model,
                                       arcface=arcface,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       args=args)
        trainer.train(train_clf_loader)

def main(args):
    preprocess()
    train(args)
    test(args)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    # ver = f'STgram_MFN'
    # args.version = ver
    # print(args.version)
    # preprocess()
    #
    # args.arcface = True
    # args.m = 0.7
    # args.s = 30
    # if args.arcface:
    #     args.version = ver + f'_ArcFace(m={args.m},s={args.s})'
    #
    # print(args.version)
    # pretrain_path = ver
    # train(args)
    # test(args)
