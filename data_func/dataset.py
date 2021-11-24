from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from data_func.view_generator import ViewGenerator
# import WavAugment.augment as augment
import numpy as np
import joblib
import torchaudio
import librosa
import re
import os

class STgram_Contrastive_Dataset(Dataset):
    def __init__(self, root_floder, ID_factor, sr,
                 win_length, hop_length, transform=None):
        with open(root_floder, 'rb') as f:
            self.file_path_list_dict = joblib.load(f)
        self.transform = transform
        self.factor = ID_factor
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))

    def __getitem__(self, item):
        wav_list, mel_list, label_list = [], [], []

        for index, key in enumerate(self.file_path_list_dict.keys()):

            file_path = self.file_path_list_dict[key][item]
            machine = file_path.split('/')[-3]
            # id_str = re.findall('id_[0-9][0-9]', file_path)
            # if machine == 'ToyCar' or machine == 'ToyConveyor':
            #     id = int(id_str[0][-1]) - 1
            # else:
            #     id = int(id_str[0][-1])
            label = self.factor[machine]
            (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

            x = x[:self.sr*10]  # (1, audio_length)
            x_wav = torch.from_numpy(x)
            x_mel = self.transform(x_wav).unsqueeze(0)

            if index == 0:
                wav_tensor = x_wav.unsqueeze(0)
                mel_tensor = x_mel.unsqueeze(0)
            else:
                wav_tensor = torch.cat((wav_tensor, x_wav.unsqueeze(0)), dim=0)
                mel_tensor = torch.cat((mel_tensor, x_mel.unsqueeze(0)), dim=0)

            label_list.append(label)


        return wav_tensor, mel_tensor, np.array(label_list)

    def __len__(self):
        for index, key in enumerate(self.file_path_list_dict.keys()):
            if index == 0:
                min_len = len(self.file_path_list_dict[key])
            else:
                if len(self.file_path_list_dict[key]) < min_len:
                    min_len = len(self.file_path_list_dict[key])

        return min_len

class Wav_Mel_ID_Dataset(Dataset):
    def __init__(self, root_floder, ID_factor, sr,
                 win_length, hop_length, transform=None):
        with open(root_floder, 'rb') as f:
            self.file_path_list = joblib.load(f)
        self.transform = transform
        self.factor = ID_factor
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))

    def __getitem__(self, item):
        file_path = self.file_path_list[item]
        machine = file_path.split('/')[-3]
        id_str = re.findall('id_[0-9][0-9]', file_path)
        if machine == 'ToyCar' or machine == 'ToyConveyor':
            id = int(id_str[0][-1]) - 1
        else:
            id = int(id_str[0][-1])
        label = int(self.factor[machine] * 7 + id)
        (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

        x = x[:self.sr*10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)
        x_mel = self.transform(x_wav).unsqueeze(0)
            # print(x.shape)


        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_path_list)



class WavMelClassifierDataset:
    def __init__(self, root_folder, sr, ID_factor, pattern="con"):
        self.root_folder = root_folder
        self.sr = sr
        self.factor = ID_factor
        self.pattern = pattern

    def get_dataset(self,
                    n_fft=1024,
                    n_mels=128,
                    win_length=1024,
                    hop_length=512,
                    power=2.0):
        if self.pattern == 'con':
            dataset = STgram_Contrastive_Dataset(self.root_folder,
                                                 self.factor,
                                                 self.sr,
                                                 win_length,
                                                 hop_length,
                                                 transform=ViewGenerator(
                                                     self.sr,
                                                     n_fft=n_fft,
                                                     n_mels=n_mels,
                                                     win_length=win_length,
                                                     hop_length=hop_length,
                                                     power=power,
                                                 ))
        else:
            dataset = Wav_Mel_ID_Dataset(self.root_folder,
                                         self.factor,
                                         self.sr,
                                         win_length,
                                         hop_length,
                                         transform=ViewGenerator(
                                             self.sr,
                                             n_fft=n_fft,
                                             n_mels=n_mels,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             power=power,
                                         ))
        return dataset



if __name__ == '__main__':

    pass

