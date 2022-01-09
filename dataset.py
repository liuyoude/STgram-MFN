import torch
from torch.utils.data import Dataset
import torchaudio
import joblib
import librosa
import re


class Generator(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


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

        x = x[:self.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)
        x_mel = self.transform(x_wav).unsqueeze(0)
        # print(x.shape)

        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_path_list)


class WavMelClassifierDataset:
    def __init__(self, root_folder, sr, ID_factor):
        self.root_folder = root_folder
        self.sr = sr
        self.factor = ID_factor

    def get_dataset(self,
                    n_fft=1024,
                    n_mels=128,
                    win_length=1024,
                    hop_length=512,
                    power=2.0):
        dataset = Wav_Mel_ID_Dataset(self.root_folder,
                                     self.factor,
                                     self.sr,
                                     win_length,
                                     hop_length,
                                     transform=Generator(
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
