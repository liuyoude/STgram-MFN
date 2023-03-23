import torch
from torch.utils.data import Dataset
import librosa
import re

import utils


class ASDDataset(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = utils.Wave2Mel(sr=args.sr, power=args.power,
                                      n_fft=args.n_fft, n_mels=args.n_mels,
                                      win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory
        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    def transform(self, filename):
        machine = filename.split('/')[-3]
        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine+'-'+id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_list)
