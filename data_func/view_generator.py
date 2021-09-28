import numpy as np
import torchaudio

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, wav_transform, sr, length
                 , n_views=2,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.wav_transform = wav_transform
        self.n_views = n_views
        self.sr = sr
        self.length = length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return [self.amplitude_to_db(
                self.mel_transform(
                self.wav_transform.apply(x,
                                         src_info={'rate': self.sr},
                                         target_info={'rate': self.sr}))).squeeze().transpose(-1,-2)
                for i in range(self.n_views)]

class ViewGenerator(object):
    """Take two random crops of one image as the query and key."""

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



class SpecViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.spec_transform = torchaudio.transforms.Spectrogram(win_length=win_length,
                                                                hop_length=hop_length,
                                                                n_fft=n_fft,
                                                                power=power)

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.spec_transform(x)