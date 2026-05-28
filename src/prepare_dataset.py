import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from torchvision.transforms.functional import resize


SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320
IMG_SIZE = 240  # EfficientNet-B1 native resolution


class BirdDataset(Dataset):
    def __init__(self, metadata_path: str, audio_dir: str, label_list: list[str], spec_dir: str | None = None):
        self.df = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.spec_dir = spec_dir
        self.label_to_idx = {l: i for i, l in enumerate(label_list)}
        self.n_classes = len(label_list)

        if spec_dir is None:
            self.mel = T.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            self.db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.spec_dir is not None:
            spec = torch.load(
                os.path.join(self.spec_dir, row["filename"].replace(".ogg", ".pt")),
                weights_only=True,
            )  # (1, N_MELS, time)
        else:
            waveform, sr = torchaudio.load(os.path.join(self.audio_dir, row["filename"]))
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            waveform = waveform.mean(dim=0, keepdim=True)
            target_len = SAMPLE_RATE * DURATION
            if waveform.shape[-1] < target_len:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
            else:
                waveform = waveform[..., :target_len]
            spec = self.db(self.mel(waveform))  # (1, N_MELS, time)

        spec = spec.repeat(3, 1, 1)                             # (3, N_MELS, time)
        spec = resize(spec, [IMG_SIZE, IMG_SIZE], antialias=True)  # (3, 240, 240)

        label = torch.zeros(self.n_classes)
        label[self.label_to_idx[row["primary_label"]]] = 1.0

        return spec, label


def get_label_list(metadata_path: str) -> list[str]:
    df = pd.read_csv(metadata_path)
    return sorted(df["primary_label"].unique().tolist())
