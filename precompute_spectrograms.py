import os
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm

METADATA = "data/train_metadata_africa_top30.csv"
AUDIO_DIR = "data/train_audio"
SPEC_DIR = "data/spectrograms"

SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320

mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
db = T.AmplitudeToDB()


def process(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    waveform = waveform.mean(dim=0, keepdim=True)
    target_len = SAMPLE_RATE * DURATION
    if waveform.shape[-1] < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
    else:
        waveform = waveform[..., :target_len]

    return db(mel(waveform))  # (1, N_MELS, time)


def main():
    df = pd.read_csv(METADATA)
    os.makedirs(SPEC_DIR, exist_ok=True)

    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing spectrograms"):
        out_path = os.path.join(SPEC_DIR, row["filename"].replace(".ogg", ".pt"))
        if os.path.exists(out_path):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        spec = process(os.path.join(AUDIO_DIR, row["filename"]))
        torch.save(spec, out_path)

    print(f"Done. Skipped {skipped} already-cached files.")


if __name__ == "__main__":
    main()
