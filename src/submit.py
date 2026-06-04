import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import timm
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.transforms.functional import resize

from prepare_dataset import (
    DURATION, HOP_LENGTH, IMG_SIZE, N_FFT, N_MELS, SAMPLE_RATE,
    get_label_list,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_DIR = Path("data/test_audio")
METADATA = "data/train_split.csv"
TEST_SPLIT = "data/test_split.csv"
SAMPLE_SUB = "data/sample_submission_test.csv"
SUBMIT_URL = "https://gooey-elk-trial.ngrok-free.dev/leaderboard"


def load_model(weights_path: str, n_classes: int, model_name: str) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model.to(DEVICE)


def waveform_to_spec(waveform: torch.Tensor) -> torch.Tensor:
    mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    db = T.AmplitudeToDB()
    spec = db(mel(waveform))
    spec = spec.repeat(3, 1, 1)
    return resize(spec, [IMG_SIZE, IMG_SIZE], antialias=True)


def predict_file(model: torch.nn.Module, ogg_path: Path) -> np.ndarray:
    waveform, sr = torchaudio.load(ogg_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0, keepdim=True)

    target_len = SAMPLE_RATE * DURATION
    if waveform.shape[-1] < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
    else:
        waveform = waveform[..., :target_len]

    spec = waveform_to_spec(waveform).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(spec)).cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Generate predictions and submit to server")
    parser.add_argument("--experiments", required=True, nargs="+", help="Path(s) to Hydra experiment output directories")
    parser.add_argument("--name", required=True, help="Participant/team name for submission")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path (default: submission.csv)")
    args = parser.parse_args()

    label_list = get_label_list(METADATA)
    test_df = pd.read_csv(TEST_SPLIT)
    filenames = list(test_df["filename"])

    all_preds: list[dict] = []
    for exp_path in args.experiments:
        exp_dir = Path(exp_path)
        cfg = OmegaConf.load(exp_dir / ".hydra" / "config.yaml")
        model = load_model(exp_dir / "best_model.pt", len(label_list), cfg.model.name)
        rows = {}
        for filename in tqdm(filenames, desc=f"Inference {exp_dir.name}"):
            rows[Path(filename).stem] = predict_file(model, AUDIO_DIR / filename)
        all_preds.append(rows)

    row_ids = [Path(f).stem for f in filenames]
    rows = {rid: np.mean([p[rid] for p in all_preds], axis=0) for rid in row_ids}

    sample = pd.read_csv(SAMPLE_SUB)
    species_cols = [c for c in sample.columns if c != "row_id"]

    df = pd.DataFrame.from_dict(rows, orient="index", columns=label_list)
    df.index.name = "row_id"
    df = (
        df[species_cols]
        .reindex(sample.set_index("row_id").index)
        .fillna(0)
        .reset_index()
    )
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(df)} rows)")

    with open(args.out, "rb") as f:
        response = requests.post(SUBMIT_URL, files={"file": f}, data={"name": args.name})
    result = response.json()
    print(f"Submission score: {result['score']:.4f} | {result['message']}")


if __name__ == "__main__":
    main()
