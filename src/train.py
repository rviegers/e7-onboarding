import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import timm
import numpy as np

from prepare_dataset import BirdDataset, get_label_list
from metric import compute_map
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METADATA = "data/train_split.csv"
AUDIO_DIR = "data/train_audio"
SPEC_DIR = "data/spectrograms"

MODEL_NAME = "efficientnet_b1"
PRETRAINED = True
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 4
SEED = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    label_list = get_label_list(METADATA)
    n_classes = len(label_list)

    dataset = BirdDataset(METADATA, AUDIO_DIR, label_list, spec_dir=SPEC_DIR)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=n_classes)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_map = -1.0
    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for specs, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} train"):
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(specs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- val ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for specs, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d} val  "):
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                logits = model(specs)
                val_loss += criterion(logits, labels).item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        val_loss /= len(val_loader)

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        map_score = compute_map(targets, preds)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | mAP={map_score:.4f}")

        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> Saved best_model.pt (mAP={best_map:.4f})")


if __name__ == "__main__":
    main()
