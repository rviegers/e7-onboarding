import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
import wandb
from sklearn.model_selection import GroupShuffleSplit

from prepare_dataset import BirdDataset, get_label_list
from metric import compute_map
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)

    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, config=dict(cfg))

    label_list = get_label_list(cfg.data.metadata)
    n_classes = len(label_list)

    dataset = BirdDataset(cfg.data.metadata, cfg.data.audio_dir, label_list, spec_dir=cfg.data.spec_dir)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.data.val_split, random_state=cfg.training.seed)
    train_idx, val_idx = next(gss.split(dataset.df, groups=dataset.df["author"]))
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = timm.create_model(cfg.model.name, pretrained=cfg.model.pretrained, num_classes=n_classes)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    best_map = -1.0
    for epoch in range(1, cfg.training.epochs + 1):
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

        if cfg.wandb.enabled:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "mAP": map_score})

        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> Saved best_model.pt (mAP={best_map:.4f})")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
