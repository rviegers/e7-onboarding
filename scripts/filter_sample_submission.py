import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"

labels = pd.read_csv(data_dir / "train_metadata_africa_top30.csv")["primary_label"].unique().tolist()

submission = pd.read_csv(data_dir / "sample_submission.csv")

keep_cols = ["row_id"] + [c for c in submission.columns if c in labels]
filtered = submission[keep_cols]

out_path = data_dir / "sample_submission_africa_top30.csv"
filtered.to_csv(out_path, index=False)
print(f"Kept {len(keep_cols) - 1} species columns (from {len(submission.columns) - 1})")
print(f"Saved to {out_path}")
