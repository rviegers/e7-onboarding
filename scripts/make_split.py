import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path(__file__).parent.parent / "data"
train_audio_dir = data_dir / "train_audio"
test_audio_dir = data_dir / "test_audio"

df = pd.read_csv(data_dir / "train_metadata_africa_top30.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.4,
    stratify=df["primary_label"],
    random_state=42,
)

label_list = sorted(df["primary_label"].unique().tolist())
other_cols = [c for c in df.columns if c not in ("filename", "primary_label", "secondary_labels")]

# move test audio files to flat data/test_audio/ (no species subfolder)
test_audio_dir.mkdir(exist_ok=True)
for filename in test_df["filename"]:
    src = train_audio_dir / filename
    dst = test_audio_dir / Path(filename).name
    if src.exists():
        shutil.move(src, dst)

# update filename column to flat name (stem.ogg, no subfolder)
test_df = test_df.copy()
test_df["filename"] = test_df["filename"].apply(lambda f: Path(f).name)

# filename first, primary/secondary last
ordered_cols = ["filename"] + other_cols + ["secondary_labels", "primary_label"]
train_df[ordered_cols].to_csv(data_dir / "train_split.csv", index=False)
test_df[ordered_cols].to_csv(data_dir / "test_split_labeled.csv", index=False)

# sample submission: filename stem as row_id, 30 species columns, all zeros
row_ids = test_df["filename"].apply(lambda f: Path(f).stem)
sample_sub = pd.DataFrame({"row_id": row_ids})
for label in label_list:
    sample_sub[label] = 0
sample_sub.to_csv(data_dir / "sample_submission_test.csv", index=False)

# answers: same format as sample submission but with ground truth 1s
answers = sample_sub.copy()
for _, row in test_df.iterrows():
    row_id = Path(row["filename"]).stem
    answers.loc[answers["row_id"] == row_id, row["primary_label"]] = 1
answers.to_csv(data_dir / "answers.csv", index=False)

# public test split: drop labels
test_cols = ["filename"] + other_cols
test_df[test_cols].to_csv(data_dir / "test_split.csv", index=False)

print(f"Train: {len(train_df)} files | Test: {len(test_df)} files")
print(f"Species in train: {train_df['primary_label'].nunique()} | test: {test_df['primary_label'].nunique()}")
print(f"Moved {len(test_df)} files to {test_audio_dir}")
print(f"Saved sample_submission_test.csv + answers.csv ({len(sample_sub)} rows x {len(label_list)} species)")
