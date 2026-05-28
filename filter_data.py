import pandas as pd

df = pd.read_csv("data/train_metadata.csv")

df_africa = df.dropna(subset=["latitude", "longitude"])
df_africa = df_africa[
    (df_africa["latitude"] >= -35.0) & (df_africa["latitude"] <= 38.0) &
    (df_africa["longitude"] >= -18.0) & (df_africa["longitude"] <= 52.0)
]

top30_labels = df_africa["primary_label"].value_counts().head(30).index.tolist()
df_final = df_africa[df_africa["primary_label"].isin(top30_labels)]

out_path = "data/train_metadata_africa_top30.csv"
df_final.to_csv(out_path, index=False)
print(f"Written {len(df_final)} rows to {out_path}")
