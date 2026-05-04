import pandas as pd
import glob
import os

data_dir = "../../data/GTEx/GTEx_tpm_per_tissue/processed"

files = glob.glob(os.path.join(data_dir, "*.csv"))

dfs = []

for f in files:
    print(f"Loading {f}")
    df = pd.read_csv(f, index_col=0)
    print(len(df.columns))
    dfs.append(df)

# łączenie po wierszach (sample)
merged_df = pd.concat(dfs, axis=0)

# zapis
merged_df.to_csv("merged_all_samples.csv")

print("Done. Shape:", merged_df.shape)