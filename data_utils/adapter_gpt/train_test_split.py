import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

# =============================
# PARAMETRY
# =============================
INPUT_PATH = "/scratch/2370352/my-research/data/censusxgene/female_lung_neuron_vs_macrophage/all_norm.h5ad"
OUTPUT_TRAIN = "/scratch/2370352/my-research/data/censusxgene/female_lung_neuron_vs_macrophage/train_norm.h5ad"
OUTPUT_TEST = "/scratch/2370352/my-research/data/censusxgene/female_lung_neuron_vs_macrophage/test_norm.h5ad"

CELL_TYPE_COLUMN = "cell_type"

N_TOTAL = 40000          # ile komórek w sumie
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =============================

print("Loading full dataset...")
adata = ad.read_h5ad(INPUT_PATH)

print("Stratified subsampling...")

y = adata.obs[CELL_TYPE_COLUMN].values

# Najpierw losujemy N_TOTAL z zachowaniem proporcji klas
subset_idx, _ = train_test_split(
    np.arange(len(adata)),
    train_size=N_TOTAL,
    stratify=y,
    random_state=RANDOM_STATE,
)

adata_subset = adata[subset_idx].copy()

print(f"Subset size: {adata_subset.n_obs}")

print("Splitting subset into train/test...")

y_subset = adata_subset.obs[CELL_TYPE_COLUMN].values

train_idx, test_idx = train_test_split(
    np.arange(len(adata_subset)),
    test_size=TEST_SIZE,
    stratify=y_subset,
    random_state=RANDOM_STATE,
)

adata_train = adata_subset[train_idx].copy()
adata_test = adata_subset[test_idx].copy()

print(f"Train size: {adata_train.n_obs}")
print(f"Test size: {adata_test.n_obs}")

print("Saving...")
adata_train.write_h5ad(OUTPUT_TRAIN)
adata_test.write_h5ad(OUTPUT_TEST)

print("Done.")