# data_pipeline.py

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp


# ============================================================
# 1. BASE SINGLE CELL DATASET (FROM TRAIN/TEST FILE)
# ============================================================

class SingleCellDataset(Dataset):
    """
    Memory-safe dataset for sparse AnnData.
    Converts one row at a time to dense tensor.
    """

    def __init__(
        self,
        h5ad_path: str,
        cell_type_column: str = "cell_type",
        encoder: OneHotEncoder = None,
        fit_encoder: bool = False,
    ):
        self.adata = ad.read_h5ad(h5ad_path)

        if cell_type_column not in self.adata.obs:
            raise ValueError(f"{cell_type_column} not found in adata.obs")

        self.cell_types = (
            self.adata.obs[cell_type_column].values.reshape(-1, 1)
        )

        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False)

        if fit_encoder:
            self.targets = encoder.fit_transform(self.cell_types)
        else:
            self.targets = encoder.transform(self.cell_types)

        self.encoder = encoder
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x_sparse = self.adata.X[idx]

        if sp.issparse(x_sparse):
            x = torch.tensor(
                x_sparse.toarray().squeeze(),
                dtype=torch.float32,
            )
        else:
            x = torch.tensor(
                np.array(x_sparse),
                dtype=torch.float32,
            )

        t = self.targets[idx]

        return x, t


# ============================================================
# 2. MIXING STRATEGY BASE CLASS
# ============================================================

class MixingStrategy:
    def sample(self, dataset: SingleCellDataset):
        raise NotImplementedError


# ============================================================
# 3. PAPER 1:1 PAIR MIX (λ ~ Uniform[0,1])
# ============================================================

class PairMix(MixingStrategy):
    """
    Exact implementation from paper:
    - mix 2 cells
    - λ ~ Uniform(0,1)
    """

    def sample(self, dataset: SingleCellDataset):

        i = random.randrange(len(dataset))
        j = random.randrange(len(dataset))

        xi, ti = dataset[i]
        xj, tj = dataset[j]

        lam = torch.rand(1)

        x_mix = lam * xi + (1 - lam) * xj
        t_mix = lam * ti + (1 - lam) * tj

        return x_mix, t_mix


# ============================================================
# 4. EXTENSIBLE N-CELL MIX (future experiments)
# ============================================================

class NCellMix(MixingStrategy):
    """
    Mix N cells using Dirichlet weights.
    """

    def __init__(self, n_cells: int):
        self.n_cells = n_cells

    def sample(self, dataset: SingleCellDataset):

        indices = random.sample(range(len(dataset)), self.n_cells)

        xs = []
        ts = []

        for idx in indices:
            x, t = dataset[idx]
            xs.append(x)
            ts.append(t)

        xs = torch.stack(xs)
        ts = torch.stack(ts)

        weights = torch.distributions.Dirichlet(
            torch.ones(self.n_cells)
        ).sample()

        x_mix = torch.sum(weights.unsqueeze(1) * xs, dim=0)
        t_mix = torch.sum(weights.unsqueeze(1) * ts, dim=0)

        return x_mix, t_mix


# ============================================================
# 5. BULK MIX DATASET WRAPPER
# ============================================================

class BulkMixDataset(Dataset):
    """
    Wraps SingleCellDataset and applies mixing strategy.
    """

    def __init__(
        self,
        base_dataset: SingleCellDataset,
        mixing_strategy: MixingStrategy,
        n_samples: int,
    ):
        self.base_dataset = base_dataset
        self.mixing_strategy = mixing_strategy
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.mixing_strategy.sample(self.base_dataset)


# ============================================================
# 6. DATALOADER FACTORY
# ============================================================

def create_dataloaders(
    train_path: str,
    test_path: str,
    cell_type_column: str = "cell_type",
    batch_size: int = 32,
    mixing: str = "pair",
    n_mix_cells: int = 2,
    n_train_samples: int = 100000,
    n_test_samples: int = 20000,
):

    # Train dataset (fit encoder here)
    train_base = SingleCellDataset(
        train_path,
        cell_type_column,
        encoder=None,
        fit_encoder=True,
    )

    # Test dataset (reuse encoder)
    test_base = SingleCellDataset(
        test_path,
        cell_type_column,
        encoder=train_base.encoder,
        fit_encoder=False,
    )

    # Select mixing strategy
    if mixing == "pair":
        strategy = PairMix()
    elif mixing == "ncell":
        strategy = NCellMix(n_mix_cells)
    else:
        raise ValueError("Unknown mixing strategy")

    train_dataset = BulkMixDataset(
        train_base,
        strategy,
        n_train_samples,
    )

    test_dataset = BulkMixDataset(
        test_base,
        strategy,
        n_test_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader