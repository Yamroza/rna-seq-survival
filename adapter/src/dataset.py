# src/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scgpt.tokenizer.gene_tokenizer import GeneVocab


class SingleCellDataset(Dataset):
    """
    Dataset tokenizujący surową macierz ekspresji do formatu scGPT.

    Każdy item zwraca dict:
        gene_ids : LongTensor  [seq_len]   – indeksy tokenów genów z vocab
        values   : FloatTensor [seq_len]   – zbinowane wartości ekspresji
        labels   : LongTensor  []          – klasa komórki (int)
    """

# src/dataset.py  –  tylko __init__ do zastąpienia

    def __init__(
        self,
        adata,
        vocab: GeneVocab,
        gene_info_path: str = None,        # ← ścieżka do gene_info_table.csv
        max_seq_len: int = 1200,
        n_bins: int = 51,
        cell_type_key: str = "cell_type",
        pad_token: str = "<pad>",
        pad_value: int = -2,
        cls_token: str = "<cls>",
    ):
        self.vocab     = vocab
        self.max_seq_len  = max_seq_len
        self.n_bins       = n_bins
        self.pad_token_id = vocab[pad_token]
        self.pad_value    = pad_value
        self.cls_token_id = vocab[cls_token]

        # ------------------------------------------------------------------
        # 1. Zbuduj słownik ENSG → symbol (jeśli podano gene_info)
        # ------------------------------------------------------------------
        ensg_to_symbol = {}
        if gene_info_path is not None:
            import pandas as pd
            gene_info = pd.read_csv(gene_info_path)
            ensg_to_symbol = dict(
                zip(gene_info["ensembl_id"], gene_info["gene_name"])
            )

        # ------------------------------------------------------------------
        # 2. Zmapuj nazwy genów z adata na symbole
        # ------------------------------------------------------------------
        raw_names = (
            adata.var["feature_name"].tolist()
            if "feature_name" in adata.var.columns
            else adata.var_names.tolist()
        )

        mapped_names = [
            ensg_to_symbol.get(g, g)   # ENSG → symbol; jeśli brak mapowania, zostaw jak jest
            for g in raw_names
        ]

        # ------------------------------------------------------------------
        # 3. Filtruj do genów obecnych w vocab
        # ------------------------------------------------------------------
        mask_in_vocab = np.array([
            isinstance(g, str) and g in vocab   # odrzuć NaN i nieznane
            for g in mapped_names
        ])

        self.gene_ids_global = np.array([
            vocab[g] for g in np.array(mapped_names)[mask_in_vocab]
        ])

        if hasattr(adata.X, "toarray"):
            X_full = adata.X.toarray()
        else:
            X_full = np.array(adata.X)

        self.X = X_full[:, mask_in_vocab]

        print(
            f"Geny w datasecie : {len(raw_names)} | "
            f"Geny w vocab     : {mask_in_vocab.sum()} | "
            f"Odrzucono        : {(~mask_in_vocab).sum()}"
        )

        # ------------------------------------------------------------------
        # 4. Binning + labelki (bez zmian)
        # ------------------------------------------------------------------
        self.X_binned = self._binning(self.X, n_bins)

        cell_types = adata.obs[cell_type_key].values
        self.type_to_id = {t: i for i, t in enumerate(np.unique(cell_types))}
        self.labels = np.array([self.type_to_id[t] for t in cell_types])


    # ------------------------------------------------------------------
    @staticmethod
    def _binning(X: np.ndarray, n_bins: int) -> np.ndarray:
        """Dyskretyzuje wartości ekspresji do n_bins+1 klas (0 = nieekspresowany)."""
        binned = np.zeros_like(X, dtype=np.int64)
        # Progi liczone globalnie po kolumnach (per gen)
        for i in range(X.shape[1]):
            col = X[:, i]
            expressed = col[col > 0]
            if len(expressed) == 0:
                continue
            # percentyle równomiernie od 0 do 100
            quantiles = np.percentile(expressed, np.linspace(0, 100, n_bins))
            quantiles = np.unique(quantiles)  # usuń duplikaty przy małej wariancji
            binned[:, i] = np.where(col == 0, 0, np.digitize(col, quantiles))
        return binned

    # ------------------------------------------------------------------
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        expr = self.X_binned[idx]          # [n_genes_in_vocab]  int
        raw  = self.X[idx]                  # [n_genes_in_vocab]  float, do sortowania

        # ------------------------------------------------------------------
        # Wybierz top-K genów według surowej ekspresji (nie binningu)
        # ------------------------------------------------------------------
        expressed_mask = raw > 0
        expressed_idx  = np.where(expressed_mask)[0]

        if len(expressed_idx) > self.max_seq_len:
            # Top max_seq_len po wartości ekspresji
            top_k = np.argsort(raw[expressed_idx])[::-1][: self.max_seq_len]
            expressed_idx = expressed_idx[top_k]

        selected_gene_ids = self.gene_ids_global[expressed_idx]   # [k]
        selected_values   = expr[expressed_idx].astype(np.float32) # [k]

        # ------------------------------------------------------------------
        # Padding do max_seq_len + dodanie tokenu CLS na początku
        # ------------------------------------------------------------------
        pad_len = self.max_seq_len - len(expressed_idx)

        # CLS token na początku (embedding komórki)
        gene_ids = np.concatenate(
            [[self.cls_token_id], selected_gene_ids,
             [self.pad_token_id] * pad_len]
        ).astype(np.int64)   # [max_seq_len + 1]

        values = np.concatenate(
            [[0.0], selected_values,                              # CLS value = 0
             [self.pad_value] * pad_len]
        ).astype(np.float32)  # [max_seq_len + 1]

        return {
            "gene_ids": torch.tensor(gene_ids, dtype=torch.long),
            "values":   torch.tensor(values,   dtype=torch.float32),
            "labels":   torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ------------------------------------------------------------------------------

def get_dataloaders(
    adata_train,
    adata_test,
    vocab: GeneVocab,
    batch_size: int = 32,
    max_seq_len: int = 1200,
    n_bins: int = 51,
    cell_type_key: str = "cell_type",
    num_workers: int = 4,
):
    train_ds = SingleCellDataset(
        adata_train, vocab,
        gene_info_path="/scratch/2370352/my-research/data/gene_info_table.csv",
        max_seq_len=max_seq_len,
        n_bins=n_bins,
        cell_type_key=cell_type_key,
    )
    test_ds = SingleCellDataset(
        adata_test, vocab,
        gene_info_path="/scratch/2370352/my-research/data/gene_info_table.csv",
        max_seq_len=max_seq_len,
        n_bins=n_bins,
        cell_type_key=cell_type_key,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # input_dim = embsize scGPT (512 dla scGPT_human)
    num_classes = len(train_ds.type_to_id)
    return train_loader, test_loader, num_classes