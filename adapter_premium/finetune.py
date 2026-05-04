import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


import warnings
warnings. filterwarnings('ignore')

from utils import get_scgpt_model, binning

torch.manual_seed(42)
np.random.seed(42)

# bulk_data_path = '../data/GTEx/processed/all_gtex_processed.csv'
bulk_data_path = '../data/GTEx/processed/gene_tpm_v11_bladder_processed.csv'

df = pd.read_csv(bulk_data_path, index_col=0)
adata = ad.AnnData(X=df.values)
adata.obs_names = df.index
adata.var_names = df.columns

model_path = "../papers/scgpt/save/whole_human"
scgpt_model, vocab = get_scgpt_model(model_path, device='cpu', eval=False, do_mvc=False)

PAD_ID = vocab['<pad>']

# Filter genes that are in the vocabulary
genes_in_vocab = [g for g in adata.var_names if g in vocab]
adata = adata[:, genes_in_vocab].copy()

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# Tokenize gene names to IDs
gene_ids = vocab(adata.var_names.tolist())

adata = adata.copy()
X = np.asarray(adata.X)

adata.X = np.stack([binning(row, 51) for row in X])

def mask_expression(values, pad_mask, mask_ratio=0.15):
    prob_matrix = torch.full(values.shape, mask_ratio)

    # nie maskuj paddingu
    prob_matrix[pad_mask] = 0

    mask = torch.bernoulli(prob_matrix).bool()

    masked_values = values.clone()
    masked_values[mask] = 0

    return masked_values, mask


def masked_mse_loss(predicts, targets, mask):
    """
    Only compute loss for masked positions.
    predicts: output from model.expr_decoder
    targets: original binned values
    mask: boolean mask from step 1
    """
    loss = F.mse_loss(predicts[mask], targets[mask].float())
    return loss

class SCDataset(Dataset):
    def __init__(self, data_matrix, gene_ids, batch_labels):
        """
        data_matrix: Macierz ekspresji (po binningu), kształt [komórki, geny]
        gene_ids: Lista ID genów ze słownika (vocab), kształt [geny]
        batch_labels: ID partii dla każdej komórki, kształt [komórki]
        """
        self.data = torch.tensor(data_matrix, dtype=torch.float32)
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        self.batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Pobieramy wartości ekspresji dla danej komórki
        values = self.data[idx]

        nonzero_mask = values > 0
        if nonzero_mask.sum() == 0:
            # fallback – np. jeden dummy token
            return (
                self.gene_ids[:1],
                values[:1],
                self.batch_labels[idx]
            )
        src = self.gene_ids[nonzero_mask]
        values = values[nonzero_mask]
        
        return src, values, self.batch_labels[idx]


def collate_fn(batch):
    src_list, values_list, batch_labels = zip(*batch)

    max_len = max(len(x) for x in src_list)

    padded_src = []
    padded_values = []
    padding_masks = []

    for src, values in zip(src_list, values_list):
        pad_len = max_len - len(src)

        padded_src.append(
            torch.cat([src, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
        )

        padded_values.append(
            torch.cat([values, torch.zeros(pad_len)])
        )

        padding_masks.append(
            torch.cat([
                torch.zeros(len(src), dtype=torch.bool),
                torch.ones(pad_len, dtype=torch.bool)
            ])
        )

    return (
        torch.stack(padded_src),
        torch.stack(padded_values),
        torch.stack(padding_masks),
        torch.tensor(batch_labels)
    )


binned_counts = adata.X
vocab_gene_ids = gene_ids
batch_indices = np.zeros(adata.shape[0], dtype=np.int64)

dataset = SCDataset(binned_counts, vocab_gene_ids, batch_indices)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Przygotowanie urządzenia (GPU jeśli dostępne, inaczej CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scgpt_model.to(device)

scgpt_model.train()
optimizer = torch.optim.Adam(scgpt_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 10

for epoch in range(num_epochs):
    scgpt_model.train()
    total_loss = 0
    n_batches = 0

    for src, values, pad_mask, batch_idx in train_loader:
        optimizer.zero_grad()

        src = src.to(device)
        values = values.to(device)
        pad_mask = pad_mask.to(device)

        masked_values, bool_mask = mask_expression(values, pad_mask)

        results = scgpt_model(
            src=src,
            values=masked_values,
            src_key_padding_mask=pad_mask,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False
        )

        preds = results["mlm_output"]
        loss = masked_mse_loss(preds, values, bool_mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(scgpt_model.parameters(), 1.0) # chatgpt recommended this for transformers
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    save_path = f"checkpoints_finetune/todo_config/scgpt_epoch_{epoch+1}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": scgpt_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, save_path)
    
    scgpt_model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, values, pad_mask, _ in val_loader:
            src = src.to(device)
            values = values.to(device)
            pad_mask = pad_mask.to(device)

            masked_values, bool_mask = mask_expression(values, pad_mask)

            results = scgpt_model(
                src=src,
                values=masked_values,
                src_key_padding_mask=pad_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False
        )
            preds = results["mlm_output"]

            loss = masked_mse_loss(preds, values, bool_mask)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Val Loss: {val_loss:.4f}")