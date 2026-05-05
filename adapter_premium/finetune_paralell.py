import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import wandb
import os
import argparse
from datetime import datetime
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import warnings
warnings. filterwarnings('ignore')

from utils import get_scgpt_model, binning

torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="gene_tpm_v11_bladder_processed.csv",)
    parser.add_argument("--data_path", type=str, default="../data/GTEx/processed/")
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mask_ratio", type=float, default=0.15)

    return parser.parse_args()

args = parse_args()

# ── WANDB SETUP ───────────────────────────────────────────────────────────
dataset_name = os.path.splitext(args.dataset)[0]
run_name = f"{dataset_name}_hvg{args.n_hvg}_bins{args.n_bins}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

is_main = int(os.environ.get("RANK", 0)) == 0
if is_main:
    wandb.init(
        project="finetuning-scgpt-on-bulk",
        name=run_name,
        config=vars(args)
    )

    config = wandb.config

# bulk_data_path = '../data/GTEx/processed/all_gtex_processed.csv'
bulk_data_path = os.path.join(args.data_path, args.dataset)

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

sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
adata = adata[:, adata.var['highly_variable']]

# Tokenize gene names to IDs
gene_ids = vocab(adata.var_names.tolist())

# save for other dataset
save_dir = f"checkpoints_finetune/{run_name}/"
os.makedirs(save_dir, exist_ok=True)

gene_list = adata.var_names.tolist()

with open(os.path.join(save_dir, "gene_set.json"), "w") as f:
    json.dump({
        "genes": gene_list,
        "n_hvg": args.n_hvg,
        "n_bins": args.n_bins
    }, f)

adata = adata.copy()
X = np.asarray(adata.X)

adata.X = np.stack([binning(row, args.n_bins) for row in X])

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

def mae(preds, targets, mask):
    return torch.mean(torch.abs(preds[mask] - targets[mask]))

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

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()


binned_counts = adata.X
vocab_gene_ids = gene_ids
batch_indices = np.zeros(adata.shape[0], dtype=np.int64)

# model
# Przygotowanie urządzenia (GPU jeśli dostępne, inaczej CPU)
local_rank = setup_ddp()
device = torch.device(f"cuda:{local_rank}")

scgpt_model.to(device)
scgpt_model = DDP(scgpt_model, device_ids=[local_rank], find_unused_parameters=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# scgpt_model.to(device)

scgpt_model.train()


dataset = SCDataset(binned_counts, vocab_gene_ids, batch_indices)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)


optimizer = torch.optim.Adam(scgpt_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = args.epochs

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    is_main = dist.get_rank() == 0

    scgpt_model.train()
    total_loss = 0
    total_loss_mae = 0
    n_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", disable=not is_main)
    for src, values, pad_mask, batch_idx in train_bar:
        optimizer.zero_grad()

        src = src.to(device)
        values = values.to(device)
        pad_mask = pad_mask.to(device)

        masked_values, bool_mask = mask_expression(values, pad_mask, mask_ratio=args.mask_ratio)

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
        loss_mae = mae(preds, values, bool_mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(scgpt_model.parameters(), 1.0) # chatgpt recommended this for transformers
        optimizer.step()

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()

        n_batches += 1
        if n_batches % 10 == 0:
            if is_main:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/loss_mae_step": loss_mae.item(),
                    "train/lr_step": optimizer.param_groups[0]["lr"]
                })

    scheduler.step()

    avg_loss = total_loss / n_batches
    avg_loss_mae = total_loss_mae / n_batches
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    if is_main:
        save_path = f"checkpoints_finetune/{run_name}/scgpt_epoch_{epoch+1}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": scgpt_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss
        }, save_path)
    
    scgpt_model.eval()
    val_loss = 0
    val_loss_mae = 0
    
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    with torch.no_grad():
        for src, values, pad_mask, _ in val_bar:
            src = src.to(device)
            values = values.to(device)
            pad_mask = pad_mask.to(device)

            masked_values, bool_mask = mask_expression(values, pad_mask, mask_ratio=args.mask_ratio)

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
            loss_mae = mae(preds, values, bool_mask)
            val_loss += loss.item()
            val_loss_mae += loss_mae.item()

    val_loss /= len(val_loader)
    val_loss_mae /= len(val_loader)
    print(f"Val Loss: {val_loss:.4f}")

    wandb.log({
        "train/loss_mse": avg_loss,
        "train/loss_mae": avg_loss_mae,
        "val/loss_mse": val_loss,
        "val/loss_mae": val_loss_mae,
        "train/lr_epoch": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1
    })

cleanup_ddp()
