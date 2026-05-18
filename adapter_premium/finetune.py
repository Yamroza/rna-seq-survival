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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


import warnings
warnings. filterwarnings('ignore')

from utils import get_scgpt_model, binning

torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="gene_tpm_v11_bladder_processed.csv",)
    parser.add_argument("--data_path", type=str, default="../data/GTEx/processed/")
    parser.add_argument("--gene_list_path", type=str, default="../data/hvg_genes_lists/TCGA-BRCA.star_tpm_hvg_2000.json")
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--gene_list", action="store_true", help="Choose genes from file")
    parser.add_argument("--lora", action="store_true", help="Finetune only subset of the model")
    parser.add_argument("--test", action="store_true", help="Ignore wandb logging in tests")

    return parser.parse_args()

args = parse_args()

# ── WANDB SETUP ───────────────────────────────────────────────────────────
dataset_name = os.path.splitext(args.dataset)[0]
if args.gene_list:
    gene_list_name = os.path.splitext(os.path.basename(args.gene_list_path))[0]
    run_name_gene_input = f"genes_from_list_{gene_list_name}"
else:
    run_name_gene_input = f"hvg{args.n_hvg}"

run_name = f"{dataset_name}_{run_name_gene_input}_bins{args.n_bins}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if args.lora:
    run_name = f"{dataset_name}_{run_name_gene_input}_bins{args.n_bins}_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not args.test:
    wandb.init(
        project="finetuning-scgpt-on-bulk",
        name=run_name,
        config=vars(args)
    )

    config = wandb.config

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

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # freeze base
        for p in self.base_layer.parameters():
            p.requires_grad = False

    # KEY: proxy attributes
    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.scaling * self.lora_B(self.lora_A(x))


def replace_linear_with_lora(model):
    replaced = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "self_attn" in name:

            parent = model
            parts = name.split(".")

            for p in parts[:-1]:
                parent = getattr(parent, p)

            setattr(parent, parts[-1], LoRALinear(module))
            replaced += 1

    print(f"[LoRA] replaced: {replaced}")

def count_params(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    print(f"Total params:      {total:,}")
    print(f"Trainable params:  {trainable:,}")
    print(f"% trainable:       {100 * trainable / total:.4f}%")


def inspect_lora(model):
    lora_count = 0
    linear_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"[LoRA OK] {name}")
            lora_count += 1

        if isinstance(module, nn.Linear):
            linear_count += 1

    print("\n=== SUMMARY ===")
    print(f"LoRA layers:   {lora_count}")
    print(f"Linear layers: {linear_count}")

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

if not args.gene_list:
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
    adata = adata[:, adata.var['highly_variable']]
else:
    with open(args.gene_list_path, 'r') as f:
        gene_list = json.load(f)
    gene_list = [g for g in gene_list if g in adata.var_names]
    adata = adata[:, gene_list]
    print(f"Using {len(gene_list)} genes from provided list")

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

binned_counts = adata.X
vocab_gene_ids = gene_ids
batch_indices = np.zeros(adata.shape[0], dtype=np.int64)

dataset = SCDataset(binned_counts, vocab_gene_ids, batch_indices)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# Przygotowanie urządzenia (GPU jeśli dostępne, inaczej CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scgpt_model.to(device)

if args.lora:
    for p in scgpt_model.parameters():
        p.requires_grad = False
    replace_linear_with_lora(scgpt_model)

    # Odblokuj decoder head — gradient MUSI przez niego przepłynąć,
    # żeby w ogóle dotrzeć do warstw LoRA w encoderze.
    for name, module in scgpt_model.named_modules():
        if "expr_decoder" in name or "value_encoder" in name:
            for p in module.parameters():
                p.requires_grad = True

    # debug
    for name, p in scgpt_model.named_parameters():
        if p.requires_grad:
            print("TRAINABLE:", name)
    inspect_lora(scgpt_model)
    count_params(scgpt_model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, scgpt_model.parameters()),
        lr=args.lr
    )
else:
    optimizer = torch.optim.Adam(scgpt_model.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = args.epochs

scgpt_model.train()
for epoch in range(num_epochs):
    scgpt_model.train()
    total_loss = 0
    total_loss_mae = 0
    n_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
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
        if n_batches % 10 == 0 and not args.test:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/loss_mae_step": loss_mae.item(),
                "train/lr_step": optimizer.param_groups[0]["lr"]
            })

    scheduler.step()

    avg_loss = total_loss / n_batches
    avg_loss_mae = total_loss_mae / n_batches
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

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

    if not args.test:
        wandb.log({
            "train/loss_mse": avg_loss,
            "train/loss_mae": avg_loss_mae,
            "val/loss_mse": val_loss,
            "val/loss_mae": val_loss_mae,
            "train/lr_epoch": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1
        })


