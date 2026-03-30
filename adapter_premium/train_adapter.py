import sys
import os
import argparse
import anndata as ad
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader, Subset

# Twoje moduły
sys.path.append(os.path.abspath(".."))
from utils import get_scgpt_model
from premium_datasets import scGPTDataset, collate_fn
from adapters import scGPTClassifier, train_epoch, eval_epoch, MLPClassifier
from mixers import NoMixer, LinearTwoCellMixer, MultiCellMixer


def main():
    # ── ARGUMENT PARSER ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="scGPT adapter training script")
    
    parser.add_argument("--model_path", type=str, default="../papers/scgpt/save/whole_human")
    parser.add_argument("--data_path",  type=str, default="data_new/train.h5ad")
    parser.add_argument("--save_path",  type=str, default="checkpoints/scgpt_adapter_checkpoint.pt")
    
    # Hiperparametry
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--dataset",    type=str, default='bulkDataset')
    parser.add_argument("--seq_length", type=int, default=2000)
    parser.add_argument('--hidden_dims',type=int, nargs='+', default=[512, 256])
    
    # Debugowanie / Szybkie testy
    parser.add_argument("--subset",     type=int,   default=None, help="Loader size for debbuging")
    
    args = parser.parse_args()

    # ── WANDB SETUP ───────────────────────────────────────────────────────────
    wandb.init(project="survival-prediction-with-adapters", config=vars(args))
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── DATA LOADING ──────────────────────────────────────────────────────────
    print(f"Loading model from {config.model_path}...")
    scgpt_model, vocab = get_scgpt_model(config.model_path, device='cpu') # scgpt ładowany na cpu, potem przerzucimy classifier
    
    print(f"Loading data from {config.data_path}...")
    adata = ad.read_h5ad(config.data_path)

    mixer_dict = {
        'scDataset': NoMixer(),
        'bulkDataset': LinearTwoCellMixer(),
        '3dataset' : MultiCellMixer()
    }

    dataset = scGPTDataset(adata, vocab, mixer=mixer_dict[config.dataset], max_genes=config.seq_length)
    print(dataset)
    if config.subset is not None:
        indices = np.random.choice(len(dataset), min(config.subset, len(dataset)), replace=False)
        dataset = Subset(dataset, indices)
        print(f"Using subset of {len(dataset)} samples.")

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False)

    # ── MODEL & OPTIMIZER ─────────────────────────────────────────────────────
    num_classes = len(adata.obs['cell_type'].unique()) # Zakładam, że tak masz etykiety

    head_model = MLPClassifier(512, num_classes, config.hidden_dims, config.dropout).to(device)
    model = scGPTClassifier(scgpt_model, head_model, num_classes=num_classes, dropout=config.dropout).to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    print(config.epochs)
    for epoch in range(config.epochs):
        train_res = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{config.epochs} | "
              f"Loss: {train_res['loss']:.4f} | Val Acc: {val_acc:.3f}")

        wandb.log({
            "adapter/epoch": epoch + 1,
            "adapter/train/loss": train_res["loss"],
            "adapter/train/acc_main": train_res["acc_main"],
            "adapter/val/loss": val_loss,
            "adapter/val/acc": val_acc,
            "adapter/lr": optimizer.param_groups[0]['lr']
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': vars(args)
            }
            torch.save(checkpoint, config.save_path)
            wandb.save(config.save_path)
            print(f"New best model saved (Acc: {val_acc:.4f})")

    wandb.finish()


if __name__ == "__main__":
    main()