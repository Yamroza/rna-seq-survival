# src/train.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# ------------------------------------------------------------------------------
# TRENING
# ------------------------------------------------------------------------------

# src/train.py

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device, epoch: int = 0):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d} [train]",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in pbar:
        gene_ids = batch["gene_ids"].to(device)
        values   = batch["values"].to(device)
        labels   = batch["labels"].to(device)

        optimizer.zero_grad()
        _, logits = model(gene_ids, values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # aktualizuj metryki
        batch_size     = labels.size(0)
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        correct       += preds.eq(labels).sum().item()
        total_samples += batch_size

        # live stats w pasku
        pbar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc":  f"{correct / total_samples:.3f}",
        })

    return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, epoch: int = 0):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d} [eval] ",
        unit="batch",
        dynamic_ncols=True,
        leave=False,   # eval znika po skończeniu, zostaje tylko train
    )

    for batch in pbar:
        gene_ids = batch["gene_ids"].to(device)
        values   = batch["values"].to(device)
        labels   = batch["labels"].to(device)

        _, logits = model(gene_ids, values)
        loss = criterion(logits, labels)

        batch_size     = labels.size(0)
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        correct       += preds.eq(labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc":  f"{correct / total_samples:.3f}",
        })

    return total_loss / total_samples, correct / total_samples


def evaluate_full(model, loader, device, id_to_type: dict):
    """
    Pełna ewaluacja: accuracy + classification_report + confusion matrix.
    Zwraca (accuracy, report_str, fig)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            gene_ids = batch["gene_ids"].to(device)
            values   = batch["values"].to(device)
            labels   = batch["labels"].to(device)

            _, logits = model(gene_ids, values)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = [id_to_type[i] for i in range(len(id_to_type))]
    acc    = (all_preds == all_labels).mean()
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Confusion matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  (acc={acc:.4f})")
    plt.tight_layout()

    return acc, report, fig


# ------------------------------------------------------------------------------
# CHECKPOINTOWANIE
# ------------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_acc, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":     epoch,
            "val_acc":   val_acc,
            "adapter":   model.adapter.state_dict(),   # tylko adapter
            "head":      model.head.state_dict(),       # + głowica
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"  💾 Checkpoint zapisany → {path}  (acc={val_acc:.4f})")


def load_checkpoint(model, optimizer, path: str, device):
    ckpt = torch.load(path, map_location=device)
    model.adapter.load_state_dict(ckpt["adapter"])
    model.head.load_state_dict(ckpt["head"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  ✓ Wczytano checkpoint z epoki {ckpt['epoch']}  (acc={ckpt['val_acc']:.4f})")
    return ckpt["epoch"], ckpt["val_acc"]