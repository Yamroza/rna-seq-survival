"""
train.py
--------
Petla treningowa MLP-A (Wang et al. 2024, Path-GPTOmic).

Poprawny pipeline (zgodnie z papierem):
    xi, xj  -- surowe wektory ekspresji dwoch komorek
    mixed   = lambda * xi + (1 - lambda) * xj   <-- PRZED scGPT
    emb     = scGPT(mixed)                       <-- zamrozony scGPT
    out     = MLP_A(emb)                         <-- trenowany
    target  = lambda * one_hot_i + (1-lambda) * one_hot_j
    loss    = MSE(out, target)

WAZNE: mixup jest na surowych danych ekspresji, NIE na embeddingach!
"""

import os
import pickle
import numpy as np
import scipy.sparse
import anndata
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, "../")
import scgpt as scg

from model import MLP_A
from data import load_and_prepare_adata
from plots import plot_training_curves


def _mixup_and_embed(adata_i, adata_j, lam, model_dir, gene_col, batch_size, device):
    """
    Zmixuj surowe dane ekspresji PRZED podaniem do scGPT.

    Paper: "we simulate the bulk RNAseq by interpolating xi, xj with (lambda*xi + (1-lambda)*xj)"
    Czyli wejsciem do scGPT jest juz zmiksowany wektor ekspresji -- nie embeddingi.

    mixed = lam * xi + (1-lam) * xj  -->  scGPT  -->  embedding
    """
    def to_dense(x):
        if scipy.sparse.issparse(x):
            return x.toarray()
        return np.array(x)

    x_i = to_dense(adata_i.X).astype(np.float32)   # (B, n_genes)
    x_j = to_dense(adata_j.X).astype(np.float32)

    lam_np = lam.cpu().numpy().reshape(-1, 1)
    x_mixed = lam_np * x_i + (1 - lam_np) * x_j   # (B, n_genes)

    # Tymczasowy AnnData z mieszanymi danymi ekspresji
    adata_mixed = anndata.AnnData(
        X=scipy.sparse.csr_matrix(x_mixed),
        var=adata_i.var.copy(),
    )
    adata_mixed.obs_names = [f"mix_{k}" for k in range(len(x_mixed))]

    embed_adata = scg.tasks.embed_data(
        adata_mixed, model_dir, gene_col=gene_col, batch_size=batch_size,
    )
    return torch.tensor(embed_adata.obsm["X_scGPT"], dtype=torch.float32).to(device)


def train_mlp_a(
    sc_data_path: str,
    model_dir: str,
    output_dir: str,
    cell_type_col: str = "cell_type",
    gene_col: str = "gene_name",
    gene_info_path: str = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    scgpt_embed_batch: int = 32,
    device: str = None,
    debug_n: int = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    print(f"[Config] Device: {device}, Epochs: {n_epochs}, LR: {lr}")

    # ── Dane ──
    adata, gene_col = load_and_prepare_adata(
        sc_data_path,
        cell_type_col=cell_type_col,
        gene_info_path=gene_info_path,
        debug_n=debug_n,
    )

    # ── Enkoder typow komorek ──
    le = LabelEncoder()
    cell_type_ids = le.fit_transform(adata.obs[cell_type_col].values)
    n_classes = len(le.classes_)
    print(f"[Data] {n_classes} typow komorek: {list(le.classes_)}")

    # ── Ustal embed_dim przez jedna probe ──
    print("[scGPT] Sprawdzam embed_dim...")
    _tmp = scg.tasks.embed_data(adata[:2], model_dir, gene_col=gene_col, batch_size=2)
    embed_dim = _tmp.obsm["X_scGPT"].shape[1]
    print(f"[scGPT] embed_dim={embed_dim}")

    # ── Model ──
    model = MLP_A(input_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    print(f"[Model] Parametry: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    n_cells = len(adata)
    indices = np.arange(n_cells)
    history = {"epoch": [], "loss": [], "lr": []}
    best_loss = float("inf")

    print(f"\n{'='*50}\nSTART TRENINGU\n{'='*50}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        np.random.shuffle(indices)
        n_pairs = n_cells // 2
        pairs_i = indices[:n_pairs]
        pairs_j = indices[n_pairs : n_pairs * 2]

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            idx_i = pairs_i[start:end]
            idx_j = pairs_j[start:end]

            # Lambda ~ Uniform(0, 1) dla kazdej pary
            lam = torch.FloatTensor(len(idx_i)).uniform_(0, 1).to(device)

            # ── MIXUP NA SUROWYCH DANYCH → scGPT → embedding ──
            emb_mixed = _mixup_and_embed(
                adata[idx_i], adata[idx_j],
                lam, model_dir, gene_col, scgpt_embed_batch, device,
            )  # (B, embed_dim)

            # Soft labels: lam * one_hot_i + (1-lam) * one_hot_j
            lam_col = lam.unsqueeze(1)
            oh_i = torch.zeros(len(idx_i), n_classes, device=device)
            oh_j = torch.zeros(len(idx_j), n_classes, device=device)
            oh_i[torch.arange(len(idx_i)), torch.tensor(cell_type_ids[idx_i], dtype=torch.long)] = 1.0
            oh_j[torch.arange(len(idx_j)), torch.tensor(cell_type_ids[idx_j], dtype=torch.long)] = 1.0
            mixed_labels = lam_col * oh_i + (1 - lam_col) * oh_j

            # Forward + backward
            optimizer.zero_grad()
            loss = loss_fn(model(emb_mixed), mixed_labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = np.mean(epoch_losses)
        current_lr = scheduler.get_last_lr()[0]
        history["epoch"].append(epoch)
        history["loss"].append(mean_loss)
        history["lr"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {mean_loss:.6f} | LR: {current_lr:.2e}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "mlp_a_best.pt"))

    torch.save(model.state_dict(), os.path.join(output_dir, "mlp_a_last.pt"))
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\n[Done] Best loss: {best_loss:.6f}")
    print(f"[Done] Wagi: {output_dir}")

    plot_training_curves(history, output_dir)
    return model, history