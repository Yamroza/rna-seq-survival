"""
MLP-A Training Pipeline for Path-GPTOmic
=========================================
Zgodnie z Wang et al. 2024 (Path-GPTOmic).

Pipeline:
1. Wczytaj dane single-cell (cellxgene census lub własne h5ad)
2. Dla każdego kroku treningu: losuj 2 komórki, zrób mixup
3. Przepuść przez scGPT (zamrożone) → MLP-A
4. Regresja na mixed cell type target
5. Zapisz MLP-A weights

Użycie:
    python train_mlp_a.py --sc_data path/to/single_cell.h5ad \
                          --model_dir path/to/scgpt/whole_human \
                          --output_dir ./mlp_a_weights
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

sys.path.insert(0, "../")
import scgpt as scg


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────

class MixupSingleCellDataset(Dataset):
    """
    Dla każdego __getitem__ zwraca jedną parę (xi, xj) z losowym lambda.
    Mixup jest robiony w collate_fn / lub tutaj - robimy go w pętli treningu
    żeby łatwo mieć dostęp do scGPT.
    
    Tutaj dataset po prostu trzyma surowe dane + one-hot labels.
    """
    def __init__(self, adata, cell_type_col="cell_type"):
        self.adata = adata
        
        # Zakoduj typy komórek jako one-hot
        self.le = LabelEncoder()
        cell_types = adata.obs[cell_type_col].values
        self.cell_type_ids = self.le.fit_transform(cell_types)
        self.n_classes = len(self.le.classes_)
        print(f"[Dataset] {len(adata)} komórek, {self.n_classes} typów: {list(self.le.classes_)}")
        
    def __len__(self):
        return len(self.adata)
    
    def get_one_hot(self, idx):
        oh = np.zeros(self.n_classes, dtype=np.float32)
        oh[self.cell_type_ids[idx]] = 1.0
        return oh
    
    def __getitem__(self, idx):
        # Zwróć indeks - embeddingi robimy batchami przez scGPT
        return idx


# ─────────────────────────────────────────────
# 2. MODEL MLP-A
# ─────────────────────────────────────────────

class MLP_A(nn.Module):
    """
    3-layer MLP zgodnie z papierem.
    Input: scGPT embedding (512 dim domyślnie)
    Output: n_cell_types (do regresji mixed label)
    
    Wewnętrznie trzymamy też "feature extractor" (przed ostatnią warstwą)
    żeby wyciągać embeddingi dla bulk RNA.
    """
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=17):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def get_embedding(self, x):
        """Użyj tego dla bulk RNA-seq po treningu."""
        with torch.no_grad():
            return self.feature_extractor(x)


# ─────────────────────────────────────────────
# 3. FUNKCJA DO EMBEDOWANIA PRZEZ scGPT
# ─────────────────────────────────────────────

def get_scgpt_embeddings_batch(adata_subset, model_dir, gene_col="gene_name", batch_size=32):
    """
    Przepuść podzbiór adata przez scGPT i zwróć embeddingi (numpy).
    """
    embed_adata = scg.tasks.embed_data(
        adata_subset,
        model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
    )
    return embed_adata.obsm["X_scGPT"]  # (n_cells, 512)


# ─────────────────────────────────────────────
# 4. TRENING
# ─────────────────────────────────────────────

def train_mlp_a(
    sc_data_path: str,
    model_dir: str,
    output_dir: str,
    cell_type_col: str = "cell_type",
    gene_col: str = "gene_name",
    gene_info_path: str = None,   # ścieżka do gene_info_table.csv
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    scgpt_embed_batch: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    precompute_embeddings: bool = True,
    debug_n: int = None,   # jeśli podane, użyj tylko pierwszych N komórek
):
    """
    Główna funkcja treningowa.
    
    precompute_embeddings=True: Przed treningiem embeduje WSZYSTKIE komórki przez scGPT.
    Szybsze, ale wymaga dużo RAM. Ustaw False jeśli masz za mało pamięci.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Config] Device: {device}, Epochs: {n_epochs}, LR: {lr}")
    
    # ── Wczytaj dane ──
    print(f"[Data] Wczytuję: {sc_data_path}")
    adata = sc.read_h5ad(sc_data_path)
    print(f"[Data] Shape: {adata.shape}")

    # ── Debug: użyj tylko N komórek ──
    if debug_n is not None:
        adata = adata[:debug_n].copy()
        print(f"[DEBUG] Używam tylko {debug_n} komórek")

    # ── Mapowanie Ensembl ID → gene symbol ──
    if gene_info_path is not None:
        print(f"[Data] Mapuję geny z: {gene_info_path}")
        gene_info = pd.read_csv(gene_info_path, index_col=0)
        ensembl_to_symbol = dict(zip(gene_info["ensembl_id"], gene_info["gene_name"]))
        adata.var["gene_name"] = adata.var["feature_id"].map(ensembl_to_symbol)
        n_mapped = adata.var["gene_name"].notna().sum()
        print(f"[Data] Zmapowano: {n_mapped}/{len(adata.var)} genów")
        adata = adata[:, adata.var["gene_name"].notna()].copy()
        print(f"[Data] Po filtrowaniu: {adata.shape}")
        gene_col = "gene_name"

    # Upewnij się że masz cell_type
    assert cell_type_col in adata.obs.columns, \
        f"Brak kolumny '{cell_type_col}' w adata.obs. Dostępne: {list(adata.obs.columns)}"
    
    # Enkoder typów komórek
    le = LabelEncoder()
    cell_type_ids = le.fit_transform(adata.obs[cell_type_col].values)
    n_classes = len(le.classes_)
    print(f"[Data] {n_classes} typów komórek: {list(le.classes_)}")
    
    # ── Pre-compute embeddingi scGPT ──
    if precompute_embeddings:
        print("[scGPT] Pre-komputuję embeddingi dla wszystkich komórek...")
        all_embeddings = get_scgpt_embeddings_batch(
            adata, model_dir, gene_col=gene_col, batch_size=scgpt_embed_batch
        )
        all_embeddings = torch.tensor(all_embeddings, dtype=torch.float32)
        embed_dim = all_embeddings.shape[1]
        print(f"[scGPT] Embeddingi: {all_embeddings.shape}")
    else:
        # Sprawdź dim na jednej próbce
        sample_emb = get_scgpt_embeddings_batch(
            adata[:2], model_dir, gene_col=gene_col, batch_size=2
        )
        embed_dim = sample_emb.shape[1]
        all_embeddings = None
    
    # ── Model ──
    model = MLP_A(input_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    model = model.to(device)
    print(f"[Model] Parametry: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()  # Regresja na mixed soft labels
    
    n_cells = len(adata)
    indices = np.arange(n_cells)
    
    # ── Historia ──
    history = {"epoch": [], "loss": [], "lr": []}
    
    print(f"\n{'='*50}")
    print("START TRENINGU")
    print(f"{'='*50}")
    
    best_loss = float("inf")
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []
        
        # Shuffle
        np.random.shuffle(indices)
        
        # Podziel na batche par
        n_pairs = n_cells // 2
        pairs_i = indices[:n_pairs]
        pairs_j = indices[n_pairs:n_pairs * 2]
        
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            idx_i = pairs_i[start:end]
            idx_j = pairs_j[start:end]
            
            # ── Pobierz embeddingi ──
            if precompute_embeddings:
                emb_i = all_embeddings[idx_i].to(device)
                emb_j = all_embeddings[idx_j].to(device)
            else:
                # Na żywo przez scGPT (wolniej)
                emb_i = torch.tensor(
                    get_scgpt_embeddings_batch(adata[idx_i], model_dir, gene_col, batch_size=scgpt_embed_batch),
                    dtype=torch.float32
                ).to(device)
                emb_j = torch.tensor(
                    get_scgpt_embeddings_batch(adata[idx_j], model_dir, gene_col, batch_size=scgpt_embed_batch),
                    dtype=torch.float32
                ).to(device)
            
            # ── Mixup ──
            # Lambda z rozkładu jednostajnego [0, 1]
            lam = torch.FloatTensor(len(idx_i)).uniform_(0, 1).to(device)
            lam_expand = lam.unsqueeze(1)  # (B, 1)
            
            # Interpolacja embeddingów
            mixed_emb = lam_expand * emb_i + (1 - lam_expand) * emb_j  # (B, 512)
            
            # ── Soft labels (one-hot zmiksowane) ──
            # Pobierz one-hot dla i i j
            oh_i = torch.zeros(len(idx_i), n_classes, device=device)
            oh_j = torch.zeros(len(idx_j), n_classes, device=device)
            oh_i[torch.arange(len(idx_i)), torch.tensor(cell_type_ids[idx_i], dtype=torch.long)] = 1.0
            oh_j[torch.arange(len(idx_j)), torch.tensor(cell_type_ids[idx_j], dtype=torch.long)] = 1.0
            
            mixed_labels = lam_expand * oh_i + (1 - lam_expand) * oh_j  # (B, n_classes)
            
            # ── Forward + Loss ──
            optimizer.zero_grad()
            logits = model(mixed_emb)  # (B, n_classes)
            
            loss = loss_fn(logits, mixed_labels)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        mean_loss = np.mean(epoch_losses)
        current_lr = scheduler.get_last_lr()[0]
        
        history["epoch"].append(epoch)
        history["loss"].append(mean_loss)
        history["lr"].append(current_lr)
        
        # ── Logging ──
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {mean_loss:.6f} | LR: {current_lr:.2e}")
        
        # ── Zapisz najlepszy model ──
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "mlp_a_best.pt"))
    
    # Zapisz ostatni
    torch.save(model.state_dict(), os.path.join(output_dir, "mlp_a_last.pt"))
    
    # Zapisz label encoder
    import pickle
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    
    print(f"\n[Done] Best loss: {best_loss:.6f}")
    print(f"[Done] Wagi zapisane w: {output_dir}")
    
    # ── Wykresy ──
    plot_training_curves(history, output_dir)
    
    return model, history


# ─────────────────────────────────────────────
# 5. WYKRESY
# ─────────────────────────────────────────────

def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["epoch"], history["loss"], color="#e74c3c", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss (Mixup Regression)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    
    # Learning rate
    axes[1].plot(history["epoch"], history["lr"], color="#3498db", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("LR Schedule (Cosine Annealing)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Wykres zapisany: {path}")


# ─────────────────────────────────────────────
# 6. INFERENCE - embeddingi dla bulk RNA-seq
# ─────────────────────────────────────────────

def embed_bulk_with_mlp_a(
    bulk_adata,
    model_dir: str,
    mlp_a_weights: str,
    output_dir: str,
    gene_col: str = "gene_name",
    hidden_dim: int = 128,
    n_classes: int = 17,
    n_random_runs: int = 20,   # Twój MoE trick - ile losowań
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Dla bulk RNA-seq:
    1. Losuje n_random_runs razy podzbiór genów (max_length=1200)
    2. Dla każdego losowania: scGPT → MLP-A → embedding
    3. Uśrednia (mean pooling po wszystkich losowaniach)
    
    Zwraca DataFrame z embeddingami.
    """
    # Wczytaj MLP-A
    # Najpierw musimy wiedzieć embed_dim - pobierz jedną próbkę
    sample_emb = get_scgpt_embeddings_batch(
        bulk_adata[:2], model_dir, gene_col=gene_col, batch_size=2
    )
    embed_dim = sample_emb.shape[1]
    
    model = MLP_A(input_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    model.load_state_dict(torch.load(mlp_a_weights, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"[Inference] {n_random_runs} losowań dla {len(bulk_adata)} próbek...")
    
    all_run_embeddings = []  # lista (n_samples, hidden_dim)
    
    for run in tqdm(range(n_random_runs), desc="Losowania"):
        # Losuj 1200 genów spośród niezerowych (jak w Twoim oryginalnym kodzie)
        # scGPT sam obsłuży max_length wewnętrznie, ale możemy też ręcznie
        run_emb_scgpt = get_scgpt_embeddings_batch(
            bulk_adata, model_dir, gene_col=gene_col, batch_size=32
        )
        
        run_emb_tensor = torch.tensor(run_emb_scgpt, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            run_emb_mlpa = model.get_embedding(run_emb_tensor)  # (n_samples, hidden_dim)
        
        all_run_embeddings.append(run_emb_mlpa.cpu().numpy())
    
    # Mean pooling po losowaniach: (n_runs, n_samples, hidden_dim) → (n_samples, hidden_dim)
    stacked = np.stack(all_run_embeddings, axis=0)  # (n_runs, n_samples, hidden_dim)
    mean_embeddings = stacked.mean(axis=0)           # (n_samples, hidden_dim)
    
    # Zapisz
    cols = [f"emb{i+1}" for i in range(mean_embeddings.shape[1])]
    df = pd.DataFrame(mean_embeddings, index=bulk_adata.obs.index, columns=cols)
    
    out_path = os.path.join(output_dir, "bulk_embeddings_mlpa.csv")
    df.to_csv(out_path)
    print(f"[Inference] Embeddingi zapisane: {out_path}, shape: {df.shape}")
    
    # Wykres wariancji między losowaniami
    plot_embedding_variance(stacked, output_dir)
    
    return df


def plot_embedding_variance(stacked_embeddings, output_dir):
    """
    Sprawdź jak stabilne są embeddingi po mean poolingu.
    stacked_embeddings: (n_runs, n_samples, hidden_dim)
    """
    # Std po run-ach dla każdej próbki i wymiaru
    std_per_sample = stacked_embeddings.std(axis=0).mean(axis=1)  # (n_samples,)
    
    # Jak zmienia się stabilność wraz z liczbą losowań
    n_runs = stacked_embeddings.shape[0]
    mean_std_over_runs = []
    for k in range(1, n_runs + 1):
        subset_mean = stacked_embeddings[:k].mean(axis=0)
        # Porównaj z full mean
        diff = np.abs(subset_mean - stacked_embeddings.mean(axis=0)).mean()
        mean_std_over_runs.append(diff)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(std_per_sample, bins=30, color="#2ecc71", edgecolor="white")
    axes[0].set_xlabel("Średnie std embeddingów (po wymiarach)")
    axes[0].set_ylabel("Liczba próbek")
    axes[0].set_title("Wariancja embeddingów między losowaniami")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, n_runs + 1), mean_std_over_runs, color="#9b59b6", linewidth=2, marker="o")
    axes[1].set_xlabel("Liczba losowań (k)")
    axes[1].set_ylabel("Różnica od pełnego mean")
    axes[1].set_title("Stabilność embeddingów vs liczba losowań")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "embedding_variance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Wykres wariancji: {path}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP-A for Path-GPTOmic")
    parser.add_argument("--sc_data", type=str, required=True,
                        help="Ścieżka do danych single-cell (.h5ad)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Ścieżka do wag scGPT (np. save/whole_human)")
    parser.add_argument("--output_dir", type=str, default="./mlp_a_output",
                        help="Gdzie zapisać wagi i wykresy")
    parser.add_argument("--cell_type_col", type=str, default="cell_type",
                        help="Nazwa kolumny z typem komórki w adata.obs")
    parser.add_argument("--gene_col", type=str, default="gene_name",
                        help="Nazwa kolumny z nazwą genu w adata.var")
    parser.add_argument("--gene_info_path", type=str, default=None,
                        help="Ścieżka do gene_info_table.csv (Ensembl→symbol mapping)")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Rozmiar ukrytych warstw MLP-A (paper: 128)")
    parser.add_argument("--no_precompute", action="store_true",
                        help="Nie pre-komputuj embeddingów (oszczędność RAM)")
    parser.add_argument("--debug_n", type=int, default=None,
                        help="DEBUG: użyj tylko pierwszych N komórek (np. 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model, history = train_mlp_a(
        sc_data_path=args.sc_data,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        cell_type_col=args.cell_type_col,
        gene_col=args.gene_col,
        gene_info_path=args.gene_info_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        precompute_embeddings=not args.no_precompute,
        debug_n=args.debug_n,
    )
    
    print("\nAby teraz zembedować bulk RNA-seq, użyj:")
    print("""
from train_mlp_a import embed_bulk_with_mlp_a
import scanpy as sc

bulk_adata = sc.read_h5ad("path/to/bulk.h5ad")
df = embed_bulk_with_mlp_a(
    bulk_adata,
    model_dir="path/to/scgpt/whole_human",
    mlp_a_weights="./mlp_a_output/mlp_a_best.pt",
    output_dir="./mlp_a_output",
    n_random_runs=20,   # Twój MoE trick
)
""")