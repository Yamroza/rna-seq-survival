"""
inference.py - Embedowanie bulk RNA-seq przez wytrenowane MLP-A.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import MLP_A
from data import get_scgpt_embeddings
from plots import plot_embedding_variance


def embed_bulk_with_mlp_a(
    bulk_adata,
    model_dir: str,
    mlp_a_weights: str,
    output_dir: str,
    gene_col: str = "gene_name",
    hidden_dim: int = 128,
    n_classes: int = None,
    n_random_runs: int = 20,
    device: str = None,
    label_encoder_path: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    if n_classes is None:
        if label_encoder_path is not None:
            with open(label_encoder_path, "rb") as f:
                le = pickle.load(f)
            n_classes = len(le.classes_)
            print(f"[Inference] n_classes={n_classes}")
        else:
            raise ValueError("Podaj n_classes lub label_encoder_path")

    sample_emb = get_scgpt_embeddings(bulk_adata[:2], model_dir, gene_col=gene_col, batch_size=2)
    embed_dim = sample_emb.shape[1]

    model = MLP_A(input_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    model.load_state_dict(torch.load(mlp_a_weights, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"[Inference] {n_random_runs} losowan dla {len(bulk_adata)} probek...")

    all_run_embeddings = []
    for _ in tqdm(range(n_random_runs), desc="Losowania"):
        run_emb_scgpt = get_scgpt_embeddings(bulk_adata, model_dir, gene_col=gene_col, batch_size=32)
        run_emb_tensor = torch.tensor(run_emb_scgpt, dtype=torch.float32).to(device)
        with torch.no_grad():
            run_emb_mlpa = model.get_embedding(run_emb_tensor)
        all_run_embeddings.append(run_emb_mlpa.cpu().numpy())

    stacked = np.stack(all_run_embeddings, axis=0)
    mean_embeddings = stacked.mean(axis=0)

    cols = [f"emb{i+1}" for i in range(mean_embeddings.shape[1])]
    df = pd.DataFrame(mean_embeddings, index=bulk_adata.obs.index, columns=cols)
    out_path = os.path.join(output_dir, "bulk_embeddings_mlpa.csv")
    df.to_csv(out_path)
    print(f"[Inference] Zapisano: {out_path}, shape: {df.shape}")

    plot_embedding_variance(stacked, output_dir)
    return df
