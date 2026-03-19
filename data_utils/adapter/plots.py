"""
plots.py
--------
Wykresy do treningu MLP-A i analizy embeddingów.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, output_dir: str):
    """Loss i LR schedule w trakcie treningu."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["loss"], color="#e74c3c", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss (Mixup Regression)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].plot(history["epoch"], history["lr"], color="#3498db", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("LR Schedule (Cosine Annealing)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Krzywe treningu: {path}")


def plot_embedding_variance(stacked_embeddings: np.ndarray, output_dir: str):
    """
    Sprawdź stabilność embeddingów po mean poolingu wielu losowań.
    stacked_embeddings: (n_runs, n_samples, hidden_dim)
    """
    std_per_sample = stacked_embeddings.std(axis=0).mean(axis=1)

    n_runs = stacked_embeddings.shape[0]
    full_mean = stacked_embeddings.mean(axis=0)
    mean_diff_over_runs = [
        np.abs(stacked_embeddings[:k].mean(axis=0) - full_mean).mean()
        for k in range(1, n_runs + 1)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(std_per_sample, bins=30, color="#2ecc71", edgecolor="white")
    axes[0].set_xlabel("Srednie std embeddingow (po wymiarach)")
    axes[0].set_ylabel("Liczba probek")
    axes[0].set_title("Wariancja embeddинgow miedzy losowaniami")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, n_runs + 1), mean_diff_over_runs,
                 color="#9b59b6", linewidth=2, marker="o")
    axes[1].set_xlabel("Liczba losowan (k)")
    axes[1].set_ylabel("Roznica od pelnego mean")
    axes[1].set_title("Stabilnosc embeddинgow vs liczba losowan")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "embedding_variance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Wariancja embeddинgow: {path}")
