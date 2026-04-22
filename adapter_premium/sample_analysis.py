"""
PCA and UMAP visualization comparing two bulk RNA-seq datasets (e.g. TCGA vs GTEx).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed — UMAP panel will be skipped.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42

STYLE = {
    'font.family':    'serif',
    'font.serif':     ['Times New Roman', 'DejaVu Serif'],
    'font.size':      9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.linewidth': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi':     300,
    'savefig.dpi':    300,
    'pdf.fonttype':   42,
    'ps.fonttype':    42,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def match_genes(data: pd.DataFrame, gtex_data: pd.DataFrame,
                index_col: str = 'Unnamed: 0') -> tuple[np.ndarray, np.ndarray, list]:
    """
    Find shared genes between two bulk DataFrames and return aligned matrices.
    Expects one non-gene index column (default 'Unnamed: 0').
    Returns (X_data, X_gtex, shared_genes).
    """
    data_genes = [c for c in data.columns if c != index_col]
    gtex_genes = [c for c in gtex_data.columns if c != index_col]
    shared     = sorted(set(data_genes) & set(gtex_genes))
    print(f"Shared genes: {len(shared):,}")

    X_data = data[shared].values.astype(np.float32)
    X_gtex = gtex_data[shared].values.astype(np.float32)
    return X_data, X_gtex, shared


def run_pca(X: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on X, return (coords, explained_variance_ratio)."""
    pca    = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def run_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.3) -> np.ndarray | None:
    """Fit UMAP on X, return 2D coords (or None if umap-learn unavailable)."""
    if not HAS_UMAP:
        return None
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=SEED,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter(ax, x, y, colors, labels, label_set):
    """Draw scatter per group with matching color."""
    color_map = dict(zip(label_set, colors))
    for lbl in label_set:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(x[idx], y[idx], c=color_map[lbl], s=8, alpha=0.6,
                   linewidths=0, rasterized=True, label=lbl)
    ax.spines[['top', 'right']].set_visible(False)


def plot_pca_umap(data: pd.DataFrame,
                  gtex_data: pd.DataFrame,
                  label_data: str = 'TCGA',
                  label_gtex: str = 'GTEx',
                  n_pca_components: int = 50,
                  index_col: str = 'Unnamed: 0',
                  save_prefix: str = 'comparison',
                  color_data: str = '#111111',
                  color_gtex: str = '#2563eb'):
    """
    Main entry point. Matches genes, runs PCA + UMAP, and produces a
    multi-panel figure saved as PDF and PNG.

    Parameters
    ----------
    data          : TCGA-style DataFrame (log2 TPM+1)
    gtex_data     : GTEx-style DataFrame (log2 TPM+1, already normalised)
    label_data    : Legend label for `data`
    label_gtex    : Legend label for `gtex_data`
    n_pca_components : Number of PCA components to compute
    index_col     : Non-gene column to exclude from matrices
    save_prefix   : Output filename prefix (no extension)
    color_data    : Scatter color for `data` points
    color_gtex    : Scatter color for `gtex_data` points
    """
    plt.rcParams.update(STYLE)

    # --- align genes & build joint matrix ---
    X_data, X_gtex, shared = match_genes(data, gtex_data, index_col)
    n_data, n_gtex = len(X_data), len(X_gtex)

    X_all  = np.vstack([X_data, X_gtex])
    labels = [label_data] * n_data + [label_gtex] * n_gtex
    colors = [color_data, color_gtex]
    groups = [label_data, label_gtex]

    # --- PCA ---
    print("Running PCA...")
    coords, var_exp = run_pca(X_all, n_components=n_pca_components)

    # --- UMAP ---
    umap_coords = run_umap(X_all)

    # --- figure layout ---
    n_panels = 4 if umap_coords is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.4),
                             gridspec_kw={'wspace': 0.38})

    # Panel A: PC1 vs PC2
    _scatter(axes[0], coords[:, 0], coords[:, 1], colors, labels, groups)
    axes[0].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
    axes[0].set_title('(A)\u2002PCA — PC1 vs PC2', pad=5, loc='left')

    # Panel B: PC1 vs PC3
    _scatter(axes[1], coords[:, 0], coords[:, 2], colors, labels, groups)
    axes[1].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC3 ({var_exp[2]*100:.1f}%)')
    axes[1].set_title('(B)\u2002PCA — PC1 vs PC3', pad=5, loc='left')

    # Panel C: cumulative explained variance
    ax = axes[2]
    cumvar = np.cumsum(var_exp) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, color='#333333', lw=1.5)
    for thresh, col in [(90, '0.55'), (95, '0.35')]:
        cross = int(np.searchsorted(cumvar, thresh))
        ax.axhline(thresh, color=col, lw=0.8, ls='--')
        if cross < len(cumvar):
            ax.axvline(cross + 1, color=col, lw=0.8, ls=':')
            ax.text(cross + 1.5, thresh - 4, f'{cross+1} PCs', fontsize=7.5, color=col)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title('(C)\u2002Scree plot (joint PCA)', pad=5, loc='left')
    ax.set_ylim(0, 101)
    ax.grid(True, axis='y', lw=0.4, ls=':', color='0.8')
    ax.spines[['top', 'right']].set_visible(False)

    # Panel D: UMAP
    if umap_coords is not None:
        _scatter(axes[3], umap_coords[:, 0], umap_coords[:, 1], colors, labels, groups)
        axes[3].set_xlabel('UMAP 1')
        axes[3].set_ylabel('UMAP 2')
        axes[3].set_title('(D)\u2002UMAP', pad=5, loc='left')

    # --- legend ---
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=6, label=lbl)
        for c, lbl in zip(colors, groups)
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.1), frameon=True,
               handletextpad=0.4, columnspacing=1.5, borderpad=0.6, fontsize=9)

    # --- save ---
    for ext in ('pdf', 'png'):
        path = f'sample_analysis_plots/{save_prefix}_pca_umap.{ext}'
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()

    # --- centroid distances (PC1-10) ---
    print(f"\nCentroid L2 distance between {label_data} and {label_gtex} (PC1-10):")
    idx_d = [i for i, l in enumerate(labels) if l == label_data]
    idx_g = [i for i, l in enumerate(labels) if l == label_gtex]
    dist  = np.linalg.norm(coords[idx_d, :10].mean(0) - coords[idx_g, :10].mean(0))
    print(f"  {dist:.3f}")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cohort = 'TCGA-KIRC'
    tissue = 'kidney_cortex'
    data      = pd.read_csv(f'../data/0_data_for_mlp_small_cohorts/{cohort}.star_tpm.csv')
    gtex_data = pd.read_csv(f'../data/GTEx/processed/gene_tpm_v11_{tissue}_processed.csv')
    
    common_genes = sorted(set(data.columns) & set(gtex_data.columns) - {'Unnamed: 0'})
    tcga_data = data[['Unnamed: 0'] + common_genes]
    gtex_data = gtex_data[['Unnamed: 0'] + common_genes]
    gtex_data[common_genes] = np.log2(gtex_data[common_genes].astype(float) + 1)

    plot_pca_umap(
        data      = tcga_data,
        gtex_data = gtex_data,
        label_data = cohort,
        label_gtex = f'GTEx {tissue}',
        save_prefix = f'tcga_gtex_{tissue}',
    )