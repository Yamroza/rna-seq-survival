"""
PCA and UMAP visualization comparing multiple bulk RNA-seq datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from typing import Optional

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

DEFAULT_COLORS = [
    '#e63946', '#2563eb', '#16a34a', '#f59e0b', '#7c3aed',
    '#0891b2', '#db2777', '#65a30d', '#ea580c', '#111111',
    '#06b6d4', '#84cc16', '#f97316', '#8b5cf6', '#ec4899',
    '#14b8a6', '#eab308', '#6366f1', '#10b981', '#ef4444',
    '#0284c7', '#a16207', '#be185d', '#15803d', '#7e22ce',
    '#b45309', '#0f766e', '#1d4ed8', '#9f1239', '#3f6212',
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_dataset(path: str, log2_transform: bool = False,
                 index_col: str = 'Unnamed: 0') -> pd.DataFrame:
    """Load CSV and optionally apply log2(x+1) to gene columns."""
    df = pd.read_csv(path)
    if log2_transform:
        gene_cols = [c for c in df.columns if c != index_col]
        df[gene_cols] = np.log2(df[gene_cols].astype(float) + 1)
    return df


def match_genes_multi(datasets: list[pd.DataFrame],
                      index_col: str = 'Unnamed: 0') -> tuple[list[np.ndarray], list]:
    """
    Intersect genes across all datasets.
    Returns (list_of_X_matrices, shared_gene_names).
    """
    gene_sets = [set(c for c in df.columns if c != index_col) for df in datasets]
    shared    = sorted(set.intersection(*gene_sets))
    print(f"Shared genes across {len(datasets)} datasets: {len(shared):,}")
    matrices  = [df[shared].values.astype(np.float32) for df in datasets]
    return matrices, shared


def run_pca(X: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA, return (coords, explained_variance_ratio)."""
    pca    = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def run_umap(X: np.ndarray, n_neighbors: int = 30,
             min_dist: float = 0.3) -> Optional[np.ndarray]:
    """Fit UMAP, return 2D coords (or None if umap-learn unavailable)."""
    if not HAS_UMAP:
        return None
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=SEED,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter(ax, x, y, colors, all_labels, groups):
    color_map = dict(zip(groups, colors))
    for lbl in groups:
        idx = [i for i, l in enumerate(all_labels) if l == lbl]
        ax.scatter(x[idx], y[idx], c=color_map[lbl], s=8, alpha=0.6,
                   linewidths=0, rasterized=True)
    ax.spines[['top', 'right']].set_visible(False)


def plot_pca_umap(datasets: list[pd.DataFrame],
                  labels: list[str],
                  colors: Optional[list[str]] = None,
                  n_pca_components: int = 50,
                  index_col: str = 'Unnamed: 0',
                  save_prefix: str = 'comparison',
                  umap_n_neighbors: int = 30,
                  umap_min_dist: float = 0.3):
    """
    Compare any number of bulk RNA-seq datasets with PCA + UMAP.

    Parameters
    ----------
    datasets         : List of DataFrames (all in log2 TPM+1 space)
    labels           : Legend label per dataset
    colors           : Hex colors per dataset; defaults to built-in palette
    n_pca_components : PCA components to compute
    index_col        : Sample-ID column to exclude from gene matrix
    save_prefix      : Output filename prefix
    umap_n_neighbors : UMAP n_neighbors
    umap_min_dist    : UMAP min_dist
    """
    assert len(datasets) == len(labels), "datasets and labels must have equal length"
    plt.rcParams.update(STYLE)

    if colors is None:
        colors = DEFAULT_COLORS[:len(datasets)]

    # align genes & stack
    matrices, _ = match_genes_multi(datasets, index_col)
    X_all       = np.vstack(matrices)
    all_labels  = [lbl for lbl, X in zip(labels, matrices) for _ in range(len(X))]

    # PCA & UMAP
    print("Running PCA...")
    coords, var_exp = run_pca(X_all, n_components=n_pca_components)
    umap_coords     = run_umap(X_all, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)

    # layout
    n_panels = 4 if umap_coords is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.4),
                             gridspec_kw={'wspace': 0.38})

    # A: PC1 vs PC2
    _scatter(axes[0], coords[:, 0], coords[:, 1], colors, all_labels, labels)
    axes[0].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
    axes[0].set_title('(A)\u2002PCA — PC1 vs PC2', pad=5, loc='left')

    # B: PC1 vs PC3
    _scatter(axes[1], coords[:, 0], coords[:, 2], colors, all_labels, labels)
    axes[1].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC3 ({var_exp[2]*100:.1f}%)')
    axes[1].set_title('(B)\u2002PCA — PC1 vs PC3', pad=5, loc='left')

    # C: scree
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

    # D: UMAP
    if umap_coords is not None:
        _scatter(axes[3], umap_coords[:, 0], umap_coords[:, 1], colors, all_labels, labels)
        axes[3].set_xlabel('UMAP 1')
        axes[3].set_ylabel('UMAP 2')
        axes[3].set_title('(D)\u2002UMAP', pad=5, loc='left')

    # legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=6, label=lbl)
        for c, lbl in zip(colors, labels)
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(labels), 5),
               bbox_to_anchor=(0.5, -0.1), frameon=True,
               handletextpad=0.4, columnspacing=1.5, borderpad=0.6, fontsize=9)

    # save
    for ext in ('pdf', 'png'):
        path = f'sample_analysis_plots/{save_prefix}_pca_umap.{ext}'
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()

    # pairwise centroid distances (PC1-10)
    print("\nPairwise centroid L2 distances (PC1-10):")
    centroids = {
        lbl: coords[[i for i, l in enumerate(all_labels) if l == lbl], :10].mean(0)
        for lbl in labels
    }
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            dist = np.linalg.norm(centroids[a] - centroids[b])
            print(f"  {a} vs {b}: {dist:.3f}")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    data_dir = '../data/0_data_for_mlp_small_cohorts'
    cohorts  = ['TCGA-BLCA', 'TCGA-KIRC', 'TCGA-LUAD', 'TCGA-OV', 'TCGA-UCEC']

    gtex_dir = '../data/GTEx/processed' #/gene_tpm_v11_{tissue}_processed.csv'
    tissues = ['bladder', 'kidney_cortex', 'ovary']
    datasets = [
        load_dataset(f'{data_dir}/{c}.star_tpm.csv', log2_transform=False)
        for c in cohorts
    ]
    gtex_datasets = [
        load_dataset(f'{gtex_dir}/gene_tpm_v11_{t}_processed.csv', log2_transform=True)
        for t in tissues
    ]
    datasets.extend(gtex_datasets)
    cohorts.extend(tissues)

    tissues = ['bladder_processed_div_1,5', 'kidney_cortex_processed_div_1,5', 'ovary_processed_div_1,5']
    gtex_datasets = [
        load_dataset(f'{gtex_dir}/gene_tpm_v11_{t}.csv', log2_transform=True)
        for t in tissues
    ]
    datasets.extend(gtex_datasets)
    cohorts.extend(tissues)

    tissues = ['bladder_nn', 'kidney_cortex_nn', 'ovary_nn']
    gtex_datasets = [
        load_dataset(f'{gtex_dir}/gene_tpm_v11_{t}.csv', log2_transform=True)
        for t in tissues
    ]
    datasets.extend(gtex_datasets)
    cohorts.extend(tissues)

    plot_pca_umap(
        datasets    = datasets,
        labels      = cohorts,
        save_prefix = 'tcga_5cohorts_gtex_3_cohorts_raw_nn_div',
    )
