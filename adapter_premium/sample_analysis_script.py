"""
PCA and UMAP visualization comparing multiple bulk RNA-seq datasets.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from typing import Optional

import umap
import anndata as ad

from pathlib import Path
from datetime import datetime
import scipy.sparse
from tqdm import tqdm

from mixers import DynamicDonorMixer, LinearTwoCellMixer

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

def load_dataset(path: str,
                 log2_transform: bool = False,
                 index_col: str = 'case_id') -> pd.DataFrame:
    """
    Supports:
    - CSV: first column = sample ID, rest = features
    - JSONL: {id, embedding}
    """

    ext = Path(path).suffix.lower()

    # ---------------- CSV ----------------
    if ext == ".csv":
        df = pd.read_csv(path)

        # pierwsza kolumna = ID (nie zawsze nazywa się Unnamed: 0)
        first_col = df.columns[0]

        df = df.rename(columns={first_col: "case_id"})

        # upewnij się, że wszystkie feature kolumny są float
        feature_cols = [c for c in df.columns if c != "case_id"]
        df[feature_cols] = df[feature_cols].astype(np.float32)

        return df

    # ---------------- JSONL ----------------
    elif ext == ".json":
        df = pd.read_json(path, lines=True)

        df = pd.concat([
            df[['id']].rename(columns={'id': 'case_id'}),
            pd.DataFrame(df['embedding'].tolist())
        ], axis=1)

        # nazwij embeddingi spójnie z CSV (emb0, emb1, ...)
        emb_cols = [c for c in df.columns if c != "case_id"]
        df.columns = ["case_id"] + [f"emb{i}" for i in range(len(emb_cols))]

        return df

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def generate_pseudobulk_from_adata(
    adata,
    n_samples: int = 500,
    n_cells: int = 50,
    random_seed: int = 42,
    mixer_type: str = 'donor_mixer',
    donor_col: str = "donor_id",
    tissue_col: str = "tissue_general"
) -> pd.DataFrame:
    """
    Generuje pseudobulk bezpośrednio z adata przy użyciu DynamicDonorMixer,
    bez użycia scGPTDataset, słowników czy DataLoaderów.
    """
    rng = np.random.default_rng(random_seed)
    
    # 1. Inicjalizacja miksera
    # Przekazujemy 'adata' jako atrapę datasetu, bo mikser potrzebuje dostępu do .X
    if mixer_type == 'donor_mixer':
        mixer = DynamicDonorMixer(adata, donor_col=donor_col, tissue_col=tissue_col, n_cells=n_cells)
    else:
        mixer = LinearTwoCellMixer()

    # Tworzymy lekki obiekt zastępujący scGPTDataset, który mikser akceptuje
    class SimpleDataset:
        def __init__(self, adata):
            self.X = adata.X
            self.labels = np.zeros(adata.n_obs) # Atrapa etykiet, mikser ich wymaga w returnie

    dataset_proxy = SimpleDataset(adata)
    
    # 2. Losowanie indeksów bazowych dla pseudobulków
    n_samples = min(n_samples, adata.n_obs)
    base_indices = rng.choice(adata.n_obs, size=n_samples, replace=False)
    
    pseudobulks = []
    print(f"Generating {n_samples} pseudobulk samples (n_cells={n_cells})...")
    
    for idx in tqdm(base_indices):
        # Wywołujemy Twój mikser bezpośrednio
        # Zwraca: mixed_row, labels, lambdas
        mixed_row, _, _ = mixer(idx, dataset_proxy)
        
        # Jeśli mixed_row jest rzadką macierzą (sparse), konwertujemy do gęstej tablicy
        if scipy.sparse.issparse(mixed_row):
            mixed_row = mixed_row.toarray().flatten()
            
        pseudobulks.append(mixed_row)
    
    # 3. Budowa wynikowego DataFrame
    X_final = np.vstack(pseudobulks)
    
    # Używamy nazw genów bezpośrednio z adata.var_names
    gene_names = adata.var['feature_name'].values.astype(str)
    df = pd.DataFrame(X_final, columns=gene_names)
    
    # Dodanie kolumny ID
    df.insert(0, 'Unnamed: 0', [f'pseudo_{n_cells}_cell_{i}' for i in range(len(df))])
    
    # Usunięcie kolumn, które są same zerami (opcjonalne, ale zalecane przed PCA)
    non_zero_cols = df.columns[(df != 0).any(axis=0)]
    df = df[non_zero_cols]
    
    print(f"Final pseudobulk shape: {df.shape}")
    return df


def match_genes_multi(datasets, index_col="case_id"):

    def detect_index_column(df):
        if index_col in df.columns:
            return index_col
        if "case_id" in df.columns:
            return "case_id"
        if "Unnamed: 0" in df.columns:
            return "Unnamed: 0"
        return df.columns[0]

    def clean(col):
        return str(col).split(".")[0].strip()

    idx_cols = [detect_index_column(df) for df in datasets]

    gene_sets = []
    cleaned_cols_all = []

    print("Step 1: Identifying shared features...")

    for df, idx in zip(datasets, idx_cols):
        cleaned_cols = []
        genes = set()

        for c in df.columns:
            if c == idx:
                cleaned_cols.append(c)
                continue
            cc = clean(c)
            cleaned_cols.append(cc)
            genes.add(cc)

        cleaned_cols_all.append(cleaned_cols)
        gene_sets.append(genes)

    shared_genes = sorted(set.intersection(*gene_sets)) if gene_sets else []

    print(f"  -> Found {len(shared_genes):,} shared features")

    if len(shared_genes) == 0:
        raise ValueError("No shared features between datasets")

    aligned = []

    print("Step 2: Aligning datasets...")

    for i, (df, idx) in enumerate(zip(datasets, idx_cols)):
        temp = df.copy()
        temp.columns = cleaned_cols_all[i]

        subset = temp.set_index(idx)
        subset = subset.reindex(columns=shared_genes, fill_value=0)

        aligned.append(subset.values.astype(np.float32))

    return aligned, shared_genes

def run_pca(X: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA, return (coords, explained_variance_ratio)."""
    pca    = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def run_umap(X: np.ndarray, n_neighbors: int = 30,
             min_dist: float = 0.3) -> Optional[np.ndarray]:
    """Fit UMAP, return 2D coords (or None if umap-learn unavailable)."""
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
        ax.scatter(x[idx], y[idx], c=color_map[lbl], s=8, alpha=0.2,
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
    
    assert len(datasets) == len(labels), "datasets and labels must have equal length"
    plt.rcParams.update(STYLE)

    # --- LOGIKA GRUPOWANIA ---
    display_labels = []
    if GROUP_BY_SOURCE:
        for lbl in labels:
            if "TCGA" in lbl: display_labels.append("TCGA")
            elif "GTEx" in lbl: display_labels.append("GTEx")
            else: display_labels.append(lbl) # Dla Pseudo zostawia oryginał
    else:
        display_labels = labels

    # Unikalne grupy do legendy i mapowania kolorów
    unique_groups = []
    for dl in display_labels:
        if dl not in unique_groups:
            unique_groups.append(dl)

    if colors is None:
        colors = DEFAULT_COLORS[:len(unique_groups)]
    
    color_map = dict(zip(unique_groups, colors))
    # -------------------------

    # align genes & stack
    matrices, _ = match_genes_multi(datasets, index_col)
    X_all       = np.vstack(matrices)
    
    # Tworzymy listę labeli dla każdego punktu (używając display_labels)
    all_point_labels = []
    for lbl, mat in zip(display_labels, matrices):
        all_point_labels.extend([lbl] * mat.shape[0])
    print(f"DEBUG: TCGA points count: {all_point_labels.count('TCGA')}")

    # PCA & UMAP
    print("Running PCA...")
    coords, var_exp = run_pca(X_all, n_components=n_pca_components)
    umap_coords     = run_umap(X_all, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)

    # layout (3 lub 4 panele)
    n_panels = 4 if umap_coords is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.4),
                             gridspec_kw={'wspace': 0.38})

    # Funkcja pomocnicza do rysowania z użyciem color_map
    def _grouped_scatter(ax, x, y, title):
        for grp in unique_groups:
            # Wybieramy indeksy punktów należących do danej grupy
            idx = [i for i, l in enumerate(all_point_labels) if l == grp]
            ax.scatter(x[idx], y[idx], c=color_map[grp], s=8, alpha=0.6,
                       linewidths=0, rasterized=True, label=grp)
        ax.set_title(title, pad=5, loc='left')
        ax.spines[['top', 'right']].set_visible(False)

    # Panele
    _grouped_scatter(axes[0], coords[:, 0], coords[:, 1], '(A)\u2002PCA — PC1 vs PC2')
    axes[0].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')

    _grouped_scatter(axes[1], coords[:, 0], coords[:, 2], '(B)\u2002PCA — PC1 vs PC3')
    axes[1].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC3 ({var_exp[2]*100:.1f}%)')

    # C: Scree plot (bez zmian)
    ax = axes[2]
    cumvar = np.cumsum(var_exp) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, color='#333333', lw=1.5)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title('(C)\u2002Scree plot', pad=5, loc='left')
    ax.set_ylim(0, 101)
    ax.spines[['top', 'right']].set_visible(False)

    # D: UMAP
    if umap_coords is not None:
        _grouped_scatter(axes[3], umap_coords[:, 0], umap_coords[:, 1], '(D)\u2002UMAP')
        axes[3].set_xlabel('UMAP 1')
        axes[3].set_ylabel('UMAP 2')

    # Legend (tylko unikalne grupy)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[grp],
               markersize=6, label=grp)
        for grp in unique_groups
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(unique_groups), 5),
               bbox_to_anchor=(0.5, -0.2), frameon=True,
               handletextpad=0.4, columnspacing=1.5, borderpad=0.6, fontsize=9)

    # save
    outdir = Path('sample_analysis_plots')
    outdir.mkdir(exist_ok=True)

    save_metadata(
        save_prefix,
        labels,
        outdir=outdir,
    )

    for ext in ('pdf', 'png'):
        path = outdir / f'{ext}/{save_prefix}_pca_umap.{ext}'
        plt.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')

    # pairwise centroid distances (PC1-10)
    print("\nPairwise centroid L2 distances (PC1-10):")
    
    # Używamy unique_groups, aby liczyć dystanse między tym co w legendzie
    # i all_point_labels, która jest teraz poprawnie zdefiniowana wewnątrz funkcji
    centroids = {
        grp: coords[[i for i, l in enumerate(all_point_labels) if l == grp], :10].mean(0)
        for grp in unique_groups
    }
    
    for i, a in enumerate(unique_groups):
        for b in unique_groups[i+1:]:
            dist = np.linalg.norm(centroids[a] - centroids[b])
            print(f"  {a} vs {b}: {dist:.3f}")

def build_save_prefix(labels: list[str], extra_tag: str = '', n_samples: int = 0, mixer: str = 'donor_mixer'):
    """
    Auto filename describing datasets.
    """
    short_labels = []
    for lbl in labels:
        cleaned = (
            lbl.replace('TCGA-', '')
               .replace('gene_tpm_v11_', '')
               .replace('processed', 'proc')
               .replace('kidney_cortex', 'kidney')
               .replace(' ', '_')
        )
        short_labels.append(cleaned)
    joined = '__'.join(short_labels)
    if extra_tag and n_samples > 0:
        return f'{extra_tag}_{joined}_{mixer}_n_samples_{n_samples}'
    elif extra_tag:
        return f'{extra_tag}_{joined}'
    return joined


def save_metadata(save_prefix, labels, outdir='sample_analysis_plots'):
    Path(outdir).mkdir(exist_ok=True)
    meta_path = Path(outdir) / f'metadata/{save_prefix}_metadata.txt'
    with open(meta_path, 'w') as f:
        f.write('Datasets used:\n')
        for lbl in labels:
            f.write(f' - {lbl}\n')
        # f.write(f'\nShared genes: {shared_genes_count}\n')
    print(f'Saved metadata: {meta_path}')

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
def print_dataset_stats(
    datasets: list[pd.DataFrame],
    labels: list[str],
    index_col: str = 'Unnamed: 0'
):
    """
    Basic per-dataset statistics for RNA-seq matrices.
    """

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for df, label in zip(datasets, labels):

        gene_cols = [c for c in df.columns if c != index_col]

        X = df[gene_cols].values.astype(np.float32)

        # per-sample stats
        sample_sums = X.sum(axis=1)
        sample_means = X.mean(axis=1)
        sample_nonzero = (X > 0).sum(axis=1)

        # global stats
        zero_fraction = (X == 0).mean()

        print(f"\nDataset: {label}")
        print("-" * 60)
        print(f"Shape:                  {X.shape}")
        print(f"N samples:              {X.shape[0]:,}")
        print(f"N genes:                {X.shape[1]:,}")

        print("\nPer-sample statistics:")
        print(f"  Mean total expression: {sample_sums.mean():.2f}")
        print(f"  Median total expr:     {np.median(sample_sums):.2f}")
        print(f"  Std total expr:        {sample_sums.std():.2f}")
        print(f"  Min total expr:        {sample_sums.min():.2f}")
        print(f"  Max total expr:        {sample_sums.max():.2f}")

        print(f"\n  Mean gene expr/sample: {sample_means.mean():.4f}")
        print(f"  Mean nonzero genes:    {sample_nonzero.mean():.1f}")
        print(f"  Median nonzero genes:  {np.median(sample_nonzero):.1f}")

        print("\nMatrix sparsity:")
        print(f"  Zero fraction:         {zero_fraction:.4f}")
        print(f"  Nonzero fraction:      {1-zero_fraction:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sample analysis")

    parser.add_argument("--tcga", type=str, nargs='*', default=[], help="List of TCGA project codes")
    parser.add_argument("--gtex", type=str, nargs='*', default=[], help="List of GTEx tissue names")
    parser.add_argument("--other", type=str, nargs='*', default=[], help="List of other dataset names")
    parser.add_argument("--other_dir", type=str, help="Directory of other files")
    parser.add_argument("--n_samples", type=int, default=0, help="For pseudobulk")
    parser.add_argument("--n_cells", type=int, nargs='+', default=[2], help="For pseudobulk")
    parser.add_argument("--group_by_source", action="store_true", help="Group labels as TCGA/GTEx") 
    parser.add_argument("--mixer_type", type=str, default='donor_mixer')

    args = parser.parse_args()

    # Config
    GROUP_BY_SOURCE = args.group_by_source
    SELECT_TCGA = args.tcga
    SELECT_GTEX = args.gtex
    LOAD_GTEX_VARIANTS = {'processed': True, 'div_1.5': False, 'nn': False}

    DATA_DIR = '../data/0_data_for_mlp_small_cohorts'
    GTEX_DIR = '../data/GTEx/processed'
    ADATA_DIR = 'data_new/blkb_common_train.h5ad'

    PSEUDO_CONFIG = {
        'n_samples': args.n_samples,  # set to 0 if you don't want any pseudobulks
        'n_cells_list': args.n_cells, # Tutaj podajesz dowolną liczbę wariantów
        'adata_path': ADATA_DIR,
    }

    # Run
    datasets = []
    labels = []

    # 0. Load other
    for dataset in args.other:
        print(f"Loading other dataset: {dataset}")
        datasets.append(load_dataset(f'{args.other_dir}/{dataset}', log2_transform=False))
        labels.append(dataset)

    # 1. Load TCGA
    for cohort in SELECT_TCGA:
        print(f"Loading TCGA: {cohort}")
        datasets.append(load_dataset(f'{DATA_DIR}/{cohort}.star_tpm.csv', log2_transform=False))
        labels.append(cohort)

    # 2. Load GTEx
    for tissue in SELECT_GTEX:
        print(f"Loading GTEx: {tissue}")
        if LOAD_GTEX_VARIANTS['processed']:
            datasets.append(load_dataset(f'{GTEX_DIR}/gene_tpm_v11_{tissue}_processed.csv', log2_transform=False))
            labels.append(f"GTEx_{tissue}")
        if LOAD_GTEX_VARIANTS['div_1.5']:
            datasets.append(load_dataset(f'{GTEX_DIR}/gene_tpm_v11_{tissue}_processed_div_1,5.csv', log2_transform=False))
            labels.append(f"GTEx_{tissue}_div_1.5")
        if LOAD_GTEX_VARIANTS['nn']:
            datasets.append(load_dataset(f'{GTEX_DIR}/gene_tpm_v11_{tissue}_nn.csv', log2_transform=False))
            labels.append(f"GTEx_{tissue}_nn")

    # 3. Generate pesudo bulks for each n_cell value
    if PSEUDO_CONFIG['n_samples'] > 0 and PSEUDO_CONFIG['n_cells_list']:
        adata = ad.read_h5ad(PSEUDO_CONFIG['adata_path'])

        for n_c in PSEUDO_CONFIG['n_cells_list']:
            print(f"\n>>> Generating PseudoBulk: n_cells={n_c}")
            pseudo_df = generate_pseudobulk_from_adata(
                adata=adata,
                n_samples=PSEUDO_CONFIG['n_samples'],
                n_cells=n_c,
                mixer_type=args.mixer_type
            )
            datasets.append(pseudo_df)
            labels.append(f"Pseudo_{n_c}cells")

    # 4. Plots
    if datasets:
        # print_dataset_stats(datasets, labels)
        print('Finished')
        save_prefix = build_save_prefix(labels, extra_tag=f'grouped_{GROUP_BY_SOURCE}', n_samples=args.n_samples, mixer=args.mixer_type)
        print('Finished 2')
        
        print(f"\nFinal datasets in plot: {labels}")
        plot_pca_umap(
            datasets=datasets,
            labels=labels,
            save_prefix=save_prefix,
        )