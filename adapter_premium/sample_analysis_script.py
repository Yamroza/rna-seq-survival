"""
PCA and UMAP visualization comparing multiple bulk RNA-seq datasets and embeddings.
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
    ext = Path(path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        first_col = df.columns[0]
        
        # 1. Standaryzacja kolumny ID (nawet jeśli była pusta / "Unnamed: 0")
        if str(first_col).startswith('emb0'):
            # Jeśli plik zaczyna się od emb0, oznacza to kompletny brak indeksu
            print(f"Warning: No explicit index column found in {path}. Generating sequential IDs.")
            df.insert(0, "case_id", [f"sample_{i}" for i in range(len(df))])
        else:
            # Pierwsza kolumna (pusta, Unnamed, lub case_id) staje się oficjalnie "case_id"
            df = df.rename(columns={first_col: "case_id"})
            
        # 2. Pobieramy kolumny z cechami (wszystkie oprócz case_id)
        feature_cols = [c for c in df.columns if c != "case_id"]
        
        # KOREKTA PRZESUNIĘCIA INDEKSÓW (emb1..emb512 -> emb0..emb511)
        # Sprawdzamy czy pierwsza cecha to 'emb1', a ostatnia to 'emb512'
        if feature_cols[0] == "emb1" and f"emb{len(feature_cols)}" in feature_cols:
            print(f"-> Detected 1-based indexing (emb1..emb{len(feature_cols)}) in {path}. Realignment to 0-based indexing applied.")
            # Mapujemy nazwy tak, aby zaczynały się od emb0
            new_feature_names = [f"emb{i}" for i in range(len(feature_cols))]
            rename_dict = dict(zip(feature_cols, new_feature_names))
            df = df.rename(columns=rename_dict)
            feature_cols = new_feature_names

        # Rzutowanie na float32
        df[feature_cols] = df[feature_cols].astype(np.float32)
        return df

    elif ext == ".json":
        df = pd.read_json(path, lines=True)
        df = pd.concat([
            df[['id']].rename(columns={'id': 'case_id'}),
            pd.DataFrame(df['embedding'].tolist())
        ], axis=1)
        emb_cols = [c for c in df.columns if c != "case_id"]
        # JSON zawsze tworzy od emb0 do emb511
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
    rng = np.random.default_rng(random_seed)
    
    if mixer_type == 'donor_mixer':
        mixer = DynamicDonorMixer(adata, donor_col=donor_col, tissue_col=tissue_col, n_cells=n_cells)
    else:
        mixer = LinearTwoCellMixer()

    class SimpleDataset:
        def __init__(self, adata):
            self.X = adata.X
            self.labels = np.zeros(adata.n_obs)

    dataset_proxy = SimpleDataset(adata)
    n_samples = min(n_samples, adata.n_obs)
    base_indices = rng.choice(adata.n_obs, size=n_samples, replace=False)
    
    pseudobulks = []
    print(f"Generating {n_samples} pseudobulk samples (n_cells={n_cells})...")
    
    for idx in tqdm(base_indices):
        mixed_row, _, _ = mixer(idx, dataset_proxy)
        if scipy.sparse.issparse(mixed_row):
            mixed_row = mixed_row.toarray().flatten()
        pseudobulks.append(mixed_row)
    
    X_final = np.vstack(pseudobulks)
    gene_names = adata.var['feature_name'].values.astype(str)
    df = pd.DataFrame(X_final, columns=gene_names)
    df.insert(0, 'Unnamed: 0', [f'pseudo_{n_cells}_cell_{i}' for i in range(len(df))])
    
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
        
        # Bezpiecznik: jeśli pierwsza kolumna to cecha embeddingu, nie używaj jej jako ID
        if str(df.columns[0]).startswith("emb"):
            raise ValueError(
                "Critical Mismatch: The script selected the first feature column as a sample ID. "
                "Ensure your CSV file contains an explicit sample index/ID column at index 0."
            )
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
    all_genes_union = set.union(*gene_sets) if gene_sets else set()
    not_shared_genes = all_genes_union - set(shared_genes)
    print(not_shared_genes)
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
    actual_components = min(n_components, X.shape[0], X.shape[1])
    pca    = PCA(n_components=actual_components, random_state=SEED)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def run_umap(X: np.ndarray, n_neighbors: int = 30,
             min_dist: float = 0.3) -> Optional[np.ndarray]:
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=SEED,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pca_umap(datasets: list[pd.DataFrame],
                  labels: list[str],
                  loaded_embeddings: Optional[list[np.ndarray]] = None,
                  embedding_labels: Optional[list[str]] = None,
                  colors: Optional[list[str]] = None,
                  n_pca_components: int = 50,
                  index_col: str = 'Unnamed: 0',
                  save_prefix: str = 'comparison',
                  umap_n_neighbors: int = 30,
                  umap_min_dist: float = 0.3):
    
    plt.rcParams.update(STYLE)

    matrices_to_stack = []
    all_point_labels = []

    # 1. Przetwarzanie standardowych zbiorów danych (z dopasowaniem cech/wymiarów)
    if datasets:
        display_labels = []
        for lbl in labels:
            if GROUP_BY_SOURCE:
                if "TCGA" in lbl: display_labels.append("TCGA")
                elif "GTEx" in lbl: display_labels.append("GTEx")
                else: display_labels.append(lbl)
            else:
                display_labels.append(lbl)

        # Dopasowanie genów/cech pomiędzy wszystkimi obiektami DataFrame
        aligned_matrices, _ = match_genes_multi(datasets, index_col)
        
        for lbl, mat in zip(display_labels, aligned_matrices):
            matrices_to_stack.append(mat)
            all_point_labels.extend([lbl] * mat.shape[0])

    # 2. Dołączenie surowych macierzy .npy (o ile ich wymiary zgadzają się z dopasowanymi cechami)
    if loaded_embeddings:
        display_emb_labels = embedding_labels if embedding_labels else [f"Emb_{i}" for i in range(len(loaded_embeddings))]
        for lbl, mat in zip(display_emb_labels, loaded_embeddings):
            # Prosta weryfikacja wymiarowości, jeśli łączymy z obiektami DataFrame
            if matrices_to_stack and mat.shape[1] != matrices_to_stack[0].shape[1]:
                print(f"Warning: Dim mismatch! Matrix {lbl} has {mat.shape[1]} features, but DataFrame has {matrices_to_stack[0].shape[1]}. Trying PCA reduction anyway.")
            matrices_to_stack.append(mat)
            all_point_labels.extend([lbl] * mat.shape[0])

    # Połączenie wszystkiego w jedną macierz zbiorczą
    X_all = np.vstack(matrices_to_stack)

    # Budowa unikalnych grup do kolorowania wykresu
    unique_groups = []
    for dl in all_point_labels:
        if dl not in unique_groups:
            unique_groups.append(dl)

    if colors is None:
        colors = DEFAULT_COLORS[:len(unique_groups)]
    color_map = dict(zip(unique_groups, colors))

    # Obliczenia PCA i UMAP
    print("Running PCA...")
    coords, var_exp = run_pca(X_all, n_components=n_pca_components)
    umap_coords     = run_umap(X_all, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)

    n_panels = 4 if umap_coords is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.4), gridspec_kw={'wspace': 0.38})

    def _grouped_scatter(ax, x, y, title):
        for grp in unique_groups:
            idx = [i for i, l in enumerate(all_point_labels) if l == grp]
            ax.scatter(x[idx], y[idx], c=color_map[grp], s=8, alpha=0.6,
                       linewidths=0, rasterized=True, label=grp)
        ax.set_title(title, pad=5, loc='left')
        ax.spines[['top', 'right']].set_visible(False)

    # Panel A: PC1 vs PC2
    _grouped_scatter(axes[0], coords[:, 0], coords[:, 1], '(A)\u2002PCA — PC1 vs PC2')
    axes[0].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')

    # Panel B: PC1 vs PC3 (lub PC2 jeśli brakuje wymiarów)
    y_dim = 2 if len(var_exp) > 2 else 1
    _grouped_scatter(axes[1], coords[:, 0], coords[:, y_dim], f'(B)\u2002PCA — PC1 vs PC{y_dim+1}')
    axes[1].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC{y_dim+1} ({var_exp[y_dim]*100:.1f}%)')

    # Panel C: Scree plot
    ax = axes[2]
    cumvar = np.cumsum(var_exp) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, color='#333333', lw=1.5)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title('(C)\u2002Scree plot', pad=5, loc='left')
    ax.set_ylim(0, 101)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel D: UMAP
    if umap_coords is not None:
        _grouped_scatter(axes[3], umap_coords[:, 0], umap_coords[:, 1], '(D)\u2002UMAP')
        axes[3].set_xlabel('UMAP 1')
        axes[3].set_ylabel('UMAP 2')

    # Legenda
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[grp],
               markersize=6, label=grp)
        for grp in unique_groups
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(unique_groups), 5),
               bbox_to_anchor=(0.5, -0.2), frameon=True,
               handletextpad=0.4, columnspacing=1.5, borderpad=0.6, fontsize=9)

    outdir = Path('sample_analysis_plots')
    outdir.mkdir(exist_ok=True)

    # Generowanie czytelnych nazw do metadanych
    all_combined_labels = labels + (embedding_labels if embedding_labels else [])
    save_metadata(save_prefix, all_combined_labels, outdir=outdir)

    for ext in ('pdf', 'png'):
        path = outdir / f'{ext}/{save_prefix}_pca_umap.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')

    print("\nPairwise centroid L2 distances (PC1-min(10, n_pcs)):")
    n_dim = min(10, coords.shape[1])
    centroids = {
        grp: coords[[i for i, l in enumerate(all_point_labels) if l == grp], :n_dim].mean(0)
        for grp in unique_groups
    }
    for i, a in enumerate(unique_groups):
        for b in unique_groups[i+1:]:
            dist = np.linalg.norm(centroids[a] - centroids[b])
            print(f"  {a} vs {b}: {dist:.3f}")


def build_save_prefix(labels: list[str], extra_tag: str = '', n_samples: int = 0, mixer: str = 'donor_mixer'):
    short_labels = []
    for lbl in labels:
        cleaned = (
            str(lbl).replace('TCGA-', '')
               .replace('gene_tpm_v11_', '')
               .replace('processed', 'proc')
               .replace('kidney_cortex', 'kidney')
               .replace(' ', '_')
        )
        short_labels.append(cleaned)
    joined = '__'.join(short_labels)
    if len(joined) > 120:  # Zabezpieczenie przed za długą nazwą pliku w systemie operacyjnym
        joined = joined[:110] + "_truncated"
    if extra_tag and n_samples > 0:
        return f'{extra_tag}_{joined}_{mixer}_n_samples_{n_samples}'
    elif extra_tag:
        return f'{extra_tag}_{joined}'
    return joined


def save_metadata(save_prefix, labels, outdir='sample_analysis_plots'):
    Path(outdir).mkdir(exist_ok=True)
    meta_path = Path(outdir) / f'metadata/{save_prefix}_metadata.txt'
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        f.write('Datasets used:\n')
        for lbl in labels:
            f.write(f' - {lbl}\n')
    print(f'Saved metadata: {meta_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sample analysis")

    parser.add_argument("--tcga", type=str, nargs='*', default=[])
    parser.add_argument("--gtex", type=str, nargs='*', default=[])
    parser.add_argument("--other", type=str, nargs='*', default=[])
    parser.add_argument("--other_dir", type=str)
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--n_cells", type=int, nargs='+', default=[2])
    parser.add_argument("--group_by_source", action="store_true") 
    parser.add_argument("--mixer_type", type=str, default='donor_mixer')
    
    parser.add_argument("--embeddings", type=str, nargs='*', default=[])
    parser.add_argument("--embedding_labels", type=str, nargs='*', default=[])

    args = parser.parse_args()

    GROUP_BY_SOURCE = args.group_by_source
    SELECT_TCGA = args.tcga
    SELECT_GTEX = args.gtex
    LOAD_GTEX_VARIANTS = {'processed': True, 'div_1.5': False, 'nn': False}

    DATA_DIR = '../data/0_data_for_mlp_small_cohorts'
    GTEX_DIR = '../data/GTEx/processed'
    ADATA_DIR = 'data_new/blkb_common_train.h5ad'

    PSEUDO_CONFIG = {
        'n_samples': args.n_samples,
        'n_cells_list': args.n_cells,
        'adata_path': ADATA_DIR,
    }

    datasets = []
    labels = []
    loaded_embeddings = []
    embedding_labels = []

    # 0. Ładowanie innych plików tekstowych (CSV/JSON) -> ZAWSZE URUCHAMIANE
    for dataset in args.other:
        print(f"Loading other dataset: {dataset}")
        full_path = f'{args.other_dir}/{dataset}' if args.other_dir else dataset
        datasets.append(load_dataset(full_path, log2_transform=False))
        labels.append(dataset)

    # 1. Ładowanie TCGA
    for cohort in SELECT_TCGA:
        print(f"Loading TCGA: {cohort}")
        datasets.append(load_dataset(f'{DATA_DIR}/{cohort}.star_tpm.csv', log2_transform=False))
        labels.append(cohort)

    # 2. Ładowanie GTEx
    for tissue in SELECT_GTEX:
        print(f"Loading GTEx: {tissue}")
        if LOAD_GTEX_VARIANTS['processed']:
            datasets.append(load_dataset(f'{GTEX_DIR}/gene_tpm_v11_{tissue}_processed.csv', log2_transform=False))
            labels.append(f"GTEx_{tissue}")

    # 3. Generowanie dynamicznych pseudo-bulków
    if PSEUDO_CONFIG['n_samples'] > 0 and PSEUDO_CONFIG['n_cells_list']:
        adata = ad.read_h5ad(PSEUDO_CONFIG['adata_path'])
        for n_c in PSEUDO_CONFIG['n_cells_list']:
            print(f"\n>>> Generating PseudoBulk: n_cells={n_c}")
            pseudo_df = generate_pseudobulk_from_adata(
                adata=adata, n_samples=PSEUDO_CONFIG['n_samples'], n_cells=n_c, mixer_type=args.mixer_type
            )
            datasets.append(pseudo_df)
            labels.append(f"Pseudo_{n_c}cells")

    # 4. Ładowanie surowych embeddingów .npy (Jeśli obecne, zostaną połączone w osi próbek z powyższymi)
    if args.embeddings:
        for path_str, label in zip(args.embeddings, args.embedding_labels if args.embedding_labels else args.embeddings):
            print(f"Loading embedding array: {path_str}")
            emb_matrix = np.load(path_str)
            print(f"  -> Loaded shape: {emb_matrix.shape}")
            loaded_embeddings.append(emb_matrix)
            embedding_labels.append(label)

    # Wywołanie rysowania jeśli cokolwiek zostało poprawnie wczytane
    if datasets or loaded_embeddings:
        print('Processing and structural analysis...')
        all_combined_labels = labels + (embedding_labels if embedding_labels else [])
        save_prefix = build_save_prefix(all_combined_labels, extra_tag=f'grouped_{GROUP_BY_SOURCE}', n_samples=args.n_samples, mixer=args.mixer_type)
        
        plot_pca_umap(
            datasets=datasets,
            labels=labels,
            loaded_embeddings=loaded_embeddings,
            embedding_labels=embedding_labels,
            save_prefix=save_prefix,
        )