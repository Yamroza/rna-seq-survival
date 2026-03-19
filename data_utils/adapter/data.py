"""
- Map Ensembl ID → gene symbol (from gene_info_table.csv)
- Wrapper na scgpt.tasks.embed_data
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, "../")
import scgpt as scg


def load_and_prepare_adata(
    sc_data_path: str,
    cell_type_col: str = "cell_type",
    gene_info_path: str = None,
    debug_n: int = None,
):
    """
    Wczytaj h5ad i opcjonalnie:
    - przytnij do debug_n komórek
    - zamapuj Ensembl ID → gene symbol (jeśli gene_info_path podane)

    Zwraca: (adata, gene_col)
    """
    print(f"[Data] Wczytuję: {sc_data_path}")
    adata = sc.read_h5ad(sc_data_path)
    print(f"[Data] Shape: {adata.shape}")

    if debug_n is not None:
        adata = adata[:debug_n].copy()
        print(f"[DEBUG] Przycinam do {debug_n} komórek")

    gene_col = "gene_name"

    if gene_info_path is not None:
        print(f"[Data] Mapuję geny z: {gene_info_path}")
        gene_info = pd.read_csv(gene_info_path, index_col=0)
        ensembl_to_symbol = dict(zip(gene_info["ensembl_id"], gene_info["gene_name"]))
        adata.var["gene_name"] = adata.var["feature_id"].map(ensembl_to_symbol)
        n_mapped = adata.var["gene_name"].notna().sum()
        print(f"[Data] Zmapowano: {n_mapped}/{len(adata.var)} genów")
        adata = adata[:, adata.var["gene_name"].notna()].copy()
        print(f"[Data] Po filtrowaniu: {adata.shape}")
    else:
        assert gene_col in adata.var.columns, (
            f"Brak kolumny '{gene_col}' w adata.var. "
            f"Podaj --gene_info_path lub upewnij się że kolumna istnieje. "
            f"Dostępne: {list(adata.var.columns)}"
        )

    assert cell_type_col in adata.obs.columns, (
        f"Brak kolumny '{cell_type_col}' w adata.obs. "
        f"Dostępne: {list(adata.obs.columns)}"
    )

    return adata, gene_col


def get_scgpt_embeddings(
    adata,
    model_dir: str,
    gene_col: str = "gene_name",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Przepuść adata przez zamrożone scGPT i zwróć embeddingi.
    Shape: (n_cells, embed_dim)
    """
    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
    )
    return embed_adata.obsm["X_scGPT"]
