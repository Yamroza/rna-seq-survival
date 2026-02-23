from pathlib import Path
import warnings

import scanpy as sc
import sys

sys.path.insert(0, "../")

import scgpt as scg
import matplotlib.pyplot as plt
import pandas as pd

plt.style.context('default')
warnings.simplefilter("ignore", ResourceWarning)

filenames = ["adata_TCGA-LUAD.star_tpm"] # source data file, in "data" folder
models = ["whole_human", "pancancer"]
highly_variables = [(False, 0), (True, 3000), (True, 6000)]

for filename in filenames:
    smaple_data_path = f'../data/0_adata_for_scgpt/{filename}.h5ad'
    adata = sc.read_h5ad(smaple_data_path)
    gene_col = "gene_name"
    batch_key = "sample"

    for model in models:
        for highly_variable in highly_variables:
            is_hvg, hvg_no = highly_variable
            if is_hvg:
                sc.pp.highly_variable_genes(adata, n_top_genes=hvg_no, flavor='seurat_v3')
                data_for_emb = adata[:, adata.var['highly_variable']]
            else:
                data_for_emb = adata
            
            model_dir = Path(f"/scratch/2370352/my-research/papers/scgpt/save/{model}")
            embed_adata = scg.tasks.embed_data(
                data_for_emb,
                model_dir,
                gene_col=gene_col,
                batch_size=64,
            )

            # zapis embeddingów do pliku

            X_emb = embed_adata.obsm["X_scGPT"]
            emb_cols = [f"emb{i+1}" for i in range(X_emb.shape[1])]
            samples = embed_adata.obs.index
            df_emb = pd.DataFrame(
                X_emb,
                index=samples,
                columns=emb_cols
            )
            df_emb = df_emb.reset_index().rename(columns={"index": "Unnamed: 0"})
            df_emb.to_csv(f"../data/scgpt_embeddings/scgpt_{filename}_{model}_hvg_{hvg_no}.csv", index=False)