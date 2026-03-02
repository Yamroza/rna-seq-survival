from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import sys

sys.path.insert(0, "../")

import scgpt as scg
import pandas as pd


warnings.simplefilter("ignore", ResourceWarning)

filenames = ["adata_TCGA-BRCA.star_tpm"] # source data file, in "data" folder "adata_TCGA-LUAD.star_tpm", 
models = ["whole_human"] #, "pancancer"]
highly_variables = [(False, 0)]#, (True, 3000), (True, 6000)]
# highly_variables = [(True, 3000), (True, 6000)]
iter = range(4,20)
# max_length = 10000

for filename in filenames:
    smaple_data_path = f'../data/0_adata_for_scgpt/{filename}.h5ad'
    adata = sc.read_h5ad(smaple_data_path)
    gene_col = "gene_name"
    batch_key = "sample"
    for i in iter:
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
                    batch_size=32,
                    # max_length= hvg_no,
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

                # Path(f"../data/scgpt_embeddings/{max_length}").mkdir(parents=True, exist_ok=True)
                df_emb.to_csv(f"../data/scgpt_embeddings/new_lengths/scgpt_{filename}_{model}_hvg_{hvg_no}_ml_1200_{i}.csv", index=False)