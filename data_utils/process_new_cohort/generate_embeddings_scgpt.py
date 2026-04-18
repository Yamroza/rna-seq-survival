import argparse
import os
import warnings
from pathlib import Path

import pandas as pd
import scanpy as sc
import scgpt as scg

# ignore noisy warnings from deep learning libraries
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ResourceWarning)

def main():
    parser = argparse.ArgumentParser(description="extract scgpt embeddings from anndata")
    
    # paths and filenames
    parser.add_argument("--filename",   type=str, default="TCGA-OV.star_tpm", help="h5ad file name without extension")
    parser.add_argument("--data_dir",   type=str, default="../../data/0_adata_for_scgpt", help="directory with input adata")
    parser.add_argument("--output_dir", type=str, default="../../data/0_data_for_mlp")
    parser.add_argument("--model_dir",  type=str, default="/scratch/2370352/my-research/papers/scgpt/save")
    
    # model and preprocessing config
    parser.add_argument("--model_name", type=str, default="whole_human", choices=["whole_human", "pancancer"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_hvg",      type=int, default=0,    help="number of HVGs, 0 means use all genes")
    parser.add_argument("--max_length", type=int, default=1200, help="max sequence length for scgpt")

    args = parser.parse_args()
    print(f"processing: adata_{args.filename} with model: {args.model_name}")

    # paths setup
    sample_data_path = os.path.join(args.data_dir, f"adata_{args.filename}.h5ad")
    model_dir = Path(args.model_dir) / args.model_name
    
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    adata = sc.read_h5ad(sample_data_path)
    gene_col = "gene_name"
    
    # step 1: preprocessing / hvg selection
    if args.n_hvg > 0:
        print(f"selecting top {args.n_hvg} HVGs")
        # scGPT usually works best with raw counts or normalized but non-log counts for embedding
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor='seurat_v3')
        data_for_emb = adata[:, adata.var['highly_variable']]
    else:
        print("using all genes for embedding")
        data_for_emb = adata

    # step 2: run embedding task
    # scgpt handles device selection internally, usually defaults to cuda if available
    embed_adata = scg.tasks.embed_data(
        data_for_emb,
        model_dir,
        gene_col=gene_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # step 3: extract features and format output
    # scGPT stores embeddings in obsm under 'X_scGPT'
    X_emb = embed_adata.obsm["X_scGPT"]
    emb_cols = [f"emb{i+1}" for i in range(X_emb.shape[1])]
    
    # map back to sample names (obs index)
    samples = embed_adata.obs.index
    df_emb = pd.DataFrame(
        X_emb,
        index=samples,
        columns=emb_cols
    )

    # format output to match the empty header requirement for samples column
    df_emb = df_emb.reset_index().rename(columns={"index": ""})

    # step 4: save results
    # filename includes key parameters to avoid overwriting
    hvg_status = f"hvg_{args.n_hvg}" if args.n_hvg > 0 else "all_genes"
    output_filename = f"scgpt_embeddings_{args.filename}_{args.model_name}_{hvg_status}_ml_{args.max_length}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    df_emb.to_csv(output_path, index=False)
    
    print(f"done! embeddings saved to: {output_path}")
    print(f"final shape: {df_emb.shape}")


if __name__ == "__main__":
    main()