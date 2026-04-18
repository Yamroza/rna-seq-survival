import argparse
import os

import anndata as ad
import gzip
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast

from choose_protein_coding import list_of_protein_coding_genes


def main():
    parser = argparse.ArgumentParser(description="prepare tcga data for mlp and scgpt models")
    
    parser.add_argument("--filename",           type=str, default="TCGA-OV.star_tpm", help="name of the input tsv file (without extension)")
    parser.add_argument("--raw_data_dir",       type=str, default="../data/raw_tsv_data")
    parser.add_argument("--data_for_mlp_dir",   type=str, default="../data/0_data_for_mlp")
    parser.add_argument("--data_for_scgpt_dir", type=str, default="../data/0_adata_for_scgpt")
    parser.add_argument("--data_for_tgpt_dir",  type=str, default="../data/0_adata_for_tgpt")
    parser.add_argument("--gene_info_file",     type=str, default="../data/gene_info.csv")
    parser.add_argument("--top_n",              type=int, default=2000, help="How many top genes to save")
    parser.add_argument("--no_save",            action="store_true", help="dry run without saving files")

    args = parser.parse_args()
    save_files = not args.no_save

    print(f"processing: {args.filename}")
    
    # step 1: prepare csv for mlp
    df_filtered = prep_data_for_mlp(
        raw_data_dir=args.raw_data_dir,
        filename=args.filename,
        data_for_mlp_dir=args.data_for_mlp_dir,
        gene_info_file=args.gene_info_file,
        save=save_files
    )
    print(f"mlp data shape: {df_filtered.shape}")

    # step 2: prepare h5ad for scgpt
    adata = prep_data_for_scgpt(
        df_filtered=df_filtered,
        data_for_scgpt_dir=args.data_for_scgpt_dir,
        filename=args.filename,
        save=save_files
    )
    print(f"scgpt adata created. n_obs: {adata.n_obs}, n_vars: {adata.n_vars}")
    
    # step 3: prepare sequences for tgpt
    sentences = prep_data_for_tgpt(
        df_filtered=df_filtered,
        data_for_tgpt_dir=args.data_for_tgpt_dir,
        filename=args.filename,
        top_n=args.top_n,
        save=save_files
    )
    print(f"tgpt sequences created. n_samples: {len(sentences)}")
    print("done.")


def prep_data_for_mlp(raw_data_dir, filename, data_for_mlp_dir, gene_info_file, save=True):
    filepath = os.path.join(raw_data_dir, f"{filename}.tsv")
    df = pd.read_csv(filepath, delimiter='\t', index_col=False)
    
    # rename for downstream compatibility
    df = df.rename(columns={'Ensembl_ID': 'Unnamed: 0-1'})

    # tcga barcodes ending with A/B/etc - keep A if duplicates exist
    col_df = pd.DataFrame({"col": df.columns})
    col_df["prefix"] = col_df["col"].str[:-1]
    col_df["suffix"] = col_df["col"].str[-1]
    col_df["is_A"] = (col_df["suffix"] == "A").astype(int)
    
    col_df = col_df.sort_values(["prefix", "is_A"], ascending=[True, False])
    selected_cols = col_df.drop_duplicates("prefix")["col"].tolist()

    if 'Unnamed: 0-1' in selected_cols:
        selected_cols.remove('Unnamed: 0-1')
    selected_cols.insert(0, 'Unnamed: 0-1')

    df = df[selected_cols]
    
    # drop the last part of tcga barcode
    df.columns = df.columns.str.split("-").str[:-1].str.join("-")

    # transpose: rows = samples, cols = genes
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    # remove ensembl version numbers (e.g., ENSG... .12 -> ENSG...)
    df.columns = df.columns.str.split(".").str[0]

    # map ensembl ids to gene symbols
    features = pd.read_csv(gene_info_file)
    id_to_symbol = dict(zip(features["feature_id"], features["feature_name"]))
    df = df.rename(columns=id_to_symbol)

    # clean up unmapped and duplicated columns
    df = df.loc[:, df.columns.notnull()]
    df = df.loc[:, ~df.columns.duplicated()]

    # filter for protein-coding genes only
    # skip the first col if it's not a gene
    gene_list = df.columns[1:] if 'Unnamed' in str(df.columns[0]) else df.columns 
    protein_coding = list_of_protein_coding_genes(gene_list)

    df_filtered = df[df.columns.intersection(protein_coding)]
    
    # drop duplicate samples just in case
    df_filtered = df_filtered[~df_filtered.index.duplicated(keep="first")]

    if save:
        os.makedirs(data_for_mlp_dir, exist_ok=True)
        out_path = os.path.join(data_for_mlp_dir, f"{filename}.csv")
        df_filtered.to_csv(out_path)
    
    return df_filtered


def prep_data_for_scgpt(df_filtered, data_for_scgpt_dir, filename, save=True):
    X = df_filtered.values.astype(np.float32)

    obs = pd.DataFrame(index=df_filtered.index)
    obs["sample"] = df_filtered.index

    var = pd.DataFrame(index=df_filtered.columns)
    var["gene_name"] = df_filtered.columns

    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    if save:
        os.makedirs(data_for_scgpt_dir, exist_ok=True)
        out_path = os.path.join(data_for_scgpt_dir, f"adata_{filename}.h5ad")
        adata.write(out_path)
        
    return adata


def prep_data_for_tgpt(df_filtered, data_for_tgpt_dir, filename, top_n=2000, save=True):
    # load tokenizer to get valid vocabulary for tgpt
    tokenizer_file = "lixiangchun/transcriptome-gpt-1024-8-16-64" 
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
    vocab = tokenizer.get_vocab()

    # dynamically get the id column (usually index 0)
    id_col = df_filtered.columns[0]
    all_genes = df_filtered.columns[1:]
    
    # keep only genes that the model has been trained on
    model_genes = set(vocab.keys())
    common_genes = [g for g in all_genes if g in model_genes]
    df_tgpt = df_filtered[[id_col] + common_genes]

    sentences = []
    gene_names = np.array(common_genes)

    # generate rank-ordered gene sequences for each sample
    for i in range(len(df_tgpt)):
        # get expression values (skipping the id column)
        row_values = df_tgpt.iloc[i, 1:].values.astype(float)
        
        # argsort sorts ascending, so we take the last top_n and reverse with [::-1]
        top_indices = np.argsort(row_values)[-top_n:][::-1]
        ranked_genes = gene_names[top_indices]
        
        # tgpt expects space-separated sequences
        sentence = " ".join(ranked_genes)
        sentences.append(sentence)

    if save:
        os.makedirs(data_for_tgpt_dir, exist_ok=True)
        out_path = os.path.join(data_for_tgpt_dir, f"{filename}.txt.gz")
        
        # write directly to compressed file
        with gzip.open(out_path, 'wt', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + "\n")
                
    return sentences


if __name__ == "__main__":
    main()