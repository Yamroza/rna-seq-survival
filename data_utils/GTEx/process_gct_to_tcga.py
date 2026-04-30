import argparse
import os
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

from choose_protein_coding import list_of_protein_coding_genes

def main():
    parser = argparse.ArgumentParser(description="Preprocess GTEx TPM data: align to TCGA gene space and keep protein-coding genes.")
    parser.add_argument("--gtex_dir", default="../../data/GTEx", help="Directory containing the raw GTEx .gct file.")
    parser.add_argument("--filename", default="gene_tpm_v11_bladder", help="GTEx filename without extension (expects a .gct file).")
    parser.add_argument("--tcga_filename", default="../../data/raw_tsv_data/TCGA-GBM.star_tpm.tsv", help="Path to the TCGA TPM file used to define the target gene set.")
    parser.add_argument("--gene_info_file", default="../../data/gene_info.csv", help="CSV mapping Ensembl IDs to gene symbols (columns: feature_id, feature_name).")
    parser.add_argument("--gene_info_table", default="../../data/gene_info_table.csv", help="CSV with gene metadata including gene_type and ensembl_id columns.")
    parser.add_argument("--save_dir", default="../../data/GTEx/processed", help="Output directory for the processed CSV.")
    
    # preprocessing flags
    parser.add_argument("--nn", action="store_true", help="No normalization: skip log2(TPM + 1)")
    parser.add_argument("--div15", action="store_true", help="Divide data by 1.5")
    
    args = parser.parse_args()

    gtex_filename_path = os.path.join(args.gtex_dir, f"{args.filename}.gct")
    df_gtex = pd.read_csv(gtex_filename_path, sep='\t', skiprows=2)
    # delete ensembleid versioning
    df_gtex['Name'] = df_gtex['Name'].str.split('.').str[0]

    # keep only TCGA genes - load TCGA example file
    df_tcga = pd.read_csv(args.tcga_filename, sep='\t')
    # delete ensemblid versioning
    df_tcga['Ensembl_ID'] = df_tcga['Ensembl_ID'].str.split('.').str[0]

    # make protein coding lists from knowledge file
    df = pd.read_csv(args.gene_info_table, index_col=0)

    duplicate_check = df.groupby('ensembl_id')['gene_type'].nunique()
    ids_with_multiple_types = duplicate_check[duplicate_check > 1].index.tolist()

    if ids_with_multiple_types:
        print(f"Znaleziono duplikaty z różnymi typami dla ID: {ids_with_multiple_types}")
        for gene_id in ids_with_multiple_types:
            names = df[df['ensembl_id'] == gene_id]['gene_name'].unique()
            print(f"  - Gen {gene_id} ({names}) występuje z wieloma typami. Zostanie zachowany jako protein_coding.")

    protein_coding_df = df[df['gene_type'] == 'protein_coding'].copy()
    protein_coding_df = protein_coding_df.drop_duplicates(subset=['ensembl_id'])

    ensembl_list = protein_coding_df['ensembl_id'].tolist()
    gene_name_list = protein_coding_df['gene_name'].tolist()

    # Printing results
    print(f"Number of protein coding ensembl: {len(ensembl_list)}")
    print(f"Number of protein coding gene_name: {len(gene_name_list)}")
    print("First 5 ID:", ensembl_list[:5])
    print("First 5 nazw:", gene_name_list[:5])

    valid_ids = set(df_tcga['Ensembl_ID'].unique())
    df_gtex_filtered = df_gtex[df_gtex['Name'].isin(valid_ids)].copy()

    # Printing results:
    print(f"Gene number before filtering: {len(df_gtex)}")
    print(f"Gene number after filtering: {len(df_gtex_filtered)}")

    lost_genes = len(df_gtex) - len(df_gtex_filtered)
    if lost_genes > 0:
        print(f"Attention: {lost_genes} gened from GTEx were lost.")

        gtex_ids = set(df_gtex['Name'].unique())
        missing_ids = gtex_ids - valid_ids
        missing_protein_coding = [gene_id for gene_id in missing_ids if gene_id in ensembl_list]
        print(f"Out of which {len(missing_protein_coding)} were protein coding genes")

    # filtering of gtex df
    df_gtex = df_gtex[df_gtex['Name'].isin(valid_ids)].reset_index(drop=True)

    # choose protein coding in the same way as in TCGA data
    df = df_gtex.drop(columns=['Description'])
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    features = pd.read_csv(args.gene_info_file)
    id_to_symbol = dict(zip(features["feature_id"], features["feature_name"]))
    df = df.rename(columns=id_to_symbol)

    df = df.loc[:, df.columns.notnull()]
    df = df.loc[:, ~df.columns.duplicated()]

    gene_list = df.columns
    protein_coding = list_of_protein_coding_genes(gene_list, args.gene_info_table, args.gene_info_file)

    df_filtered = df[df.columns.intersection(protein_coding)]
    df_filtered = df_filtered[~df_filtered.index.duplicated(keep="first")]

    # Conversion to numbers
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')

    # --- Normalization logic ---
    suffix = ""
    if args.nn:
        print("Skipping log2 normalization (--nn active).")
        suffix += "_nn"
    else:
        print("Applying log2(TPM + 1) normalization.")
        df_filtered = np.log2(df_filtered + 1)

    if args.div15:
        print("Dividing data by 1.5 (--div15 active).")
        df_filtered = df_filtered / 1.5
        suffix += "_div15"


    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = f"{args.filename}{suffix}_processed.csv"
    save_path = os.path.join(args.save_dir, save_filename)
    df_filtered.to_csv(save_path)
    print(f"Saved processed data to: {save_path}")

if __name__ == "__main__":
    main()