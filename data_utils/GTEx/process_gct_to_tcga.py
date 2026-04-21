import os
import pandas as pd

import sys
sys.path.append("..")

from choose_protein_coding import list_of_protein_coding_genes

gene_info_file = "../../data/gene_info.csv"
save_dir = f"../../data/GTEx/processed"

filepath = '../../data/GTEx'
filename = 'gene_tpm_v11_bladder'
df_gtex = pd.read_csv(f"{filepath}/{filename}.gct", sep='\t', skiprows=2)

# delete ensemblid versioning
df_gtex['Name'] = df_gtex['Name'].str.split('.').str[0]

# keep only TCGA genes - load TCGA example file
filename = '../../data/raw_tsv_data/TCGA-GBM.star_tpm.tsv'
df_tcga = pd.read_csv(filename, sep='\t')

# delete ensemblid versioning
df_tcga['Ensembl_ID'] = df_tcga['Ensembl_ID'].str.split('.').str[0]
df_tcga.head()

# make protein coding lists from knowledge file

df = pd.read_csv('../../data/gene_info_table.csv', index_col=0)

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

features = pd.read_csv(gene_info_file)
id_to_symbol = dict(zip(features["feature_id"], features["feature_name"]))
df = df.rename(columns=id_to_symbol)

df = df.loc[:, df.columns.notnull()]
df = df.loc[:, ~df.columns.duplicated()]

gene_list = df.columns
protein_coding = list_of_protein_coding_genes(gene_list)

df_filtered = df[df.columns.intersection(protein_coding)]

df_filtered = df_filtered[~df_filtered.index.duplicated(keep="first")]

os.makedirs(save_dir, exist_ok=True)
# Zapisujemy gotowy plik, np. z dopiskiem '_processed'
save_path = os.path.join(save_dir, f"{filename}_processed.csv")
df_filtered.to_csv(save_path)
print(f"Saved processed data to: {save_path}")