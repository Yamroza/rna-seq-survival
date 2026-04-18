import pandas as pd
import json

def list_of_protein_coding_genes(gene_list: list):
    genes = pd.Index(gene_list)

    # load gene info data, drop duplacates
    gene_info_original = pd.read_csv('../../data/gene_info_table.csv')
    gene_info = gene_info_original.drop(columns=['ensembl_id', 'Unnamed: 0'])
    gene_info = gene_info.drop_duplicates()

    # choose 1 gene_type for ambiguous genes
    gene_info["priority"] = (gene_info["gene_type"] == "protein_coding").astype(int)
    gene_info = gene_info.sort_values(
        ["gene_name", "priority"],
        ascending=[True, False]
    )
    gene_info = gene_info.drop_duplicates("gene_name")
    gene_info = gene_info.drop(columns="priority")

    gene_type_map = gene_info.set_index("gene_name")["gene_type"]
    mapped_types = genes.map(gene_type_map)
    missing_genes = genes[mapped_types.isna()]

    # another gene_info for missed genes
    gene_ensembl = pd.read_csv('../../data/gene_info.csv')
    symbol_to_ens_map = gene_ensembl.set_index("feature_name")["feature_id"]
    missing_ens = missing_genes.map(symbol_to_ens_map)
    gene_info_ens = gene_info_original.drop(columns=['gene_name', 'Unnamed: 0'])
    gene_info_ens = gene_info_ens.drop_duplicates()

    # choose 1 gene_type for ambiguous genes
    gene_info_ens["priority"] = (gene_info_ens["gene_type"] == "protein_coding").astype(int)
    gene_info_ens = gene_info_ens.sort_values(
        ["ensembl_id", "priority"],
        ascending=[True, False]
    )
    gene_info_ens = gene_info_ens.drop_duplicates("ensembl_id")
    gene_info_ens = gene_info_ens.drop(columns="priority")

    # creat mapping
    gene_type_map_ens = gene_info_ens.set_index("ensembl_id")["gene_type"]
    missing_types_from_ens = missing_ens.map(gene_type_map_ens)

    mapped_types = pd.Series(mapped_types, index=genes)
    mapped_types.loc[missing_genes] = missing_types_from_ens.values
    protein_coding_genes = mapped_types[mapped_types == "protein_coding"].index.tolist()

    return protein_coding_genes
