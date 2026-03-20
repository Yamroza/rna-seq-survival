import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

hvg_list = [4500, 4800, 5200, 5500, 5800] #3000, 5000, 8000]
data_for_mlp_dir = "../data/0_data_for_mlp/TCGA-BRCA.star_tpm.csv"

df = pd.read_csv(data_for_mlp_dir, index_col=0)

for hvg in hvg_list:
    if hvg > 0:
        variances = df.var(axis=0)
        top_genes = variances.sort_values(ascending=False).head(hvg).index
        df_x = df[top_genes]
    else:
        df_x = df

    data_for_mlp_hvg = f"../data/0_data_for_mlp/TCGA-BRCA.star_tpm_hvg_{hvg}.csv"
    df_x.to_csv(data_for_mlp_hvg)
