import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n_components_list = [40, 50, 75, 100, 150]  # możesz zmienić np. na 50 / 100 / 200
hvg_list = [0, 1500, 3000, 5000, 8000]
data_for_mlp_dir = "../data/0_data_for_mlp/TCGA-BRCA.star_tpm.csv"

df = pd.read_csv(data_for_mlp_dir, index_col=0)

for n_components in n_components_list:
    for hvg in hvg_list:
        if hvg > 0:
            variances = df.var(axis=0)
            top_genes = variances.sort_values(ascending=False).head(3000).index
            df_x = df[top_genes]
        else:
            df_x = df

        X = df.values  # macierz (n_samples, n_genes)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, index=df.index, columns=pca_columns)
        
        exp_var = pca.explained_variance_ratio_.sum()

        data_for_mlp_pca = f"../data/1_data_for_mlp_pca/TCGA-BRCA.star_tpm_pca_{n_components}_hvg_{hvg}_var{round(exp_var,3)}.csv"
        df_pca.to_csv(data_for_mlp_pca)
