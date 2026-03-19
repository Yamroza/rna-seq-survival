import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

n_components_list = [50, 75, 100, 150]
hvg_list = [0, 1500, 3000, 5000, 8000]

data_for_mlp_dir = "../data/0_data_for_mlp/TCGA-BRCA.star_tpm.csv"

df = pd.read_csv(data_for_mlp_dir, index_col=0)

for n_components in n_components_list:
    for hvg in hvg_list:

        # === HVG selection ===
        if hvg > 0:
            variances = df.var(axis=0)
            top_genes = variances.sort_values(ascending=False).head(hvg).index
            df_x = df[top_genes]
        else:
            df_x = df

        # === DATA ===
        X = df_x.values

        # === SCALING ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # === ICA ===
        ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=1000,
            tol=0.001
        )

        X_ica = ica.fit_transform(X_scaled)

        # === DF ===
        ica_columns = [f"IC{i+1}" for i in range(n_components)]
        df_ica = pd.DataFrame(X_ica, index=df.index, columns=ica_columns)

        # ICA nie ma explained_variance jak PCA
        data_for_mlp_ica = f"../data/1_data_for_mlp_ica/TCGA-BRCA.star_tpm_ica_{n_components}_hvg_{hvg}.csv"

        df_ica.to_csv(data_for_mlp_ica)