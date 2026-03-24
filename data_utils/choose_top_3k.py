import json
import pandas as pd
from collections import Counter


def load_gene_runs(filepath):
    runs = []
    with open(filepath) as f:
        for line in f:
            runs.append(json.loads(line))
    return runs

df = pd.read_csv('../data/0_data_for_mlp/TCGA-BRCA.star_tpm.csv')
df_all_dataset = pd.read_csv('../data/0_data_for_mlp_shap/top_shap_3k_training.csv')
all_columns = set(df_all_dataset.columns)

for outer in range(5):
    runs = load_gene_runs(f"../data/clinical_data/brca_clinical/shap_experiment/outer_{outer}/outer_{outer}.jsonl")
    # top 1000 genes from each fold
    gene_lists = [run["genes"][:1000] for run in runs]
    top_concat = [element for gene_list in gene_lists for element in gene_list]

    gene_sets = {run["run_name"]: set(run["genes"]) for run in runs}
    common_all = set.intersection(*gene_sets.values())
    print(len(common_all))

    top_concat.extend(common_all)

    N_TOP = 3000
    # === ranking ===
    gene_lists = [run["genes"][:6000] for run in runs]
    n_runs = len(gene_lists)
    all_genes = [g for lst in gene_lists for g in lst]
    freq = Counter(all_genes)
    grouped = {}
    for gene, count in freq.items():
        grouped.setdefault(count, []).append(gene)
    selected = []
    used = set()

    # od najbardziej konsensusowych do najmniej
    for k in range(n_runs, 0, -1):
        if k in grouped:
            for gene in grouped[k]:
                if gene not in used:
                    selected.append(gene)
                    used.add(gene)
                if len(selected) >= N_TOP:
                    break
        if len(selected) >= N_TOP:
            break
    
    print('Overlap:', len(set(top_concat) & set(selected)))
    print("Outer: ", outer, "Num chosen genes: ", len(set(top_concat)))

    columns = list(set(top_concat))
    columns = set(top_concat)
    print("Match with original: ", len(columns & all_columns), '(', round(len(columns & all_columns)/len(columns)*100, 2), '%)')
    # columns.insert(0, 'Unnamed: 0')
    # df_selected = df[columns]
    # df_selected.to_csv(f"../data/0_data_for_mlp_testing/top_shap_3k_training_outer_{outer}.csv", index=False)
