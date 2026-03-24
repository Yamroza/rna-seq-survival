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

N_TOP = 3000

for outer in range(5):

    runs = load_gene_runs(
        f"../data/clinical_data/brca_clinical/shap_experiment/outer_{outer}/outer_{outer}.jsonl"
    )

    # === zbieramy top geny z każdego folda ===
    gene_lists = [run["genes"] for run in runs]
    n_runs = len(gene_lists)

    # flatten
    all_genes = [g for lst in gene_lists for g in lst]

    # === liczymy częstość występowania ===
    freq = Counter(all_genes)

    # === grupujemy po liczbie foldów, w których gen występuje ===
    grouped = {}
    for gene, count in freq.items():
        grouped.setdefault(count, []).append(gene)

    print(grouped)
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

    print(f"Outer {outer}: selected {len(selected)} genes")

    # # fallback jeśli nie dobijesz do 3000
    # if len(selected) < N_TOP:
    #     remaining = [g for g in freq.keys() if g not in used]
    #     selected.extend(remaining[:N_TOP - len(selected)])

    # # === filtr DF ===
    # columns = [col for col in selected if col in df.columns]
    # columns.insert(0, 'Unnamed: 0')

    # df_selected = df[columns]

    # df_selected.to_csv(
    #     f"../data/0_data_for_mlp_testing/top_shap_3k_training_outer_{outer}.csv",
    #     index=False
    # )