from argparse import Namespace
import json
import numpy as np
import pandas as pd
import shap
import torch

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# root projektu
sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "ts_survival_prediction", "src")
sys.path.insert(0, SRC_DIR)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from ts_survival_prediction.src.model.FNN import FNN
from ts_survival_prediction.src.survival.losses import CoxLoss
from ts_survival_prediction.src.utils.data_utils import obtain_dataloader
from ts_survival_prediction.src.data.survival_dataset import RNASurvivalDataset

def save_genes(importance_df, top_n, run_name, filepath):
    subset = importance_df.head(top_n)
    genes = subset['gene'].to_list()
    entry = {
        "run_name": run_name,
        "genes": genes
    }
    with open(filepath, "a") as f:  # append mode
        f.write(json.dumps(entry) + "\n")

device = 'cpu'

for outer in range(5):
    exp_code = f'mlp_lasso_shap_experiment/outer_{outer}'

    for fold in range(5):

        args = Namespace(
            model="mlp",
            task="dss_survival_brca",
            data_source=f"/scratch/2370352/my-research/data/clinical_data/brca_clinical/shap_experiment/outer_{outer}",
            exp_code=exp_code,
            omics_type="TCGA-BRCA.star_tpm.csv",  # <- możesz zmienić dynamicznie
            data_type="csv",
            expression_data_path="/scratch/2370352/my-research/data/0_data_for_mlp_testing",
            folds=fold,
            target_col="dss_survival_days"
        )

        loss = CoxLoss()
        model = FNN(20260, 'mlp', [256, 256], loss)
        model.from_pretrained(f'/scratch/2370352/my-research/ts_survival_prediction/src/results/dss_survival_brca/{exp_code}/Fold_{fold}/model_checkpoint.pth', device=device)
        model.eval()

        # 1. Inicjalizacja datasetów
        train_dataset = RNASurvivalDataset(args, 'train', 'rna', 0)
        test_dataset = RNASurvivalDataset(args, 'test', 'rna', 0)

        # 2. Przygotowanie danych "tła" (Background Data)
        # SHAP używa ich do ustalenia "stanu zerowego". Bierzemy 100 próbek z treningu.
        background_data_np = train_dataset.df_rna.values[:200]
        background_data = torch.tensor(background_data_np, dtype=torch.float32).to(device)

        # 3. Przygotowanie danych testowych do wyjaśnienia
        # Wyciągamy wartości z DataFrame'u testowego
        test_data_np = test_dataset.df_rna.values
        test_data = torch.tensor(test_data_np, dtype=torch.float32).to(device)

        # ###
        # train_data_np = train_dataset.df_rna.values
        # train_data = torch.tensor(train_data_np, dtype=torch.float32).to(device)

        # background_data_new = train_data_np[:300]
        # background_data_new_tensor = torch.tensor(background_data_new, dtype=torch.float32).to(device)
        # train_data_new = train_data_np[300:]
        # train_data_new_tensor = torch.tensor(train_data_new, dtype=torch.float32).to(device)
        # ###

        # 4. Wyciągnięcie nazw genów (z nazw kolumn w DataFrame)
        gene_names = test_dataset.df_rna.columns.tolist()

        # 6. Obliczenia SHAP
        print("Inicjalizacja DeepExplainer...")
        original_forward = model.forward
        model.forward = model.forward_no_loss
        explainer = shap.DeepExplainer(model, background_data)
        print("Done")

        shap_values = explainer.shap_values(test_data)

        # 7. Formatowanie wyników pod wykres
        # PyTorch z SHAP często zwraca listę (po jednej tablicy na klasę wyjściową).
        # Dla num_classes=1 interesuje nas zerowy indeks.
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[0]
        else:
            shap_values_to_plot = shap_values

        # Upewniamy się, że shap_values_to_plot jest dwuwymiarowe (pacjenci x geny)
        # Jeśli ma kształt (N, 1, 20000), to wyciągamy środek
        if len(shap_values_to_plot.shape) == 3:
            shap_values_to_plot = shap_values_to_plot.reshape(shap_values_to_plot.shape[0], -1)

        # Oblicz średnią i wymuś 1D
        mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0).flatten()

        # Sprawdźmy długości - muszą być identyczne!
        print(f"Liczba nazw genów: {len(gene_names)}")
        print(f"Liczba wartości SHAP: {len(mean_abs_shap)}")

        # Stwórz tabelę
        importance_df = pd.DataFrame({
            'gene': gene_names,
            'mean_abs_shap': mean_abs_shap
        })

        importance_df = importance_df.sort_values(by='mean_abs_shap', ascending=False)
        save_genes(importance_df, 6000, f'fold_{fold}_lasso', f'outer_{outer}.jsonl')
