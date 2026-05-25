import os
import argparse
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

import warnings
warnings.filterwarnings('ignore')

from utils import get_scgpt_model, binning

# Konfiguracja ścieżek do metryk survivalowych
tsa_root = os.path.abspath("../ts_survival_prediction/src")
if tsa_root not in sys.path:
    sys.path.insert(0, tsa_root)

from ts_survival_prediction.src.utils.test_utils import compute_survival_metrics
from ts_survival_prediction.src.utils.general_utils import set_seed, save_json

# Ustawienie ziarna dla powtarzalności
torch.manual_seed(42)
np.random.seed(42)

PAD_ID = None

# ── DEFINICJA STRATY COXA ─────────────────────────────────────────────────
class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hazards, survival_times, censorships):
        survival_times, idx = torch.sort(survival_times, dim=0, descending=True)
        hazards = hazards[idx].squeeze(-1)
        censorships = censorships[idx].squeeze(-1)

        log_risk = torch.logcumsumexp(hazards, dim=0)
        uncensored_loss = censorships * (hazards - log_risk)
        loss = -torch.sum(uncensored_loss) / (torch.sum(censorships) + 1e-8)
        return loss

# ── DATASET POD K-FOLD DLA scGPT ──────────────────────────────────────────
class ScGPTSurvivalFoldDataset(Dataset):
    def __init__(self, df_rna_binned, df_clinical_fold, gene_ids, target_col):
        # Dopasowanie po 'case_id'
        common_patients = sorted(list(set(df_rna_binned.index).intersection(set(df_clinical_fold['case_id']))))
        
        self.df_rna = df_rna_binned.loc[common_patients]
        self.df_clinical = df_clinical_fold.set_index('case_id').loc[common_patients].reset_index()
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        
        self.survival_times = self.df_clinical[target_col].values.astype(np.float32)
        censorship_col = target_col.split('_')[0] + '_censorship'
        if censorship_col not in self.df_clinical.columns:
            censorship_col = [c for c in self.df_clinical.columns if 'censor' in c.lower() or 'status' in c.lower()][0]
            
        self.censorships = self.df_clinical[censorship_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df_rna)

    def __getitem__(self, idx):
        values = torch.tensor(self.df_rna.iloc[idx].values, dtype=torch.float32)
        nonzero_mask = values > 0
        if nonzero_mask.sum() == 0:
            return self.gene_ids[:1], values[:1], self.survival_times[idx], self.censorships[idx], self.df_clinical.iloc[idx]['case_id']
            
        src = self.gene_ids[nonzero_mask]
        values = values[nonzero_mask]
        return src, values, self.survival_times[idx], self.censorships[idx], self.df_clinical.iloc[idx]['case_id']

def collate_fn(batch):
    src_list, values_list, survival_times, censorships, case_ids = zip(*batch)
    max_len = max(len(x) for x in src_list)

    padded_src = []
    padded_values = []
    padding_masks = []

    for src, values in zip(src_list, values_list):
        pad_len = max_len - len(src)
        padded_src.append(torch.cat([src, torch.full((pad_len,), PAD_ID, dtype=torch.long)]))
        padded_values.append(torch.cat([values, torch.zeros(pad_len)]))
        padding_masks.append(torch.cat([torch.zeros(len(src), dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)]))

    return (
        torch.stack(padded_src),
        torch.stack(padded_values),
        torch.stack(padding_masks),
        torch.tensor(survival_times, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(censorships, dtype=torch.float32).unsqueeze(-1),
        case_ids
    )

# ── ARCHITEKTURA MODELU scGPT DLA SURVIVALU ───────────────────────────────
class ScGPTSurvivalModel(nn.Module):
    def __init__(self, base_scgpt, emb_dim=512):
        super().__init__()
        self.base_model = base_scgpt
        self.survival_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1, bias=False)
        )

    def forward(self, src, values, src_key_padding_mask):
        outputs = self.base_model(
            src=src,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            CLS=True,
            CCE=False, MVC=False, ECS=False
        )
        cell_emb = outputs["cell_emb"]
        hazard_ratio = self.survival_head(cell_emb)
        return hazard_ratio

# ── LORA IMPLEMENTATION ───────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base_layer.parameters():
            p.requires_grad = False

    @property
    def weight(self): return self.base_layer.weight
    @property
    def bias(self): return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.scaling * self.lora_B(self.lora_A(x))

def replace_linear_with_lora(model):
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "self_attn" in name:
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module))
            replaced += 1
    print(f"[LoRA] Wymieniono {replaced} warstw liniowych w Transformerze.")

# ── FUNKCJA TRENINGU POJEDYNCZEGO FOLDA ───────────────────────────────────
def survival_train_fold(args, fold, device, df_rna_binned, gene_ids):
    print(f"\n=== ROZPOCZĘCIE TRENINGU: FOLD {fold} ===")
    
    fold_train_path = os.path.join(args.data_source, f"splits/{fold}/train_filtered.csv")
    df_clinical_train = pd.read_csv(fold_train_path)
    
    train_dataset = ScGPTSurvivalFoldDataset(df_rna_binned, df_clinical_train, gene_ids, args.target_col)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    base_scgpt, _ = get_scgpt_model(args.model_path, device='cpu', eval=False, do_mvc=False)
    model = ScGPTSurvivalModel(base_scgpt)
    model.to(device)
    
    if args.lora:
        for p in model.base_model.parameters():
            p.requires_grad = False
        replace_linear_with_lora(model.base_model)
        for p in model.survival_head.parameters():
            p.requires_grad = True
        for name, module in model.base_model.named_modules():
            if "value_encoder" in name:
                for p in module.parameters():
                    p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    criterion = CoxLoss()
    
    result_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}/")
    os.makedirs(result_dir_fold, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for src, values, pad_mask, survival_times, censorships, _ in train_loader:
            optimizer.zero_grad()
            src, values, pad_mask = src.to(device), values.to(device), pad_mask.to(device)
            survival_times, censorships = survival_times.to(device), censorships.to(device)
            
            hazards = model(src=src, values=values, src_key_padding_mask=pad_mask)
            loss = criterion(hazards, survival_times, censorships)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(result_dir_fold, "model_checkpoint.pth"))
            
    print(f"Zakończono Fold {fold} | Najlepsza strata: {best_loss:.4f}")
    return {"train_loss": best_loss}

# ── FUNKCJA TESTOWANIA POJEDYNCZEGO FOLDA ──────────────────────────────────
def survival_test_fold(args, fold, device, df_rna_binned, gene_ids):
    print(f"=== ROZPOCZĘCIE TESTOWANIA: FOLD {fold} ===")
    
    df_clinical_test = pd.read_csv(os.path.join(args.data_source, f"splits/{fold}/test_filtered.csv"))
    df_clinical_train = pd.read_csv(os.path.join(args.data_source, f"splits/{fold}/train_filtered.csv"))
    
    test_dataset = ScGPTSurvivalFoldDataset(df_rna_binned, df_clinical_test, gene_ids, args.target_col)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    base_scgpt, _ = get_scgpt_model(args.model_path, device='cpu', eval=False, do_mvc=False)
    model = ScGPTSurvivalModel(base_scgpt)
    
    if args.lora:
        replace_linear_with_lora(model.base_model)
        
    result_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}/")
    model.load_state_dict(torch.load(os.path.join(result_dir_fold, "model_checkpoint.pth"), map_location=device))
    model.to(device)
    model.eval()
    
    criterion = CoxLoss()
    
    all_hazards, all_times, all_censorships = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for src, values, pad_mask, survival_times, censorships, _ in test_loader:
            src, values, pad_mask = src.to(device), values.to(device), pad_mask.to(device)
            survival_times, censorships = survival_times.to(device), censorships.to(device)
            
            hazards = model(src=src, values=values, src_key_padding_mask=pad_mask)
            loss = criterion(hazards, survival_times, censorships)
            total_loss += loss.item()
            
            all_hazards.append(hazards.cpu().numpy())
            all_times.append(survival_times.cpu().numpy())
            all_censorships.append(censorships.cpu().numpy())
            
    avg_test_loss = total_loss / len(test_loader)
    
    all_hazards = np.concatenate(all_hazards).squeeze(-1)
    all_times = np.concatenate(all_times).squeeze(-1)
    all_censorships = np.concatenate(all_censorships).squeeze(-1)
    
    censorship_col = args.target_col.split('_')[0] + '_censorship'
    if censorship_col not in df_clinical_train.columns:
        censorship_col = [c for c in df_clinical_train.columns if 'censor' in c.lower() or 'status' in c.lower()][0]
        
    # TUTAJ NASTĄPIŁA ZMIANA: 'survival_time' zastąpiono kluczem 'time'
    survival_info_train = {
        'censorship': df_clinical_train[censorship_col].values.astype(np.int32),
        'time': df_clinical_train[args.target_col].values.astype(np.float32)
    }
    
    c_index, c_index_ipcw = compute_survival_metrics(all_censorships, all_times, all_hazards, survival_info_train)
    
    results = {
        "loss": avg_test_loss,
        "c_index": float(c_index),
        "c_index_ipcw": float(c_index_ipcw)
    }
    
    if not args.test:
        wandb.log({
            f"survival_test/fold_{fold}_loss": avg_test_loss,
            f"survival_test/fold_{fold}_c_index": c_index,
            f"survival_test/fold_{fold}_c_index_ipcw": c_index_ipcw,
        })
        
    save_json(result_dir_fold, "test_summary.json", results)
    return results


# ── PĘTLA K-FOLD PIPELINE (TRAIN + TEST) ──────────────────────────────────
def k_fold_pipeline(args, device, df_rna_binned, gene_ids):
    final_res = {}
    losses = []
    c_indices = []
    c_indices_ipcw = []

    for i in range(args.folds):
        if args.mode == "train":
            _ = survival_train_fold(args, i, device, df_rna_binned, gene_ids)
            
        fold_results = survival_test_fold(args, i, device, df_rna_binned, gene_ids)
        final_res[f'Fold{i}'] = fold_results

        losses.append(fold_results["loss"])
        c_indices.append(fold_results["c_index"])
        c_indices_ipcw.append(fold_results["c_index_ipcw"])

    summary = {
        "loss": {"avg": float(np.mean(losses)), "std": float(np.std(losses))},
        "c_index": {"avg": float(np.mean(c_indices)), "std": float(np.std(c_indices))},
        "c_index_ipcw": {"avg": float(np.mean(c_indices_ipcw)), "std": float(np.std(c_indices_ipcw))}
    }
    final_res["Summary"] = summary
    
    save_json(args.result_dir, f'Final_results_{args.omics_type}.json', final_res)
    
    print("\n" + "="*50)
    print(f" FINAL K-FOLD PERFORMANCE SUMMARY ({args.folds} FOLDS) ")
    print("="*50)
    print(f"Metric       | Average    ± Standard Dev.")
    print("-"*50)
    print(f"Cox Loss     | {summary['loss']['avg']:.4f}     ± {summary['loss']['std']:.4f}")
    print(f"C-Index      | {summary['c_index']['avg']:.4f}     ± {summary['c_index']['std']:.4f}")
    print(f"C-Index IPCW | {summary['c_index_ipcw']['avg']:.4f}     ± {summary['c_index_ipcw']['std']:.4f}")
    print("="*50 + "\n")
    
    return final_res

# ── GŁÓWNY PUNKT WEJŚCIA ──────────────────────────────────────────────────
def main(args):
    if not args.test:
        wandb.init(project="scgpt-finetuning-kfold-survival", config=vars(args))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.result_dir = os.path.join(args.result_dir, args.task, args.exp_code)
    args.log_dir = os.path.join(args.log_dir, args.task, args.exp_code)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("Inicjalizacja słownika i przetwarzanie matrycy rna...")
    _, vocab = get_scgpt_model(args.model_path, device='cpu', eval=True, do_mvc=False)
    global PAD_ID
    PAD_ID = vocab['<pad>']

    # Wczytanie i zapakowanie w obiekt AnnData identycznie jak w oryginalnym pliku maskującym
    df_rna = pd.read_csv(args.expression_data_path, index_col=0)
    adata = ad.AnnData(X=df_rna.values)
    adata.obs_names = df_rna.index
    adata.var_names = df_rna.columns

    # Odfiltrowanie genów obecnych w słowniku scGPT
    genes_in_vocab = [g for g in adata.var_names if g in vocab]
    adata = adata[:, genes_in_vocab].copy()

    # Logika wyboru genów: z pliku JSON lub automatyczne HVG przy użyciu Scanpy
    if not args.gene_list:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
        adata = adata[:, adata.var['highly_variable']]
        print(f"Wyselekcjonowano top {args.n_hvg} genów jako HVG za pomocą Scanpy.")
    else:
        if not args.gene_list_path:
            raise ValueError("Flaga --gene_list wymaga zdefiniowania ścieżki --gene_list_path!")
        with open(args.gene_list_path, 'r') as f:
            gene_list = json.load(f)
        gene_list = [g for g in gene_list if g in adata.var_names]
        adata = adata[:, gene_list]
        print(f"Używam {len(gene_list)} genów z zewnętrznej listy pliku JSON.")

    gene_ids = vocab(adata.var_names.tolist())

    # Przetwarzanie i transformacja przez binning ekspresji
    X = np.asarray(adata.X)
    X_binned = np.stack([binning(row, n_bins=args.n_bins) for row in X])
    df_rna_binned = pd.DataFrame(X_binned, index=adata.obs_names, columns=adata.var_names)

    # Uruchomienie zintegrowanego pipeline'u
    k_fold_pipeline(args, device, df_rna_binned, gene_ids)
    
    if not args.test:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival Prediction with scGPT Adapter - 5 Fold')

    # Parametry uczenia i preprocessingu
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
    parser.add_argument('--loss_fn', type=str, default='cox')
    parser.add_argument('--n_hvg', type=int, default=2000, help="Liczba genów do wyboru przez Scanpy, jeśli brak pliku JSON")
    parser.add_argument('--n_bins', type=int, default=51, help="Liczba przedziałów binningu ekspresji genów")

    # Ścieżki danych i eksperymentu
    parser.add_argument('--task', type=str, default='dss_survival_brca')
    parser.add_argument('--target_col', type=str, default='dss_survival_days')
    parser.add_argument('--data_source', type=str, required=True, help="Ścieżka do katalogu nadrzędnego klinicznego (brca_clinical)")
    parser.add_argument('--expression_data_path', type=str, required=True, help="Ścieżka bezpośrednia do pliku TCGA-BRCA.star_tpm.csv")
    
    # Warunkowy wybór genów
    parser.add_argument('--gene_list', action="store_true", help="Jeśli włączone, wymusza ładowanie listy genów z pliku JSON zamiast kalkulacji HVG")
    parser.add_argument('--gene_list_path', type=str, default=None, help="Ścieżka do pliku .json z listą genów")
    parser.add_argument('--model_path', default="../papers/scgpt/save/whole_human")
    
    # Dyrektywy zapisu i konfiguracji
    parser.add_argument('--result_dir', default='results')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--exp_code', type=str, default='scgpt_test')
    parser.add_argument('--omics_type', default='rna_clean_norm')
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    main(args)