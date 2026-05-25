import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import wandb

import warnings
warnings.filterwarnings('ignore')

from utils import get_scgpt_model, binning

# Ustalamy ziarno dla powtarzalności wyników
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    # Dane wejściowe (Matryca ekspresji + Dane kliniczne)
    parser.add_argument("--expression_dataset", type=str, required=True, help="np. TCGA-BRCA.star_tpm.csv")
    parser.add_argument("--clinical_dataset", type=str, required=True, help="Ścieżka do pliku z przeżywalnością, np. train_filtered.csv")
    parser.add_argument("--data_path", type=str, default="../data/")
    parser.add_argument("--gene_list_path", type=str, default="../data/hvg_genes_lists/TCGA-BRCA.star_tpm_hvg_2000.json")
    
    # Parametry scGPT i przeżywalności
    parser.add_argument("--target_col", type=str, default="OS_time", help="Kolumna z czasem przeżycia")
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--n_bins", type=int, default=51)
    
    # Parametry uczenia
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lora", action="store_true", help="Użyj LoRA dla warstw uwagi scGPT")
    parser.add_argument("--test", action="store_true", help="Ignoruj logowanie do wandb podczas testów lokalnych")

    return parser.parse_args()

args = parse_args()

# ── WANDB SETUP ───────────────────────────────────────────────────────────
cohort_name = os.path.splitext(args.expression_dataset)[0]
run_name = f"survival_{cohort_name}_bins{args.n_bins}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if args.lora:
    run_name = f"survival_{cohort_name}_bins{args.n_bins}_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not args.test:
    wandb.init(
        project="scgpt-finetuning-survival",
        name=run_name,
        config=vars(args)
    )

# ── LOSS FUNCTION (COX PARTIAL LOG-LIKELIHOOD) ────────────────────────────
class CoxLoss(nn.Module):
    """
    Ujemna funkcja wiarygodności częściowej Coxa dla danych z cenzurowaniem.
    """
    def __init__(self):
        super().__init__()

    def forward(self, hazards, survival_times, censorships):
        """
        hazards: [batch_size, 1] - logarytm wskaźnika hazardu przewidziany przez model
        survival_times: [batch_size, 1] - rzeczywisty czas przeżycia
        censorships: [batch_size, 1] - status cenzurowania (1 = zdarzenie/zgon, 0 = pacjent ocenzurowany)
        """
        # Sortowanie po czasie przeżycia malejąco (wymóg dla konstrukcji zbioru ryzyka)
        survival_times, idx = torch.sort(survival_times, dim=0, descending=True)
        hazards = hazards[idx].squeeze(-1)
        censorships = censorships[idx].squeeze(-1)

        # Obliczanie log-sum-exp dla zbioru ryzyka (skumulowany mianownik)
        log_risk = torch.logcumsumexp(hazards, dim=0)
        
        # Wyliczanie straty częściowej tylko dla nieocenzurowanych pacjentów (u których wystąpiło zdarzenie)
        uncensored_loss = censorships * (hazards - log_risk)
        loss = -torch.sum(uncensored_loss) / (torch.sum(censorships) + 1e-8)
        return loss

# ── DATASET CLASS ─────────────────────────────────────────────────────────
class ScGPTSurvivalDataset(Dataset):
    def __init__(self, df_rna, df_clinical, gene_ids, target_col):
        """
        df_rna: DataFrame z surową lub znormalizowaną ekspresją, z indeksem 'case_id'
        df_clinical: DataFrame kliniczny zawierający 'case_id', czas przeżycia i status cenzurowania
        gene_ids: Lista ID tokenów genów odpowiadająca kolumnom df_rna
        """
        # Znajdujemy wspólnych pacjentów
        common_patients = sorted(list(set(df_rna.index).intersection(set(df_clinical['case_id']))))
        
        self.df_rna = df_rna.loc[common_patients]
        self.df_clinical = df_clinical.set_index('case_id').loc[common_patients].reset_index()
        
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        
        # Przygotowanie etykiet survivalowych
        self.survival_times = self.df_clinical[target_col].values.astype(np.float32)
        censorship_col = target_col.split('_')[0] + '_censorship'
        self.censorships = self.df_clinical[censorship_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df_rna)

    def __getitem__(self, idx):
        # Pobranie zbindowanych wartości dla pacjenta
        values = torch.tensor(self.df_rna.iloc[idx].values, dtype=torch.float32)
        
        # Filtrowanie zer (podejście scGPT: bierzemy tylko geny o ekspresji > 0)
        nonzero_mask = values > 0
        if nonzero_mask.sum() == 0:
            return self.gene_ids[:1], values[:1], self.survival_times[idx], self.censorships[idx]
            
        src = self.gene_ids[nonzero_mask]
        values = values[nonzero_mask]
        
        return src, values, self.survival_times[idx], self.censorships[idx]

def collate_fn(batch):
    src_list, values_list, survival_times, censorships = zip(*batch)
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
        torch.tensor(censorships, dtype=torch.float32).unsqueeze(-1)
    )

# ── MODEL HEAD FOR REGRESSION ─────────────────────────────────────────────
class ScGPTSurvivalModel(nn.Module):
    """
    Wrapper na scGPT, ekstrahujący embedding pacjenta i mapujący go na prognozę ryzyka.
    """
    def __init__(self, base_scgpt, emb_dim=512):
        super().__init__()
        self.base_model = base_scgpt
        # Prosta głowa regresyjna mapująca embedding komórki/pacjenta na log-hazard
        self.survival_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1, bias=False) # Brak biasu ułatwia stabilizację optymalizacji Coxa
        )

    def forward(self, src, values, src_key_padding_mask):
        # Pobieramy reprezentację z transformera
        outputs = self.base_model(
            src=src,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            CLS=True, # Prosimy model o wygenerowanie reprezentacji globalnej (CLS)
            CCE=False, MVC=False, ECS=False
        )
        
        # Pobieramy embedding z tokena CLS reprezentującego całościowy profil pacjenta
        cell_emb = outputs["cell_emb"] # Kształt: [batch_size, emb_dim]
        
        # Przejście przez głowę Coxa
        hazard_ratio = self.survival_head(cell_emb)
        return hazard_ratio

# ── LORA IMPLEMENTATION (ZACHOWANA Z TWOJEGO PLIKU) ───────────────────────
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
    print(f"[LoRA] Wymieniono warstw liniowych: {replaced}")

# ── DATA PREPARATION ──────────────────────────────────────────────────────
# Wczytanie matrycy ekspresji i pliku klinicznego
df_rna = pd.read_csv(os.path.join(args.data_path, args.expression_dataset), index_col=0)
df_clinical = pd.read_csv(os.path.join(args.data_path, args.clinical_dataset))

# Ładowanie wag bazowych scGPT
model_path = "../papers/scgpt/save/whole_human"
base_scgpt, vocab = get_scgpt_model(model_path, device='cpu', eval=False, do_mvc=False)
PAD_ID = vocab['<pad>']

# Czyszczenie i dopasowanie genów do słownika scGPT
genes_in_vocab = [g for g in df_rna.columns if g in vocab]
df_rna = df_rna[genes_in_vocab]

if not args.gene_list_path:
    # Selektujemy najbardziej zmienne geny (HVG)
    # Tworzymy tymczasowy obiekt AnnData do selekcji scanpy
    import anndata as ad
    import scanpy as sc
    adata_tmp = ad.AnnData(X=df_rna.values)
    adata_tmp.var_names = df_rna.columns
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=args.n_hvg)
    selected_genes = adata_tmp.var_names[adata_tmp.var['highly_variable']].tolist()
    df_rna = df_rna[selected_genes]
else:
    with open(args.gene_list_path, 'r') as f:
        gene_list = json.load(f)
    selected_genes = [g for g in gene_list if g in df_rna.columns]
    df_rna = df_rna[selected_genes]
    print(f"Używam {len(selected_genes)} genów z podanej listy strukturalnej.")

gene_ids = vocab(df_rna.columns.tolist())

# Wykonanie Binningu danych ekspresji (zastępuje StandardScaler)
X_binned = np.stack([binning(row, args.n_bins) for row in df_rna.values])
df_rna_binned = pd.DataFrame(X_binned, index=df_rna.index, columns=df_rna.columns)

# Inicjalizacja podwójnego datasetu
full_dataset = ScGPTSurvivalDataset(df_rna_binned, df_clinical, gene_ids, args.target_col)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# ── MODEL & OPTIMIZER SETUP ───────────────────────────────────────────────
model = ScGPTSurvivalModel(base_scgpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if args.lora:
    for p in model.base_model.parameters():
        p.requires_grad = False
    replace_linear_with_lora(model.base_model)
    
    # Odblokowanie modułów kodera wartości i nowo dodanej głowy regresyjnej Coxa
    for p in model.survival_head.parameters():
        p.requires_grad = True
    for name, module in model.base_model.named_modules():
        if "value_encoder" in name:
            for p in module.parameters():
                p.requires_grad = True

# Definiujemy optymalizator zbierający wyłącznie parametry wymagające aktualizacji
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = CoxLoss()

# ── TRAINING LOOP ─────────────────────────────────────────────────────────
save_dir = f"checkpoints_survival/{run_name}/"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train Survival]")
    for src, values, pad_mask, survival_times, censorships in train_bar:
        optimizer.zero_grad()

        src = src.to(device)
        values = values.to(device)
        pad_mask = pad_mask.to(device)
        survival_times = survival_times.to(device)
        censorships = censorships.to(device)

        # Forward pass (brak maskowania, model widzi pełny binned profil)
        hazards = model(src=src, values=values, src_key_padding_mask=pad_mask)
        
        loss = criterion(hazards, survival_times, censorships)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 5 == 0 and not args.test:
            wandb.log({
                "train/cox_loss_step": loss.item(),
                "train/lr_step": optimizer.param_groups[0]["lr"]
            })

    scheduler.step()
    avg_train_loss = total_loss / n_batches
    print(f"Epoch {epoch+1}/{args.epochs} | Średnia strata treningowa Cox: {avg_train_loss:.4f}")

    # Zapis punktu kontrolnego
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_train_loss
    }, os.path.join(save_dir, f"scgpt_survival_epoch_{epoch+1}.pt"))

    # ── VALIDATION LOOP ─────────────────────────────────────────────────────
    model.eval()
    val_loss = 0
    
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val Survival]")
    with torch.no_grad():
        for src, values, pad_mask, survival_times, censorships in val_bar:
            src = src.to(device)
            values = values.to(device)
            pad_mask = pad_mask.to(device)
            survival_times = survival_times.to(device)
            censorships = censorships.to(device)

            hazards = model(src=src, values=values, src_key_padding_mask=pad_mask)
            loss = criterion(hazards, survival_times, censorships)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Walidacyjna strata Cox: {avg_val_loss:.4f}")

    if not args.test:
        wandb.log({
            "train/cox_loss_epoch": avg_train_loss,
            "val/cox_loss_epoch": avg_val_loss,
            "epoch": epoch + 1
        })

print("Dostrajanie modelu pod przeżywalność zakończone sukcesem!")