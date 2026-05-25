import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import sys

adapter_root = "/storage/scratch/2370352/my-research/adapter_premium"

if adapter_root not in sys.path:
    sys.path.insert(0, adapter_root)
    # Dodajmy też katalog nadrzędny, na wypadek gdyby import szukał z poziomu root:
    sys.path.insert(0, os.path.dirname(adapter_root)) 
from adapter_premium.utils import get_scgpt_model

# --- Kopia Twojej implementacji LoRA z poprzedniego skryptu ---
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

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
    print(f"[LoRA] Replaced {replaced} linear layers.")


# --- Główny Wrapper Modelu scGPT do Survival Prediction ---
class scGPTSurvivalModel(nn.Module):
    def __init__(self, model_path, gene_names, n_bins=51, lora=True, loss_fn=None):
        super().__init__()
        self.n_bins = n_bins
        self.loss_fn = loss_fn
        
        # 1. Załaduj bazowy model scGPT
        print(f"[scGPT] Loading core model from {model_path}...")
        self.scgpt_model, self.vocab = get_scgpt_model(model_path, device='cpu', eval=False, do_mvc=False)
        self.PAD_ID = self.vocab['<pad>']
        
        # 2. Mapowanie genów z Twojego datasetu na słownik scGPT
        self.gene_names = list(gene_names)
        self.gene_ids_all = np.array([self.vocab[g] if g in self.vocab else self.PAD_ID for g in self.gene_names])
        self.valid_gene_mask = self.gene_ids_all != self.PAD_ID
        
        print(f"[scGPT] Dataset genes: {len(gene_names)} | Valid in vocab: {sum(self.valid_gene_mask)}")
        
        # 3. Głowa regresyjna do szacowania ryzyka Coxa (scGPT ukryty wymiar to zazwyczaj 512)
        hidden_dim = self.scgpt_model.config.d_model if hasattr(self.scgpt_model, 'config') else 512
        self.survival_head = nn.Linear(hidden_dim, 1)
        
        # 4. Obsługa lory / zamrażania wag
        if lora:
            for p in self.scgpt_model.parameters():
                p.requires_grad = False
            replace_linear_with_lora(self.scgpt_model)
            
            # Odblokowujemy wejścia wartościowe i enkoder/dekoder warstw pomocniczych
            for name, module in self.scgpt_model.named_modules():
                if "value_encoder" in name:
                    for p in module.parameters():
                        p.requires_grad = True
        
        # Głowa survivalowa zawsze musi kalkulować gradienty
        for p in self.survival_head.parameters():
            p.requires_grad = True

    def _torch_binning(self, x):
        """ Szybki, wektorowy binning realizowany bezpośrednio na GPU w PyTorch. """
        # x kształt: [batch_size, num_genes]
        # Pozostawiamy 0 jako zero, wartości dodatnie mapujemy na przedziały 1 do n_bins-1
        eps = 1e-8
        binned = torch.zeros_like(x, dtype=torch.long)
        mask = x > 0
        if mask.any():
            # Normalizacja min-max wierszami dla niezerowych wartości
            min_val = x.min(dim=1, keepdim=True)[0]
            max_val = x.max(dim=1, keepdim=True)[0]
            norm_x = (x - min_val) / (max_val - min_val + eps)
            binned[mask] = (norm_x[mask] * (self.n_bins - 2)).long() + 1
        return binned

    def forward(self, rna, label=None, censorship=None):
        device = rna.device
        batch_size = rna.shape[0]
        
        # 1. Filtrowanie i binning surowych wartości RNA ekspresji
        rna_filtered = rna[:, self.valid_gene_mask]
        gene_ids_filtered = torch.tensor(self.gene_ids_all[self.valid_gene_mask], dtype=torch.long, device=device)
        
        binned_rna = self._torch_binning(rna_filtered)
        
        # 2. Dynamiczne pakowanie rzadkich danych (Sparse item collation) dla scGPT
        # Zbieramy tylko te geny, które w danej próbce mają ekspresję > 0
        src_list = []
        values_list = []
        
        for i in range(batch_size):
            nonzero_mask = binned_rna[i] > 0
            if nonzero_mask.sum() == 0:
                # Fallback w przypadku braku jakiejkolwiek ekspresji
                src_list.append(gene_ids_filtered[:1])
                values_list.append(binned_rna[i, :1])
            else:
                src_list.append(gene_ids_filtered[nonzero_mask])
                values_list.append(binned_rna[i, nonzero_mask])
                
        max_len = max(len(x) for x in src_list)
        
        # Tworzenie tensorów paddingu
        padded_src = torch.full((batch_size, max_len), self.PAD_ID, dtype=torch.long, device=device)
        padded_values = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        pad_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device) # True oznacza padding dla scGPT
        
        for i, (src, val) in enumerate(zip(src_list, values_list)):
            length = len(src)
            padded_src[i, :length] = src
            padded_values[i, :length] = val
            pad_mask[i, :length] = False
            
        # 3. Wywołanie enkodera transformera scGPT
        # scGPT zwraca słownik; pobieramy ukryte stany transformera lub pre-wyliczony cell embedding
        scgpt_outputs = self.scgpt_model(
            src=padded_src,
            values=padded_values,
            src_key_padding_mask=pad_mask,
            CLS=False, CCE=False, MVC=False, ECS=False
        )
        
        # Pobieramy reprezentację komórkową. Jeśli nie ma klucza 'cell_emb', uśredniamy wyjścia tokenów genu
        if "cell_emb" in scgpt_outputs:
            cell_embeddings = scgpt_outputs["cell_emb"]
        else:
            # Rezerwowe uśrednianie wyjść z transformera z uwzględnieniem maski paddingu
            token_embeddings = scgpt_outputs["transformer_output"] # [Batch, SeqLen, HiddenDim]
            analysis_mask = (~pad_mask).unsqueeze(-1).float()
            cell_embeddings = (token_embeddings * analysis_mask).sum(dim=1) / analysis_mask.sum(dim=1).clamp(min=1)

        # 4. Przejście przez głowę survivalową w celu kalkulacji hazardu (risk score)
        risk = self.survival_head(cell_embeddings) # Kształt: [Batch, 1]
        
        # 5. Budowa słowników wyjściowych dopasowanych do Twojej pętli train_loop
        output_results = {'risk': risk}
        log_dict = {}
        
        if label is not None and censorship is not None and self.loss_fn is not None:
            # Obliczenie straty Coxa bezpośrednio na podstawie wyjściowego score
            loss = self.loss_fn(risk, label, censorship)
            output_results['loss'] = loss
            log_dict['loss'] = loss.item()
            
        return output_results, log_dict