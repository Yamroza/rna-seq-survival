from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int] = [512, 256], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> dict:
        if return_embedding:
            hidden_features = self.net[:-1](x)
            return {"embedding": hidden_features}

        logits = self.net(x)
        probs  = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


# TODO zmieniń nazwe na lepszą
# ── SCGPT + MLP ───────────────────────────────────────────────────────────────
class scGPTClassifier(nn.Module):
    def __init__(self, scgpt_model, head_model, num_classes: int, emb_dim: int = 512,
                 hidden_dims: list[int] = [512, 256], dropout: float = 0.1):
        super().__init__()

        self.encoder = scgpt_model
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = head_model

    @torch.no_grad()
    def encode(self, src, values, mask):
        return self.encoder(src, values, src_key_padding_mask=mask)  # (B, emb_dim)

    def forward(self, src, values, mask, return_embedding=False):
        emb = self.encode(src, values, mask)["cell_emb"]
        return self.classifier(emb, return_embedding)


# ── MIXUP LOSS ────────────────────────────────────────────────────────────────
def mixup_loss(logits, labels, lambda_):
    """
    logits:  (B, num_classes)
    labels:  (B, 2)  — [:, 0] oryginalna komórka, [:, 1] mixup partner
    lambda_: (B,)    — współczynnik interpolacji
    """
    criterion   = nn.CrossEntropyLoss(reduction="none")
    loss_main   = criterion(logits, labels[0]) 
    loss_random = criterion(logits, labels[1])
    return (lambda_ * loss_main + (1 - lambda_) * loss_random).mean()

def universal_mixup_loss(logits, labels, lambdas):
    # Upewnij się, że mają wymiar [B, N]
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    if lambdas.dim() == 1:
        lambdas = lambdas.unsqueeze(1)

    B, C = logits.shape
    N = min(labels.shape[1], lambdas.shape[1])

    labels = labels[:, :N]
    lambdas = lambdas[:, :N]

    # Tworzymy soft target: [B, C]
    soft_targets = torch.zeros(B, C, device=logits.device)

    # Zamieniamy labels -> one-hot i ważymy lambdami
    soft_targets.scatter_add_(
        dim=1,
        index=labels.long(),
        src=lambdas
    )

    # log_softmax zamiast wielu CrossEntropy
    log_probs = F.log_softmax(logits, dim=1)

    # Soft cross-entropy
    loss = -(soft_targets * log_probs).sum(dim=1)

    return loss.mean()


# ── TRAIN EPOCH ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    if hasattr(model, "encoder"):
        model.encoder.eval()

    total_loss = 0
    correct_main, total = 0, 0

    for src, values, mask, labels, lambdas in tqdm(loader):
        # Przenoszenie na device (labels i lambdas to tensory B x N)
        src = src.to(device)
        values = values.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        lambdas = lambdas.to(device)

        # Forward i Loss
        optimizer.zero_grad()
        out = model(src, values, mask)
        logits = out["logits"]
        
        # universal_mixup_loss sam iteruje po wszystkich etykietach w labels
        loss = universal_mixup_loss(logits, labels, lambdas)        
        loss.backward()
        optimizer.step()

        # Statystyki - Soft Accuracy uwzględniające wszystkie klasy z Mixupu
        with torch.no_grad():
            preds = logits.argmax(-1) # Kształt: [Batch]
            batch_size = preds.size(0)
            num_mixed = labels.shape[1]
            
            batch_soft_correct = 0.0
            
            # Sprawdzamy trafienia dla każdej domieszkanej klasy
            for i in range(num_mixed):
                curr_labels = labels[:, i]
                curr_lambdas = lambdas[:, i]
                
                # Gdzie predykcja zgadza się z i-tą etykietą (0 lub 1)
                match = (preds == curr_labels).float()
                
                # Mnożymy trafienia przez wagę (lambdę) tej klasy
                batch_soft_correct += (match * curr_lambdas).sum().item()
                
            correct_main += batch_soft_correct # Używamy starej zmiennej, albo zmień nazwę na soft_correct
            total += batch_size
            total_loss += loss.item()

    return {
        "loss": total_loss / len(loader),
        "acc": correct_main / total,
    }

# ── EVAL EPOCH ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()

    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for src, values, mask, labels, _ in tqdm(loader):
        src, values, mask = src.to(device), values.to(device), mask.to(device)
        label_main = torch.stack([l[0] for l in labels]).to(device)

        out        = model(src, values, mask)
        total_loss += criterion(out["logits"], label_main).item()
        correct    += (out["logits"].argmax(-1) == label_main).sum().item()
        total      += len(label_main)

    return total_loss / len(loader), correct / total


def load_trained_model(checkpoint_path, scgpt_model, num_classes=21, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_config = checkpoint['config']
    num_classes = checkpoint.get('num_classes', 21)
    
    head_model = MLPClassifier(
        input_dim=saved_config.get('emb_dim', 512), 
        num_classes=num_classes,
        hidden_dims=saved_config.get('hidden_dims', [512, 256]),
        dropout=saved_config['dropout']
    )
    model = scGPTClassifier(
        scgpt_model, 
        head_model, 
        num_classes=num_classes, 
        dropout=saved_config['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, saved_config
