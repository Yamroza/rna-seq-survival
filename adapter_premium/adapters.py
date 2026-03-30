from tqdm import tqdm
import torch
import torch.nn as nn


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
    # Wymuś, żeby oba miały min. 2 wymiary (Batch, N)
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    if lambdas.dim() == 1:
        lambdas = lambdas.unsqueeze(1)

    # ZABEZPIECZENIE: num_mixed musi być mniejsze lub równe liczbie kolumn w obu tensorach
    num_mixed = min(labels.shape[1], lambdas.shape[1])
    batch_size = logits.shape[0]
    
    total_loss = torch.zeros(batch_size, device=logits.device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for i in range(num_mixed):
        curr_labels = labels[:, i].long()
        curr_lambdas = lambdas[:, i]
        
        loss_comp = criterion(logits, curr_labels)
        total_loss += curr_lambdas * loss_comp
        
    return total_loss.mean()


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

        # Statystyki - tylko dla głównej etykiety (indeks 0)
        with torch.no_grad():
            preds = logits.argmax(-1)
            lab_main = labels[:, 0]
            correct_main += (preds == lab_main).sum().item()
            total += lab_main.size(0)
            total_loss += loss.item()

    return {
        "loss": total_loss / len(loader),
        "acc":  correct_main / total,
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


def load_trained_model(checkpoint_path, scgpt_model, num_classes=7, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_config = checkpoint['config']
    
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
