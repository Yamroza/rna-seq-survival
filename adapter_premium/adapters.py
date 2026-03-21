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
    
    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)
        probs  = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}

# ── SCGPT + MLP ───────────────────────────────────────────────────────────────
class scGPTClassifier(nn.Module):
    def __init__(self, scgpt_model, num_classes: int, emb_dim: int = 512,
                 hidden_dims: list[int] = [512, 256], dropout: float = 0.1):
        super().__init__()

        self.encoder = scgpt_model
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = MLPClassifier(emb_dim, num_classes, hidden_dims, dropout)

    @torch.no_grad()
    def encode(self, src, values, mask):
        return self.encoder(src, values, src_key_padding_mask=mask)  # (B, emb_dim)

    def forward(self, src, values, mask):
        emb = self.encode(src, values, mask)["cell_emb"]
        return self.classifier(emb)


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


# ── TRAIN EPOCH ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    model.encoder.eval()

    total_loss = 0
    correct_main, correct_rand, total = 0, 0, 0

    for src, values, mask, labels, lambda_ in tqdm(loader):
        src, values, mask = src.to(device), values.to(device), mask.to(device)

        lab1 = torch.tensor([l[0] for l in labels]).to(device)
        lab2 = torch.tensor([l[1] for l in labels]).to(device)
        labels = (lab1, lab2)
        lambda_tensor = torch.tensor(lambda_, dtype=torch.float32).to(device)

        out  = model(src, values, mask)
        loss = mixup_loss(out["logits"], labels, lambda_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = out["logits"].argmax(-1)
        total_loss   += loss.item()
        correct_main += (preds == lab1).sum().item()
        correct_rand += (preds == lab2).sum().item()
        total        += len(labels[0])

    return {
        "loss":     total_loss / len(loader),
        "acc_main": correct_main / total,
        "acc_rand": correct_rand / total,
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
