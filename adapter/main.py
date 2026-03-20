# main.py

import torch
import scanpy as sc

from src.dataset import get_dataloaders
from src.model import scGPTAdapterModel
from src.train import (
    train_one_epoch, evaluate, evaluate_full,
    save_checkpoint, load_checkpoint,
)
from src.loader import load_scgpt_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. DANE
    adata_train = sc.read_h5ad("data/train.h5ad")
    adata_test  = sc.read_h5ad("data/test.h5ad")

    # 2. SCGPT
    scgpt, vocab = load_scgpt_model(
        model_dir="/scratch/2370352/my-research/papers/scgpt/save/whole_human",
        device=device,
    )

    # 3. DATALOADERS
    train_loader, test_loader, num_classes = get_dataloaders(
        adata_train, adata_test,
        vocab=vocab,
        batch_size=32,
        max_seq_len=1200,
    )

    # 4. MODEL
    SCGPT_EMBSIZE = 512
    model = scGPTAdapterModel(
        scgpt_model=scgpt,
        input_dim=SCGPT_EMBSIZE,
        latent_dim=256,
        num_classes=num_classes,
        vocab=vocab,
    ).to(device)

    # 5. OPTIMIZER + SCHEDULER
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6,
    )

    # 6. TRENING
    best_val_acc = 0.0

    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch=epoch,
        )
        val_loss, val_acc = evaluate(
            model, test_loader, device, epoch=epoch,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d} │ "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} │ "
            f"val   loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        # Zapisz najlepszy checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                path="checkpoints/best_model.pt",
            )

    # 7. FINALNA EWALUACJA
    print("\n── Finalna ewaluacja ──")

    # Wczytaj najlepszy model przed raportem
    load_checkpoint(model, optimizer=None, path="checkpoints/best_model.pt", device=device)

    # id_to_type potrzebny do nazw klas w raporcie
    id_to_type = {v: k for k, v in train_loader.dataset.type_to_id.items()}

    acc, report, fig = evaluate_full(model, test_loader, device, id_to_type)

    print(f"\nTest accuracy: {acc:.4f}")
    print(report)
    fig.savefig("checkpoints/confusion_matrix.png", dpi=150)
    print("Confusion matrix zapisana → checkpoints/confusion_matrix.png")


if __name__ == "__main__":
    main()