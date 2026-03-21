import sys
import os
import argparse
import json

import anndata as ad
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(".."))
from utils import get_scgpt_model, binning
from premium_datasets import scDataset, bulkDataset, predictionDataset, collate_fn
from adapters import scGPTClassifier, MLPClassifier, train_epoch, eval_epoch, load_trained_model


def main():
    # ── ARGUMENT PARSER ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning Script")
    
    parser.add_argument("--model_path", type=str, default="../papers/scgpt/save/whole_human")
    parser.add_argument("--data_path",  type=str, default="data_new/train.h5ad")
    parser.add_argument("--check_path", type=str, default="checkpoints/scgpt_adapter_checkpoint.pt")
    parser.add_argument("--save_path",  type=str, default="checkpoints/scgpt_adapter_checkpoint.pt")
    parser.add_argument("--dataset",    type=str, default='predictionDataset')

    # Debugowanie / Szybkie testy
    parser.add_argument("--subset",     type=int,   default=None, help="Loader size for debbuging")
    
    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {config.model_path}...")
    scgpt_model, vocab = get_scgpt_model(config.model_path, device='cpu') # scgpt ładowany na cpu, potem przerzucimy classifier

    # ---- SETUP -----
    print(f"Loading data from {config.data_path}...")
    df = pd.read_csv(config.data_path)

    dataset_dict = {
        'scDataset': scDataset,
        'bulkDataset': bulkDataset,
        'predictionDataset': predictionDataset
    }

    model, _ = load_trained_model(config.check_path, scgpt_model, device=device)
    dataset = dataset_dict[config.dataset](df, vocab)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True,  num_workers=0)

    # ---- GET EMB ----
    all_embeddings = []
    all_ids = []

    print(f"Generating embeddings...")
    with torch.no_grad():
        for src, values, mask, ids, _ in tqdm(loader):
            src, values, mask = src.to(device), values.to(device), mask.to(device)
            output = model(src, values, mask, return_embedding=True)

            emb = output["embedding"]

            all_embeddings.append(emb.cpu().numpy())
            all_ids.append(ids)

    flat_ids = [str(item) for sublist in all_ids for item in sublist]
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    results_dict = {
        cell_id: emb.tolist() 
        for cell_id, emb in zip(flat_ids, final_embeddings)
    }

    # ----- SAVE ------
    with open(config.save_path, 'w', encoding='utf-8') as f:
        for cell_id, emb in results_dict.items():
            record = {"id": cell_id, "embedding": emb}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()