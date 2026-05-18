import sys
import os
import argparse
import json
from pathlib import Path

import warnings
warnings. filterwarnings('ignore')

import anndata as ad
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(".."))
from utils import binning
from premium_datasets import scGPTDataset, predictionDataset, collate_fn, finetunedSCGPTDataset
from adapters import scGPTClassifier, MLPClassifier, train_epoch, eval_epoch, load_trained_model

scgpt_root = os.path.abspath("../papers/scgpt")
if scgpt_root not in sys.path:
    sys.path.insert(0, scgpt_root)

from papers.scgpt.scgpt.tokenizer import GeneVocab
from papers.scgpt.scgpt.model import TransformerModel



def get_scgpt_model(
    checkpoint_dir: str,
    model_name: str,
    additional_files_dir: str,
    device: str = 'cuda',
    use_fast_transformer: bool = True,
    do_mvc: bool = True
) -> torch.nn.Module:
    """
    Get scGPT model and vocab from given path
    """
    # LOAD MODEL
    checkpoint_dir = Path(checkpoint_dir)
    additional_files_dir = Path(additional_files_dir)
    vocab_file = additional_files_dir / "vocab.json"
    model_config_file = additional_files_dir / "args.json"
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    checkpoint = torch.load(checkpoint_path)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    vocab.set_default_index(vocab["<pad>"])
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=do_mvc,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, vocab


def main():
    # ── ARGUMENT PARSER ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning Script")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_finetune/merged_all_samples_hvg2000_bins51_20260505_100001")
    parser.add_argument("--model_name", type=str, default="scgpt_epoch_18")
    parser.add_argument("--additional_files_dir", type=str, default="../papers/scgpt/save/whole_human")
    parser.add_argument("--data_path",  type=str, default="../data/0_data_for_mlp_small_cohorts/TCGA-BRCA.star_tpm.csv")
    parser.add_argument("--save_path",  type=str, default="../data/0_data_for_mlp_finetuned_scgpt/embeddings.json")

    parser.add_argument("--dataset",    type=str, default='predictionDataset')
    # Debugowanie / Szybkie testy
    parser.add_argument("--subset",     type=int,   default=None, help="Loader size for debbuging")
    parser.add_argument("--all_genes", action="store_true", help="Ignore genes from gene file, take all")

    
    
    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {config.checkpoint_dir}...")
    scgpt_model, vocab = get_scgpt_model(
        checkpoint_dir=config.checkpoint_dir,
        model_name= config.model_name,
        additional_files_dir=config.additional_files_dir,
        device=device
    )

    # ---- SETUP -----
    print(f"Loading data from {config.data_path}...")
    df = pd.read_csv(config.data_path)

    gene_list_path = Path(config.checkpoint_dir)
    gene_list_file = gene_list_path / "gene_set.json"
    dataset = finetunedSCGPTDataset(df, vocab, gene_list_file, config.all_genes)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True,  num_workers=0)

    # ---- GET EMB ----
    all_embeddings = []
    all_ids = []

    print(f"Generating embeddings...")
    with torch.no_grad():
        for src, values, mask, ids, _ in tqdm(loader):
            src, values, mask = src.to(device), values.to(device), mask.to(device)
            output = scgpt_model(
                src=src,
                values=values,
                src_key_padding_mask=mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False
            )
            emb = output["cell_emb"]

            all_embeddings.append(emb.cpu().numpy())
            all_ids.append(ids)

    flat_ids = [item for sublist in all_ids for item in sublist]
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    results_dict = {
        cell_id: emb.tolist() 
        for cell_id, emb in zip(flat_ids, final_embeddings)
    }

    # ----- SAVE ------
#   --checkpoint_dir checkpoints_finetune/merged_all_samples_genes_from_list_star_tpm_hvg_3000_bins51_20260505_201712 \
    run_name = os.path.basename(config.checkpoint_dir)
    config.save_path = config.save_path + run_name + '.json'
    if config.all_genes:
        config.save_path = config.save_path.replace(".json", "_all_genes.json")
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    with open(config.save_path, 'w', encoding='utf-8') as f:
        for cell_id, emb in results_dict.items():
            record = {"id": cell_id, "embedding": emb}
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()