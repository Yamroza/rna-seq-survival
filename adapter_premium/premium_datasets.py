
import sys
import os

sys.path.append(os.path.abspath(".."))

import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import binning
from mixers import LinearTwoCellMixer


class predictionDataset(Dataset):
    def __init__(self, data, vocab, max_genes=2000, n_bins=51):
        self.data = data[:]
        self.data.columns = [vocab[col] for col in self.data.columns]
        self.data = self.data.rename(columns={self.data.columns[0]: 'id'})

        self.max_genes = max_genes
        self.n_bins = n_bins
        self.gene2id = vocab.stoi if hasattr(vocab, "stoi") else vocab.get_stoi()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        row_id = self.data.iloc[idx].id

        gene_ids = self.data.columns[1:]
        values = row.values[1:].astype(np.float32)

        if len(values) > self.max_genes:
            topk = np.argsort(values)[-self.max_genes:]
            gene_ids = gene_ids[topk]
            values = values[topk]

        src = torch.tensor(gene_ids, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        values = binning(values, self.n_bins)
        mask = torch.zeros(len(src), dtype=torch.bool)

        return src, values, mask, row_id, ('lambda_')


class scGPTDataset(Dataset):
    def __init__(self, adata, vocab, mixer=None, max_genes=2000, n_bins=51):
        self.adata = adata
        self.X = adata.X.tocsr()
        self.max_genes = max_genes
        self.n_bins = n_bins
        self.samples = len(adata.obs["cell_type"])
        
        # default 2 cell mixing
        self.mixer = mixer if mixer is not None else LinearTwoCellMixer()

        # GeneVocab → dict
        self.gene2id = vocab.stoi if hasattr(vocab, "stoi") else vocab.get_stoi()

        # Gene mapping
        self.var_to_vocab = np.array([
            self.gene2id.get(g, self.gene2id.get("<unk>", 0))
            for g in adata.var['feature_name'].tolist()
        ])
        self.cell_types = adata.obs["cell_type"].astype("category")
        self.labels = self.cell_types.cat.codes.values.astype(np.int64)
        
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # WYKORZYSTANIE MIXERA: Tutaj dzieje się magia mieszania
        row, labels, lambda_info = self.mixer(idx, self)

        gene_ids = self.var_to_vocab[row.indices]
        values = row.data.astype(np.float32)

        # top-k genes
        if len(values) > self.max_genes:
            topk = np.argsort(values)[-self.max_genes:]
            gene_ids = gene_ids[topk]
            values = values[topk]

        src = torch.tensor(gene_ids, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        values = binning(values, self.n_bins)
        mask = torch.zeros(len(src), dtype=torch.bool)

        return src, values, mask, labels, lambda_info


def collate_fn(batch):
    # Rozpakowanie batcha
    srcs, values, _, labels_list, lambdas_list = zip(*batch)
    batch_size = len(batch)
    max_len = max(len(x) for x in srcs)

    # 1. Padding dla danych wejściowych (bez zmian)
    src_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    val_pad = torch.zeros(batch_size, max_len, dtype=torch.float)
    mask_pad = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i in range(batch_size):
        L = len(srcs[i])
        src_pad[i, :L] = srcs[i]
        val_pad[i, :L] = values[i]
        mask_pad[i, :L] = False 

    # 2. Inteligentna obsługa Etykiet i Wag
    # Sprawdzamy pierwszy element. Jeśli to string, zakładamy tryb "ID/Inference"
    first_label = labels_list[0]
    
    if isinstance(first_label, str):
        # TRYB PREDYKCJI: labels_list to lista ID (stringów)
        labels_final = list(labels_list) 
        lambdas_final = list(lambdas_list) # tu będzie Twoje 'lambda_'
    else:
        # TRYB TRENINGU: labels_list to listy/tensory z etykietami
        labels_tensors = [torch.as_tensor(l, dtype=torch.long).flatten() for l in labels_list]
        lambdas_tensors = [torch.as_tensor(lam, dtype=torch.float32).flatten() for lam in lambdas_list]

        labels_final = pad_sequence(labels_tensors, batch_first=True, padding_value=0)
        lambdas_final = pad_sequence(lambdas_tensors, batch_first=True, padding_value=0.0)

    return src_pad, val_pad, mask_pad, labels_final, lambdas_final