
import sys
import os

sys.path.append(os.path.abspath(".."))

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import binning


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


class scDataset(Dataset):
    def __init__(self, adata, vocab, max_genes=2000, n_bins=51):
        self.adata = adata
        self.X = adata.X.tocsr()
        self.max_genes = max_genes
        self.n_bins = n_bins

        # GeneVocab → dict
        self.gene2id = vocab.stoi if hasattr(vocab, "stoi") else vocab.get_stoi()

        # map var_names → vocab ids
        self.var_to_vocab = np.array([
            self.gene2id.get(g, self.gene2id.get("<unk>", 0))
            for g in adata.var['feature_name'].tolist()
        ])
        self.cell_types = adata.obs["cell_type"].astype("category")
        self.label2id = {k: i for i, k in enumerate(self.cell_types.cat.categories)}
        self.labels = self.cell_types.cat.codes.values.astype(np.int64)
        
    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        row = self.X[idx]

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
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return src, values, mask, label, ('lambda_')
    

class bulkDataset(Dataset):
    def __init__(self, adata, vocab, max_genes=2000, n_bins=51):
        self.adata = adata
        self.X = adata.X.tocsr()
        self.max_genes = max_genes
        self.n_bins = n_bins
        self.samples = len(adata.obs["cell_type"])

        # GeneVocab → dict
        self.gene2id = vocab.stoi if hasattr(vocab, "stoi") else vocab.get_stoi()

        # map var_names → vocab ids
        self.var_to_vocab = np.array([
            self.gene2id.get(g, self.gene2id.get("<unk>", 0))
            for g in adata.var['feature_name'].tolist()
        ])
        self.cell_types = adata.obs["cell_type"].astype("category")
        self.label2id = {k: i for i, k in enumerate(self.cell_types.cat.categories)}
        self.labels = self.cell_types.cat.codes.values.astype(np.int64)
        
    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        row = self.X[idx]

        random_idx = random.randint(0, self.samples-1)
        random_row = self.X[random_idx]
        lambda_ = random.randint(0, 100) / 100

        row = lambda_ * row + (1-lambda_) * random_row

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
        labels = torch.tensor([self.labels[idx], self.labels[random_idx]], dtype=torch.long)

        return src, values, mask, labels, lambda_


def collate_fn(batch):
    srcs, values, _, labels, lambda_ = zip(*batch)
    max_len = max(len(x) for x in srcs)
    batch_size = len(batch)

    src_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    val_pad = torch.zeros(batch_size, max_len, dtype=torch.float)
    mask_pad = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding

    for i in range(batch_size):
        L = len(srcs[i])

        src_pad[i, :L] = srcs[i]
        val_pad[i, :L] = values[i]
        mask_pad[i, :L] = False 

    return src_pad, val_pad, mask_pad, labels, lambda_
