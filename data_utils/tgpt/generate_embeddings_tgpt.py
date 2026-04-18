import argparse
import gzip
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


class LineDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines
        self.regex = re.compile(r'\-|\.')
    def __getitem__(self, i):
        return self.regex.sub('_', self.lines[i])
    def __len__(self):
        return len(self.lines)


def main():
    parser = argparse.ArgumentParser(description="extract tgpt embeddings from gene sequences")
    
    parser.add_argument("--filename",           type=str, default="TCGA-OV.star_tpm", help="name of the input tsv file (without extension)")
    parser.add_argument("--data_for_mlp_dir",   type=str, default="../data/0_data_for_mlp")
    parser.add_argument("--data_for_tgpt_dir",  type=str, default="../data/0_adata_for_tgpt")
    parser.add_argument("--output_dir",         type=str, default="../data/0_data_for_mlp")
    parser.add_argument("--max_len",            type=int, default=64, help='number of top genes used for analysis')
    parser.add_argument("--batch_size",         type=int, default=64)

    args = parser.parse_args()
    print(f"processing: {args.filename}")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # paths
    text_file = os.path.join(args.data_for_tgpt_dir, f"{args.filename}.txt.gz") ## Gene symbols ranked by expression
    mlp_csv_path = os.path.join(args.data_for_mlp_dir, f"{args.filename}.csv")
    output_csv = os.path.join(args.output_dir, f"tgpt_embeddings_{args.filename}_max_len_{args.max_len}.csv")

    df = pd.read_csv(mlp_csv_path)

    # load model and tokenizer
    # using lixiangchun's gpt-based transcriptome model
    checkpoint = "lixiangchun/transcriptome-gpt-1024-8-16-64" ## Pretrained model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

    model = GPT2LMHeadModel.from_pretrained(checkpoint,output_hidden_states = True).transformer
    model = model.to(device)
    model.eval()

    # step 1: run inference
    embeddings = extract_tgpt_embeddings(
        text_file=text_file,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=args.max_len,
        batch_size=args.batch_size
    )

    # step 2: map back to sample names
    # we read the original mlp csv to get the correct sample order
    df_original = pd.read_csv(mlp_csv_path)
    sample_names = df_original.iloc[:, 0].values

    # step 3: format and save
    emb_cols = [f"emb{i+1}" for i in range(embeddings.shape[1])]
    df_output = pd.DataFrame(embeddings, columns=emb_cols)
    
    # insert sample names at the beginning with an empty header as requested
    df_output.insert(0, "", sample_names)

    df_output.to_csv(output_csv, index=False)
    
    print(f"done! embeddings saved to: {output_csv}")
    print(f"final shape: {df_output.shape}")


def extract_tgpt_embeddings(text_file, model, tokenizer, device, max_len=64, batch_size=64):
    """
    passes gene sequences through tgpt and extracts mean hidden states.
    """
    lines = [s.decode().strip() for s in gzip.open(text_file, "r").readlines()]

    ds = LineDataset(lines)
    dl = DataLoader(ds, batch_size=batch_size)

    Xs = []
    for a in tqdm(dl, total=len(dl)):
        batch = tokenizer(a, max_length= max_len, truncation=True, padding=True, return_tensors="pt")

        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            x = model(**batch)
        
        eos_idxs = batch.attention_mask.sum(dim=1) - 1
        xx = x.last_hidden_state
        
        result_list = [[] for i in range(len(xx))]

        for j, item in enumerate(xx):
            result_list[j] = item[1:int(eos_idxs[j]),:].mean(dim =0).tolist()
            
        Xs.extend(result_list)
        
    return np.stack(Xs)


if __name__ == "__main__":
    main()