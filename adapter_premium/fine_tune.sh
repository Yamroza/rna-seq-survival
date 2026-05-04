#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=finetune_scgpt

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

echo "Using python from: $(which python)"

python finetune.py \
  --dataset merged_all_samples.csv \
  --data_path ../data/GTEx/GTEx_tpm_per_tissue/processed/ \
  --n_hvg 4000