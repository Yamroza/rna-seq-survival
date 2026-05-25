#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=finetune_22k
#SBATCH --time=9-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt2

echo "Using python from: $(which python)"

python finetune.py \
  --dataset merged_all_samples.csv \
  --data_path ../data/GTEx/GTEx_tpm_per_tissue/processed/ \
  --n_hvg 22000 \
  --batch_size 8 \
  --lora \
  --test


  # --dataset merged_all_samples.csv \
  # --dataset gene_tpm_v11_bladder_processed.csv \

  # --n_hvg 6000
# --n_hvg 3000
# torchrun --nproc_per_node=1 finetune.py \
# python finetune.py \
  # --gene_list \
