#!/bin/bash

#SBATCH --job-name=scGPT_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpua6000
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

# ── ENV ────────────────────────────────────────────────────────────────
source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

# cd adapter_premium

export WANDB_PROJECT="scGPT-classification"

# ── URUCHOMIENIE TRENINGU ─────────────────────────────────────────────────────
python train_adapter.py \
    --lr 0.00005 \
    --batch_size 128 \
    --epochs 100 \
    --dropout 0.1\
    --save_path "checkpoints/best_scgpt_model.pt" \
    --data_path "data_new/train.h5ad" \
