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

LR=0.00005
BATCH_SIZE=128
EPOCHS=100
DROPOUT=0.5
DATA_PATH="data_new/train.h5ad"
DATASET="bulkDataset"

SAVE_NAME="best_scgpt_lr_${LR}_bs_${BATCH_SIZE}_ep_${EPOCHS}_drop_${DROPOUT}.pt"
SAVE_PATH="checkpoints/${SAVE_NAME}"

python train_adapter.py \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --dropout $DROPOUT \
    --save_path $SAVE_PATH \
    --data_path $DATA_PATH
