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

export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
export WANDB_RESUME="allow"

echo "====================================================="
echo "TRAINING ADAPTER"
echo "====================================================="

LR=${LR:-0.00005}
HIDDEN=${HIDDEN:-"512 256"}
SEQ_LENGTH=${SEQ_LENGTH:-2000}

# LR=0.00005
# SEQ_LENGTH=2000
# HIDDEN="512 256"

BATCH_SIZE=128
EPOCHS=20
DROPOUT=0.5
DATA_PATH="data_new/blkb_common_train.h5ad"
DATASET="donorDataset"

SAVE_NAME="best_scgpt_lr_${LR}_bs_${BATCH_SIZE}_ep_${EPOCHS}_drop_${DROPOUT}_seqlen_${SEQ_LENGTH}_hiddims_${HIDDEN// /_}.pt"
SAVE_PATH="checkpoints/${SAVE_NAME}"

python train_adapter.py \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --dropout $DROPOUT \
    --save_path $SAVE_PATH \
    --data_path $DATA_PATH \
    --seq_length $SEQ_LENGTH \
    --hidden_dims $HIDDEN \
    # --subset 100

echo "====================================================="
echo "GENERATING EMBEDDINGS"
echo "====================================================="

FILENAME="TCGA-BRCA.star_tpm"
DATA_PATH="../data/0_data_for_mlp/${FILENAME}.csv"
EMB_SAVE_PATH="embeddings/${SAVE_NAME}_${FILENAME}.json"

python get_adapter_embeddings.py \
    --data_path $DATA_PATH \
    --check_path $SAVE_PATH \
    --save_path $EMB_SAVE_PATH \

echo "====================================================="
echo "SURVIVAL PREDICTION"
echo "====================================================="

conda deactivate
conda activate myenv

DATA_TYPE='json'
DATA_SOURCE="/scratch/2370352/my-research/data/clinical_data/brca_clinical"
TASK="dss_survival_brca"
MODEL="mlp"
EXP_CODE="adapter_premium"
OMICS_TYPE="${SAVE_NAME}_${FILENAME}.json"
EXPR_PATH="embeddings"
MAX_EPOCHS=50

# TRAIN
python ../ts_survival_prediction/src/main.py \
    --model $MODEL \
    --max_epochs $MAX_EPOCHS \
    --mode train \
    --task $TASK \
    --data_source $DATA_SOURCE \
    --exp_code $EXP_CODE \
    --omics_type $OMICS_TYPE \
    --data_type $DATA_TYPE \
    --folds 5 \
    --expression_data_path $EXPR_PATH

# TEST
python ../ts_survival_prediction/src/main.py \
    --model $MODEL \
    --mode test \
    --task $TASK \
    --data_source $DATA_SOURCE \
    --data_type $DATA_TYPE \
    --exp_code $EXP_CODE \
    --omics_type $OMICS_TYPE \
    --expression_data_path $EXPR_PATH
