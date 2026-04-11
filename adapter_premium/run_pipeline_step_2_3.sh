#!/bin/bash

#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpua6000
#SBATCH --time=3-12:00:00
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

LR=${LR:-0.0001}
HIDDEN=${HIDDEN:-"512 512"}
SEQ_LENGTH=${SEQ_LENGTH:-2000}
DROPOUT=${DROPOUT:-0}

# LR=0.00005
# SEQ_LENGTH=2000
# HIDDEN="512 256"
# DROPOUT=0.5

BATCH_SIZE=128
EPOCHS=20
# DATA_PATH="data_new/train.h5ad"
DATA_PATH="data_new/blkb_common_train.h5ad"
DATASET="donorDataset"

SAVE_CONFIG="first_scgpt_lr_${LR}_bs_${BATCH_SIZE}_ep_${EPOCHS}_drop_${DROPOUT}_seqlen_${SEQ_LENGTH}_hiddims_${HIDDEN// /_}"
SAVE_PATH="checkpoints/the_great/best_scgpt_lr_0.0001_bs_128_ep_20_drop_0_seqlen_2000_hiddims_512_512_20.pt"

echo "====================================================="
echo "GENERATING EMBEDDINGS"
echo "====================================================="

FILENAME="TCGA-BRCA.star_tpm"
DATA_PATH="../data/0_data_for_mlp/${FILENAME}.csv"
EMB_SAVE_DIR="embeddings/the_great/${SAVE_CONFIG}"
# EMB_SAVE_DIR="embeddings/test"
EMB_SAVE_PATH="${EMB_SAVE_DIR}/${FILENAME}.json"

CHECK_PATH="${SAVE_PATH}_${EPOCHS}"

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
EXP_CODE="adapter_premium_donor"
OMICS_TYPE="${FILENAME}.json"
EXPR_PATH="embeddings/the_great"
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
    --expression_data_path $EMB_SAVE_DIR

# TEST
python ../ts_survival_prediction/src/main.py \
    --model $MODEL \
    --mode test \
    --task $TASK \
    --data_source $DATA_SOURCE \
    --data_type $DATA_TYPE \
    --exp_code $EXP_CODE \
    --omics_type $OMICS_TYPE \
    --expression_data_path $EMB_SAVE_DIR
