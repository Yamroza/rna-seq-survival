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

LR=${LR:-0.00005}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-25}
DROPOUT=${DROPOUT:-0.5}
DATA_PATH=${DATA_PATH:-"data_new/train.h5ad"}
SEQ_LENGTH=${SEQ_LENGTH:-2000}
HIDDEN=${HIDDEN:-"512 256"}
DATASET=${DATASET:-"bulkDataset"}
EXP_CODE=${EXP_CODE:-"adapter_premium_donor"}

# LR=0.00005
# BATCH_SIZE=128
# EPOCHS=25
# DROPOUT=0.5
# DATA_PATH="data_new/train.h5ad"
# SEQ_LENGTH=2000
# HIDDEN="512 256"
# DATASET="bulkDataset"

DATA_FILENAME=$(basename "$DATA_PATH")
DATA_FILENAME="${DATA_FILENAME%.*}"

SAVE_CONFIG="scgpt_dataset_${DATASET}_train_data_${DATA_FILENAME}_lr_${LR}_bs_${BATCH_SIZE}_ep_${EPOCHS}_drop_${DROPOUT}_seqlen_${SEQ_LENGTH}_hiddims_${HIDDEN// /_}"
SAVE_NAME="epoch.pt"
SAVE_PATH="checkpoints/the_great/${SAVE_CONFIG}/${SAVE_NAME}"

echo "====================================================="
echo "STARTING EVALUATION LOOP FOR ALL EPOCHS"
echo "====================================================="

# Pętla iterująca po każdej epoce
for (( i=1; i<=$EPOCHS; i++ ))
do
    echo "--- PROCESSING EPOCH $i ---"
    
    # Dynamicznie ustawiamy ścieżkę do checkpointu danej epoki
    CHECK_PATH="${SAVE_PATH%.pt}_${i}.pt"
    
    # Sprawdzamy, czy plik istnieje, żeby uniknąć błędów
    if [ ! -f "$CHECK_PATH" ]; then
        echo "Warning: Checkpoint $CHECK_PATH not found, skipping..."
        continue
    fi

    # 1. GENERATING EMBEDDINGS
    FILENAME="TCGA-BRCA.star_tpm"
    DATA_PATH="../data/0_data_for_mlp/${FILENAME}.csv"
    
    # Dodajemy numer epoki do folderu zapisu, żeby embeddingi się nie nadpisywały
    EMB_SAVE_DIR="embeddings/the_great/${SAVE_CONFIG}/epoch_${i}"
    EMB_SAVE_PATH="${EMB_SAVE_DIR}/${FILENAME}.json"

    echo "Generating embeddings ..."
    conda activate scgpt # Upewnij się, że jesteś w dobrym env
    python get_adapter_embeddings.py \
        --data_path $DATA_PATH \
        --check_path $CHECK_PATH \
        --save_path $EMB_SAVE_PATH

    # 2. SURVIVAL PREDICTION
    conda deactivate
    conda activate myenv

    DATA_TYPE='json'
    DATA_SOURCE="/scratch/2370352/my-research/data/clinical_data/brca_clinical"
    TASK="dss_survival_brca"
    MODEL="mlp"
    OMICS_TYPE="${FILENAME}.json"
    MAX_EPOCHS_SURV=50
    
    # Dodajemy epoch do EXP_CODE, żeby w WandB/logach widzieć różnicę
    CURRENT_EXP_CODE="${EXP_CODE}/${SAVE_CONFIG}/epoch_${i}"

    echo "Survival prediction ..."
    # TRAIN SURVIVAL
    python ../ts_survival_prediction/src/main.py \
        --model $MODEL \
        --max_epochs $MAX_EPOCHS_SURV \
        --mode train \
        --task $TASK \
        --data_source $DATA_SOURCE \
        --exp_code "$CURRENT_EXP_CODE" \
        --omics_type $OMICS_TYPE \
        --data_type $DATA_TYPE \
        --folds 5 \
        --expression_data_path $EMB_SAVE_DIR

    # TEST SURVIVAL
    python ../ts_survival_prediction/src/main.py \
        --model $MODEL \
        --mode test \
        --task $TASK \
        --data_source $DATA_SOURCE \
        --data_type $DATA_TYPE \
        --exp_code "$CURRENT_EXP_CODE" \
        --omics_type $OMICS_TYPE \
        --expression_data_path $EMB_SAVE_DIR

done

echo "====================================================="
echo "GENERATING FINAL PLOT"
echo "====================================================="

# Ścieżka do wyników (tam gdzie są foldery epoch_1, epoch_2...)
RESULTS_BASE_DIR="results/the_great/${EXP_CODE}/${SAVE_CONFIG}"

python acc_vs_c_index_check_vis.py --results_dir "$RESULTS_BASE_DIR"


echo "====================================================="
echo "ALL EXPERIMENTS FINISHED"
echo "====================================================="