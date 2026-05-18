#!/bin/bash
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
EPOCHS=${EPOCHS:-30}
DROPOUT=${DROPOUT:-0.5}
DATA_PATH=${DATA_PATH:-"data_new/train.h5ad"}
SEQ_LENGTH=${SEQ_LENGTH:-2000}
HIDDEN=${HIDDEN:-"512 256"}
DATASET=${DATASET:-"bulkDataset"}
EXP_CODE=${EXP_CODE:-"adapter_premium_donor"}

DATA_FILENAME=$(basename "$DATA_PATH")
DATA_FILENAME="${DATA_FILENAME%.*}"

SAVE_CONFIG="merged_all_samples_hvg2000_bins51_lora_20260506_152407"
CHECKPOINT_BASE_DIR="checkpoints_finetune/${SAVE_CONFIG}"

FILENAME="TCGA-BRCA.star_tpm"
DATA_PATH="../data/0_data_for_mlp/${FILENAME}.csv"

echo "====================================================="
echo "STARTING EVALUATION LOOP FOR ALL EPOCHS"
echo "====================================================="

# Pętla iterująca po każdej epoce
for (( i=1; i<=$EPOCHS; i++ ))
do
    echo "--- PROCESSING EPOCH $i ---"
    
    # Nazwa modelu odpowiadająca danej epoce (bez rozszerzenia .pt dla skryptu)
    MODEL_NAME="scgpt_epoch_${i}"
    
    # Sprawdzamy, czy plik checkpointu istnieje na dysku (z rozszerzeniem .pt)
    if [ ! -f "${CHECKPOINT_BASE_DIR}/${MODEL_NAME}.pt" ]; then
        echo "Warning: Checkpoint ${CHECKPOINT_BASE_DIR}/${MODEL_NAME}.pt not found, skipping..."
        continue
    fi

    # 1. GENERATING EMBEDDINGS
    # Dodajemy numer epoki do folderu zapisu, żeby embeddingi się nie nadpisywały
    EMB_SAVE_DIR="embeddings/finetuned/${SAVE_CONFIG}/epoch_${i}/"

    # echo "Generating embeddings ..."
    # conda activate scgpt # Upewnij się, że jesteś w dobrym env
    # python get_finetuned_scgpt_embeddings.py \
    #     --checkpoint_dir "$CHECKPOINT_BASE_DIR" \
    #     --model_name "$MODEL_NAME" \
    #     --save_path "$EMB_SAVE_DIR"

    # Pobierz nazwę pliku json z folderu embeddingów
    OMICS_TYPE=$(find "$EMB_SAVE_DIR" -maxdepth 1 -name "*.json" -exec basename {} \; | head -n 1)

    if [ -z "$OMICS_TYPE" ]; then
        echo "Error: No JSON file found in $EMB_SAVE_DIR"
        exit 1
    fi

    echo "Detected OMICS_TYPE: $OMICS_TYPE"

    # 2. SURVIVAL PREDICTION
    conda deactivate
    conda activate myenv

    DATA_TYPE='json'
    DATA_SOURCE="/scratch/2370352/my-research/data/clinical_data/brca_clinical"
    TASK="dss_survival_brca"
    MODEL="mlp"
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
        --omics_type "$OMICS_TYPE" \
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
        --omics_type "$OMICS_TYPE" \
        --expression_data_path $EMB_SAVE_DIR

done

echo "====================================================="
echo "GENERATING FINAL PLOT"
echo "====================================================="

# Ścieżka do wyników (tam gdzie są foldery epoch_1, epoch_2...)
RESULTS_BASE_DIR="results/finetuned/${EXP_CODE}/${SAVE_CONFIG}"

python acc_vs_c_index_check_vis.py --results_dir "$RESULTS_BASE_DIR"


echo "====================================================="
echo "ALL EXPERIMENTS FINISHED"
echo "====================================================="