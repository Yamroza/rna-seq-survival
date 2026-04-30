#!/bin/bash
#SBATCH --partition=gpua6000
source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate myenv

echo "Using python from:"
which python

EXPR_PATH="/scratch/2370352/my-research/data/0_data_for_mlp"
DATA_TYPE='csv'

# Zmienne bazowe dla ścieżek
CLINICAL_BASE_DIR="/scratch/2370352/my-research/data/clinical_data"

# mlp snn pathway_mlp pathway_snn gene_dimaf
MODEL="mlp"
EXP_CODE="limiting_training_set"

LIMIT_SIZES=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
SEEDS=(2 12 22 32 42)

for file in "$EXPR_PATH"/*; do
    for SEED in "${SEEDS[@]}"; do
        for LIMIT_SIZE in "${LIMIT_SIZES[@]}"; do

            omics_type=$(basename "$file")
            EXP_CODE_TMP="${EXP_CODE}/${SEED}/${LIMIT_SIZE}"

            echo "=============================="
            echo "Processing file: $omics_type"
            
            # 1. Wyciąganie nazwy kohorty za pomocą Regex (szukamy 'TCGA-COŚ')
            if [[ "$omics_type" =~ TCGA-([A-Z]+) ]]; then
                # BASH_REMATCH[1] przechowuje to, co było w nawiasach (czyli np. BLCA, OV)
                cohort_upper="${BASH_REMATCH[1]}"
                
                # 2. Zamiana na małe litery (np. BLCA -> blca)
                cohort_lower=$(echo "$cohort_upper" | tr '[:upper:]' '[:lower:]')
            else
                echo "WARNING: Could not extract cohort name (TCGA-...) from $omics_type. Skipping..."
                echo "=============================="
                continue
            fi

            # 3. Dynamiczne budowanie ścieżki i nazwy zadania
            DATA_SOURCE="${CLINICAL_BASE_DIR}/${cohort_lower}_clinical"
            TASK="dss_survival_${cohort_lower}"

            echo "Detected cohort : $cohort_upper"
            echo "Data source     : $DATA_SOURCE"
            echo "Task            : $TASK"
            echo "=============================="

            # TRAIN
            python main.py \
                --model $MODEL \
                --mode train \
                --task $TASK \
                --data_source $DATA_SOURCE \
                --exp_code $EXP_CODE_TMP \
                --omics_type "$omics_type" \
                --data_type $DATA_TYPE \
                --folds 5 \
                --expression_data_path "$EXPR_PATH" \
                --seed $SEED \
                --train_subset $LIMIT_SIZE

            # TEST
            python main.py \
                --model $MODEL \
                --mode test \
                --task $TASK \
                --data_source $DATA_SOURCE \
                --data_type $DATA_TYPE \
                --exp_code $EXP_CODE_TMP \
                --omics_type "$omics_type" \
                --expression_data_path "$EXPR_PATH"
        done
    done
done