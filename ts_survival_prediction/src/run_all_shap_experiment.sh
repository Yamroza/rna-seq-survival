#!/bin/bash
#SBATCH --partition=gpua6000

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate myenv

echo "Using python from:"
which python

EXPR_PATH="/scratch/2370352/my-research/data/0_data_for_mlp_testing"
DATA_TYPE='csv'

BASE_DATA_SOURCE="/scratch/2370352/my-research/data/clinical_data/brca_clinical/shap_experiment"
TASK="dss_survival_brca"

MODEL="mlp"
BASE_EXP_CODE="mlp_lasso_shap_experiment"

N_OUTER=5

for outer in $(seq 0 $((N_OUTER-1))); do

    echo "=============================="
    echo "Running for outer split: $outer"
    echo "=============================="

    # 🔥 dynamiczne ścieżki
    DATA_SOURCE="${BASE_DATA_SOURCE}/outer_${outer}_held"
    EXP_CODE="${BASE_EXP_CODE}/outer_${outer}_held"

    for file in "$EXPR_PATH"/*; do
        omics_type=$(basename "$file")

        echo "=============================="
        echo "Running for omics_type: $omics_type | outer: $outer"
        echo "=============================="

        # TRAIN
        python main.py \
            --model $MODEL \
            --mode train \
            --task $TASK \
            --data_source $DATA_SOURCE \
            --exp_code $EXP_CODE \
            --omics_type "$omics_type" \
            --data_type $DATA_TYPE \
            --folds 1 \
            --expression_data_path "$EXPR_PATH"

        # TEST
        python main.py \
            --model $MODEL \
            --mode test \
            --task $TASK \
            --data_source $DATA_SOURCE \
            --data_type $DATA_TYPE \
            --exp_code $EXP_CODE \
            --omics_type "$omics_type" \
            --expression_data_path "$EXPR_PATH"

    done
done