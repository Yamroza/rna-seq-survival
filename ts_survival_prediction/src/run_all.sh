#!/bin/bash

EXPR_PATH="/scratch/2370352/my-research/data/0_data_for_mlp"
#0_data_for_mlp"
#scgpt_embeddings"

DATA_SOURCE="/scratch/2370352/my-research/data/clinical_data/brca_clinical"
TASK="dss_survival_brca"

# mlp snn pathway_mlp pathway_snn gene_dimaf
MODEL="pathway_mlp"
EXP_CODE="pathway_mlp"

# NETWORK_SIZE="big"
AGGREGATION_TYPE="concat"

for file in "$EXPR_PATH"/*; do
    omics_type=$(basename "$file")

    echo "=============================="
    echo "Running for omics_type: $omics_type"
    echo "=============================="

    # TRAIN
    python main.py \
        --model $MODEL \
        --mode train \
        --task $TASK \
        --data_source $DATA_SOURCE \
        --exp_code $EXP_CODE \
        --omics_type "$omics_type" \
        --folds 5 \
        --aggregation_type $AGGREGATION_TYPE \
        --expression_data_path "$EXPR_PATH"
        # --network_size $NETWORK_SIZE \


    # TEST
    python main.py \
        --model $MODEL \
        --mode test \
        --task $TASK \
        --data_source $DATA_SOURCE \
        --exp_code $EXP_CODE \
        --omics_type "$omics_type" \
        --aggregation_type $AGGREGATION_TYPE \
        --expression_data_path "$EXPR_PATH"
        # --network_size $NETWORK_SIZE \

done