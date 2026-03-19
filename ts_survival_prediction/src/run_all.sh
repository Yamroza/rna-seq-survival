#!/bin/bash
#SBATCH --partition=gpua6000
source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate myenv

echo "Using python from:"
which python

EXPR_PATH="/scratch/2370352/my-research/data/1_data_for_mlp_pca"
# 0_data_for_mlp"
#scgpt_embeddings"

# luad_clinical, brca_clinical
DATA_SOURCE="/scratch/2370352/my-research/data/clinical_data/brca_clinical"
TASK="dss_survival_brca"

# mlp snn pathway_mlp pathway_snn gene_dimaf
MODEL="mlp"
EXP_CODE="mlp_pca_128_128"

# NETWORK_SIZE="big"
# AGGREGATION_TYPE="concat"

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
        --expression_data_path "$EXPR_PATH"


    # TEST
    python main.py \
        --model $MODEL \
        --mode test \
        --task $TASK \
        --data_source $DATA_SOURCE \
        --exp_code $EXP_CODE \
        --omics_type "$omics_type" \
        --expression_data_path "$EXPR_PATH"

done