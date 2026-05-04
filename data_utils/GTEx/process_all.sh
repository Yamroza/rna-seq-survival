#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=finetune_scgpt

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

echo "Using python from: $(which python)"

DATA_DIR="../../data/GTEx/GTEx_tpm_per_tissue"
SAVE_DIR="../../data/GTEx/GTEx_tpm_per_tissue/processed"

for file in "$DATA_DIR"/*.gct; do
    filename=$(basename "$file")
    filename="${filename%.*}"
    
    echo "Processing $filename ..."
    
    python process_gct_to_tcga.py \
        --filename "$filename" \
        --gtex_dir "$DATA_DIR" \
        --save_dir "$SAVE_DIR"
done