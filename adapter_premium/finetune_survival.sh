#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=finetune_surv
#SBATCH --time=9-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt2

echo "Using python from: $(which python)"

# python3 finetune_survival.py \
#     --expression_dataset ../data/0_data_for_mlp/TCGA-BRCA.star_tpm.csv \
#     --clinical_dataset ../data/clinical_data/brca_clinical/splits/0/train_filtered.csv \
#     --data_path /scratch/2370352/my-research/data \
#     --gene_list_path ../data/hvg_genes_lists/TCGA-BRCA.star_tpm_hvg_2000.json \
#     --target_col dss_survival_days \
#     --n_hvg 2000 \
#     --n_bins 51 \
#     --batch_size 32 \
#     --lr 1e-4 \
#     --epochs 30 \
#     --lora \
#     --test

python3 finetune_survival_kfold.py \
    --data_source ../data/clinical_data/brca_clinical \
    --expression_data_path ../data/0_data_for_mlp/TCGA-BRCA.star_tpm.csv \
    --task dss_survival_brca \
    --target_col dss_survival_days \
    --folds 5 \
    --mode train \
    --max_epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --lora \
    --exp_code scgpt_survival_run \
    --test

    # --gene_list_path ../data/hvg_genes_lists/TCGA-BRCA.star_tpm_hvg_2000.json \
