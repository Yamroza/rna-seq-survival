#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=finetune_scgpt_12000

#SBATCH --time=9-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

echo "Using python from: $(which python)"

python get_finetuned_scgpt_embeddings.py \
  --checkpoint_dir checkpoints_finetune/merged_all_samples_genes_from_list_../data/hvg_genes_lists/TCGA-BRCA.star_tpm_hvg_3000_bins51_20260505_201712 \
  --model_name scgpt_epoch_25 \
  --save_path ../data/0_data_for_mlp_finetuned_scgpt/ \
  --all_genes
