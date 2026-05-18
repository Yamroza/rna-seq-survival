#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=get_embeddings

#SBATCH --time=9-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

echo "Using python from: $(which python)"

python get_finetuned_scgpt_embeddings.py \
  --checkpoint_dir checkpoints_finetune/merged_all_samples_hvg2000_bins51_20260505_100001 \
  --model_name scgpt_epoch_30 \
  --save_path ../data/0_data_for_mlp_finetuned_scgpt/ 
  # \
  # --all_genes
