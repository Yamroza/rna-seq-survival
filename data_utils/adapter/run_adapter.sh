#!/bin/bash
#SBATCH --partition=gpua6000
source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

echo "Using python from:"
which python

python adapter_claude.py \
  --sc_data /scratch/2370352/my-research/data/censusxgene/female_lung_neuron_vs_macrophage.h5ad \
  --model_dir /scratch/2370352/my-research/papers/scgpt/save/whole_human \
  --output_dir ./mlp_a_output \
  --n_epochs 50 \
  --hidden_dim 128 \
  --gene_info_path /scratch/2370352/my-research/data/gene_info_table.csv \
  --gene_col "feature_name" \
  --debug_n 10000