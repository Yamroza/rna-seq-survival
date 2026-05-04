#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=finetune_scgpt

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate scgpt

echo "Using python from: $(which python)"

python finetune.py