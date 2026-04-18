#!/bin/bash
#SBATCH --partition=gpua6000
source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate myenv

# Funkcja do wyświetlania separatorów dla lepszej czytelności logów
print_step() {
    echo "============================================================"
    echo "STEP: $1"
}

echo "Starting pipeline for filename: ${FILENAME}"
echo "Using python from: $(which python)"

#=========CONFIG============
FILENAME="TCGA-BLCA.star_tpm"
GUNZIP_FILENAME="../../data/raw_tsv_data/${FILENAME}.tsv.gz"
COHORT="blca"

# clinical
N_FOLDS=5

# tgpt
TGPT_MAX_LEN=64
TGPT_TOP_N=200

# scgpt
MODEL_NAME="whole_human"
N_HVG=0
MAX_LENGTH=2000
BATCH_SIZE=32

#=========RUN================

print_step "UNZIPPING DATA"
echo "Unzipping ${GUNZIP_FILENAME}..."
gunzip -k $GUNZIP_FILENAME  # Dodałem -k (keep), żeby nie usuwało oryginału, jeśli wolisz

print_step "PREPROCESSING (unify_star_tpm)"
python unify_star_tpm.py --filename $FILENAME --top_n $TGPT_TOP_N

print_step "CLINICAL DATA PREPARATION"
echo "Creating clinical data for cohort: ${COHORT}"
python create_clinical_data.py --cohort $COHORT

echo "Splitting data into ${N_FOLDS} folds..."
python split_data_into_folds.py --cohort $COHORT --n_folds $N_FOLDS

print_step "FOUNDATION MODELS: tGPT EMBEDDINGS"
echo "Running tGPT with max_len=${TGPT_MAX_LEN}"
python generate_embeddings_tgpt.py --filename $FILENAME --max_len $TGPT_MAX_LEN

print_step "FOUNDATION MODELS: scGPT EMBEDDINGS"
echo "Switching conda environment to 'scgpt'..."
conda deactivate
conda activate scgpt

echo "Running scGPT with model=${MODEL_NAME}, n_hvg=${N_HVG}, batch_size=${BATCH_SIZE}"
python generate_embeddings_scgpt.py \
    --filename $FILENAME \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --n_hvg $N_HVG \
    --batch_size $BATCH_SIZE

print_step "PIPELINE FINISHED SUCCESSFULLY"