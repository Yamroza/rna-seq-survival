#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=pca_umap_plots
#SBATCH --output=logs_plots_%j.out

source /scratch/2370352/conda/etc/profile.d/conda.sh
conda activate myenv

echo "Using python from: $(which python)"

# Definicja mapowania
declare -A tcga_to_gtex
tcga_to_gtex["TCGA-BLCA"]="bladder"
tcga_to_gtex["TCGA-OV"]="ovary"
tcga_to_gtex["TCGA-BRCA"]="breast_mammary_tissue"
tcga_to_gtex["TCGA-KIRC"]="kidney_cortex kidney_medulla"
tcga_to_gtex["TCGA-CESC"]="cervix_endocervix cervix_ectocervix"
tcga_to_gtex["TCGA-LUAD"]="lung"
tcga_to_gtex["TCGA-UCEC"]="uterus"

# Zmienne pomocnicze (wszystkie kohorty do eksperymentów "mix")
ALL_TCGA="${!tcga_to_gtex[@]}"
ALL_GTEX="bladder ovary breast_mammary_tissue kidney_cortex kidney_medulla cervix_endocervix cervix_ectocervix lung uterus"

# ==============================================================================
# 1. GTEx vs TCGA - the same cohorts (1 vs 1 / 1 vs 2)
# ==============================================================================
echo -e "\n>>> SCENARIO 1: Matching cohorts (TCGA vs GTEx)"
for tcga in $ALL_TCGA; do
    tissues=${tcga_to_gtex[$tcga]}
    echo "Running: $tcga vs $tissues"
    python sample_analysis_script.py \
        --tcga "$tcga" \
        --gtex $tissues
done

# ==============================================================================
# 2. GTEx vs TCGA - some mixed data from different cohorts
# ==============================================================================
echo -e "\n>>> SCENARIO 2: Mixed data from different cohorts"
# Wybieramy np. 3 różne nowotwory i 3 losowe tkanki
python sample_analysis_script.py \
    --tcga TCGA-BLCA TCGA-LUAD TCGA-BRCA \
    --gtex bladder lung breast_mammary_tissue

# ==============================================================================
# 3. All TCGA vs All GTEx vs Various Pseudobulks
# ==============================================================================
echo -e "\n>>> SCENARIO 3: All TCGA vs All GTEx vs Pseudobulks (Grouped by Source)"
# Tutaj używamy flagi --group_by_source, żeby na wykresie zbiło nam to w legendzie 
# do "TCGA", "GTEx" i "Pseudo" (dzięki logice w Twoim skrypcie).
python sample_analysis_script.py \
    --tcga $ALL_TCGA \
    --gtex $ALL_GTEX \
    --n_samples 500 \
    --n_cells 2 10 100 500 1000\
    --group_by_source

# ==============================================================================
# 4. TCGA vs Various Pseudobulks (Brak GTEx)
# ==============================================================================
echo -e "\n>>> SCENARIO 4: All TCGA vs Various Pseudobulks"
python sample_analysis_script.py \
    --tcga $ALL_TCGA \
    --n_samples 500 \
    --n_cells 2 10 100 500 1000 \
    --group_by_source

# ==============================================================================
# 5. Only TCGA data
# ==============================================================================
echo -e "\n>>> SCENARIO 5: Only TCGA data"
python sample_analysis_script.py \
    --tcga $ALL_TCGA

# ==============================================================================
# 6. Only GTEx data
# ==============================================================================
echo -e "\n>>> SCENARIO 6: Only GTEx data"
python sample_analysis_script.py \
    --gtex $ALL_GTEX

echo -e "\n>>> ALL JOBS FINISHED SUCCESSFULLY!"