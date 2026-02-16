# Data 

This folder contains all data files, from preprocessing to dataset classes. 

-  The `files` folder contains all data files, containing the gene sets for all pathways (in `hallmarks_signatures.csv`), and cohort specific files (i.e. `tcga_blca`, `tcga_brca`, `tcga_luad`, `tcga_kirc`). These cohort specific files consist of all train and test splits (for 5-fold cross validation, stratified on sites) and clinical data of all samples (`clinical_data_all_filtered.csv`).
- The `preprocess_TCGA.py` file contains all steps for preprocessing the data.
- The `preprocess_TCGA.ipynb` file is the notebook version of the file above with some additional options to visualize the data.
- The `survival_dataset.py` contains the dataset class for the transcriptomics data used during training and testing. 

To preprocess the BRCA data, download the raw data and run the `preprocess_TCGA.py` file. Below, the steps for all 4 supported TCGA cohorts are shown. To remove all genes that are never in te Hallmark Gene Sets, add --remove when running `preprocess_TCGA.py`. For all types, first run

```
cd src/data
conda activate myenv
```

## TCGA-BRCA
```
curl -o files/tcga_brca/HiSeqV2_PANCAN_BRCA.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip files/tcga_brca/HiSeqV2_PANCAN_BRCA.gz
python preprocess_TCGA.py --data brca --name rna_data
```

## TCGA-BLCA
```
curl -o files/tcga_blca/HiSeqV2_PANCAN_BLCA.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BLCA.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip files/tcga_blca/HiSeqV2_PANCAN_BLCA.gz
python preprocess_TCGA.py --data blca --name rna_data
```

## TCGA-LUAD
```
curl -o files/tcga_luad/HiSeqV2_PANCAN_LUAD.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip files/tcga_luad/HiSeqV2_PANCAN_LUAD.gz
python preprocess_TCGA.py --data luad --name rna_data
```

## TCGA-KIRC
```
curl -o files/tcga_kirc/HiSeqV2_PANCAN_KIRC.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.KIRC.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip files/tcga_kirc/HiSeqV2_PANCAN_KIRC.gz
python preprocess_TCGA.py --data kirc --name rna_data
```

