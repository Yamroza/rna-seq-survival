How to transform raw tsv data downloaded straight from XENA portal for baselines & scGPT embeddings?
1. Download data from XENA, eg. https://xenabrowser.net/datapages/?dataset=TCGA-LUAD.star_tpm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
2. Move it to raw data folder in data folder
3. gunzip filename.tsv.gz
4. Run unify_star_rpm.ipynb, change raw file name on top
5. Run make_embeddings_scgpt.ipynb, change file name on top
6. Download clinical data from CBioportal, eg. https://www.cbioportal.org/study/clinicalData?id=luad_tcga_gdc
7. Run create_clinical_data.ipynb, change cohort name on top
8. Run split_data_into_folds.ipynb, change cohort name on top
