How to transform raw tsv data downloaded straight from XENA portal for baselines & scGPT embeddings?
1. Download data from XENA - https://xenabrowser.net/datapages/?hub=https://gdc.xenahubs.net:443,
 eg. LUAD -  https://xenabrowser.net/datapages/?dataset=TCGA-LUAD.star_tpm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
2. Move it to data/raw_tsv_data folder
3. Download clinical data from CBioportal - https://www.cbioportal.org/datasets
 eg. LUAD - https://www.cbioportal.org/study/clinicalData?id=luad_tcga_gdc
4. Run "process_new_cohort.sh" script




OLD:
1. Download data from XENA - https://xenabrowser.net/datapages/?hub=https://gdc.xenahubs.net:443,
 eg. LUAD -  https://xenabrowser.net/datapages/?dataset=TCGA-LUAD.star_tpm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
2. Move it to data/raw_tsv_data folder
3. gunzip filename.tsv.gz
4. + Run unify_star_tpm.ipynb, change raw file name on top
5. + Run generate_embeddings_scgpt.py specify file name
6. + Run generate_embeddings_tgpt.py specify file name
7. / Download clinical data from CBioportal, eg. https://www.cbioportal.org/study/clinicalData?id=luad_tcga_gdc
8. + Run create_clinical_data.ipynb, specify cohort name
9. Run split_data_into_folds.ipynb, change cohort name on top

