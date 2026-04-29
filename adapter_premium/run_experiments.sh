#!/bin/bash
#SBATCH --partition=gpua6000
#SBATCH --job-name=scGPT_train

LRS=(0.0001) # 0.00005 0.00001)
HIDDENS=("512 512") # "512 128")
SEQ_LENGTH=(2000) # (1000 2000 3000)
DROPOUTS=(0)
DATA_PATH="data_new/blkb_simp_100k_path_train.h5ad"
DATASET="bulkDataset"
EXP_CODE="adapter_premium"

for LR in "${LRS[@]}"; do
    for SEQ in "${SEQ_LENGTH[@]}"; do
        for HIDDEN in "${HIDDENS[@]}"; do
            for DROPOUT in "${DROPOUTS[@]}"; do
                
                HIDDEN_ID=${HIDDEN// /_}
                JOB_NAME="scGPT_LR${LR}_SEQ${SEQ}_H${HIDDEN_ID}_D${DROPOUT}"
                
                echo "Running job: $JOB_NAME"
                
                sbatch --job-name=$JOB_NAME \
                    --export=ALL,LR=$LR,HIDDEN="$HIDDEN",SEQ_LENGTH=$SEQ,DROPOUT=$DROPOUT,DATA_PATH=$DATA_PATH,DATASET=$DATASET,BATCH_SIZE=$BATCH_SIZE,EPOCHS=$EPOCHS,EXP_CODE=$EXP_CODE\
                    run_pipeline.sh
                
                sleep 1
            done
        done
    done
done
