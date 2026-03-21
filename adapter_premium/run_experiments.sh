#!/bin/bash

LRS=(0.00001 0.00005 0.0001)
HIDDENS=("512 512" "512 256" "512 128")
SEQ_LENGTH=(1000 2000 3000)

for LR in "${LRS[@]}"; do
    for SEQ in "${SEQ_LENGTH[@]}"; do
        for HIDDEN in "${HIDDENS[@]}"; do
            
            HIDDEN_ID=${HIDDEN// /_}
            JOB_NAME="scGPT_LR${LR}_SEQ${SEQ}_H${HIDDEN_ID}"
            
            echo "Running job: $JOB_NAME"
            
            sbatch --job-name=$JOB_NAME \
                   --export=ALL,LR=$LR,HIDDEN="$HIDDEN",SEQ_LENGTH=$SEQ \
                   run_pipeline.sh
            
            sleep 1
        done
    done
done
