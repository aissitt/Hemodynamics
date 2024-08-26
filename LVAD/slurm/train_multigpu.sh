#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################

##################################
####### Resources Request ########
##################################

# Use GPU partition (gpu1 and gpu2) or other partition (e.g.: short)
# Find more usable partitions with 'sinfo -a'
#SBATCH --partition=gpu1

# Configure the number of nodes (in partition above)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Configure the number of GPUs to use all available GPUs on the node
#SBATCH --gpus-per-node=4
# Set time limit 3 days
#SBATCH --time=3-00:00:00
#SBATCH --job-name=LVAD_Train_MultiGPU

# Prepare a timestamp for directory naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories for this run
OUTPUT_BASE_DIR=/path/to/your/Hemodynamics/LVAD/outputs
RUN_DIR=${OUTPUT_BASE_DIR}/train_run_${TIMESTAMP}

# Direct output and error files to the run-specific directory
#SBATCH --output=${OUTPUT_BASE_DIR}/LVAD_Train_MultiGPU_%j.out
#SBATCH --error=${OUTPUT_BASE_DIR}/LVAD_Train_MultiGPU_%j.err

echo "Starting training job on $(hostname) at $(date)"

# Load conda environment named 'hemodynamics'
. ./env.sh hemodynamics

# Move to the LVAD directory containing the scripts
cd /path/to/your/Hemodynamics/LVAD

# Parse the training type argument
TRAIN_TYPE=$1

# Define data path and training/validation indices
DATA_PATH=/path/to/your/LVAD_data
TRAIN_INDICES_START=0
TRAIN_INDICES_END=36
VAL_INDICES_START=36
VAL_INDICES_END=40

if [[ "$TRAIN_TYPE" == "data_driven" ]]; then
    TRAIN_SCRIPT="train.py"
elif [[ "$TRAIN_TYPE" == "physics" ]]; then
    TRAIN_SCRIPT="train.py"
else
    echo "Invalid training type specified. Use 'data_driven' or 'physics'."
    exit 1
fi

start=$(date +%s)

# Train the model using all available GPUs, specified indices, and data path
python $TRAIN_SCRIPT --output-dir ${RUN_DIR} --mode ${TRAIN_TYPE} --data-path ${DATA_PATH} --train-indices ${TRAIN_INDICES_START} ${TRAIN_INDICES_END} --val-indices ${VAL_INDICES_START} ${VAL_INDICES_END}

end=$(date +%s)

runtime=$((end-start))
echo "Runtime: $runtime seconds"
echo "Training job completed at $(date)"
