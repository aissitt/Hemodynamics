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
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH --job-name=LVAD_Train_MultiGPU

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE_DIR=/path/to/your/Hemodynamics/LVAD/outputs
RUN_DIR=${OUTPUT_BASE_DIR}/train_run_${TIMESTAMP}

#SBATCH --output=${RUN_DIR}/LVAD_Train_MultiGPU_%j.out
#SBATCH --error=${RUN_DIR}/LVAD_Train_MultiGPU_%j.err

echo "Starting training job on $(hostname) at $(date)"

# Load conda environment named 'hemodynamics'
. ./env.sh hemodynamics

# Move to the LVAD directory containing the scripts
cd /path/to/your/Hemodynamics/LVAD

# Set environment variables for data paths
export INPUT_DATA_PATH="/path/to/your/Hemodynamics/LVAD/LVAD_data/lvad_rdfs_inlets.npy"
export OUTPUT_DATA_PATH="/path/to/your/Hemodynamics/LVAD/LVAD_data/lvad_vels.npy"

# Parse the training type argument
TRAIN_TYPE=$1

if [[ "$TRAIN_TYPE" == "data" ]]; then
    TRAIN_SCRIPT="train.py"
elif [[ "$TRAIN_TYPE" == "physics" ]]; then
    TRAIN_SCRIPT="train.py"
else
    echo "Invalid training type specified. Use 'data' or 'physics'."
    exit 1
fi

start=$(date +%s)

# Ensure the run directory is created only once
mkdir -p ${RUN_DIR}

# Train the model using all available GPUs
python $TRAIN_SCRIPT --output-dir ${RUN_DIR} --mode ${TRAIN_TYPE}

end=$(date +%s)

runtime=$((end-start))
echo "Runtime: $runtime seconds"
echo "Training job completed at $(date)"
