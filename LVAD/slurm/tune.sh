#!/bin/bash

##################################
####### Resources Request ########
##################################

#SBATCH --partition=gpu1  # Partition to use
#SBATCH --nodes=1         # Number of nodes
#SBATCH --ntasks-per-node=4  # Number of tasks per node
#SBATCH --gpus-per-node=4    # Number of GPUs per node
#SBATCH --time=14-00:00:00    # Time limit (14 day)
#SBATCH --job-name=LVAD_Tune_MultiGPU  # Job name
#SBATCH --exclusive
# Create output directory for logs if it doesn't exist
LOG_DIR="/path/to/your/Hemodynamics/LVAD/tuning_logs"
mkdir -p $LOG_DIR

# SLURM output and error files
#SBATCH --output=${LOG_DIR}/tune_%j.out
#SBATCH --error=${LOG_DIR}/tune_%j.err

echo "Starting tuning job on $(hostname) at $(date)"

# Load conda environment
. ./env.sh hemodynamics

# Move to the LVAD directory containing the scripts
cd /home1/aissitt2019/Hemodynamics/LVAD

# Set environment variables for data paths
export INPUT_DATA_PATH="/path/to/your/LVAD/LVAD_data/lvad_rdfs_inlets.npy"
export OUTPUT_DATA_PATH="/path/to/your/LVAD/LVAD_data/lvad_vels.npy"

# Set N_TRIALS environment variable
export N_TRIALS=100

# nvidia-smi -l 300 & # Log GPU memory usage every 5 mins

# Run tuning with the specified mode (data or physics)
python tune.py --output-dir "/path/to/your/Hemodynamics/LVAD/tuning_outputs" --mode "$1" --config "/path/to/your/Hemodynamics/LVAD/config.json"
