#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################

##################################
####### Resources Request ########
##################################

# Use GPU partition (gpu1 and gpu2) or other partition (e.g.: short)
#SBATCH --partition=gpu2

# Configure the number of nodes (in partition above)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Configure the number of GPUs to use all available GPUs on the node
#SBATCH --gpus-per-node=4
# Set time limit 3 days
#SBATCH --time=3-00:00:00
#SBATCH --job-name=LVAD_Eval_MultiGPU

# Prepare a timestamp for directory naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Define the training run directory to be evaluated
TRAIN_RUN_DIR=$(ls -dt /path/to/your/Hemodynamics/LVAD/outputs/train_run_* | head -n 1)

# Create directories for this run under the corresponding training run folder
RUN_DIR=${TRAIN_RUN_DIR}/eval_run_${TIMESTAMP}
mkdir -p ${RUN_DIR}/images
mkdir -p ${RUN_DIR}/logs

# Direct output and error files to the run-specific directory
#SBATCH --output=${TRAIN_RUN_DIR}/LVAD_Eval_MultiGPU_%j.out
#SBATCH --error=${TRAIN_RUN_DIR}/LVAD_Eval_MultiGPU_%j.err

echo "Starting evaluation job on $(hostname) at $(date)"

# Load conda environment named 'hemodynamics'
. ./env.sh hemodynamics

# Move to the LVAD directory containing the scripts
cd /path/to/your/Hemodynamics/LVAD

# Set environment variables for data paths
export INPUT_DATA_PATH="/path/to/your/Hemodynamics/LVAD/LVAD_data/lvad_rdfs_inlets.npy" # Expects shape (x, 128, 128, 128, 2) where x is the number of samples, 128x128x128 represents the geometry, and 2 represents rdf and inlet values
export OUTPUT_DATA_PATH="/path/to/your/Hemodynamics/LVAD/LVAD_data/lvad_vels.npy" # Expects shape (x, 128, 128, 128, 3) where x is the number of samples, 128x128x128 represents the geometry, and 3 represents velocity components

# Parse the evaluation type argument
EVAL_TYPE=$1

# Optional: Specify a model path argument
MODEL_PATH_ARG=$2

# Check if a specific model path is provided; if not, default to the most recent model
if [[ -z "$MODEL_PATH_ARG" ]]; then
    if [[ "$EVAL_TYPE" == "data" ]]; then
        MODEL_PATH="${TRAIN_RUN_DIR}/lvad_model_*.h5"
    elif [[ "$EVAL_TYPE" == "physics" ]]; then
        MODEL_PATH="${TRAIN_RUN_DIR}/lvad_model_*.h5"
    else
        echo "Invalid evaluation type specified. Use 'data' or 'physics'."
        exit 1
    fi

    # Find the most recent model file for evaluation
    LATEST_MODEL=$(ls -t $MODEL_PATH 2>/dev/null | head -n 1)
    if [[ -z "$LATEST_MODEL" ]]; then
        echo "No model file found matching pattern $MODEL_PATH"
        exit 1
    fi
else
    # Use the provided model path argument
    LATEST_MODEL=$MODEL_PATH_ARG
fi

start=$(date +%s)

# Evaluate the model using all available GPUs
python eval.py --model-path "$LATEST_MODEL" --mode ${EVAL_TYPE} --output-dir ${RUN_DIR}

end=$(date +%s)

runtime=$((end-start))
echo "Runtime: $runtime seconds"
echo "Evaluation job completed at $(date)"
