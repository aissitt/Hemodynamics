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
#SBATCH --job-name=LVAD_Eval_MultiGPU

# Prepare a timestamp for directory naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories for this run
OUTPUT_BASE_DIR=/your/path/to/Hemodynamics/LVAD/outputs
RUN_DIR=${OUTPUT_BASE_DIR}/eval_run_${TIMESTAMP}

# Direct output and error files to the run-specific directory
#SBATCH --output=${OUTPUT_BASE_DIR}/LVAD_Eval_MultiGPU_%j.out
#SBATCH --error=${OUTPUT_BASE_DIR}/LVAD_Eval_MultiGPU_%j.err

echo "Starting evaluation job on $(hostname) at $(date)"

# Load conda environment named 'blood'
. ./env.sh blood

# Move to the LVAD directory containing the scripts
cd /your/path/to/Hemodynamics/LVAD

# Parse the evaluation type argument
EVAL_TYPE=$1

# Define test indices
TEST_INDICES_START=40
TEST_INDICES_END=50

if [[ "$EVAL_TYPE" == "data_driven" ]]; then
    MODEL_PATH="outputs/train_run_*/lvad_model_*.h5"
elif [[ "$EVAL_TYPE" == "physics" ]]; then
    MODEL_PATH="outputs/train_run_*/lvad_model_*.h5"
else
    echo "Invalid evaluation type specified. Use 'data_driven' or 'physics'."
    exit 1
fi

# Find the most recent model file for evaluation
LATEST_MODEL=$(ls -t $MODEL_PATH | head -n 1)

start=$(date +%s)

# Evaluate the model using all available GPUs and specified indices
python eval.py --model-path $LATEST_MODEL --mode ${EVAL_TYPE} --output-dir ${RUN_DIR} --test-indices ${TEST_INDICES_START} ${TEST_INDICES_END}

end=$(date +%s)

runtime=$((end-start))
echo "Runtime: $runtime seconds"
echo "Evaluation job completed at $(date)"
