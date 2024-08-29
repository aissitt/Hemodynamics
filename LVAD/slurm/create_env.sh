#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################

##################################
####### Resources Request ########
##################################

# Use GPU partition (gpu1 and gpu2) or other partition (e.g., short)
# Find more usable partitions with 'sinfo -a'
#SBATCH --partition=gpu2

# Configure the number of nodes (in partition above)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Configure the number of GPUs to use all available GPUs on the node
#SBATCH --gpus-per-node=1
# Set time limit for the job
#SBATCH --time=1-00:00:00
#SBATCH --job-name=Create_Env

# Direct output and error files to the run-specific directory
#SBATCH --output=create_env_%j.out
#SBATCH --error=create_env_%j.err

echo "Starting environment setup job on $(hostname) at $(date)"

##################################
########## Environment ###########
##################################

# Set the path to your project directory
REPO_PATH="${REPO_PATH:-/home1/aissitt2019/Hemodynamics/LVAD}"
ENV_YML_PATH="$REPO_PATH/environment.yml"
REQ_TXT_PATH="$REPO_PATH/requirements.txt"

# Attempt to find Conda automatically if not provided via environment variable
CONDA_BASE=$(conda info --base)
ENV_NAME="hemodynamics"
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"

echo "Detected conda base directory: $CONDA_BASE"
echo "Environment YAML file located at: $ENV_YML_PATH"
echo "Requirements file located at: $REQ_TXT_PATH"
echo "Environment will be created at: $ENV_PATH"

# Load the conda environment
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create the conda environment from the environment.yml file
conda env create --file="$ENV_YML_PATH" --name="$ENV_NAME"

# Activate the environment
conda activate "$ENV_NAME"

# Install additional pip packages from requirements.txt
pip install -r "$REQ_TXT_PATH"

echo "Environment setup completed at $(date)"
