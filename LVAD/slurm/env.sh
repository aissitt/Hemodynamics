#!/bin/bash

env_name="${1:-base}"

source ~/.bashrc
source $(conda info --base)/etc/profile.d/mamba.sh
eval "$(conda shell.bash hook)"
conda activate "$env_name"
shift
"$@"