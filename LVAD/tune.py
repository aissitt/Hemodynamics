import os
import json
import argparse
import optuna
import gc
from tensorflow.keras import backend as K
import numpy as np
from datetime import datetime
from train import train_tuning as train
from utils import create_output_directories
import tensorflow as tf

# Enable dynamic memory allocation for GPUs to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Directory to save tuning results')
parser.add_argument('--mode', required=True, choices=['data', 'physics'], help='Training mode: data-driven or physics-informed')
parser.add_argument('--config', help='Path to custom config file', default='config.json')
parser.add_argument('--n-trials', type=int, default=500, help='Number of trials for Optuna optimization')
parser.add_argument('--seed', type=int, help='Set a seed for reproducibility')

args = parser.parse_args()

# Set the seed for reproducibility if specified
if args.seed:
    print(f"Setting seed: {args.seed}")
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
else:
    print("No seed specified. Results may vary.")

# Print the number of trials
print(f"Number of trials: {args.n_trials}")

# Load base config
with open(args.config, 'r') as f:
    base_config = json.load(f)

# Ensure mode is part of the config
base_config["mode"] = args.mode

# Resolve environment variables for input and output data paths
base_config["training"]["input_data"] = os.getenv("INPUT_DATA_PATH", base_config["training"]["input_data"])
base_config["training"]["output_data"] = os.getenv("OUTPUT_DATA_PATH", base_config["training"]["output_data"])

# Generate output directories based on the mode and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"tuning_run_{timestamp}"
output_dir, logs_dir, images_dir = create_output_directories(args.output_dir, run_name)

print(f"Created output directories: {output_dir}, {logs_dir}, {images_dir}")

# Define the study for hyperparameter optimization
study = optuna.create_study(direction="minimize")

# Best loss log path
best_loss_log = os.path.join(logs_dir, "best_loss.txt")

def update_best_loss(val_loss, trial_number):
    """Updates the best loss value and trial number in the log file."""
    with open(best_loss_log, 'w') as f:
        f.write(f"Best loss: {val_loss}\nTrial number: {trial_number}\n")
    print(f"Updated best loss to {val_loss} at trial {trial_number}")

# Define the objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])

    # Set static values for simplicity
    activation = 'relu'
    batch_norm = False
    attention = False
    dropout_rate = 0.0
    l2_reg = 0.0

    # Physics-specific parameters
    if args.mode == 'physics':
        lambda_data = trial.suggest_float("lambda_data", 0.1, 2)
        lambda_continuity = trial.suggest_float("lambda_continuity", 1e-5, 1, log=True)
        lambda_vorticity_focused = trial.suggest_float("lambda_vorticity_focused", 0.1, 1.5)
        lambda_momentum = trial.suggest_float("lambda_momentum", 1e-3, 1, log=True)
        lambda_gradient_penalty = trial.suggest_float("lambda_gradient_penalty", 1e-3, 1.5, log=True)
        huber_delta = trial.suggest_float("huber_delta", 1e-3, 1.5)

        config = base_config.copy()
        config["training"]["learning_rate"] = learning_rate
        config["training"]["batch_size"] = batch_size
        config["loss_parameters"]["physics_informed"]["lambda_data"] = lambda_data
        config["loss_parameters"]["physics_informed"]["lambda_continuity"] = lambda_continuity
        config["loss_parameters"]["physics_informed"]["lambda_vorticity_focused"] = lambda_vorticity_focused
        config["loss_parameters"]["physics_informed"]["lambda_momentum"] = lambda_momentum
        config["loss_parameters"]["physics_informed"]["lambda_gradient_penalty"] = lambda_gradient_penalty
        config["loss_parameters"]["physics_informed"]["huber_delta"] = huber_delta
    else:
        huber_delta = trial.suggest_float("huber_delta", 0.01, 1.0)

        config = base_config.copy()
        config["training"]["learning_rate"] = learning_rate
        config["training"]["batch_size"] = batch_size
        config["loss_parameters"]["data_driven"]["huber_delta"] = huber_delta

    # Include seed in the config if provided
    if args.seed:
        config["training"]["seed"] = args.seed

    arch_params = {
        "activation": activation,
        "batch_norm": batch_norm,
        "dropout_rate": dropout_rate,
        "l2_reg": l2_reg,
        "attention": attention,
    }

    try:
        trial_output_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(trial_output_dir, exist_ok=True)
        val_loss = train(config, arch_params, trial_output_dir)

        if not np.isfinite(val_loss):
            raise ValueError(f"val_loss is not finite: {val_loss}")

        trial.set_user_attr("val_loss", val_loss)
        print(f"Trial {trial.number}: val_loss = {val_loss}")

        # If this is the best loss so far, update the best loss log and config
        if study.best_value is None or val_loss < study.best_value:
            update_best_loss(val_loss, trial.number)  # Update best loss and trial number in log

            # Save the best parameters
            best_params_file = os.path.join(logs_dir, f'best_params_{args.mode}.json')
            with open(best_params_file, 'w') as f:
                json.dump(trial.params, f, indent=4)
            print(f"Saved best parameters to {best_params_file}")

            # Save the best configuration
            best_config_file = os.path.join(logs_dir, f'best_config_{args.mode}.json')
            best_config = config.copy()
            best_config["model"] = arch_params
            best_config["model"]["input_shape"] = config["model"]["input_shape"]
            with open(best_config_file, 'w') as f:
                json.dump(best_config, f, indent=4)
            print(f"Saved best configuration to {best_config_file}")
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        val_loss = float('inf')
    finally:
        K.clear_session()
        gc.collect()

    return val_loss

# Optimize the study
study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

# Final best parameters and configuration save
best_params_file = os.path.join(logs_dir, f'best_params_{args.mode}_final.json')
with open(best_params_file, 'w') as f:
    json.dump(study.best_params, f, indent=4)
print(f"Saved final best parameters to {best_params_file}")

best_config = base_config.copy()
best_config.update(study.best_params)
best_config["model"]["input_shape"] = base_config["model"]["input_shape"]

if args.seed:
    best_config["training"]["seed"] = args.seed

best_config_path = os.path.join(output_dir, "best_config.json")
with open(best_config_path, 'w') as f:
    json.dump(best_config, f, indent=4)

print(f"Saved final best configuration to {best_config_path}")
