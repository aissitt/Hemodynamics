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

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Directory to save tuning results')
parser.add_argument('--mode', required=True, choices=['data', 'physics'], help='Training mode: data-driven or physics-informed')
parser.add_argument('--config', help='Path to custom config file', default='config.json')
parser.add_argument('--n-trials', type=int, default=100, help='Number of trials for Optuna optimization')
args = parser.parse_args()

print(f"Number of trials: {args.n_trials}")

# Load base config
with open(args.config, 'r') as f:
    base_config = json.load(f)

# Ensure mode is part of the config
base_config["mode"] = args.mode  # Add mode directly into the config

# Generate output directories based on the mode (data/physics) and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Desired run directory structure: mode/tuning_run_<timestamp>
output_base_dir = os.path.join(args.output_dir, args.mode, f'tuning_run_{timestamp}')

# Generate the run_name to be just "tuning_<timestamp>"
run_name = f"tuning_{timestamp}"

# Create the directories using utils
output_dir, logs_dir, images_dir = create_output_directories(output_base_dir, run_name)

print(f"Created output directories: {output_dir}, {logs_dir}, {images_dir}")

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for tuning
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])
    
    # Make activation dynamic
    activation = trial.suggest_categorical("activation", ['relu', 'leaky_relu', 'elu'])  # Dynamic values
    
    # Turn off batch normalization and attention to save memory
    batch_norm = False  # Static value
    attention = False   # Static value
    
    # Fixed dropout and l2 regularization to 0
    dropout_rate = 0.0  # Fixed at 0
    l2_reg = 0.0  # Fixed at 0

    # Physics-specific parameters
    if args.mode == 'physics':
        lambda_data = trial.suggest_float("lambda_data", 0.1, 2)
        lambda_continuity = trial.suggest_float("lambda_continuity", 1e-6, 1, log=True)
        lambda_vorticity_focused = trial.suggest_float("lambda_vorticity_focused", .1, 1.5)
        lambda_momentum = trial.suggest_float("lambda_momentum", 1e-2, 1.5, log=True)
        lambda_gradient_penalty = trial.suggest_float("lambda_gradient_penalty", 1e-3, 1.5, log=True)
        huber_delta = trial.suggest_float("huber_delta", 1e-3, 1.5)  

        # Update config with suggested hyperparameters
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
        # Data-specific parameter: tune huber delta
        huber_delta = trial.suggest_float("huber_delta", 0.01, 1.0)

        # Update config with suggested hyperparameters
        config = base_config.copy()
        config["training"]["learning_rate"] = learning_rate
        config["training"]["batch_size"] = batch_size
        config["loss_parameters"]["data_driven"]["huber_delta"] = huber_delta

    # Architecture-related parameters with reduced complexity but dynamic activation
    arch_params = {
        "activation": activation,  # Dynamic activation
        "batch_norm": batch_norm,  # Turned off
        "dropout_rate": dropout_rate,  # Fixed at 0
        "l2_reg": l2_reg,  # Fixed at 0
        "attention": attention,  # Turned off
    }

    try:
        # Train the model with the current set of hyperparameters
        val_loss = train(config, arch_params, output_dir)
        
        # Validate that val_loss is a finite number
        if not np.isfinite(val_loss):
            raise ValueError(f"val_loss is not finite (NaN or inf): {val_loss}")

        trial.set_user_attr("val_loss", val_loss)  # Store validation loss in trial attributes
        print(f"Trial {trial.number}: val_loss = {val_loss}")
        
        # Save best parameters dynamically after each successful trial
        if study.best_value is None or (val_loss is not None and val_loss < study.best_value):
            best_params_file = os.path.join(logs_dir, f'best_params_{args.mode}.json')
            with open(best_params_file, 'w') as f:
                json.dump(trial.params, f, indent=4)
            print(f"Saved best parameters so far to {best_params_file}")
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        val_loss = float('inf')  # Set to infinity if the trial fails
    finally:
        # Clear memory and release GPU resources after each trial
        del config, arch_params  # Free variables
        K.clear_session()
        gc.collect()

    return val_loss

# Create a study for hyperparameter optimization with failure handling
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

# Save the best hyperparameters to a file at the end
best_params = study.best_params
best_params_file = os.path.join(logs_dir, f'best_params_{args.mode}_final.json')  # Save in logs directory
with open(best_params_file, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Best hyperparameters saved to {best_params_file}")
