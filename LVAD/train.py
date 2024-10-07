import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from tensorflow.keras import backend as K
from datetime import datetime
import argparse
import json
import time
from model import unet_model
from loss import data_driven_loss, physics_informed_loss
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import load_and_split_data, plot_training_history, create_output_directories, save_hyperparameters, log_runtime
from keras.saving import register_keras_serializable

# Enable dynamic memory allocation for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Base directory to save training results')
parser.add_argument('--mode', choices=['data', 'physics'], required=True, help='Training mode: data-driven or physics-informed')
parser.add_argument('--config', help='Path to custom config file', default='config.json')
parser.add_argument('--use-tuning', action='store_true', help='Flag to use tuned hyperparameters from best_params.json')
args = parser.parse_args()

# Load configuration
if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)
else:
    with open('config.json', 'r') as f:
        config = json.load(f)

# Add mode from args to config
config["mode"] = args.mode  # This ensures "mode" is available in the config dictionary.

# If using environment variables, replace placeholders with their values
if config.get("use_env_vars", False):
    config["training"]["input_data"] = os.getenv("INPUT_DATA_PATH")
    config["training"]["output_data"] = os.getenv("OUTPUT_DATA_PATH")

# Check if best_params.json exists for tuning and load it
if args.use_tuning:
    best_params_path = os.path.join(args.output_dir, f'best_params_{args.mode}.json')  # Separate for data/physics
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
            # Update the config with the best hyperparameters
            config["training"]["learning_rate"] = best_params.get("learning_rate", config["training"]["learning_rate"])
            config["training"]["batch_size"] = best_params.get("batch_size", config["training"]["batch_size"])
            if args.mode == 'physics':
                config["loss_parameters"]["physics_informed"]["lambda_data"] = best_params.get("lambda_data", config["loss_parameters"]["physics_informed"]["lambda_data"])
                config["loss_parameters"]["physics_informed"]["lambda_continuity"] = best_params.get("lambda_continuity", config["loss_parameters"]["physics_informed"]["lambda_continuity"])
                config["loss_parameters"]["physics_informed"]["lambda_vorticity_focused"] = best_params.get("lambda_vorticity_focused", config["loss_parameters"]["physics_informed"]["lambda_vorticity_focused"])
                config["loss_parameters"]["physics_informed"]["lambda_momentum"] = best_params.get("lambda_momentum", config["loss_parameters"]["physics_informed"]["lambda_momentum"])
                config["loss_parameters"]["physics_informed"]["lambda_gradient_penalty"] = best_params.get("lambda_gradient_penalty", config["loss_parameters"]["physics_informed"]["lambda_gradient_penalty"]) 
    else:
        print(f"No best_params_{args.mode}.json found. Using default config values.")

# Create directories for output
output_dir = args.output_dir
logs_dir = os.path.join(output_dir, "logs")
images_dir = os.path.join(output_dir, "images")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Load and split data using indices provided in config
trainX_np, trainY_np, valX_np, valY_np = load_and_split_data(
    config["training"]["input_data"],
    config["training"]["output_data"],
    config["training"]["train_indices"],
    config["training"]["val_indices"]
)

# Define named functions for loss
@register_keras_serializable()
def data_loss_fn(y_true, y_pred):
    return data_driven_loss(y_true, y_pred, config)

@register_keras_serializable()
def physics_loss_fn(y_true, y_pred):
    return physics_informed_loss(y_true, y_pred, config)

# Define custom objects for saving and loading
custom_objects = {
    "data_loss_fn": data_loss_fn,
    "physics_loss_fn": physics_loss_fn,
    "RMSEPerComponent": RMSEPerComponent,
    "NRMSEPerComponent": NRMSEPerComponent,
    "MAEPerComponent": MAEPerComponent,
    "NMAEPerComponent": NMAEPerComponent
}

# Define the tuning-compatible train function
def train_tuning(config, arch_params=None, output_dir=None):
    # Clear memory and session before starting training to avoid memory buildup
    K.clear_session()
    gc.collect()
    
    # Strategy for distributed training across GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_shape = tuple(config["model"]["input_shape"])
        model = unet_model(
            input_shape,
            activation=arch_params.get("activation", "relu") if arch_params else "relu",
            # Keep batch_norm and dropout fixed for tuning to avoid complexity
            batch_norm=False,  
            dropout_rate=0.0, 
            l2_reg=0.0,
            attention=False  # Turn off attention to reduce memory usage
        )

        # Compile the model with the appropriate loss function based on the mode
        if config["mode"] == 'data':
            model.compile(
                loss=data_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )
        else:
            model.compile(
                loss=physics_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )

        # Callbacks
        checkpoint_path = os.path.join(output_dir, f"lvad_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', mode='min')

        # Train the model
        history = model.fit(
            trainX_np, trainY_np,
            validation_data=(valX_np, valY_np),
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"] * strategy.num_replicas_in_sync,
            callbacks=[early_stopping, checkpoint]
        )
    
    # Clear memory explicitly after training to avoid memory leakage
    K.clear_session()
    gc.collect()

    # Get the best validation loss
    val_loss = min(history.history['val_loss'])
    
    # Return only the numerical value of val_loss for compatibility with tuning frameworks
    return float(val_loss)

# Define the full-functionality train function
def train_full(config, arch_params=None, output_dir=None):
    # Strategy for distributed training across GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_shape = tuple(config["model"]["input_shape"])
        model = unet_model(
            input_shape,
            activation=arch_params.get("activation", "relu") if arch_params else "relu",
            batch_norm=arch_params.get("batch_norm", False) if arch_params else False,
            dropout_rate=arch_params.get("dropout_rate", 0.0) if arch_params else 0.0,
            l2_reg=arch_params.get("l2_reg", 0.0) if arch_params else 0.0,
            attention=arch_params.get("attention", False) if arch_params else False,
        )

        # Compile the model with the appropriate loss function based on the mode
        if config["mode"] == 'data':
            model.compile(
                loss=data_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )
        else:
            model.compile(
                loss=physics_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )

        # Callbacks
        checkpoint_path = os.path.join(output_dir, f"lvad_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss', mode='min')

        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', mode='min')

        # Train the model
        history = model.fit(
            trainX_np, trainY_np,
            validation_data=(valX_np, valY_np),
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"] * strategy.num_replicas_in_sync,
            callbacks=[early_stopping, checkpoint]
        )


    # Return the history object and validation loss for further processing
    return history, min(history.history['val_loss'])

# Main function to handle either tuning or full mode
if __name__ == '__main__':
    start_time = time.time()    

    arch_params = None  # Placeholder for architecture parameters if needed

    if args.use_tuning:
        # Use tuning-compatible function
        val_loss = train_tuning(config, arch_params, output_dir)
    else:
        # Run the full functionality for single runs
        history, val_loss = train_full(config, arch_params, output_dir)
    
    end_time = time.time()
    total_time = end_time - start_time

    # Now plot the history and save metrics
    plot_training_history(history, images_dir)

    # Save hyperparameters
    save_hyperparameters(config, output_dir)

    # Log runtime
    log_runtime(history.epoch[-1] + 1, logs_dir)

    print(f"Training complete. Total time: {total_time:.2f} seconds. Final validation loss: {val_loss}")
