import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import argparse
import json
from model import unet_model
from loss import data_driven_loss, physics_informed_loss
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import load_and_split_data, plot_training_history, create_output_directories, save_hyperparameters, log_runtime

# Enable dynamic memory allocation for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# If using environment variables, replace placeholders with their values
if config.get("use_env_vars", False):
    config["training"]["input_data"] = os.getenv("INPUT_DATA_PATH")
    config["training"]["output_data"] = os.getenv("OUTPUT_DATA_PATH")

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Base directory to save training results')
parser.add_argument('--mode', choices=['data', 'physics'], required=True, help='Training mode: data-driven or physics-informed')
args = parser.parse_args()

# Use the output directory provided by the SLURM script, do not create a nested directory
output_dir = args.output_dir
logs_dir = os.path.join(output_dir, "logs")
images_dir = os.path.join(output_dir, "images")

# Create necessary directories if they don't exist
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

print(f"Using output directory: {output_dir}, logs directory: {logs_dir}, images directory: {images_dir}")

# Load and split data using indices provided in config
trainX_np, trainY_np, valX_np, valY_np = load_and_split_data(
    config["training"]["input_data"],
    config["training"]["output_data"],
    config["training"]["train_indices"],
    config["training"]["val_indices"]
)

# Define named functions for loss
def data_loss_fn(y_true, y_pred):
    return data_driven_loss(y_true, y_pred, config)

def physics_loss_fn(y_true, y_pred):
    return physics_informed_loss(y_true, y_pred, config)

# Strategy for distributed training across GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    input_shape = tuple(config["model"]["input_shape"])
    model = unet_model(input_shape)

    # Compile the model with the appropriate loss function based on the mode
    if args.mode == 'data':
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = ModelCheckpoint(os.path.join(output_dir, f"lvad_model_{timestamp}.h5"), save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        trainX_np, trainY_np,
        validation_data=(valX_np, valY_np),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"] * strategy.num_replicas_in_sync,
        callbacks=[early_stopping, checkpoint]
    )

# Plot training history
plot_training_history(history, images_dir)

# Save hyperparameters
save_hyperparameters(config, output_dir)

# Log runtime
log_runtime(history.epoch[-1] + 1, logs_dir)
