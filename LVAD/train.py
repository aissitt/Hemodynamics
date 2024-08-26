import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import argparse
from model import unet_model
from loss import data_driven_loss, physics_informed_loss
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import load_and_split_data, plot_training_history, create_output_directories, save_hyperparameters, log_runtime

# Enable dynamic memory allocation for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Directory to save training results')
parser.add_argument('--mode', choices=['data', 'physics'], required=True, help='Training mode: data-driven or physics-informed')
parser.add_argument('--train-indices', required=True, nargs=2, type=int, help='Start and end indices for the training data')
parser.add_argument('--val-indices', required=True, nargs=2, type=int, help='Start and end indices for the validation data')
args = parser.parse_args()

# Load and split data using indices provided by the user
lvad_data_path = '/path/to/your/LVAD_data'
trainX_np, trainY_np, valX_np, valY_np = load_and_split_data(lvad_data_path, train_indices=tuple(args.train_indices), val_indices=tuple(args.val_indices))

# Create directories for this run
output_dir, logs_dir, images_dir = create_output_directories(args.output_dir, "train_run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

# Strategy for distributed training across GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    input_shape = (128, 128, 128, 2)
    model = unet_model(input_shape)

    # Select loss function based on mode
    loss = data_driven_loss if args.mode == 'data' else physics_informed_loss

    # Compile the model
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=0.001),
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
        epochs=10000,
        batch_size=1 * strategy.num_replicas_in_sync,
        callbacks=[early_stopping, checkpoint]
    )

# Plot training history
plot_training_history(history, images_dir)

# Save hyperparameters
hyperparameters = {
    'learning_rate': 0.001,
    'batch_size': 1 * strategy.num_replicas_in_sync,
    'epochs': 10000,
    'loss': args.mode
}
save_hyperparameters(hyperparameters, output_dir)

# Log runtime
log_runtime(history.epoch[-1] + 1, logs_dir)
