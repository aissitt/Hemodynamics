import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from model import unet_model
from loss import data_driven_loss, physics_informed_loss
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import load_dataset, plot_training_history

# Enable dynamic memory allocation for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Directory to save training results')
parser.add_argument('--mode', choices=['data', 'physics'], required=True, help='Training mode: data-driven or physics-informed')
args = parser.parse_args()

# Load datasets
lvad_data_path = '/home1/aissitt2019/LVAD/LVAD_data'
trainX_np = load_dataset(os.path.join(lvad_data_path, 'lvad_rdfs_inlets.npy'))[:36]
trainY_np = load_dataset(os.path.join(lvad_data_path, 'lvad_vels.npy'))[:36]
valX_np = load_dataset(os.path.join(lvad_data_path, 'lvad_rdfs_inlets.npy'))[36:40]
valY_np = load_dataset(os.path.join(lvad_data_path, 'lvad_vels.npy'))[36:40]

# Create directories for this run
output_dir = args.output_dir
logs_dir = os.path.join(output_dir, "logs")
images_dir = os.path.join(output_dir, "images")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Strategy for distributed training across GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    input_shape = (128, 128, 128, 2)  # Example input shape
    model = unet_model(input_shape)

    # Select loss function based on mode
    if args.mode == 'data':
        loss = data_driven_loss
    else:
        loss = physics_informed_loss

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
