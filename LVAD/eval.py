import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from model import unet_model
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import load_dataset, plot_training_history, create_output_directories

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train-dir', required=True, help='Directory of the corresponding training run')
parser.add_argument('--mode', choices=['data', 'physics'], required=True, help='Evaluation mode: data-driven or physics-informed')
args = parser.parse_args()

# Define base directories
lvad_data_path = '/home1/aissitt2019/LVAD/LVAD_data'

# Load the data
valX_np = load_dataset(os.path.join(lvad_data_path, 'lvad_rdfs_inlets.npy'))[36:40]
valY_np = load_dataset(os.path.join(lvad_data_path, 'lvad_vels.npy'))[36:40]

# Get the latest model from the training directory
model_files = [f for f in os.listdir(args.train_dir) if f.endswith('.h5')]
latest_model_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(args.train_dir, x)))
latest_model_path = os.path.join(args.train_dir, latest_model_file)

# Load the model
model = load_model(latest_model_path, custom_objects={
    'RMSEPerComponent': RMSEPerComponent,
    'NRMSEPerComponent': NRMSEPerComponent,
    'MAEPerComponent': MAEPerComponent,
    'NMAEPerComponent': NMAEPerComponent
})

# Create directories for evaluation results within the training directory
eval_dir = os.path.join(args.train_dir, 'evaluation')
logs_dir = os.path.join(eval_dir, "logs")
images_dir = os.path.join(eval_dir, "images")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Evaluate the model
results = model.evaluate(valX_np, valY_np)
print(f"Evaluation results: {results}")

# If there are additional plots or logs, save them in the evaluation directory
# Example: plot_training_history(history, images_dir)
# Note: Replace 'history' with appropriate variable if plotting training history

print(f"Evaluation completed. Results saved in {eval_dir}")
