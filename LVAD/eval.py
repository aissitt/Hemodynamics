import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json
from model import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from loss import data_driven_loss, physics_informed_loss
from utils import load_dataset, apply_mask, compute_errors, compute_high_error_metrics, plot_error_maps

# Define custom objects for loading the model
custom_objects = {
    "RMSEPerComponent": RMSEPerComponent,
    "NRMSEPerComponent": NRMSEPerComponent,
    "MAEPerComponent": MAEPerComponent,
    "NMAEPerComponent": NMAEPerComponent,
    "data_driven_loss": data_driven_loss,
    "physics_informed_loss": physics_informed_loss
}

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True, help='Path to the trained model file')
parser.add_argument('--mode', required=True, choices=['data', 'physics'], help='Type of the model')
parser.add_argument('--output-dir', required=True, help='Directory to save evaluation results')
parser.add_argument('--config-file', required=True, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration file
with open(args.config_file) as f:
    config = json.load(f)

# Extract test indices from the config
test_indices = tuple(config['training']['test_indices'])

# Load dataset
testX_np = load_dataset(config['training']['input_data'])[test_indices[0]:test_indices[1]]
testY_np = load_dataset(config['training']['output_data'])[test_indices[0]:test_indices[1]]

# Extract RDF component to create a mask
rdf = testX_np[..., 0]
mask = rdf > 0

# Load the best model
best_model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

# Make predictions on the sample batch
predictions = best_model.predict(testX_np)

# Compute errors and metrics
abs_errors, peak_error, peak_error_coords = compute_errors(testY_np, predictions, mask)
high_error_count, high_error_percentage = compute_high_error_metrics(abs_errors, config['training']['high_error_threshold'])

# Log results
metrics = {
    "Peak Error": float(peak_error),
    "Coordinates of Peak Error": [int(coord) for coord in peak_error_coords],
    "Number of high error points": int(high_error_count),
    "Percentage of high error points": float(high_error_percentage)
}

metrics_file = os.path.join(args.output_dir, "logs", "metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=4)

# Plot and save error maps
images_dir = os.path.join(args.output_dir, "images")
os.makedirs(images_dir, exist_ok=True)
plot_error_maps(testY_np, predictions, abs_errors, peak_error_coords, images_dir, mask)

print(f"Evaluation completed. Results saved to {args.output_dir}")
