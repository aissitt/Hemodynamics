import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from loss import data_driven_loss, physics_informed_loss
from utils import load_dataset, apply_mask, compute_errors, compute_high_error_metrics, plot_error_maps

# Define named functions for loss (same as in train.py)
def data_loss_fn(y_true, y_pred):
    return data_driven_loss(y_true, y_pred, config)

def physics_loss_fn(y_true, y_pred):
    return physics_informed_loss(y_true, y_pred, config)

# Define custom objects for loading the model
custom_objects = {
    "RMSEPerComponent": RMSEPerComponent,
    "NRMSEPerComponent": NRMSEPerComponent,
    "MAEPerComponent": MAEPerComponent,
    "NMAEPerComponent": NMAEPerComponent,
    "data_loss_fn": data_loss_fn,
    "physics_loss_fn": physics_loss_fn
}

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# If using environment variables, replace placeholders with their values
if config.get("use_env_vars", False):
    config["training"]["input_data"] = os.getenv("INPUT_DATA_PATH")
    config["training"]["output_data"] = os.getenv("OUTPUT_DATA_PATH")

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True, help='Path to the trained model file')
parser.add_argument('--mode', required=True, choices=['data', 'physics'], help='Type of the model')
parser.add_argument('--output-dir', required=True, help='Directory to save evaluation results')
parser.add_argument('--test-indices', nargs=2, type=int, help='Start and end indices for the test data')
args = parser.parse_args()

# Determine test indices to use
if args.test_indices:
    test_indices = args.test_indices
else:
    test_indices = config["training"]["test_indices"]

# Load dataset using test indices from config or arguments
testX_np = load_dataset(config["training"]["input_data"])[test_indices[0]:test_indices[1]]
testY_np = load_dataset(config["training"]["output_data"])[test_indices[0]:test_indices[1]]

# Extract RDF component to create a mask
rdf = testX_np[..., 0]
mask = rdf > 0

# Load the best model
best_model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

# Evaluate the model on test data to obtain metrics
results = best_model.evaluate(testX_np, testY_np, return_dict=True)
print(f"Evaluation results: {results}")

# Convert any NumPy arrays in results to lists for JSON serialization
for key in results:
    if isinstance(results[key], np.ndarray):
        results[key] = results[key].tolist()

# Compute additional error metrics
abs_errors, peak_error, peak_error_coords = compute_errors(testY_np, best_model.predict(testX_np), mask)
high_error_threshold = config["loss_parameters"].get("high_error_threshold", 0.01)  # Default to 0.01 if not specified
high_error_count, high_error_percentage = compute_high_error_metrics(abs_errors, high_error_threshold)

# Log results
metrics = {
    "Peak Error": float(peak_error),
    "Coordinates of Peak Error": [int(coord) for coord in peak_error_coords],
    "Number of high error points": int(high_error_count),
    "Percentage of high error points": float(high_error_percentage)
}

# Combine evaluation results with custom metrics
metrics.update(results)

# Ensure logs directory exists
logs_dir = os.path.join(args.output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Save metrics to a JSON file
metrics_file = os.path.join(logs_dir, "metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_file}")

# Plot and save error maps
images_dir = os.path.join(args.output_dir, "images")
os.makedirs(images_dir, exist_ok=True)
plot_error_maps(testY_np, best_model.predict(testX_np), abs_errors, peak_error_coords, images_dir, mask)

print(f"Evaluation completed. Results saved to {args.output_dir}")
