import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import manual_rmse, manual_nrmse, manual_mae, manual_nmae

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Directory to save evaluation results')
parser.add_argument('--model-path', required=True, help='Path to the trained model file')
parser.add_argument('--eval-type', choices=['data_driven', 'physics'], required=True, help='Evaluation type: data_driven or physics')
args = parser.parse_args()

# Define base directories
lvad_data_path = '/home1/aissitt2019/LVAD/LVAD_data'

def load_dataset(filename):
    # Load dataset from given filename
    file_path = os.path.join(lvad_data_path, filename)
    return np.load(file_path)

# Load test data
testX_np = load_dataset('lvad_rdfs_inlets.npy')[40:]  # Adjusted indices for test data
testY_np = load_dataset('lvad_vels.npy')[40:]

# Create directories for this run
output_dir = args.output_dir
logs_dir = os.path.join(output_dir, "logs")
images_dir = os.path.join(output_dir, "images")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Load the trained model
model = load_model(args.model_path, custom_objects={
    'RMSEPerComponent': RMSEPerComponent,
    'NRMSEPerComponent': NRMSEPerComponent,
    'MAEPerComponent': MAEPerComponent,
    'NMAEPerComponent': NMAEPerComponent
})

# Perform model prediction
predictions = model.predict(testX_np)

# Calculate metrics
rmse = manual_rmse(testY_np, predictions)
nrmse = manual_nrmse(testY_np, predictions)
mae = manual_mae(testY_np, predictions)
nmae = manual_nmae(testY_np, predictions)

# Log results
results = {
    'RMSE': rmse.numpy().tolist(),
    'NRMSE': nrmse.numpy().tolist(),
    'MAE': mae.numpy().tolist(),
    'NMAE': nmae.numpy().tolist()
}

# Save results to a JSON file
results_file = os.path.join(output_dir, 'evaluation_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Evaluation results saved to {results_file}")

# Plot predictions vs ground truth
def plot_predictions_vs_ground_truth(y_true, y_pred, output_dir):
    for i in range(3):  # Assuming 3 components in velocity
        plt.figure()
        plt.plot(y_true[:, :, :, :, i].flatten(), label='True')
        plt.plot(y_pred[:, :, :, :, i].flatten(), label='Predicted')
        plt.title(f'Component {i+1} - True vs Predicted')
        plt.xlabel('Sample index')
        plt.ylabel('Velocity')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'component_{i+1}_true_vs_predicted.png'))
        plt.close()

plot_predictions_vs_ground_truth(testY_np, predictions, images_dir)

print(f"Plots saved to {images_dir}")

if args.eval_type == 'physics':
    # Additional physics-informed evaluation can be implemented here
    from utils import compute_vorticity, compute_continuity_loss, compute_momentum_loss, compute_vorticity_magnitude

    vorticity_pred = compute_vorticity(predictions)
    continuity_loss = compute_continuity_loss(predictions)
    momentum_loss = compute_momentum_loss(predictions, nu=1.0)  # Example nu value; adjust accordingly

    vorticity_magnitude = compute_vorticity_magnitude(vorticity_pred)
    
    # Save additional physics results to JSON
    physics_results = {
        'Continuity Loss': continuity_loss.numpy(),
        'Momentum Loss': momentum_loss.numpy(),
        'Vorticity Magnitude': vorticity_magnitude.numpy().tolist()
    }

    physics_results_file = os.path.join(output_dir, 'physics_evaluation_results.json')
    with open(physics_results_file, 'w') as f:
        json.dump(physics_results, f, indent=4)

    print(f"Physics evaluation results saved to {physics_results_file}")
