import numpy as np
import os
import matplotlib.pyplot as plt
import json

def load_dataset(file_path):
    # Load a NumPy dataset from a specified file path.
    return np.load(file_path)

def plot_training_history(history, output_dir):
    # Dynamically plot all metrics available in the training history.
    epochs = range(1, len(history.history['loss']) + 1)
    metrics = [key for key in history.history.keys() if 'val_' not in key]  # Extract metric names without validation prefix

    num_metrics = len(metrics)
    plt.figure(figsize=(18, 6 * num_metrics))  # Adjust height dynamically based on the number of metrics

    for i, metric in enumerate(metrics):
        plt.subplot(num_metrics, 1, i + 1)
        plt.plot(epochs, history.history[metric], label=f'Train {metric}')
        if f'val_{metric}' in history.history:
            plt.plot(epochs, history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()

    # Save the plot to the output directory.
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def save_hyperparameters(hyperparameters, output_dir):
    # Save hyperparameters to a JSON file in the specified output directory.
    hyperparameter_file = os.path.join(output_dir, 'hyperparameters.json')
    with open(hyperparameter_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to {hyperparameter_file}")

def log_runtime(runtime, logs_dir):
    # Log the runtime of the training process to a file in the logs directory.
    runtime_log_file = os.path.join(logs_dir, 'runtime.log')
    with open(runtime_log_file, 'w') as f:
        f.write(f"Total runtime: {runtime} seconds\n")
    print(f"Runtime logged to {runtime_log_file}")

def create_output_directories(base_dir, run_name):
    # Create directories for output, logs, and images under the base directory.
    output_dir = os.path.join(base_dir, run_name)
    logs_dir = os.path.join(output_dir, "logs")
    images_dir = os.path.join(output_dir, "images")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"Created output directories: {output_dir}, {logs_dir}, {images_dir}")
    return output_dir, logs_dir, images_dir

def load_and_split_data(input_data_path, output_data_path, train_indices, val_indices):
    # Load and split data into training and validation sets based on provided indices.
    lvad_inlets_np = load_dataset(input_data_path)
    lvad_vels_np = load_dataset(output_data_path)

    trainX_np = lvad_inlets_np[train_indices[0]:train_indices[1]]
    trainY_np = lvad_vels_np[train_indices[0]:train_indices[1]]
    valX_np = lvad_inlets_np[val_indices[0]:val_indices[1]]
    valY_np = lvad_vels_np[val_indices[0]:val_indices[1]]

    print(f"Loaded and split data: trainX {trainX_np.shape}, trainY {trainY_np.shape}, valX {valX_np.shape}, valY {valY_np.shape}")
    return trainX_np, trainY_np, valX_np, valY_np

def apply_mask(data, mask):
    # Apply a binary mask to the data.
    expanded_mask = np.repeat(mask[..., np.newaxis], data.shape[-1], axis=-1)
    return np.ma.masked_where(~expanded_mask, data)

def compute_errors(true_values, predicted_values, mask):
    # Compute absolute errors, peak error, and coordinates of peak error.
    true_values_masked = apply_mask(true_values, mask)
    predicted_values_masked = apply_mask(predicted_values, mask)

    abs_errors = np.abs(true_values_masked - predicted_values_masked)
    
    peak_error = np.max(abs_errors)
    peak_error_coords = np.unravel_index(np.argmax(abs_errors), abs_errors.shape)
    
    return abs_errors, peak_error, peak_error_coords

def compute_high_error_metrics(abs_errors, threshold):
    # Compute the number and percentage of points with high errors above a threshold.
    high_error_mask = abs_errors > threshold
    high_error_count = np.sum(high_error_mask)
    total_count = abs_errors.count()  # Use count() instead of size for masked array
    high_error_percentage = (high_error_count / total_count) * 100
    return high_error_count, high_error_percentage

def plot_error_maps(true_values, predicted_values, abs_errors, peak_error_coords, output_dir, mask):
    # Plot error maps for true values, predicted values, and absolute errors around the peak error location.
    peak_batch, peak_y, peak_x, peak_z, peak_component = peak_error_coords
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    true_slice = np.ma.masked_where(~mask[peak_batch, :, peak_y, :], true_values[peak_batch, :, peak_y, :, peak_component])
    predicted_slice = np.ma.masked_where(~mask[peak_batch, :, peak_y, :], predicted_values[peak_batch, :, peak_y, :, peak_component])
    abs_error_slice = abs_errors[peak_batch, :, peak_y, :, peak_component]
    
    vmin = min(true_slice.min(), predicted_slice.min())
    vmax = max(true_slice.max(), predicted_slice.max())
    
    im_true = axs[0].imshow(true_slice, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'True Values Slice @ Y={peak_y}')
    fig.colorbar(im_true, ax=axs[0])
    
    im_pred = axs[1].imshow(predicted_slice, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title(f'Predicted Values Slice @ Y={peak_y}')
    fig.colorbar(im_pred, ax=axs[1])
    
    im_err = axs[2].imshow(abs_error_slice, cmap='hot', aspect='auto')
    axs[2].set_title(f'Absolute Errors Slice @ Y={peak_y}')
    fig.colorbar(im_err, ax=axs[2])
    
    im_err.set_clim(0, abs_error_slice.max())
    
    plt.suptitle(f'Error Analysis at Peak Error Coordinates {peak_error_coords}')
    plt.savefig(os.path.join(output_dir, 'error_maps_peak_error.png'))
    plt.close()
