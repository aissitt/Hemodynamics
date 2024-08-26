import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset(file_path):
    # Load a NumPy dataset from the specified file path.
    return np.load(file_path)

def plot_training_history(history, output_dir):
    # Plot training history for loss and metrics.
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(18, 12))

    # Plot training and validation loss over epochs.
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot RMSE per component if available.
    if 'rmse_per_component' in history.history:
        plt.subplot(3, 1, 2)
        plt.plot(epochs, history.history['rmse_per_component'], label='Train RMSE Per Component')
        plt.plot(epochs, history.history['val_rmse_per_component'], label='Validation RMSE Per Component')
        plt.title('RMSE Per Component Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()

    # Plot MAE per component if available.
    if 'mae_per_component' in history.history:
        plt.subplot(3, 1, 3)
        plt.plot(epochs, history.history['mae_per_component'], label='Train MAE Per Component')
        plt.plot(epochs, history.history['val_mae_per_component'], label='Validation MAE Per Component')
        plt.title('MAE Per Component Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()

    # Save the plot to the output directory.
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def save_hyperparameters(hyperparameters, output_dir):
    # Save hyperparameters to a JSON file.
    import json
    hyperparameter_file = os.path.join(output_dir, 'hyperparameters.json')
    with open(hyperparameter_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to {hyperparameter_file}")

def log_runtime(runtime, logs_dir):
    # Log the runtime of the training process to a file.
    runtime_log_file = os.path.join(logs_dir, 'runtime.log')
    with open(runtime_log_file, 'w') as f:
        f.write(f"Total runtime: {runtime} seconds\n")
    print(f"Runtime logged to {runtime_log_file}")

def create_output_directories(base_dir, run_name):
    # Create directories for output, logs, and images.
    output_dir = os.path.join(base_dir, run_name)
    logs_dir = os.path.join(output_dir, "logs")
    images_dir = os.path.join(output_dir, "images")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"Created output directories: {output_dir}, {logs_dir}, {images_dir}")
    return output_dir, logs_dir, images_dir

def load_and_split_data(data_path, train_indices, val_indices):
    # Load and split data into training and validation sets based on indices.
    lvad_inlets_np = load_dataset(os.path.join(data_path, 'lvad_rdfs_inlets.npy'))
    lvad_vels_np = load_dataset(os.path.join(data_path, 'lvad_vels.npy'))

    trainX_np = lvad_inlets_np[train_indices[0]:train_indices[1]]
    trainY_np = lvad_vels_np[train_indices[0]:train_indices[1]]
    valX_np = lvad_inlets_np[val_indices[0]:val_indices[1]]
    valY_np = lvad_vels_np[val_indices[0]:val_indices[1]]

    print(f"Loaded and split data: trainX {trainX_np.shape}, trainY {trainY_np.shape}, valX {valX_np.shape}, valY {valY_np.shape}")
    return trainX_np, trainY_np, valX_np, valY_np
