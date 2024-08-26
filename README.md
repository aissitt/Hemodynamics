# Hemodynamics

This repository contains code for training and evaluating deep learning models for hemodynamic simulations, specifically focusing on Left Ventricular Assist Device (LVAD) data. The repository supports both data-driven and physics-informed models using a 3D U-Net architecture.

## Project Structure

```
Hemodynamics/                         
│
├── LVAD/                             
│   ├── config.json                   # Configuration file for training and evaluation
│   ├── environment.yml               # Conda environment file for setting up dependencies
│   ├── loss.py                       # Script containing custom loss functions
│   ├── metrics.py                    # Script containing custom metrics
│   ├── model.py                      # Script defining the model architecture (UNet)
│   ├── train.py                      # Training script
│   ├── eval.py                       # Evaluation script
│   ├── utils.py                      # Utility functions used across scripts
│   └── slurm/                        
│       ├── train_multigpu.sh         # SLURM script for launching training jobs
│       ├── eval_multigpu.sh          # SLURM script for launching evaluation jobs
│       └── env.sh                    # Script for setting up the environment
│
├── outputs/                          
│   ├── train_run_YYYYMMDD_HHMMSS/    
│   │   ├── lvad_model_YYYYMMDD_HHMMSS.h5  # Saved model file from the training run
│   │   ├── logs/                     
│   │   │   ├── metrics.json          # JSON file containing evaluation metrics
│   │   │   └── runtime.log           # Log file containing runtime information
│   │   ├── images/                   
│   │   │   └── training_history.png  # Plot of training history
│   │   └── eval_run_YYYYMMDD_HHMMSS/ 
│   │       ├── logs/                 
│   │       │   └── metrics.json      # JSON file containing evaluation metrics for this evaluation run
│   │       ├── images/               
│   │       │   └── error_maps_peak_error.png  # Plot of error maps at peak error
│   │       └── ...                   
│   ├── ...                           
│
├── README.md                         # README file with project details and usage instructions
└── requirements.txt                  # List of required Python packages (if using pip)
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda
- NVIDIA GPU with CUDA support
- [SLURM](https://slurm.schedmd.com/) for job scheduling

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Hemodynamics.git
   cd Hemodynamics/LVAD
   ```

2. **Set Up the Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate Hemodynamics
   ```

3. **Configure Environment Variables** (if using `use_env_vars` in `config.json`):
   ```bash
   export INPUT_DATA_PATH=/path/to/input/data
   export OUTPUT_DATA_PATH=/path/to/output/data
   ```

## Training and Evaluation

### Training

To train a model, use the `train_multigpu.sh` script with SLURM:

```bash
sbatch slurm/train_multigpu.sh data_driven  # For data-driven model
sbatch slurm/train_multigpu.sh physics      # For physics-informed model
```

Training parameters like epochs, batch size, learning rate, and data paths are configured in the `config.json` file.

### Evaluation

To evaluate a model, use the `eval_multigpu.sh` script with SLURM:

```bash
sbatch slurm/eval_multigpu.sh data_driven  # For data-driven model
sbatch slurm/eval_multigpu.sh physics      # For physics-informed model
```

By default, the evaluation script uses the most recent model saved in the corresponding `train_run` directory. You can specify a different model by modifying the `eval_multigpu.sh` script or passing the `--model-path` argument.

## Configuration

The `config.json` file contains all the configurations for training and evaluation:

```json
{
  "use_env_vars": true, 
  "training": {
    "epochs": 10000,
    "batch_size": 4,
    "learning_rate": 0.001,
    "train_indices": [0, 36],
    "val_indices": [36, 40],
    "test_indices": [40, 50],
    "input_data": "$INPUT_DATA_PATH",  
    "output_data": "$OUTPUT_DATA_PATH" 
  },
  "loss_parameters": {
    "data_driven": {
      "huber_delta": 0.1
    },
    "physics_informed": {
      "lambda_data": 1.0,
      "lambda_continuity": 0.001,
      "lambda_vorticity_focused": 0.1,
      "threshold_vorticity": 0.0437,
      "lambda_momentum": 0.001,
      "nu": 3.5e-6
    }
  },
  "model": {
    "input_shape": [128, 128, 128, 2]
  }
}
```

## Loss Functions

### Data-Driven Loss (Huber Loss)

The Huber loss for each velocity component (u, v, w) is calculated as:

$$
L_{\text{Huber}} = \frac{1}{3} \left( \text{Huber}(u, \hat{u}) + \text{Huber}(v, \hat{v}) + \text{Huber}(w, \hat{w}) \right)
$$

where the Huber loss is defined as:

$$
\text{Huber}(x, y) = 
\begin{cases} 
0.5 \cdot (x - y)^2 & \text{if } |x - y| < \delta \\
\delta \cdot (|x - y| - 0.5 \cdot \delta) & \text{otherwise}
\end{cases}
$$

### Physics-Informed Loss

The physics-informed loss combines several components:

$$
L_{\text{Physics}} = \lambda_{\text{data}}L_{\text{data}} + \lambda_{\text{continuity}}L_{\text{continuity}} + \lambda_{\text{vorticity}}L_{\text{vorticity}} + \lambda_{\text{momentum}}L_{\text{momentum}}
$$

where:

- **\(L_{\text{data}}\):** Huber loss on velocity components  
- **\(L_{\text{continuity}}\):** Continuity loss to enforce incompressibility  
- **\(L_{\text{vorticity}}\):** Vorticity-focused loss to emphasize high-vorticity regions  
- **\(L_{\text{momentum}}\):** Partial momentum loss capturing convective and diffusive terms of the Navier-Stokes equations (note: only specific components are used, not the full Navier-Stokes momentum equation).  
