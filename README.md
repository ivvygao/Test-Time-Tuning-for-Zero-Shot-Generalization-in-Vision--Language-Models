# Dynamic Shot Capacity Adjustment = Overview
This project implements a mechanism designed to dynamically adjust shot capacity based on class-specific accuracy to improve model performance during inference.
The framework utilizes PyTorch and supports Distributed Data Parallel (DDP) training across multiple GPUs.

## Features
Dynamic Shot Capacity Adjustment: Dynamically adjusts the shot capacity based on real-time inference accuracy metrics to optimize performance.
Distributed Training: Leverages multiple GPUs to accelerate the adaptation process using PyTorch's DistributedDataParallel.
Visualization Tools: Includes functionality to generate and display confusion matrices, ROC curves, and other relevant visualizations to analyze model performance.
WandB Integration: Utilizes Weights & Biases for tracking experiments, logging performance metrics, and visualizing training progress.
Prerequisites

Before you begin, ensure you have met the following requirements:

Python 3.8+
PyTorch 1.7+
torchvision
scikit-learn
matplotlib
wandb: For experiment tracking and visualization.
Installation
Clone the repository and install the required packages:

```bash
git clone https://ivvygao/dynamic_shot_capacity_adjustment.git
cd project_tda
pip install -r requirements.txt
```

## Configuration
Adjust the configuration parameters for the DSA runner in the config.yaml file. This file contains settings for the model, dataset, and adaptation parameters.

33 Output
The script will output various performance metrics and save visualization results in the specified output directory. Check the Weights & Biases dashboard for real-time monitoring and analysis.
