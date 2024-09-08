# Dynamic Shot Capacity Adjustment Overview
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
cd Dynamic_Shot_Capacity_Adjustment
```

## Configuration
Adjust the configuration parameters for the DSA runner in the config.yaml file. This file contains settings for the model, dataset, and adaptation parameters.

## Output
The script will output various performance metrics and save visualization results in the specified output directory. Check the Weights & Biases dashboard for real-time monitoring and analysis.

## Datasets
I suggest putting all datasets under the same folder (say `$dataset`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$dataset/
|–– imagenet/
|–– imagenet-a/
|–– imagenetv2/
|–– imagenet-a/
```

### ImageNetV2
- Create a folder named `imagenetv2/` under `$DATA`.
- Go to this github repo https://github.com/modestyachts/ImageNetV2.
- Download the matched-frequency dataset from https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz and extract it to `$DATA/imagenetv2/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenetv2/`.

The directory structure should look like
```
imagenetv2/
|–– imagenetv2-matched-frequency-format-val/
|–– classnames.txt
```

### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `$DATA/imagenet-sketch`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-sketch/`.

The directory structure should look like
```
imagenet-sketch/
|–– images/ # contains 1,000 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-A
- Create a folder named `imagenet-adversarial/` under `$DATA`.
- Download the dataset from https://github.com/hendrycks/natural-adv-examples and extract it to `$DATA/imagenet-adversarial/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-adversarial/`.

The directory structure should look like
```
imagenet-adversarial/
|–– imagenet-a/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-R
- Create a folder named `imagenet-rendition/` under `$DATA`.
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `$DATA/imagenet-rendition/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-rendition/`.

The directory structure should look like
```
imagenet-rendition/
|–– imagenet-r/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
