# YOLO Optimization Framework User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Framework Overview](#framework-overview)
4. [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
5. [Fine-tuning](#fine-tuning)
6. [Data Augmentation](#data-augmentation)
7. [Running the Complete Pipeline](#running-the-complete-pipeline)
8. [Working with Configurations](#working-with-configurations)
9. [Model Optimization](#model-optimization)
10. [Standardized Models](#standardized-models)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Features](#advanced-features)
13. [Performance Tips](#performance-tips)
14. [Appendix](#appendix)

---

## 1. Introduction <a name="introduction"></a>

The YOLO Optimization Framework is a comprehensive toolkit for optimizing YOLOv8 object detection models through Neural Architecture Search (NAS) and fine-tuning. This framework allows you to:

- Automatically discover optimal model architectures tailored to your specific dataset
- Apply transfer learning and fine-tuning techniques to pretrained models
- Optimize model size and speed through pruning and quantization
- Leverage advanced data augmentation strategies
- Create standardized high-performance models

This user manual provides detailed instructions on how to use each component of the framework, explains key concepts, and offers best practices for obtaining the best results.

## 2. Installation <a name="installation"></a>

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- PyTorch 1.10 or higher

### Required Dependencies
```bash
# Core dependencies
pip install torch>=1.10.0 torchvision>=0.11.0
pip install ultralytics>=8.0.0  # For YOLOv8 functionality
pip install pyyaml>=6.0
pip install scikit-optimize>=0.9.0  # For Bayesian optimization
pip install pandas>=1.3.0  # For results processing
pip install albumentations>=1.3.0  # For advanced data augmentation

# Optional dependencies
pip install tensorboard>=2.9.0  # For visualization
```

### Installing the Framework
1. Clone the repository:
```bash
git clone https://github.com/your-organization/yolo-optimization.git
cd yolo-optimization
```

2. Install the package:
```bash
pip install -e .  # Install in development mode
```

## 3. Framework Overview <a name="framework-overview"></a>

The YOLO Optimization Framework consists of several interconnected modules:

### Core Modules
- **Neural Architecture Search (NAS)**: Automatically discovers optimal model architectures.
- **Fine-tuning**: Applies transfer learning to pretrained models or NAS results.
- **Data Augmentation**: Enhances training data to improve model robustness.
- **Pipeline Integration**: Combines NAS and fine-tuning into a single workflow.

### Supporting Modules
- **Configuration Management**: Handles YAML configuration files.
- **Model Architecture**: Creates custom YOLO architectures.
- **Pruning**: Reduces model size by removing less important weights.
- **Quantization**: Converts models to lower precision for faster inference.
- **Standardized Models**: Predefined high-performing architectures.

### Key Files
- `run_pipeline.py`: Main entry point for the end-to-end optimization pipeline
- `optimization.py`: Implements neural architecture search algorithms
- `fine_tuning.py`: Implements fine-tuning and transfer learning
- `data_augmentation.py`: Implements data augmentation strategies
- `model_architecture.py`: Creates custom YOLO architectures
- `config_utils.py`: Handles configuration loading and management
- `standard_yolo_architecture.py`: Provides standard model architectures
- `yolo_optimizer.py`: Alternative entry point with additional options

### Directory Structure
```
yolo-optimization/
├── run_pipeline.py        # Main entry point
├── optimization.py        # NAS implementation
├── fine_tuning.py         # Fine-tuning implementation
├── data_augmentation.py   # Data augmentation utilities
├── model_architecture.py  # Model architecture utilities
├── config_utils.py        # Configuration utilities
├── standard_yolo_architecture.py  # Standard models
├── yolo_optimizer.py      # Alternative entry point
├── nas_config.yaml        # Default NAS configuration
└── fine_tuning_config.yaml  # Default fine-tuning configuration
```

## 4. Neural Architecture Search (NAS) <a name="neural-architecture-search-nas"></a>

Neural Architecture Search (NAS) automatically discovers optimal model architectures tailored to your specific dataset.

### Basic Usage

To run NAS standalone:

```bash
python yolo_optimizer.py --mode search --config nas_config.yaml --data path/to/dataset.yaml
```

Parameters:
- `--mode search`: Specifies that only the NAS process should be run
- `--config`: Path to the NAS configuration file
- `--data`: Path to your dataset YAML file (YOLOv8 format)
- `--batch`: (Optional) Batch size for training (default: 4)
- `--threshold`: (Optional) Performance threshold for early stopping (default: 0.85)
- `--low-memory`: (Optional) Flag to use lower memory settings

### Configuration Options

The `nas_config.yaml` file contains all the parameters for the NAS process:

#### Directory Settings
```yaml
directories:
  base_dir: "yolo_opt_results"  # Base directory for all outputs
  best_model_dir: "best_models"  # Directory for best models
  results_dir: "results"  # Directory for results
```

#### Search Space
```yaml
search_space:
  depth_mult: [0.25, 1.0]       # Depth multiplier range
  width_mult: [0.25, 1.0]       # Width multiplier range
  kernel_size: [3, 7]           # Kernel size range
  num_channels: [16, 96]        # Base channel count range
  resolution: [256, 640]        # Input resolution range
  use_complex: [0, 1]           # Whether to use complex architecture
  activation: [0, 3]            # Activation function options
  use_cbam: [0, 1]              # Whether to use attention
  dropout_rate: [0.0, 0.5]      # Dropout rate range
  use_gating: [0, 1]            # Whether to use gating
  bottleneck_ratio: [0.25, 0.75]  # Bottleneck ratio range
  num_heads: [1, 8]             # Number of attention heads
  skip_connections: [0, 2]      # Skip connection type
  use_eca: [0, 1]               # Whether to use ECA
```

#### Optimization Settings
```yaml
optimization:
  init_points: 2          # Initial random points
  n_iter: 5               # Number of iterations
  performance_threshold: 0.75  # Early stopping threshold
  early_stopping: true    # Whether to use early stopping
  patience: 3             # Early stopping patience
```

#### Training Settings
```yaml
training:
  epochs: 10              # Training epochs per trial
  patience: 5             # Training early stopping patience
  optimizer: "AdamW"      # Optimizer
  batch_size: 16          # Batch size
  save: true              # Whether to save models
  verbose: true           # Verbose training output
  warmup_epochs: 3        # Warmup epochs
  lr0: 0.01               # Initial learning rate
  lrf: 0.01               # Final learning rate ratio
  use_ema: true           # Use EMA for weights
```

### Understanding NAS Results

After running NAS, the following outputs are generated:

1. **Best Model File**: Located in the `best_model_dir` directory with a name like `best_model_trial{N}_map{score}_{timestamp}.pt`

2. **Architecture Configuration**: YAML files describing the architecture, located in the same directory

3. **Performance Summary**: HTML and CSV files in the `results_dir` directory with detailed performance metrics for all trials

4. **Logs**: Detailed logs of the NAS process in the `logs` directory

### Example Use Cases

#### Finding a Small, Fast Model
```yaml
search_space:
  depth_mult: [0.25, 0.5]       # Smaller depth
  width_mult: [0.25, 0.5]       # Narrower width
  resolution: [256, 384]        # Lower resolution
```

#### Finding a High-Performance Model
```yaml
search_space:
  depth_mult: [0.5, 1.0]        # Deeper networks
  width_mult: [0.5, 1.0]        # Wider networks
  resolution: [448, 640]        # Higher resolution
optimization:
  performance_threshold: 0.9    # Higher threshold
```

## 5. Fine-tuning <a name="fine-tuning"></a>

Fine-tuning allows you to adapt pretrained models or NAS-discovered architectures to your specific dataset using transfer learning.

### Basic Usage

To run fine-tuning standalone:

```bash
# Fine-tune a specific model
python fine_tuning.py --config fine_tuning_config.yaml --model path/to/model.pt --dataset path/to/dataset.yaml

# Fine-tune the YOLOv8n pretrained model
python fine_tuning.py --yolov8n_only --dataset path/to/dataset.yaml
```

Parameters:
- `--config`: Path to fine-tuning configuration file
- `--model`: Path to the model to fine-tune
- `--dataset`: Path to your dataset YAML file
- `--yolov8n_only`: Flag to fine-tune only the YOLOv8n pretrained model
- `--create_config`: Generate a default configuration file

### Fine-tuning Methods

The framework supports several fine-tuning strategies:

1. **Full Fine-tuning**: Fine-tune all layers of the model
```yaml
name: 'full_finetune'
description: 'Fine-tune all layers of YOLOv8n'
freeze_layers: null
learning_rate: 0.001
epochs: 20
```

2. **Head-only Training**: Freeze the backbone and train only the detection head
```yaml
name: 'head_only'
description: 'Transfer learning - train only detection head'
freeze_layers: 'backbone'
learning_rate: 0.01
epochs: 15
```

3. **Progressive Unfreezing**: Gradually unfreeze layers during training
```yaml
name: 'progressive_unfreezing'
description: 'Progressive transfer learning strategy'
progressive: true
phases:
  - {freeze_layers: 'backbone', epochs: 5, learning_rate: 0.01}
  - {freeze_layers: -10, epochs: 10, learning_rate: 0.001}
  - {freeze_layers: null, epochs: 10, learning_rate: 0.0001}
```

4. **Domain Adaptation**: Focus on final layers with higher learning rate
```yaml
name: 'domain_adaptation'
description: 'Adapt YOLOv8n to new domain'
freeze_layers: -5
learning_rate: 0.02
epochs: 10
```

### Configuration Options

The `fine_tuning_config.yaml` file specifies fine-tuning parameters:

```yaml
directories:
  base_dir: 'yolo_fine_tuning'
  models_dir: 'models'
  results_dir: 'results'

dataset:
  path: 'data.yaml'

base_models:
  - {name: 'yolov8n.pt', description: 'YOLOv8 Nano Pretrained Model'}

fine_tuning:
  methods:
    # Method definitions here...

training:
  batch_size: 16
  patience: 5
  optimizer: 'AdamW'
  device: null  # Auto-detect
  image_size: 640
  save_period: 5
```

### Understanding Fine-tuning Results

After running fine-tuning, the following outputs are generated:

1. **Fine-tuned Models**: Located in the specified results directory, with subdirectories for each trial

2. **Results Summary**: JSON file with detailed results for all fine-tuning methods and models

3. **Best Model**: Identified in the summary with metrics like mAP@50, precision, and recall

### Example Use Cases

#### Fine-tuning for a Small Dataset
```yaml
fine_tuning:
  methods:
    - name: 'head_only'
      description: 'Transfer learning for small datasets'
      freeze_layers: 'backbone'
      learning_rate: 0.01
      epochs: 20
```

#### Fine-tuning for a New Domain
```yaml
fine_tuning:
  methods:
    - name: 'progressive_domain_adaptation'
      description: 'Gradual adaptation to new domain'
      progressive: true
      phases:
        - {freeze_layers: 'backbone', epochs: 5, learning_rate: 0.01}
        - {freeze_layers: -10, epochs: 5, learning_rate: 0.005}
        - {freeze_layers: null, epochs: 10, learning_rate: 0.001}
```

## 6. Data Augmentation <a name="data-augmentation"></a>

Data augmentation enhances your training dataset by applying various transformations to improve model robustness and performance.

### Basic Usage

Data augmentation is integrated into both NAS and fine-tuning processes. You can configure it in the respective configuration files:

```yaml
# In nas_config.yaml or fine_tuning_config.yaml
augmentation:
  # Standard YOLOv8 augmentations
  use_mosaic: true
  mosaic_prob: 0.8
  use_mixup: false
  use_copy_paste: false
  
  # Basic augmentations
  hsv_h: 0.015       # HSV Hue augmentation
  hsv_s: 0.5         # HSV Saturation
  hsv_v: 0.3         # HSV Value
  degrees: 0.0       # Rotation degrees
  translate: 0.1     # Translation
  scale: 0.4         # Scaling
  shear: 0.0         # Shear
  perspective: 0.0   # Perspective
  flipud: 0.0        # Vertical flip
  fliplr: 0.5        # Horizontal flip
  
  # Advanced augmentations (via Albumentations)
  use_albumentations: true
  random_brightness_contrast: true
  brightness_limit: 0.2
  contrast_limit: 0.2
  random_brightness_contrast_prob: 0.2
  blur: false
  cutout: true
  cutout_holes: 4
  cutout_height: 8
  cutout_width: 8
  cutout_prob: 0.3
```

### Available Augmentation Techniques

The framework supports these augmentation categories:

1. **Geometric Transformations**:
   - Translation
   - Scaling
   - Flipping (horizontal/vertical)
   - Rotation
   - Shear
   - Perspective transformations

2. **Color Transformations**:
   - HSV adjustments (hue, saturation, brightness)
   - Contrast adjustments
   - Random brightness

3. **Advanced Techniques**:
   - Mosaic: Combines 4 images into one
   - Cutout: Randomly masks regions of the image
   - MixUp: Blends two images together
   - Copy-Paste: Copies objects between images

### Customizing for Your Dataset

Different datasets benefit from different augmentation strategies:

#### For Small Datasets
```yaml
augmentation:
  use_mosaic: true
  mosaic_prob: 1.0  # Increased probability
  use_mixup: true   # Enable MixUp
  mixup_prob: 0.2
  scale: 0.5        # More aggressive scaling
  fliplr: 0.5
  hsv_s: 0.7        # More aggressive saturation
  hsv_v: 0.4        # More aggressive brightness
  cutout: true      # Enable Cutout
  cutout_prob: 0.5  # Increased probability
```

#### For Domain-Specific Data (e.g., Aerial Imagery)
```yaml
augmentation:
  use_mosaic: true
  rotate: 0.5       # Enable rotation (good for orientation-agnostic cases)
  degrees: 180      # Full rotation
  scale: 0.6        # More aggressive scaling
  use_albumentations: true
  random_brightness_contrast: true
  brightness_limit: 0.3
  contrast_limit: 0.3
```

## 7. Running the Complete Pipeline <a name="running-the-complete-pipeline"></a>

The complete pipeline combines Neural Architecture Search (NAS) and fine-tuning into a single workflow.

### Basic Usage

```bash
python run_pipeline.py --dataset path/to/dataset.yaml --nas_config nas_config.yaml --ft_config fine_tuning_config.yaml
```

Parameters:
- `--dataset`: Path to your dataset YAML file (required)
- `--nas_config`: Path to NAS configuration file (default: nas_config.yaml)
- `--ft_config`: Path to fine-tuning configuration file (default: fine_tuning_config.yaml)
- `--skip_nas`: Skip the NAS step and use an existing model
- `--skip_ft`: Skip the fine-tuning step
- `--model_path`: Path to existing model (when skipping NAS)
- `--batch`: Batch size for training
- `--low_memory`: Use lower memory settings
- `--output_dir`: Override output directory
- `--yolov8n_only`: Focus only on fine-tuning the YOLOv8n pretrained model

### Pipeline Workflow

The complete pipeline follows these steps:

1. **Configuration Loading**: Load and validate all configuration files
2. **Neural Architecture Search** (if not skipped):
   - Run architecture search based on the NAS configuration
   - Find the best model architecture for your dataset
   - Save the best model to the specified directory
3. **Fine-tuning** (if not skipped):
   - Use the best model from NAS (or specified model)
   - Apply various fine-tuning strategies
   - Find the best fine-tuning approach
   - Save the best fine-tuned model
4. **Results Summary**: Create a unified summary of the entire optimization process

### Example Commands

#### Full Pipeline with Default Settings
```bash
python run_pipeline.py --dataset data/custom_dataset.yaml
```

#### Skip NAS, Fine-tune Existing Model
```bash
python run_pipeline.py --dataset data/custom_dataset.yaml --skip_nas --model_path models/yolov8n.pt
```

#### YOLOv8n-only Mode
```bash
python run_pipeline.py --dataset data/custom_dataset.yaml --yolov8n_only
```

#### Custom Output Directory
```bash
python run_pipeline.py --dataset data/custom_dataset.yaml --output_dir my_custom_results
```

### Understanding Pipeline Results

After running the complete pipeline, the following outputs are generated:

1. **Best Model from NAS**: Located in the `best_models` directory
2. **Best Fine-tuned Model**: Located in the fine-tuning results directory
3. **Unified Summary**: JSON file with results from both NAS and fine-tuning
4. **Performance Metrics**: mAP@50, precision, recall for all models
5. **Log Files**: Detailed logs of the entire process

## 8. Working with Configurations <a name="working-with-configurations"></a>

The framework uses YAML configuration files to control all aspects of the optimization process.

### Configuration File Structure

#### NAS Configuration (`nas_config.yaml`)
```yaml
# Directory structure
directories:
  base_dir: "yolo_opt_results"
  best_model_dir: "best_models"
  results_dir: "results"

# Dataset configuration
dataset_path: "data.yaml"

# Search space for architecture optimization
search_space:
  # Parameters and ranges...

# Optimization settings
optimization:
  # Settings...

# Training settings
training:
  # Settings...

# Data augmentation settings
augmentation:
  # Settings...
```

#### Fine-tuning Configuration (`fine_tuning_config.yaml`)
```yaml
# Directory structure
directories:
  base_dir: "yolo_fine_tuning"
  models_dir: "models"
  results_dir: "results"

# Dataset configuration
dataset:
  path: "data.yaml"

# Base models
base_models:
  # Models to fine-tune...

# Fine-tuning methods
fine_tuning:
  methods:
    # Method definitions...

# Training settings
training:
  # Settings...
```

### Creating Default Configurations

You can generate default configuration files:

```bash
# Create default fine-tuning configuration
python fine_tuning.py --create_config

# Creating a standard NAS configuration is not directly supported,
# but you can copy and modify the provided nas_config.yaml
```

### Customizing Configurations

You can customize configurations for your specific needs:

#### For Limited GPU Memory
```yaml
# In nas_config.yaml
training:
  batch_size: 4  # Smaller batch size
  resolution: [256, 384]  # Smaller input size

# In fine_tuning_config.yaml
training:
  batch_size: 8
  image_size: 416
```

#### For Quick Experimentation
```yaml
# In nas_config.yaml
optimization:
  n_iter: 5  # Fewer iterations
  performance_threshold: 0.7  # Lower threshold

training:
  epochs: 5  # Fewer epochs per trial

# In fine_tuning_config.yaml
fine_tuning:
  methods:
    - name: 'quick_test'
      description: 'Quick fine-tuning test'
      freeze_layers: 'backbone'
      learning_rate: 0.01
      epochs: 5
```

#### For Maximum Performance
```yaml
# In nas_config.yaml
optimization:
  n_iter: 20  # More iterations
  performance_threshold: 0.9  # Higher threshold

training:
  epochs: 30  # More epochs per trial
  batch_size: 32  # Larger batch size (if memory allows)

# In fine_tuning_config.yaml
fine_tuning:
  methods:
    - name: 'full_finetune'
      description: 'Thorough fine-tuning'
      freeze_layers: null  # Train all layers
      learning_rate: 0.001
      epochs: 50
```

## 9. Model Optimization <a name="model-optimization"></a>

The framework provides additional optimization techniques for reducing model size and improving inference speed.

### Running Model Optimization

```bash
python yolo_optimizer.py --mode optimize --model path/to/your/model.pt --config nas_config.yaml
```

Parameters:
- `--mode optimize`: Specifies that only optimization should be run
- `--model`: Path to the model to optimize
- `--config`: Path to the configuration file
- `--skip-pruning`: Skip the pruning step
- `--skip-quantization`: Skip the quantization step
- `--data`: Path to dataset for validating optimized models

### Available Optimization Techniques

#### Pruning
Pruning reduces model size by removing less important weights:

```yaml
# In nas_config.yaml
pruning:
  method: "magnitude"  # Pruning method (magnitude or structured)
  amount: 0.3          # Percentage of weights to prune
  iterations: 3        # Number of iterative pruning rounds
  retrain_epochs: 5    # Epochs to retrain after each round
```

#### Quantization
Quantization converts models to lower precision for faster inference:

```yaml
# In nas_config.yaml
quantization:
  int8: true          # Enable INT8 quantization
  fp16: true          # Enable FP16 quantization
  calibration_steps: 100  # Steps for calibration
```

### Understanding Optimization Results

After running model optimization, the following outputs are generated:

1. **Pruned Models**: Located in the `pruned_models` directory
2. **Quantized Models**: Located in the `quantized_models` directory
3. **Performance Metrics**: Comparison of original, pruned, and quantized models

## 10. Standardized Models <a name="standardized-models"></a>

The framework includes standardized YOLO architectures that have been optimized for different use cases.

### Using Standardized Models

```bash
python standard_yolo_architecture.py --type standard --data path/to/dataset.yaml --output models
```

Parameters:
- `--type`: Model type (standard, efficient, high_performance)
- `--data`: Path to dataset YAML file
- `--output`: Output directory
- `--epochs`: Number of training epochs
- `--batch`: Batch size for training
- `--classes`: Number of classes for detection

### Available Model Types

1. **Standard Model**: Balanced architecture for general use
```python
# Key parameters
depth_mult = 0.58
width_mult = 0.60
num_channels = 64
dropout_rate = 0.29
input_size = 352
```

2. **Efficient Model**: Smaller architecture for resource-constrained environments
```python
# Key parameters
depth_mult = 0.33
width_mult = 0.50
num_channels = 48
input_size = 320
```

3. **High-Performance Model**: Larger architecture for maximum accuracy
```python
# Key parameters
depth_mult = 0.67
width_mult = 0.75
num_channels = 80
input_size = 416
```

### Example Use Cases

#### Embedded Devices
```bash
python standard_yolo_architecture.py --type efficient --data path/to/dataset.yaml --output embedded_models
```

#### High-Accuracy Applications
```bash
python standard_yolo_architecture.py --type high_performance --data path/to/dataset.yaml --output high_accuracy_models --epochs 200
```

## 11. Troubleshooting <a name="troubleshooting"></a>

### Common Issues and Solutions

#### Out of Memory Errors
**Symptoms**:
- CUDA out of memory error
- Process killed during training

**Solutions**:
1. Reduce batch size: Use `--batch 4` or lower
2. Use `--low_memory` flag
3. Reduce model complexity in configuration:
```yaml
search_space:
  depth_mult: [0.25, 0.5]  # Lower depth
  width_mult: [0.25, 0.5]  # Narrower width
  resolution: [256, 384]   # Smaller input size
```

#### Slow Training
**Symptoms**:
- Training takes much longer than expected

**Solutions**:
1. Reduce epochs in configuration:
```yaml
training:
  epochs: 5  # Fewer epochs per trial
```
2. Reduce search iterations:
```yaml
optimization:
  n_iter: 5  # Fewer iterations
```
3. Use a smaller resolution:
```yaml
training:
  image_size: 320  # Smaller input size
```

#### Poor Model Performance
**Symptoms**:
- Low mAP scores
- Model doesn't detect objects well

**Solutions**:
1. Increase training duration:
```yaml
training:
  epochs: 50  # More epochs
```
2. Adjust data augmentation:
```yaml
augmentation:
  use_mosaic: true
  mosaic_prob: 1.0
  scale: 0.5
  hsv_s: 0.7
  hsv_v: 0.4
```
3. Try different fine-tuning methods:
```yaml
fine_tuning:
  methods:
    - name: 'progressive_unfreezing'
      progressive: true
      phases:
        - {freeze_layers: 'backbone', epochs: 10, learning_rate: 0.01}
        - {freeze_layers: -10, epochs: 15, learning_rate: 0.001}
        - {freeze_layers: null, epochs: 20, learning_rate: 0.0001}
```

#### Import Errors
**Symptoms**:
- ModuleNotFoundError
- ImportError

**Solutions**:
1. Check all dependencies are installed:
```bash
pip install -r requirements.txt
```
2. Ensure you're running from the correct directory
3. Check PYTHONPATH includes the project directory

### Logging and Debugging

The framework creates detailed logs that can help diagnose issues:

1. **Console Output**: Basic progress information
2. **Log Files**: Detailed logs in the `logs` directory
3. **TensorBoard**: Visualizations in `runs/yolo_optimization`

To enable more verbose logging, modify logging settings in the code:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
```

To view TensorBoard visualizations:
```bash
tensorboard --logdir=runs/yolo_optimization
```

## 12. Advanced Features <a name="advanced-features"></a>

### Custom Model Architectures

You can create custom YOLO architectures by modifying `model_architecture.py`:

```python
def create_custom_yolo_config(depth_mult, width_mult, kernel_size, num_channels):
    # Customize parameters for your specific needs
    # ...
```

### Working with Custom Datasets

Prepare your dataset in the YOLOv8 format:

1. Create a dataset YAML file:
```yaml
# dataset.yaml
path: /path/to/dataset  # Root directory
train: images/train     # Train images (relative to path)
val: images/val         # Validation images
test: images/test       # Test images

# Classes
nc: 3  # Number of classes
names: ['class1', 'class2', 'class3']
```

2. Organize your data:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

3. Format your labels in the YOLO format:
```
class_id x_center y_center width height
```

### Multi-GPU Training

The framework supports multi-GPU training:

```yaml
# In nas_config.yaml or fine_tuning_config.yaml
training:
  device: [0, 1, 2, 3]  # Use 4 GPUs
```

### Custom Callbacks

You can add custom callbacks by modifying the training functions:

```python
# Add to train_trial_model in optimization.py
from ultralytics.yolo.utils.callbacks import callbacks

# Define custom callback
def on_train_epoch_end(trainer):
    print(f"Custom callback: Epoch {trainer.epoch} ended")

# Register callback
callbacks.on_train_epoch_end = on_train_epoch_end
```

## 13. Performance Tips <a name="performance-tips"></a>

### Optimizing for Speed

1. **Use smaller models**:
```yaml
search_space:
  depth_mult: [0.25, 0.4]
  width_mult: [0.25, 0.4]
  resolution: [256, 320]
```

2. **Enable quantization**:
```yaml
quantization:
  int8: true
  fp16: true
```

3. **Use efficient augmentation**:
```yaml
augmentation:
  use_mosaic: true
  use_mixup: false  # Disable computationally expensive augmentations
  use_copy_paste: false
```

4. **Set appropriate batch size**:
Start with a smaller batch size and increase until you hit memory limits.

### Optimizing for Accuracy

1. **Use larger models**:
```yaml
search_space:
  depth_mult: [0.6, 1.0]
  width_mult: [0.6, 1.0]
  resolution: [416, 640]
```

2. **Use comprehensive augmentation**:
```yaml
augmentation:
  use_mosaic: true
  use_mixup: true
  hsv_s: 0.7
  hsv_v: 0.4
  scale: 0.5
  translate: 0.2
  cutout: true
```

3. **Train for longer**:
```yaml
training:
  epochs: 100
  patience: 30  # Longer patience for early stopping
```

4. **Use progressive learning rates**:
```yaml
training:
  warmup_epochs: 5
  lr0: 0.01
  lrf: 0.001
```

### Optimizing for Size-Constrained Environments

1. **Use efficient models**:
```bash
python standard_yolo_architecture.py --type efficient --data path/to/dataset.yaml
```

2. **Apply pruning**:
```yaml
pruning:
  method: "magnitude"
  amount: 0.5        # Higher pruning amount
  iterations: 5
```

3. **Use INT8 quantization**:
```yaml
quantization:
  int8: true
  fp16: false
```

## 14. Appendix <a name="appendix"></a>

### Dataset Preparation Guidelines

For best results, prepare your dataset following these guidelines:

1. **Image Quality**:
   - Resolution: At least 640x640 pixels for high-quality detection
   - Format: JPEG or PNG (JPEG preferred for training efficiency)
   - Variety: Include different lighting conditions, angles, and backgrounds

2. **Annotation Guidelines**:
   - Label all instances of objects
   - Use tight bounding boxes
   - Be consistent with similar objects
   - Include occlusion cases
   - Cover different object sizes

3. **Dataset Distribution**:
   - Training set: 70-80% of data
   - Validation set: 10-15% of data
   - Test set: 10-15% of data
   - Ensure class balance across splits

### Configuration Reference

Comprehensive list of all configuration parameters:

#### NAS Configuration
| Parameter | Section | Description | Default |
|-----------|---------|-------------|---------|
| `base_dir` | directories | Base directory for all outputs | "yolo_opt_results" |
| `best_model_dir` | directories | Directory for best models | "best_models" |
| `depth_mult` | search_space | Depth multiplier range | [0.25, 1.0] |
| `width_mult` | search_space | Width multiplier range | [0.25, 1.0] |
| `epochs` | training | Number of training epochs per trial | 10 |
| `batch_size` | training | Training batch size | 16 |
| ... | ... | ... | ... |

#### Fine-tuning Configuration
| Parameter | Section | Description | Default |
|-----------|---------|-------------|---------|
| `base_dir` | directories | Base directory for fine-tuning | "yolo_fine_tuning" |
| `freeze_layers` | fine_tuning.methods | Layers to freeze | null |
| `learning_rate` | fine_tuning.methods | Learning rate | 0.001 |
| `epochs` | fine_tuning.methods | Training epochs | 20 |
| ... | ... | ... | ... |

### Command Line Reference

All available command-line arguments for the main scripts:

#### run_pipeline.py
```
--dataset DATASET        Path to dataset configuration (data.yaml)
--nas_config NAS_CONFIG  Path to NAS configuration file
--ft_config FT_CONFIG    Path to fine-tuning configuration file
--skip_nas               Skip Neural Architecture Search
--model_path MODEL_PATH  Path to existing model (when skipping NAS)
--batch BATCH            Batch size for training
--low_memory             Use lower memory settings
--output_dir OUTPUT_DIR  Override output directory
--skip_ft                Skip fine-tuning
--yolov8n_only           Focus only on fine-tuning YOLOv8n
```

#### yolo_optimizer.py
```
--mode {search,optimize,full}  Operation mode
--model MODEL                  Path to the model to optimize
--config CONFIG                Path to the configuration file
--batch BATCH                  Batch size for training
--low-memory                   Start with low memory configuration
--threshold THRESHOLD          Performance threshold for early stopping
--skip-pruning                 Skip pruning step
--skip-quantization            Skip quantization step
--data DATA                    Path to the dataset YAML file
```

#### standard_yolo_architecture.py
```
--type {standard,efficient,high_performance}  Model type
--data DATA                                   Path to dataset
--output OUTPUT                               Output directory
--epochs EPOCHS                               Number of training epochs
--batch BATCH                                 Batch size for training
--classes CLASSES                             Number of classes
```

### Further Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Data Augmentation Best Practices](https://albumentations.ai/docs/)

---

Thank you for using the YOLO Optimization Framework. For additional support, bug reports, or feature requests, please open an issue on the GitHub repository.