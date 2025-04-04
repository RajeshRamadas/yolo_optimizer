# YOLO Architecture Optimizer

A comprehensive tool for optimizing YOLOv8 models through neural architecture search, structured pruning, and quantization.

## Overview

This project provides a framework for automatically optimizing YOLO object detection models with three key capabilities:

1. **Neural Architecture Search (NAS)**: Efficiently discover optimal model architectures using Bayesian optimization.
2. **Structured Pruning**: Remove redundant channels to reduce model size while preserving accuracy.
3. **Quantization**: Convert models to INT8 or FP16 precision for faster inference.

## Features

- **YAML-based Configuration**: All search spaces and optimization parameters defined in a single config file
- **Bayesian Optimization**: Efficient hyperparameter search with early stopping
- **Performance Tracking**: TensorBoard integration for visualizing optimization progress
- **Low-memory Mode**: Start with a minimal model for memory-constrained environments
- **Comprehensive Logging**: Detailed logs of all optimizations and results
- **Parameter Export**: JSON export of best architectures and training configurations
- **Modular Design**: Each optimization component can be used independently

## Installation

```bash
# Clone the repository
git clone https://github.com/RajeshRamadas/yolo-optimizer.git
cd yolo-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- PyTorch 1.10+
- Ultralytics (for YOLOv8)
- Bayesian Optimization
- TensorBoard
- ONNX Runtime (for quantized model benchmarking)

## Usage

The optimizer can be run in three modes:

### 1. Architecture Search Only

```bash
python yolo_optimizer.py --mode search --config nas_config.yaml
```

### 2. Model Optimization Only (Pruning & Quantization)

```bash
python yolo_optimizer.py --mode optimize --model path/to/your/model.pt --config nas_config.yaml
```

### 3. Complete Pipeline (Search + Optimization)

```bash
python yolo_optimizer.py --mode full --config nas_config.yaml
```

### 3. Complete Pipeline (Search + Optimization + dataset)

```bash
python yolo_optimizer.py --mode full --config nas_config.yaml --data path/to/your/data.yaml
```

### Additional Options

- `--threshold 0.75`: Set custom performance threshold for early stopping (default: 0.85)
- `--low-memory`: Start with minimal configuration to conserve memory
- `--skip-pruning`: Skip the pruning step
- `--skip-quantization`: Skip the quantization step

## Configuration

The search space and optimization parameters are defined in `nas_config.yaml`. Here's an example:

```yaml
# Search space definition - [min_value, max_value]
search_space:
  # Architecture parameters
  resolution: [320, 520]     # Input image size
  depth_mult: [0.33, 0.75]   # Depth multiplier (scaling of layers)
  width_mult: [0.25, 0.75]   # Width multiplier (scaling of channels)
  kernel_size: [3, 5]        # Kernel size for specific layers
  num_channels: [16, 128]    # Base number of channels
  
  # Training hyperparameters
  lr0: [1.0e-5, 1.0e-2]      # Initial learning rate
  momentum: [0.6, 0.98]      # Momentum for optimizer
  batch_size: [8, 16]        # Training batch size
  iou_thresh: [0.4, 0.9]     # IoU threshold for NMS
  weight_decay: [0.0001, 0.01] # Weight decay regularization
```

See `nas_config.yaml` for a full example.

## Project Structure

```
yolo-optimizer/
├── yolo_optimizer.py      # Main script
├── nas_config.yaml        # Configuration file
├── config_utils.py        # YAML configuration utilities
├── model_architecture.py  # YOLO architecture functions
├── optimization.py        # Bayesian optimization implementation
├── pruning.py             # Model pruning functions
├── quantization.py        # Model quantization functions
├── logs/                  # Log files
├── best_model/            # Best architectures from search
├── pruned_models/         # Pruned model outputs
└── quantized_models/      # Quantized model outputs
```

## Output Files

The optimizer generates several output files:

- `best_model/best_model_{map50}.pt`: Best model weights from architecture search
- `best_model/nas_results.json`: Detailed search results and best parameters
- `best_model/optimization_history.json`: Complete history of all trials
- `pruned_models/pruned_{amount}_{map50}.pt`: Pruned models with different pruning levels
- `quantized_models/{model_name}_int8.onnx`: INT8 quantized models
- `quantized_models/{model_name}_fp16.onnx`: FP16 quantized models
- `logs/yolo_opt_{timestamp}.log`: Detailed log file

## Example Results

A successful optimization typically yields the following improvements:

| Model | mAP50 | Parameters | Inference Time |
|-------|-------|------------|----------------|
| Original | 0.8534 | 3,145,728 | 15.2ms |
| Architecture Search | 0.8612 | 1,862,436 | 12.1ms |
| Pruned (30%) | 0.8503 | 1,325,672 | 9.8ms |
| INT8 Quantized | 0.8482 | 1,325,672 | 5.3ms |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) for the optimization framework
