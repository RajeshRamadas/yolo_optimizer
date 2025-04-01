# YOLOv8 Neural Architecture Search and Fine-Tuning Pipeline

This project combines Neural Architecture Search (NAS) with advanced fine-tuning strategies for YOLOv8 models. It allows you to either search for optimal architectures and then fine-tune them, or focus specifically on transfer learning with YOLOv8n pretrained model.

## Setup

1. Ensure all required files are in the same directory:
   - `optimization.py` (NAS implementation)
   - `model_architecture.py` (YOLO architecture builder)
   - `config_utils.py` (Configuration utilities)
   - `data_augmentation.py` (Data augmentation functions)
   - `fine_tuning.py` (Fine-tuning implementation)
   - `run_pipeline.py` (Main integration script)
   - `nas_config.yaml` (NAS configuration)
   - `fine_tuning_config.yaml` (Fine-tuning configuration)

2. Install dependencies:
   ```bash
   pip install ultralytics torch torchvision pyyaml tensorboard
   ```

## Usage Options

### Option 1: Full Pipeline (NAS + Fine-tuning)

Run the complete pipeline to perform NAS and then fine-tune the best model:

```bash
python run_pipeline.py --dataset path/to/data.yaml
```

### Option 2: NAS Only

Run only the Neural Architecture Search:

```bash
python run_pipeline.py --dataset path/to/data.yaml --skip_ft
```

### Option 3: Fine-tuning Only (using a specific model)

Skip NAS and fine-tune an existing model:

```bash
python run_pipeline.py --dataset path/to/data.yaml --skip_nas --model_path path/to/model.pt
```

### Option 4: YOLOv8n Transfer Learning Only

Focus on fine-tuning the YOLOv8n pretrained model with multiple strategies:

```bash
python run_pipeline.py --dataset path/to/data.yaml --yolov8n_only
```

## Additional Parameters

- `--nas_config`: Path to NAS configuration file (default: nas_config.yaml)
- `--ft_config`: Path to fine-tuning configuration file (default: fine_tuning_config.yaml)
- `--batch`: Batch size for training (default: 8)
- `--low_memory`: Use lower memory settings for NAS
- `--output_dir`: Override output directory

## Example Usage

### For custom object detection dataset:

```bash
python run_pipeline.py --dataset data/custom_dataset.yaml --yolov8n_only
```

### For complete optimization with larger batch size:

```bash
python run_pipeline.py --dataset data/custom_dataset.yaml --batch 16
```

## Output Structure

- `best_models/`: Contains the best models from NAS
- `yolo_fine_tuning/results/`: Contains fine-tuned models and results
- `runs/`: TensorBoard logs (if TensorBoard is available)
- `complete_optimization_summary_*.json`: Unified summary of the entire process

## Monitoring Progress

- Check `yolo_optimization.log` and `yolo_fine_tuning.log` for detailed progress
- View JSON result files for metrics and performance comparisons