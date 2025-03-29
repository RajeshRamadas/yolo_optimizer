"""
YOLO Architecture and Model Optimization with YAML Configuration.

This script allows for:
1. Neural Architecture Search (NAS) via Bayesian optimization
2. Model pruning for size reduction
3. Model quantization (INT8, FP16) for inference speedup

# Run with default threshold (0.85)
python yolo_optimizer.py --mode search --config nas_config.yaml

# Run with custom threshold (0.75)
python yolo_optimizer.py --mode search --threshold 0.75 --config nas_config.yaml

# Run with low memory settings and early stopping
python yolo_optimizer.py --low-memory --threshold 0.80 --config nas_config.yaml

# Just optimize an existing model
python yolo_optimizer.py --mode optimize --model path/to/your/model.pt --config nas_config.yaml
"""

import os
import logging
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse

# Import utility modules
from config_utils import (
    load_yaml_config,
    get_directories,
    get_dataset_path
)
from optimization import (
    run_architecture_search
    # Remove the 'writer' import from here
)
from pruning import prune_model
from quantization import run_model_quantization

# Global variables for configuration
config = None
DATASET_PATH = None
BEST_MODEL_DIR = None
PRUNED_MODEL_DIR = None
QUANTIZED_MODEL_DIR = None
writer = None  # Define writer as a global variable


def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"yolo_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # TensorBoard setup
    global writer
    writer = SummaryWriter(log_dir="runs/yolo_optimization")
    return writer


def load_configuration(config_path):
    global config, BEST_MODEL_DIR, PRUNED_MODEL_DIR, QUANTIZED_MODEL_DIR
    
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Set up directories
    directories = get_directories(config)
    BEST_MODEL_DIR = directories.get('best_model_dir', 'best_model')
    PRUNED_MODEL_DIR = directories.get('pruned_model_dir', 'pruned_models')
    QUANTIZED_MODEL_DIR = directories.get('quantized_model_dir', 'quantized_models')
    
    # Set dataset path only if not already set by command line
    global DATASET_PATH
    if DATASET_PATH is None:
        DATASET_PATH = get_dataset_path(config)
    
    logging.info(f"Configuration loaded from {config_path}")
    logging.info(f"Using dataset: {DATASET_PATH}")
    logging.info(f"Best model directory: {BEST_MODEL_DIR}")
    logging.info(f"Pruned models directory: {PRUNED_MODEL_DIR}")
    logging.info(f"Quantized models directory: {QUANTIZED_MODEL_DIR}")
    
    return config


def run_model_optimization(model_path):
    """
    Run the model optimization process (pruning and quantization)

    Args:
        model_path: Path to the model to optimize

    Returns:
        Dictionary with paths to optimized models
    """
    # Check if model exists
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        return None

    results = {
        "original": model_path
    }

    # Run pruning if not skipped
    if not args.skip_pruning:
        logging.info("Running model pruning...")
        try:
            pruned_model = prune_model(model_path, config, DATASET_PATH, PRUNED_MODEL_DIR)
            results["pruned"] = pruned_model
            logging.info(f"Pruning completed. Best pruned model: {pruned_model}")

            # Use pruned model for quantization if available
            model_for_quant = pruned_model if pruned_model else model_path
        except Exception as e:
            logging.error(f"Error during pruning: {e}")
            import traceback
            logging.error(traceback.format_exc())
            model_for_quant = model_path
    else:
        logging.info("Pruning step skipped as requested.")
        model_for_quant = model_path
        results["pruned"] = model_path

    # Run quantization if not skipped
    if not args.skip_quantization:
        logging.info("Running model quantization...")
        try:
            quantization_results = run_model_quantization(model_for_quant, config, DATASET_PATH, QUANTIZED_MODEL_DIR)
            results.update({
                "int8": quantization_results.get("int8"),
                "fp16": quantization_results.get("fp16")
            })
        except Exception as e:
            logging.error(f"Error during quantization: {e}")
            import traceback
            logging.error(traceback.format_exc())
    else:
        logging.info("Quantization step skipped as requested.")

    # Log optimization results
    logging.info("Model optimization results:")
    for key, value in results.items():
        logging.info(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    # Set PyTorch CUDA memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLO Architecture and Model Optimization")
    parser.add_argument("--mode", type=str, choices=["search", "optimize", "full"], default="full",
                        help="Operation mode: 'search' for architecture search, 'optimize' for model optimization, 'full' for both")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model to optimize (required for 'optimize' mode)")
    parser.add_argument("--config", type=str, default="nas_config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument("--low-memory", action="store_true",
                        help="Start with low memory configuration for initial test")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Performance threshold (mAP50) for early stopping (default: 0.85)")
    parser.add_argument("--skip-pruning", action="store_true",
                        help="Skip pruning step")
    parser.add_argument("--skip-quantization", action="store_true",
                        help="Skip quantization step")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the dataset YAML file (overrides the config file)")

    args = parser.parse_args()

    # Setup logging and TensorBoard
    setup_logging()
    
    # Load configuration
    load_configuration(args.config)

    if args.data:
        DATASET_PATH = args.data
        logging.info(f"Override dataset path from command line: {DATASET_PATH}")

    try:
        # Override threshold if provided
        performance_threshold = args.threshold

        best_model_from_search = None

        if args.mode == "search" or args.mode == "full":
            # Pass the writer to run_architecture_search
            best_model_from_search = run_architecture_search(config, DATASET_PATH, BEST_MODEL_DIR, args,
                                                             performance_threshold, writer)

            if best_model_from_search:
                logging.info(f"Architecture search completed. Best model: {best_model_from_search}")
            else:
                logging.error("Architecture search failed to produce a valid model.")
                if args.mode == "full":
                    logging.error("Cannot proceed with optimization without a valid model.")
                    sys.exit(1)

        # Determine which model to optimize
        model_to_optimize = None
        if args.mode == "full" and best_model_from_search:
            model_to_optimize = best_model_from_search
        elif args.mode == "optimize":
            if not args.model:
                logging.error("--model parameter is required for 'optimize' mode")
                sys.exit(1)
            model_to_optimize = args.model

        # Run optimization if needed
        if model_to_optimize:
            logging.info(f"Starting model optimization process for: {model_to_optimize}")
            optimization_results = run_model_optimization(model_to_optimize)

        logging.info("YOLO optimization process completed successfully")

    except Exception as e:
        logging.error(f"Error during optimization process: {e}")
        import traceback

        logging.error(traceback.format_exc())
        sys.exit(1)