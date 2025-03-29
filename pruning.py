"""
Model pruning and optimization functions for YOLO models.
"""

import os
import logging
from time import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from copy import deepcopy
from ultralytics import YOLO

from config_utils import get_pruning_params
from model_architecture import count_parameters


def apply_structured_pruning(model, amount=0.3):
    """
    Apply structured pruning to a YOLO model to remove entire channels

    Args:
        model: YOLO model to prune
        amount: Fraction of channels to prune (0-1)

    Returns:
        Pruned model
    """
    logging.info(f"Applying structured pruning with amount {amount}")

    # Get all Conv2d layers in the model
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'weight'):
            # Skip depthwise convolutions or bottleneck layers with few channels
            if module.out_channels > 8 and module.groups == 1:
                conv_layers.append((name, module))

    logging.info(f"Found {len(conv_layers)} convolutional layers for structured pruning")

    # Apply channel pruning to selected layers
    pruned_count = 0
    for name, module in conv_layers:
        # Skip some layers to avoid pruning too much
        if pruned_count >= len(conv_layers) * 0.7:  # Only prune up to 70% of eligible layers
            break

        # Apply L1 structured pruning on output channels
        try:
            # Calculate layer-specific pruning amount based on channels
            layer_amount = min(amount, 0.5)  # Don't prune more than 50% of any layer

            prune.ln_structured(module, name='weight', amount=layer_amount, n=1, dim=0)

            # Make pruning permanent
            prune.remove(module, 'weight')
            logging.info(f"Applied structured pruning to layer: {name} (amount={layer_amount:.2f})")
            pruned_count += 1
        except Exception as e:
            logging.warning(f"Couldn't apply pruning to layer {name}: {e}")

    logging.info(f"Successfully pruned {pruned_count} layers with structured pruning")
    return model


def finetune_model(model, data_path, epochs=10, img_size=640):
    """
    Fine-tune a model after pruning

    Args:
        model: YOLO model to fine-tune
        data_path: Path to the dataset
        epochs: Number of epochs for fine-tuning
        img_size: Image size for training

    Returns:
        Fine-tuned model
    """
    logging.info(f"Fine-tuning model for {epochs} epochs")

    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=16,
            lr0=0.001,  # Lower learning rate for fine-tuning
            lrf=0.01,  # Final learning rate fraction
            device="cuda" if torch.cuda.is_available() else "cpu",
            optimizer="AdamW",
            patience=5,
            save=True,
            verbose=True
        )

        logging.info(f"Fine-tuning completed, mAP50: {results.metrics.get('mAP50(B)', 0.0):.4f}")
        return model
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        return model


def evaluate_model(model, data_path, img_size=640):
    """
    Evaluate the model performance

    Args:
        model: YOLO model to evaluate
        data_path: Path to the validation data
        img_size: Image size for validation

    Returns:
        Metrics dict
    """
    logging.info("Evaluating model performance...")
    start_time = time()
    metrics = model.val(data=data_path, imgsz=img_size)
    eval_time = time() - start_time

    logging.info(f"Evaluation completed in {eval_time:.2f} seconds")
    logging.info(f"mAP50: {metrics.box.map50:.4f}")
    logging.info(f"mAP50-95: {metrics.box.map:.4f}")

    return metrics.box


def benchmark_inference(model_path, img_size=640, num_runs=100):
    """
    Benchmark inference speed

    Args:
        model_path: Path to the model file
        img_size: Image size for inference
        num_runs: Number of inference runs for averaging

    Returns:
        Average inference time in ms
    """
    logging.info(f"Benchmarking inference speed for {model_path}")

    try:
        # Load the model
        model = YOLO(model_path)

        # Create a dummy input
        device = next(model.parameters()).device
        dummy_input = torch.rand(1, 3, img_size, img_size).to(device)

        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # Measure inference time
        total_time = 0
        with torch.no_grad():
            for _ in range(num_runs):
                start = time()
                _ = model(dummy_input)
                total_time += (time() - start)

        avg_time = (total_time / num_runs) * 1000  # Convert to ms
        logging.info(f"Average inference time: {avg_time:.2f}ms ({1000 / avg_time:.2f} FPS)")

        return avg_time
    except Exception as e:
        logging.error(f"Error during inference benchmarking: {e}")
        return 0


def benchmark_inference_model(model, img_size=640, num_runs=100):
    """
    Benchmark inference speed for a model object

    Args:
        model: YOLO model object
        img_size: Image size for inference
        num_runs: Number of inference runs for averaging

    Returns:
        Average inference time in ms
    """
    logging.info(f"Benchmarking inference speed for model")

    try:
        # Create a dummy input
        device = next(model.parameters()).device
        dummy_input = torch.rand(1, 3, img_size, img_size).to(device)

        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # Measure inference time
        total_time = 0
        with torch.no_grad():
            for _ in range(num_runs):
                start = time()
                _ = model(dummy_input)
                total_time += (time() - start)

        avg_time = (total_time / num_runs) * 1000  # Convert to ms
        logging.info(f"Average inference time: {avg_time:.2f}ms ({1000 / avg_time:.2f} FPS)")

        return avg_time
    except Exception as e:
        logging.error(f"Error during inference benchmarking: {e}")
        return 0


def prune_model(model_path, config, dataset_path, pruned_model_dir):
    """
    Prune a YOLO model with different structured pruning levels and evaluate performance

    Args:
        model_path: Path to the trained YOLO model
        config: Configuration dictionary
        dataset_path: Path to validation data
        pruned_model_dir: Directory to save pruned models

    Returns:
        Path to the best pruned model
    """
    logging.info(f"Starting enhanced structured pruning process for {model_path}")

    # Get pruning parameters from config
    pruning_config = get_pruning_params(config)
    prune_amounts = pruning_config.get('prune_amounts', [0.1, 0.2, 0.3])
    finetune_enabled = pruning_config.get('finetune', True)
    finetune_epochs = pruning_config.get('finetune_epochs', 10)

    # Load the model
    model = YOLO(model_path)
    original_params = count_parameters(model.model)
    logging.info(f"Original model has {original_params:,} parameters")

    # Evaluate the original model
    original_metrics = evaluate_model(model, dataset_path)
    best_map50 = original_metrics.map50
    best_model_path = model_path
    best_prune_amount = 0

    # Create results dictionary for tracking
    results = {
        "original": {
            "params": original_params,
            "map50": original_metrics.map50,
            "map": original_metrics.map
        }
    }

    # Try different pruning amounts
    for amount in prune_amounts:
        try:
            logging.info(f"Testing structured pruning amount: {amount}")

            # Create a copy of the model
            pruned_model = deepcopy(model)

            # Apply structured pruning
            pruned_model.model = apply_structured_pruning(pruned_model.model, amount)

            # Fine-tune if enabled
            if finetune_enabled:
                logging.info(f"Fine-tuning pruned model")
                pruned_model = finetune_model(pruned_model, dataset_path, epochs=finetune_epochs)

            # Count parameters
            pruned_params = count_parameters(pruned_model.model)
            logging.info(
                f"Pruned model has {pruned_params:,} parameters ({pruned_params / original_params:.2%} of original)")

            # Evaluate
            pruned_metrics = evaluate_model(pruned_model, dataset_path)

            # Calculate inference speedup
            original_time = benchmark_inference(model_path, num_runs=20)
            pruned_time = benchmark_inference_model(pruned_model, num_runs=20)
            speedup = original_time / pruned_time if pruned_time > 0 else 0

            # Save results
            results[f"pruned_{amount}"] = {
                "params": pruned_params,
                "map50": pruned_metrics.map50,
                "map": pruned_metrics.map,
                "param_reduction": 1 - (pruned_params / original_params),
                "speedup": speedup
            }

            # Check if this is the best model (combining accuracy and speedup)
            # We want models that are accurate AND fast
            quality_score = pruned_metrics.map50 * speedup

            if pruned_metrics.map50 > best_map50 * 0.95 and speedup > 0.8:  # Allow up to 5% accuracy drop
                # Save the model
                pruned_model_path = os.path.join(pruned_model_dir, f"pruned_{amount}_{pruned_metrics.map50:.4f}.pt")
                pruned_model.export(format="torchscript", path=pruned_model_path)
                logging.info(f"Saved pruned model to {pruned_model_path}")

                if quality_score > (best_map50 * 1.0):  # Adjust this threshold as needed
                    best_map50 = pruned_metrics.map50
                    best_model_path = pruned_model_path
                    best_prune_amount = amount
                    logging.info(
                        f"New best pruned model: {amount} pruning with mAP50 {pruned_metrics.map50:.4f} and speedup {speedup:.2f}x")

        except Exception as e:
            logging.error(f"Error pruning with amount {amount}: {e}")

    # Print summary
    logging.info("\nStructured Pruning Summary:")
    for key, value in results.items():
        if key == "original":
            logging.info(f"Original: mAP50={value['map50']:.4f}, Parameters={value['params']:,}")
        else:
            param_reduction = value.get("param_reduction", 0)
            speedup = value.get("speedup", 0)
            logging.info(f"{key}: mAP50={value['map50']:.4f}, Parameters={value['params']:,} "
                         f"({param_reduction:.2%} reduction), Speedup: {speedup:.2f}x")

    logging.info(f"Best pruned model: {best_prune_amount} pruning with mAP50 {best_map50:.4f}")
    logging.info(f"Best pruned model saved at: {best_model_path}")

    return best_model_path