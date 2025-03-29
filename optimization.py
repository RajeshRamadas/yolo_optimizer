"""
Bayesian optimization functions for YOLO model architecture search.
"""

import os
import json
import logging
from datetime import datetime
from time import time
import torch
import numpy as np
from ultralytics import YOLO
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from config_utils import get_training_params, save_nas_results, get_initial_test_params, get_optimization_params, \
    get_search_bounds
from model_architecture import modify_yolo_architecture, log_model_architecture
from results_tracker import NASResultsTracker, track_nas_result

# Global variables for tracking
best_map50 = 0.0
best_model_path = None
writer = None

# Create a global tracker for NAS results
tracker = NASResultsTracker(output_dir="nas_results")

def yolo_train(resolution, depth_mult, width_mult, kernel_size, num_channels,
               lr0=0.01, momentum=0.937, batch_size=4, iou_thresh=0.7, weight_decay=0.0005,
               config=None, dataset_path=None, best_model_dir=None):
    """
    Train a YOLO model with the specified architecture parameters

    Args:
        resolution: Input image size
        depth_mult: Depth multiplier for scaling layers
        width_mult: Width multiplier for scaling channels
        kernel_size: Kernel size for specific layers
        num_channels: Base number of channels
        lr0: Initial learning rate
        momentum: SGD momentum
        batch_size: Training batch size
        iou_thresh: IoU threshold for NMS
        weight_decay: Weight decay regularization
        config: Full configuration dictionary
        dataset_path: Path to the dataset
        best_model_dir: Directory to save the best model

    Returns:
        mAP50 score of the trained model
    """
    global best_map50, best_model_path, writer, tracker
    start_time = time()
    iteration_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log the actual received parameters before any processing
    logging.info(f"Raw input parameters: resolution={resolution}, depth_mult={depth_mult}, "
                 f"width_mult={width_mult}, kernel_size={kernel_size}, num_channels={num_channels}, "
                 f"lr0={lr0}, momentum={momentum}, batch_size={batch_size}, "
                 f"iou_thresh={iou_thresh}, weight_decay={weight_decay}")

    try:
        # Convert and validate parameters
        depth_mult, width_mult = map(float, [depth_mult, width_mult])

        # Ensure resolution is divisible by 32 (YOLO requirement)
        resolution = int(round(resolution / 32) * 32)
        resolution = max(128, min(1280, resolution))  # Keep within reasonable bounds

        # Ensure kernel size is odd and within bounds
        kernel_size = max(3, min(7, int(round(kernel_size))))
        if kernel_size % 2 == 0:  # Ensure kernel size is odd
            kernel_size += 1

        # Round num_channels to multiples of 16 for better hardware optimization
        num_channels = max(16, min(256, int(round(num_channels / 16) * 16)))

        # Round batch_size to nearest integer (ensure it's at least 1)
        batch_size = max(1, int(round(batch_size)))

        # Ensure other hyperparameters are within reasonable bounds
        lr0 = max(1e-6, min(1e-1, float(lr0)))
        momentum = max(0.5, min(0.99, float(momentum)))
        iou_thresh = max(0.1, min(0.95, float(iou_thresh)))
        weight_decay = max(1e-6, min(0.1, float(weight_decay)))

        # Get training parameters from config
        train_config = get_training_params(config)
        optimizer = train_config.get('optimizer', 'AdamW')
        epochs = train_config.get('epochs', 50)
        patience = train_config.get('patience', 5)
        save = train_config.get('save', True)
        verbose = train_config.get('verbose', True)

        logging.info(f"Trial {iteration_id} parameters: "
                     f"resolution={resolution}, depth_mult={depth_mult:.2f}, "
                     f"width_mult={width_mult:.2f}, kernel_size={kernel_size}, "
                     f"num_channels={num_channels}, batch_size={batch_size}, "
                     f"lr0={lr0:.6f}, momentum={momentum:.3f}, "
                     f"iou_thresh={iou_thresh:.2f}, weight_decay={weight_decay:.6f}")

        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            logging.error(f"Dataset file not found at: {dataset_path}")
            return 0.0

        custom_model_path = modify_yolo_architecture(depth_mult, width_mult, kernel_size, num_channels, best_model_dir)
        log_model_architecture(custom_model_path)

        model = YOLO(custom_model_path, task='detect')

        # Enable gradient checkpointing if available
        try:
            if hasattr(model.model, 'model'):
                for m in model.model.model.modules():
                    if hasattr(m, 'checkpoint') and callable(m.checkpoint):
                        m.checkpoint = True
                        logging.info(f"Enabled gradient checkpointing on {m.__class__.__name__}")
        except Exception as e:
            logging.warning(f"Could not enable gradient checkpointing: {e}")

        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=resolution,
            lr0=lr0,
            momentum=momentum,
            batch=batch_size,
            optimizer=optimizer,
            iou=iou_thresh,
            weight_decay=weight_decay,
            device="cuda" if torch.cuda.is_available() else "cpu",
            patience=patience,
            save=save,
            project="bayesian_opt",
            name=f"trial_{iteration_id}",
            verbose=verbose
        )

        # Log training time
        training_time = time() - start_time
        logging.info(f"Trial {iteration_id} completed in {training_time:.2f} seconds")

        # Access mAP50 correctly based on the current YOLO version
        # Try different ways to access mAP50 based on YOLO version
        map50 = 0.0
        try:
            # First try the box.map50 attribute (newer versions)
            map50 = results.box.map50
            logging.info(f"Retrieved mAP50 from results.box.map50: {map50:.4f}")
        except Exception:
            try:
                # Try accessing metrics dictionary (older versions)
                map50 = results.metrics.get("mAP50(B)", 0.0)
                logging.info(f"Retrieved mAP50 from results.metrics dictionary: {map50:.4f}")
            except Exception:
                # If all else fails, try to parse from validation directory
                try:
                    val_dir = os.path.join("bayesian_opt", f"trial_{iteration_id}")
                    results_file = os.path.join(val_dir, "results.csv")
                    if os.path.exists(results_file):
                        with open(results_file, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # Skip header
                                parts = lines[-1].strip().split(",")
                                if len(parts) >= 4:
                                    map50 = float(parts[3])  # Usually in the 4th column
                                    logging.info(f"Retrieved mAP50 from results.csv: {map50:.4f}")
                except Exception as e:
                    logging.error(f"Failed to parse mAP50 from results file: {e}")
                    map50 = 0.0

        logging.info(f"Trial {iteration_id} mAP50: {map50:.4f}")

        # Write to TensorBoard
        if writer:
            writer.add_scalar('mAP50', map50, int(iteration_id.replace('_', '')))
            writer.add_scalar('Training Time', training_time, int(iteration_id.replace('_', '')))

        # Save the best model
        if map50 > best_map50:
            best_map50 = map50
            best_model_path = os.path.join(best_model_dir, f"best_model_{map50:.4f}.pt")
            try:
                # Copy the best weights file directly
                val_dir = os.path.join("bayesian_opt", f"trial_{iteration_id}")
                weights_path = os.path.join(val_dir, "weights", "best.pt")
                if os.path.exists(weights_path):
                    import shutil
                    shutil.copy(weights_path, best_model_path)
                    logging.info(f"New best model with mAP50={map50:.4f} saved to {best_model_path}")
                else:
                    logging.warning(f"Could not find weights file at {weights_path}")
            except Exception as e:
                logging.error(f"Failed to save model: {e}")

        # Track the result in the NAS tracker
        params = {
            "resolution": resolution,
            "depth_mult": depth_mult,
            "width_mult": width_mult,
            "kernel_size": kernel_size,
            "num_channels": num_channels,
            "lr0": lr0,
            "momentum": momentum,
            "batch_size": batch_size,
            "iou_thresh": iou_thresh,
            "weight_decay": weight_decay
        }
        track_nas_result(tracker, params, map50, training_time, iteration_id)

        return map50
    except Exception as e:
        logging.error(f"Trial {iteration_id} failed: {e}")
        return 0.0


# Create a proper callback class for early stopping
class EarlyStopCallback:
    def __init__(self, threshold, early_stop_dict):
        self.threshold = threshold
        self.early_stop_dict = early_stop_dict
        
    def update(self, event, instance):
        if instance.max["target"] >= self.threshold:
            logging.info(f"Early stopping threshold reached: {instance.max['target']:.4f} >= {self.threshold}")
            self.early_stop_dict["should_stop"] = True


def run_optimization(config, dataset_path, best_model_dir, init_points=5, n_iter=25, performance_threshold=0.85):
    """
    Run Bayesian optimization to find the best architecture parameters

    Args:
        config: Configuration dictionary
        dataset_path: Path to the dataset
        best_model_dir: Directory to save the best model
        init_points: Number of initial random points to sample
        n_iter: Number of iterations for optimization
        performance_threshold: Early stopping threshold for mAP50 (0-1)

    Returns:
        Optimizer with results
    """
    # Get search bounds from configuration
    pbounds = get_search_bounds(config)

    # Print the bounds we're using for clarity
    logging.info(f"Optimization bounds: {pbounds}")
    logging.info(f"Early stopping threshold: mAP50 >= {performance_threshold}")

    # Create a wrapper for the yolo_train function that includes the config and paths
    def yolo_train_wrapper(**kwargs):
        return yolo_train(**kwargs, config=config, dataset_path=dataset_path, best_model_dir=best_model_dir)

    # Initialize optimizer
    optimizer = BayesianOptimization(f=yolo_train_wrapper, pbounds=pbounds, random_state=42, verbose=2)
    logs_path = os.path.join(best_model_dir, "bayesian_opt_logs.json")
    logger = JSONLogger(path=logs_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Create a flag for early stopping
    early_stop = {"should_stop": False}

    # Create and subscribe the early stopping callback object
    early_stop_cb = EarlyStopCallback(performance_threshold, early_stop)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, early_stop_cb)

    try:
        # Load previous optimization results if available
        if os.path.exists(logs_path):
            logging.info("Found previous optimization logs, loading previous results...")
            with open(logs_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            prev_result = json.loads(line)
                            optimizer.register(
                                params=prev_result.get("params", {}),
                                target=prev_result.get("target", 0.0)
                            )
                            logging.info(f"Loaded previous result with target {prev_result.get('target', 0.0)}")

                            # Check if any previous result already meets our threshold
                            if prev_result.get("target", 0.0) >= performance_threshold:
                                early_stop["should_stop"] = True
                                logging.info(
                                    f"Previous result already meets performance threshold: {prev_result.get('target', 0.0):.4f}")
                        except Exception as e:
                            logging.error(f"Error loading line from JSON log: {e}")
    except Exception as e:
        logging.error(f"Error loading previous optimization results: {e}")

    # Start optimization with early stopping check
    if not early_stop["should_stop"]:
        try:
            # Initial random exploration phase
            for i in range(init_points):
                if early_stop["should_stop"]:
                    logging.info("Early stopping during initial points")
                    break
                
                # Try different ways to call suggest based on the library version
                try:
                    # Try to use suggest() with no arguments
                    next_point = optimizer.suggest()
                except TypeError:
                    # Fallback to completely random points if suggest() doesn't work
                    next_point = {}
                    for param, bounds in pbounds.items():
                        next_point[param] = np.random.uniform(bounds[0], bounds[1])
                        
                target = yolo_train_wrapper(**next_point)
                optimizer.register(params=next_point, target=target)

            # Bayesian optimization phase
            for i in range(n_iter):
                if early_stop["should_stop"]:
                    logging.info("Early stopping during iteration")
                    break
                
                # Try different ways to call suggest based on the library version
                try:
                    # Try to use suggest() with no arguments
                    next_point = optimizer.suggest()
                except TypeError:
                    # Fallback to completely random points if suggest() doesn't work
                    next_point = {}
                    for param, bounds in pbounds.items():
                        next_point[param] = np.random.uniform(bounds[0], bounds[1])
                
                target = yolo_train_wrapper(**next_point)
                optimizer.register(params=next_point, target=target)
                
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Return None or partial results instead of failing completely
            if hasattr(optimizer, 'max') and optimizer.max:
                return optimizer
            # Create a default result with the best we found so far
            if best_map50 > 0:
                return {"max": {"params": {}, "target": best_map50}}
            return None
    else:
        logging.info("Skipping optimization as performance threshold already met")

    return optimizer


def run_architecture_search(config, dataset_path, best_model_dir, args, performance_threshold=0.85):
    """
    Run the Bayesian architecture search process with early stopping

    Args:
        config: Configuration dictionary
        dataset_path: Path to the dataset
        best_model_dir: Directory to save the best model
        args: Command line arguments
        performance_threshold: Early stopping threshold for mAP50 (0-1)

    Returns:
        Path to the best model
    """
    global best_map50, best_model_path, tracker

    # Get initial test parameters from config
    initial_test_params = get_initial_test_params(config)

    # Run initial test if low memory flag is set
    if args.low_memory and initial_test_params:
        logging.info(f"Running initial test with low memory parameters: {initial_test_params}")
        test_result = yolo_train(**initial_test_params, config=config, dataset_path=dataset_path,
                                 best_model_dir=best_model_dir)

        # If initial test already meets threshold, save these parameters and skip optimization
        if test_result >= performance_threshold:
            logging.info(
                f"Initial test result {test_result:.4f} already meets performance threshold {performance_threshold}. Skipping optimization.")

            # Save the architecture parameters from initial test
            save_nas_results(initial_test_params, test_result, None, config)

            return best_model_path

    # Get optimization parameters from config
    opt_params = get_optimization_params(config)
    init_points = opt_params.get('init_points', 3)
    n_iter = opt_params.get('n_iter', 15)

    # Run Bayesian optimization with early stopping
    optimizer = run_optimization(config, dataset_path, best_model_dir, init_points, n_iter, performance_threshold)
    
    # Handle case where optimization might have failed
    if optimizer is None:
        logging.error("Optimization failed to produce valid results")
        return best_model_path
        
    # Try to get best parameters, with fallback for incomplete optimization
    try:
        if hasattr(optimizer, 'max'):
            best_params = optimizer.max["params"]
            best_score = optimizer.max["target"]
        else:
            best_params = optimizer["max"]["params"]
            best_score = optimizer["max"]["target"]
    except (TypeError, KeyError, AttributeError):
        # If optimizer doesn't have a max property or it's incomplete
        logging.warning("Couldn't obtain proper optimization results, using best found so far")
        if best_map50 > 0:
            best_params = initial_test_params or {}  # Use initial test params as fallback
            best_score = best_map50
        else:
            # No successful model at all - return None
            logging.error("No valid model was found during optimization")
            return None

    # Save the results
    try:
        save_nas_results(best_params, best_score, optimizer, config)
    except Exception as e:
        logging.error(f"Error saving NAS results: {e}")

    # Generate performance report and visualizations
    try:
        report = tracker.generate_report()
        
        # Print top models table
        logging.info("\nTop Performing Models:")
        logging.info(tracker.get_results_table(top_n=5))
        
        # Log report locations
        logging.info(f"NAS results CSV saved to: {report['results_csv']}")
        if report.get('progress_plot'):
            logging.info(f"NAS progress plot saved to: {report['progress_plot']}")
        if report.get('parameter_effects_plot'):
            logging.info(f"NAS parameter effects plot saved to: {report['parameter_effects_plot']}")
        logging.info(f"NAS detailed report saved to: {report['report_text']}")
    except Exception as e:
        logging.error(f"Error generating performance report: {e}")

    logging.info(f"Best model saved to: {best_model_path}")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best score: {best_score}")

    return best_model_path