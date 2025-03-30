"""
Neural Architecture Search for YOLO models using Bayesian optimization.
"""

import os
import logging
import torch
import numpy as np
import yaml
import time
from functools import partial
from ultralytics import YOLO
import random
import shutil
from datetime import datetime

# Import model_architecture functions
from model_architecture import (
    create_custom_yolo_config,
    create_complex_yolo_config,
    save_model_config as save_arch_config,
    log_model_architecture
)

# Try to import optional dependencies
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Using random search instead.")

# Constants
MAX_TRIALS = 50
EARLY_STOP_PATIENCE = 5


def log_to_tensorboard(writer, tag, value, step, default=0):
    """
    Safely log metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        tag: The name of the metric
        value: The value to log (can be None)
        step: The step/iteration number
        default: Default value to use if value is None
    """
    if writer is None:
        return
        
    try:
        if value is not None:
            writer.add_scalar(tag, value, step)
        else:
            writer.add_scalar(tag, default, step)
    except Exception as e:
        logging.warning(f"Could not log {tag} to TensorBoard: {e}")


def get_range(range_list):
    """Convert a range list [min, max] to a tuple (min, max)"""
    if isinstance(range_list, list) and len(range_list) == 2:
        return (range_list[0], range_list[1])
    return range_list  # Return as is if not a range list


def generate_search_space_from_config(config):
    """
    Define the search space for neural architecture search from the config file.
    
    Args:
        config: Configuration dictionary loaded from nas_config.yaml
        
    Returns:
        Dictionary or list with parameter names and their ranges.
    """
    search_config = config.get("search_space", {})
    
    if not search_config:
        logging.warning("Search space not found in config, using default search space")
        return generate_default_search_space()
    
    # Get activation and skip connections mappings
    activation_mapping = config.get("activation_mapping", {
        0: "SiLU", 1: "ReLU", 2: "Mish", 3: "LeakyReLU"
    })
    
    skip_connections_mapping = config.get("skip_connections_mapping", {
        0: "standard", 1: "dense", 2: "residual"
    })
    
    if SKOPT_AVAILABLE:
        # Using scikit-optimize space for Bayesian optimization
        search_space = []
        
        # Add architecture parameters
        if "resolution" in search_config:
            search_space.append(Integer(*get_range(search_config["resolution"]), name='resolution'))
        
        if "depth_mult" in search_config:
            search_space.append(Real(*get_range(search_config["depth_mult"]), name='depth_mult'))
        
        if "width_mult" in search_config:
            search_space.append(Real(*get_range(search_config["width_mult"]), name='width_mult'))
        
        if "kernel_size" in search_config:
            search_space.append(Integer(*get_range(search_config["kernel_size"]), name='kernel_size'))
        
        if "num_channels" in search_config:
            search_space.append(Integer(*get_range(search_config["num_channels"]), name='num_channels'))
        
        # Add training hyperparameters if needed
        
        # Add advanced architecture options
        if "use_complex" in search_config:
            search_space.append(Integer(*get_range(search_config["use_complex"]), name='use_complex'))
        
        if "activation" in search_config:
            activation_range = get_range(search_config["activation"])
            activation_choices = list(range(activation_range[0], activation_range[1]+1))
            search_space.append(Categorical(activation_choices, name='activation'))
        
        if "use_cbam" in search_config:
            search_space.append(Integer(*get_range(search_config["use_cbam"]), name='use_cbam'))
        
        if "dropout_rate" in search_config:
            search_space.append(Real(*get_range(search_config["dropout_rate"]), name='dropout_rate'))
        
        if "use_gating" in search_config:
            search_space.append(Integer(*get_range(search_config["use_gating"]), name='use_gating'))
        
        if "bottleneck_ratio" in search_config:
            search_space.append(Real(*get_range(search_config["bottleneck_ratio"]), name='bottleneck_ratio'))
        
        if "num_heads" in search_config:
            num_heads_range = get_range(search_config["num_heads"])
            if isinstance(num_heads_range, tuple) and len(num_heads_range) == 2:
                search_space.append(Integer(*num_heads_range, name='num_heads'))
            else:
                # If it's a list of choices
                search_space.append(Categorical(num_heads_range, name='num_heads'))
        
        if "skip_connections" in search_config:
            skip_range = get_range(search_config["skip_connections"])
            skip_choices = list(range(skip_range[0], skip_range[1]+1))
            search_space.append(Categorical(skip_choices, name='skip_connections'))
        
        if "use_eca" in search_config:
            search_space.append(Integer(*get_range(search_config["use_eca"]), name='use_eca'))
    else:
        # Simple ranges for random search
        search_space = {}
        
        # Add architecture parameters
        if "resolution" in search_config:
            search_space['resolution'] = get_range(search_config["resolution"])
        
        if "depth_mult" in search_config:
            search_space['depth_mult'] = get_range(search_config["depth_mult"])
        
        if "width_mult" in search_config:
            search_space['width_mult'] = get_range(search_config["width_mult"])
        
        if "kernel_size" in search_config:
            search_space['kernel_size'] = get_range(search_config["kernel_size"])
        
        if "num_channels" in search_config:
            search_space['num_channels'] = get_range(search_config["num_channels"])
        
        # Add advanced architecture options
        if "use_complex" in search_config:
            search_space['use_complex'] = get_range(search_config["use_complex"])
        
        if "activation" in search_config:
            activation_range = get_range(search_config["activation"])
            if isinstance(activation_range, tuple) and len(activation_range) == 2:
                search_space['activation'] = list(range(activation_range[0], activation_range[1]+1))
            else:
                search_space['activation'] = activation_range
        
        if "use_cbam" in search_config:
            search_space['use_cbam'] = get_range(search_config["use_cbam"])
        
        if "dropout_rate" in search_config:
            search_space['dropout_rate'] = get_range(search_config["dropout_rate"])
        
        if "use_gating" in search_config:
            search_space['use_gating'] = get_range(search_config["use_gating"])
        
        if "bottleneck_ratio" in search_config:
            search_space['bottleneck_ratio'] = get_range(search_config["bottleneck_ratio"])
        
        if "num_heads" in search_config:
            search_space['num_heads'] = get_range(search_config["num_heads"])
        
        if "skip_connections" in search_config:
            skip_range = get_range(search_config["skip_connections"])
            if isinstance(skip_range, tuple) and len(skip_range) == 2:
                search_space['skip_connections'] = list(range(skip_range[0], skip_range[1]+1))
            else:
                search_space['skip_connections'] = skip_range
        
        if "use_eca" in search_config:
            search_space['use_eca'] = get_range(search_config["use_eca"])
    
    logging.info(f"Generated search space from config: {search_space}")
    return search_space


def generate_default_search_space():
    """
    Generate a default search space when config doesn't have one.
    
    Returns:
        Default search space
    """
    if SKOPT_AVAILABLE:
        # Using scikit-optimize space
        search_space = [
            Integer(2, 5, name='num_blocks'),
            Integer(2, 8, name='num_repeats'),
            Integer(32, 128, name='width_multiple'),
            Integer(320, 640, name='input_size'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['silu', 'relu', 'leaky_relu'], name='activation'),
            Integer(16, 64, name='bottleneck_width')
        ]
    else:
        # Simple ranges for random search
        search_space = {
            'num_blocks': (2, 5),
            'num_repeats': (2, 8),
            'width_multiple': (32, 128),
            'input_size': (320, 640),
            'dropout_rate': (0.1, 0.5),
            'activation': ['silu', 'relu', 'leaky_relu'],
            'bottleneck_width': (16, 64)
        }
    
    return search_space


def sample_parameters_from_config(search_space, optimizer=None, iteration=0, config=None):
    """
    Sample parameters from the search space defined in the config.
    
    Args:
        search_space: Search space definition from config
        optimizer: Bayesian optimizer (if available)
        iteration: Current iteration number
        config: Full configuration dictionary
        
    Returns:
        Dictionary of sampled parameters
    """
    params = {}
    
    # Get activation and skip connections mappings
    activation_mapping = config.get("activation_mapping", {
        0: "SiLU", 1: "ReLU", 2: "Mish", 3: "LeakyReLU"
    })
    
    skip_connections_mapping = config.get("skip_connections_mapping", {
        0: "standard", 1: "dense", 2: "residual"
    })
    
    if SKOPT_AVAILABLE and optimizer:
        # Get suggestion from Bayesian optimizer
        suggestion = optimizer.ask()
        
        # Map suggestion to parameter dictionary
        for param_index, param_name in enumerate(param.name for param in search_space):
            params[param_name] = suggestion[param_index]
    else:
        # Random sampling for each parameter
        for param_name, param_range in search_space.items():
            if param_name in ['use_complex', 'use_cbam', 'use_gating', 'use_eca']:
                # Binary parameters
                params[param_name] = random.randint(*param_range) if isinstance(param_range, tuple) else random.choice(param_range)
            elif param_name in ['activation', 'skip_connections', 'num_heads']:
                # Categorical parameters
                params[param_name] = random.choice(param_range)
            elif param_name in ['dropout_rate', 'bottleneck_ratio', 'depth_mult', 'width_mult']:
                # Float parameters
                params[param_name] = random.uniform(*param_range)
            else:
                # Integer parameters
                params[param_name] = random.randint(*param_range)
    
    # Ensure resolution is divisible by 32 if present
    if 'resolution' in params:
        params['resolution'] = (params['resolution'] // 32) * 32
    
    # Convert boolean flags from integers to booleans
    for key in ['use_complex', 'use_cbam', 'use_gating', 'use_eca']:
        if key in params:
            params[key] = bool(params[key])
    
    # Map activation and skip_connections indices to their string values if needed
    if 'activation' in params and isinstance(params['activation'], int):
        params['activation'] = activation_mapping.get(params['activation'], "SiLU")
    
    if 'skip_connections' in params and isinstance(params['skip_connections'], int):
        params['skip_connections'] = skip_connections_mapping.get(params['skip_connections'], "standard")
    
    return params


def create_model_configuration_from_params(params, config):
    """
    Create a model configuration based on the sampled parameters using model_architecture functions.
    
    Args:
        params: Dictionary of sampled parameters
        config: Full configuration dictionary
        
    Returns:
        Model configuration dictionary
    """
    # Map parameters to model architecture requirements
    depth_mult = params.get('depth_mult', 1.0)
    width_mult = params.get('width_mult', 0.5)
    kernel_size = params.get('kernel_size', 5)
    num_channels = params.get('num_channels', 64)
    
    use_complex = params.get('use_complex', False)
    activation = params.get('activation', "SiLU")
    use_cbam = params.get('use_cbam', False)
    dropout_rate = params.get('dropout_rate', 0.0)
    use_gating = params.get('use_gating', False)
    bottleneck_ratio = params.get('bottleneck_ratio', 0.5)
    num_heads = params.get('num_heads', 4)
    skip_connections = params.get('skip_connections', "standard")
    use_eca = params.get('use_eca', False)
    
    # Create model configuration using appropriate function
    if use_complex:
        model_config = create_complex_yolo_config(
            depth_mult=depth_mult,
            width_mult=width_mult,
            kernel_size=kernel_size,
            num_channels=num_channels,
            activation=activation,
            use_cbam=use_cbam,
            dropout_rate=dropout_rate,
            use_gating=use_gating,
            bottleneck_ratio=bottleneck_ratio,
            num_heads=num_heads,
            skip_connections=skip_connections,
            use_eca=use_eca
        )
    else:
        model_config = create_custom_yolo_config(
            depth_mult=depth_mult,
            width_mult=width_mult,
            kernel_size=kernel_size,
            num_channels=num_channels
        )
    
    # Add input image size if specified
    if 'resolution' in params:
        model_config['input_size'] = params['resolution']
    
    return model_config


def save_model_config(config_yaml, trial_dir, trial_num):
    """
    Save model configuration to a YAML file.
    
    Args:
        config_yaml: Model configuration dictionary
        trial_dir: Directory to save configuration
        trial_num: Trial number
        
    Returns:
        Path to the saved configuration file
    """
    os.makedirs(trial_dir, exist_ok=True)
    config_path = os.path.join(trial_dir, f"model_trial_{trial_num}.yaml")
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        logging.info(f"Saved model configuration to {config_path}")
        return config_path
    except Exception as e:
        logging.error(f"Error saving model configuration: {e}")
        return None

# Replace the train_trial_model function in optimization.py with this updated version

def train_trial_model(config_path, dataset_path, trial_dir, trial_num, args, global_config):
    """
    Train a model with the given configuration.
    
    Args:
        config_path: Path to model configuration file
        dataset_path: Path to dataset
        trial_dir: Directory to save trial results
        trial_num: Trial number
        args: Command-line arguments
        global_config: Global configuration dictionary from nas_config.yaml
        
    Returns:
        Path to the trained model
    """
    logging.info(f"Training trial model {trial_num}")
    
    # Log the model architecture
    log_model_architecture(config_path)
    
    # Determine image size from configuration or use default
    try:
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        img_size = model_config.get('input_size', 640)
    except:
        img_size = 640
    
    # Set a lower batch size for initial testing if low memory flag is set
    batch_size = 4 if args.low_memory else args.batch

    # Get training parameters from the global config file
    training_config = global_config.get("training", {})
    epochs = training_config.get("epochs", 50)  # Default to 50 if not specified
    patience = training_config.get("patience", 5)  # Default to 5 if not specified
    optimizer = training_config.get("optimizer", "AdamW")  # Default to AdamW if not specified
    save = training_config.get("save", True)  # Default to True if not specified
    verbose = training_config.get("verbose", True)  # Default to True if not specified

    logging.info(f"Training with parameters from config - epochs: {epochs}, optimizer: {optimizer}, patience: {patience}")
    
    # Set up project and name for saving
    project_dir = os.path.join(trial_dir, f'trial_{trial_num}')
    run_name = f'model_{trial_num}'
    
    # Create the expected paths where the model will be saved
    run_dir = os.path.join(project_dir, run_name)
    weights_dir = os.path.join(run_dir, 'weights')
    best_path = os.path.join(weights_dir, 'best.pt')
    
    # Initialize YOLOv8 with custom configuration
    try:
        model = YOLO(config_path)
    except Exception as e:
        logging.error(f"Error initializing YOLO model with config {config_path}: {e}")
        # Fall back to default model
        logging.info("Falling back to default YOLOv8n model")
        model = YOLO('yolov8n.yaml')
    
    # Set up training parameters
    train_args = {
        'data': dataset_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'optimizer': optimizer,
        'patience': patience,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'project': project_dir,
        'name': run_name,
        'exist_ok': True,
        'verbose': verbose,
        'save': save
    }
    
    # Apply data augmentation if configured
    try:
        from data_augmentation import get_augmentation_params, apply_augmentation_config_to_training
        
        # Get augmentation parameters from config
        aug_params = get_augmentation_params(global_config)
        
        if aug_params:
            logging.info(f"Applying data augmentation: {aug_params}")
            train_args = apply_augmentation_config_to_training(train_args, aug_params, os.path.join(trial_dir, f'trial_{trial_num}'))
        else:
            logging.info("No augmentation configuration found. Using default training settings.")
    except ImportError:
        logging.warning("Data augmentation module not available. Continuing without augmentation.")
    except Exception as e:
        logging.error(f"Error applying augmentation: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Train model
    try:
        results = model.train(**train_args)
        
        # Check if the best model file exists based on the expected path
        if os.path.exists(best_path):
            logging.info(f"Trial {trial_num} completed successfully. Best model found at: {best_path}")
            
            # Save the model architecture alongside the model weights
            arch_dir = os.path.join(weights_dir, "architecture")
            os.makedirs(arch_dir, exist_ok=True)
            arch_path = os.path.join(arch_dir, f"model_trial_{trial_num}_arch.yaml")
            
            # Copy the original config
            try:
                shutil.copy(config_path, arch_path)
                logging.info(f"Saved model architecture to {arch_path}")
            except Exception as e:
                logging.error(f"Error saving model architecture: {e}")
            
            return best_path, results
        else:
            # If best.pt doesn't exist, try to find any .pt files in the weights directory
            if os.path.exists(weights_dir):
                pt_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                if pt_files:
                    best_path = os.path.join(weights_dir, pt_files[0])
                    logging.info(f"No 'best.pt' found, using {pt_files[0]} instead")
                    return best_path, results
            
            logging.warning(f"Trial {trial_num} did not produce a valid model.")
            return None, None
    except Exception as e:
        logging.error(f"Error training trial model {trial_num}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None
        
def update_bayesian_optimizer(optimizer, suggestion, map50):
    """
    Update Bayesian optimizer with results.
    
    Args:
        optimizer: Bayesian optimizer instance
        suggestion: Parameter suggestion that was evaluated
        map50: mAP@50 result
        
    Returns:
        Updated optimizer
    """
    # Bayesian optimization tries to minimize, so we negate mAP
    optimizer.tell(suggestion, -map50)
    return optimizer


def save_best_model(model_path, best_model_dir, map50, trial_num):
    """
    Save the best model found during architecture search.
    
    Args:
        model_path: Path to the model to save
        best_model_dir: Directory to save the best model
        map50: mAP@50 of the model
        trial_num: Trial number
        
    Returns:
        Path to the saved best model
    """
    os.makedirs(best_model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_name = f"best_model_trial{trial_num}_map{map50:.4f}_{timestamp}.pt"
    best_model_path = os.path.join(best_model_dir, best_model_name)
    
    # Copy the model file
    shutil.copy(model_path, best_model_path)
    logging.info(f"Saved best model to: {best_model_path}")
    
    # Also save the model architecture
    try:
        # Look for the architecture file in the same directory as model_path
        arch_dir = os.path.join(os.path.dirname(model_path), "architecture")
        if os.path.exists(arch_dir):
            arch_files = [f for f in os.listdir(arch_dir) if f.startswith(f"model_trial_{trial_num}")]
            if arch_files:
                arch_path = os.path.join(arch_dir, arch_files[0])
                best_arch_path = os.path.join(best_model_dir, f"best_model_trial{trial_num}_arch_{timestamp}.yaml")
                shutil.copy(arch_path, best_arch_path)
                logging.info(f"Saved best model architecture to: {best_arch_path}")
        else:
            logging.warning(f"Architecture directory not found for model {model_path}")
    except Exception as e:
        logging.error(f"Error saving model architecture: {e}")
    
    return best_model_path

def evaluate_trial(model_path, dataset_path):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to validation dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not model_path or not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return {"map50": 0.0}
    
    try:
        model = YOLO(model_path)
        
        # Run validation
        val_results = model.val(data=dataset_path)
        
        # Extract metrics (handling different versions of YOLO)
        try:
            # Newer YOLO versions
            # Handle case where metrics might be arrays (one value per class)
            if hasattr(val_results.box, 'map50'):
                map50 = val_results.box.map50
                # If map50 is an array with multiple values, use the mean
                if hasattr(map50, 'shape') and len(map50.shape) > 0 and map50.shape[0] > 1:
                    map50 = float(map50.mean())
                else:
                    map50 = float(map50)
            else:
                map50 = 0.0
                
            if hasattr(val_results.box, 'p'):
                precision = val_results.box.p
                # If precision is an array with multiple values, use the mean
                if hasattr(precision, 'shape') and len(precision.shape) > 0 and precision.shape[0] > 1:
                    precision = float(precision.mean())
                else:
                    precision = float(precision)
            else:
                precision = 0.0
                
            if hasattr(val_results.box, 'r'):
                recall = val_results.box.r
                # If recall is an array with multiple values, use the mean
                if hasattr(recall, 'shape') and len(recall.shape) > 0 and recall.shape[0] > 1:
                    recall = float(recall.mean())
                else:
                    recall = float(recall)
            else:
                recall = 0.0
                
            metrics = {
                "map50": map50,
                "precision": precision,
                "recall": recall
            }
        except Exception as e:
            logging.warning(f"Error extracting specific metrics: {e}. Trying alternative method.")
            # Older YOLO versions or different structure
            try:
                metrics = {
                    "map50": getattr(val_results, "map50", 0.0),
                    "precision": getattr(val_results, "precision", 0.0),
                    "recall": getattr(val_results, "recall", 0.0)
                }
                
                # Convert any numpy arrays to scalar float values
                for key in metrics:
                    value = metrics[key]
                    if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] > 1:
                        metrics[key] = float(value.mean())
                    else:
                        try:
                            metrics[key] = float(value)
                        except (TypeError, ValueError):
                            metrics[key] = 0.0
                            
            except Exception as e2:
                logging.error(f"Error extracting metrics from alternative method: {e2}")
                # Fallback
                metrics = {"map50": 0.0, "precision": 0.0, "recall": 0.0}
        
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return {"map50": 0.0, "precision": 0.0, "recall": 0.0}


def run_architecture_search(config, dataset_path, best_model_dir, args, performance_threshold, writer=None):
    """
    Run neural architecture search using Bayesian optimization
    
    Args:
        config: Configuration dictionary
        dataset_path: Path to dataset
        best_model_dir: Directory to save best models
        args: Command-line arguments
        performance_threshold: Threshold for early stopping
        writer: TensorBoard SummaryWriter instance
    
    Returns:
        Path to the best model found
    """
    logging.info("Starting neural architecture search")
    
    # Create directory for trial configurations
    trial_dir = os.path.join(os.getcwd(), "nas_trials")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Define search space from config
    search_space = generate_search_space_from_config(config)
    
    # Initialize optimizer if scikit-optimize is available
    if SKOPT_AVAILABLE:
        optimizer = Optimizer(search_space)
    else:
        optimizer = None
    
    # Initialize tracking variables
    best_map50 = 0.0
    best_model_path = None
    
    # Get the number of iterations from the config
    opt_config = config.get("optimization", {})
    num_iterations = opt_config.get("n_iter", MAX_TRIALS)
    logging.info(f"Will run {num_iterations} search iterations")
    
    no_improvement_count = 0
    
    # Run search iterations
    for iteration in range(num_iterations):
        logging.info(f"\n--- Starting Trial {iteration + 1}/{num_iterations} ---")
        
        # Sample parameters from the config-based search space
        params = sample_parameters_from_config(search_space, optimizer, iteration, config)
        
        # Log parameters
        logging.info(f"Trial parameters: {params}")
        
        # Create model configuration using the model_architecture functions
        model_config = create_model_configuration_from_params(params, config)
        
        # Save the model configuration
        config_path = save_model_config(model_config, trial_dir, iteration)
        
        if not config_path:
            logging.error(f"Failed to save model configuration for trial {iteration}")
            continue
        
        # Train trial model with the configuration
        model_path, results = train_trial_model(config_path, dataset_path, trial_dir, iteration, args, config)
        
        # Evaluate model
        if model_path:
            metrics = evaluate_trial(model_path, dataset_path)
            
            # Get metrics - these should already be Python scalars from our evaluate_trial function
            map50 = metrics.get("map50", 0.0)
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            
            logging.info(f"Trial {iteration + 1} results - mAP@50: {map50:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Log to TensorBoard
            log_to_tensorboard(writer, "search/map50", map50, iteration)
            log_to_tensorboard(writer, "search/precision", precision, iteration)
            log_to_tensorboard(writer, "search/recall", recall, iteration)
            
            # Update optimizer with results
            if SKOPT_AVAILABLE and optimizer:
                optimizer = update_bayesian_optimizer(optimizer, params, map50)
            
            # Check if this is the best model so far
            if map50 > best_map50:
                best_map50 = map50
                best_model_path = save_best_model(model_path, best_model_dir, map50, iteration)
                no_improvement_count = 0
                
                # Early stopping if we reach performance threshold
                perf_threshold = opt_config.get("performance_threshold", performance_threshold)
                if map50 >= perf_threshold:
                    logging.info(f"Performance threshold {perf_threshold} reached. Stopping search.")
                    break
            else:
                no_improvement_count += 1
            
            # Early stopping if no improvement for several iterations
            if no_improvement_count >= EARLY_STOP_PATIENCE:
                logging.info(f"No improvement for {EARLY_STOP_PATIENCE} iterations. Stopping search.")
                break
        else:
            logging.warning(f"Trial {iteration + 1} failed to produce a valid model.")
    
    # Log final results
    if best_model_path:
        logging.info(f"Neural architecture search completed successfully.")
        logging.info(f"Best model: {best_model_path} with mAP@50: {best_map50:.4f}")
        
        # Log to TensorBoard
        log_to_tensorboard(writer, "search/best_map50", best_map50, num_iterations)
    else:
        logging.error("No valid model found during architecture search.")
    
    return best_model_path