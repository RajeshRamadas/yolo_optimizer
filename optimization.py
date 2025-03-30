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

MODEL_FINGERPRINTS = set()

# Import model_architecture functions
from model_architecture import (
    create_custom_yolo_config,
    create_complex_yolo_config,
    save_model_config as save_arch_config,
    log_model_architecture,
    check_model_initialization,  # NEW: Added import for model initialization check
    get_model_fingerprint,       # NEW: Added import for model fingerprint
    compare_model_structures     # NEW: Added import for model comparison
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

# NEW: Track model fingerprints to detect duplicates
MODEL_FINGERPRINTS = set()


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
    # Reset random seed for each iteration to ensure diversity
    import time
    random.seed(int(time.time() * 1000) + iteration)  # Use current time plus iteration as seed
    
    params = {}
    
    # Get activation and skip connections mappings
    activation_mapping = config.get("activation_mapping", {
        0: "SiLU", 1: "ReLU", 2: "LeakyReLU", 3: "Hardswish"
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
    
    # Debug log sampled parameters to verify diversity
    logging.info(f"Trial {iteration} - Sampled parameters: depth_mult={params.get('depth_mult', 'N/A')}, " 
                f"width_mult={params.get('width_mult', 'N/A')}, " 
                f"activation={params.get('activation', 'N/A')}, "
                f"dropout_rate={params.get('dropout_rate', 'N/A')}")
    
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
    
    # NEW: Check for model initialization issues
    init_success, init_error = check_model_initialization(config_path)
    if not init_success:
        logging.warning(f"Model initialization check failed: {init_error}")
        logging.warning("Will attempt to train anyway, but may fall back to default model")
    else:
        logging.info("Model initialization check passed successfully")
    
    # NEW: Check for duplicate model architecture
    if config_path.endswith('.yaml'):
        fp_success, fingerprint = get_model_fingerprint(config_path)
        if fp_success:
            if fingerprint in MODEL_FINGERPRINTS:
                logging.warning(f"Duplicate model architecture detected with fingerprint: {fingerprint}")
                logging.warning("This model has the same fundamental structure as a previously tested model")
            else:
                MODEL_FINGERPRINTS.add(fingerprint)
                logging.info(f"New model architecture fingerprint: {fingerprint}")
    
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
        # NEW: Log a success message
        logging.info(f"Successfully initialized model with config: {config_path}")
        
        # NEW: Log model size information
        if hasattr(model, 'model'):
            param_count = sum(p.numel() for p in model.model.parameters())
            logging.info(f"Model has {param_count:,} parameters")
            
            # Log the first layer structure as a sanity check
            if hasattr(model.model, 'model') and len(model.model.model) > 0:
                first_layer = model.model.model[0]
                logging.info(f"First layer: {type(first_layer).__name__}")
    except Exception as e:
        logging.error(f"Error initializing YOLO model with config {config_path}: {e}")
        # Fall back to default model
        logging.warning("Falling back to default YOLOv8n model")
        model = YOLO('yolov8n.yaml')
        
        # NEW: Track fallback occurrences
        global_config['_fallback_count'] = global_config.get('_fallback_count', 0) + 1
        logging.warning(f"This is fallback #{global_config['_fallback_count']} in this search session")
    
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
        Path to the best model found (or fallback model if all trials fail)
    """
    logging.info("Starting neural architecture search")
    
    # Create directory for trial configurations
    trial_dir = os.path.join(os.getcwd(), "nas_trials")
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Initialize list to keep track of all successful models
    all_models = []  # List of (model_path, map50, precision, recall) tuples for all successful models
    
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
    
    # NEW: Initialize counters for tracking model diversity and success rates
    fallback_count = 0
    unique_models_count = 0
    model_initialization_failures = 0
    
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
        
        # Update counters for tracking diversity
        if '_fallback_count' in config and config['_fallback_count'] > fallback_count:
            fallback_count = config['_fallback_count']
            model_initialization_failures += 1
        
        # Track number of unique models (based on fingerprints)
        unique_models_count = len(MODEL_FINGERPRINTS)
        
        # Log diversity stats
        logging.info(f"Model Diversity Stats - Trial {iteration + 1}:")
        logging.info(f"  Unique model architectures: {unique_models_count}")
        logging.info(f"  Model initialization failures: {model_initialization_failures}")
        logging.info(f"  Fallback to default model: {fallback_count} times")
        
        # Evaluate model
        if model_path:
            metrics = evaluate_trial(model_path, dataset_path)
            
            # Get metrics - these should already be Python scalars from our evaluate_trial function
            map50 = metrics.get("map50", 0.0)
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            
            logging.info(f"Trial {iteration + 1} results - mAP@50: {map50:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Add to list of all successful models
            all_models.append((model_path, map50, precision, recall))
            
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
    
    # Check if we have a best model
    if best_model_path and os.path.exists(best_model_path):
        logging.info(f"Neural architecture search completed successfully.")
        logging.info(f"Best model: {best_model_path} with mAP@50: {best_map50:.4f}")
        logging.info(f"Search Statistics Summary:")
        logging.info(f"  Total iterations ran: {iteration + 1} of {num_iterations}")
        logging.info(f"  Unique model architectures: {unique_models_count}")
        logging.info(f"  Model initialization failures: {model_initialization_failures}")
        logging.info(f"  Fallback to default model: {fallback_count} times")
        
        # Log to TensorBoard
        log_to_tensorboard(writer, "search/best_map50", best_map50, num_iterations)
        
        # Generate performance summary
        create_performance_summary(all_models, best_model_path, config)
        
        return best_model_path
    else:
        # No best model - try to find any successful model
        logging.warning("Best model not found or was not properly saved.")
        
        # Try to find another successful model from our list
        if all_models:
            # Sort by map50 score and get the best one
            all_models.sort(key=lambda x: x[1], reverse=True)
            best_alternative = all_models[0]
            
            logging.info(f"Using alternative model with mAP@50: {best_alternative[1]:.4f}")
            # Copy it to the best_model_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alternative_path = os.path.join(best_model_dir, f"alternative_model_map{best_alternative[1]:.4f}_{timestamp}.pt")
            try:
                shutil.copy(best_alternative[0], alternative_path)
                logging.info(f"Saved alternative model to: {alternative_path}")
                
                # Generate performance summary
                create_performance_summary(all_models, alternative_path, config)
                
                return alternative_path
            except Exception as e:
                logging.error(f"Failed to save alternative model: {e}")
        
        # If all else fails, create a fallback default model
        logging.error("No valid models were produced during architecture search. Creating fallback model.")
        try:
            # Create a default YOLOv8 model
            fallback_model = YOLO('yolov8n.pt')
            logging.info("Created default YOLOv8n model as fallback")
            
            # Save it to the best_model_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_path = os.path.join(best_model_dir, f"fallback_default_model_{timestamp}.pt")
            
            # If using a pre-trained model, simply copy it
            if hasattr(fallback_model, 'ckpt_path') and os.path.exists(fallback_model.ckpt_path):
                shutil.copy(fallback_model.ckpt_path, fallback_path)
            # Otherwise, save the model directly
            else:
                fallback_model.save(fallback_path)
                
            logging.info(f"Saved fallback model to: {fallback_path}")
            
            # Generate performance summary
            if all_models:
                create_performance_summary(all_models, fallback_path, config)
            
            return fallback_path
        except Exception as e:
            logging.error(f"Failed to create or save fallback model: {e}")
            logging.error(f"Error details: {str(e)}")
            logging.error("Architecture search failed completely.")
            return None


def create_performance_summary(all_models, best_model_path, config):
    """
    Create a summary table of all model performances from architecture search
    
    Args:
        all_models: List of tuples (model_path, map50, precision, recall)
        best_model_path: Path to the best model
        config: Configuration dictionary
        
    Returns:
        Tuple of paths (html_path, csv_path) to the created summary files
    """
    try:
        import pandas as pd
        from datetime import datetime
        import re
        
        # Get results directory from config
        directories = config.get('directories', {})
        results_dir = directories.get('results_dir', 'results')
        if not os.path.isabs(results_dir):
            base_dir = directories.get('base_dir', '.')
            results_dir = os.path.join(base_dir, results_dir)
        
        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a DataFrame to store results
        results = []
        
        # Extract trial number from model paths
        pattern = r'trial[_]?(\d+)'
        
        # Process each model
        for model_data in all_models:
            model_path = model_data[0]
            map50 = model_data[1]
            precision = model_data[2] if len(model_data) > 2 else None
            recall = model_data[3] if len(model_data) > 3 else None
            
            # Get filename only
            filename = os.path.basename(model_path)
            
            # Extract trial number
            match = re.search(pattern, model_path)
            trial_num = int(match.group(1)) if match else -1
            
            # Check if this is the best model
            is_best = os.path.normpath(model_path) == os.path.normpath(best_model_path)
            
            # Add to results
            results.append({
                'Trial': trial_num,
                'Model Filename': filename,
                'mAP@50': round(map50, 4) if map50 is not None else None,
                'Precision': round(precision, 4) if precision is not None else None,
                'Recall': round(recall, 4) if recall is not None else None,
                'Is Best Model': '✓' if is_best else ''
            })
        
        # Convert to DataFrame and sort by mAP (descending)
        df = pd.DataFrame(results)
        df = df.sort_values(by=['mAP@50'], ascending=False)
        
        # Save as CSV
        csv_path = os.path.join(results_dir, f'architecture_search_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as formatted HTML with styling
        html_path = os.path.join(results_dir, f'architecture_search_results_{timestamp}.html')
        
        # Create styled HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Architecture Search Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th {{ background-color: #333366; color: white; text-align: left; padding: 8px; }}
                td {{ border: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #ddd; }}
                .best-model {{ background-color: #e6ffe6; font-weight: bold; }}
                .summary {{ margin-top: 30px; padding: 10px; background-color: #f8f8f8; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>YOLO Architecture Search Results</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total models trained: {len(results)}</p>
                <p>Best model: {os.path.basename(best_model_path) if best_model_path else 'None'}</p>
                <p>Best mAP@50: {df['mAP@50'].max() if not df.empty else 'N/A'}</p>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <h2>Performance Table</h2>
        """
        
        # Convert DataFrame to HTML with custom formatting
        table_html = df.to_html(index=False, classes='dataframe')
        
        # Add highlighting for the best model
        for i, row in df.iterrows():
            if row['Is Best Model'] == '✓':
                # Replace the row with a version that has the "best-model" class
                row_html = f'<tr class="best-model">'
                for cell in row.values:
                    row_html += f"<td>{cell}</td>"
                row_html += '</tr>'
                
                table_row = f'<tr>'
                for cell in row.values:
                    table_row += f"<td>{cell}</td>"
                table_row += '</tr>'
                
                table_html = table_html.replace(table_row, row_html)
        
        html_content += table_html
        html_content += """
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Performance summary saved to {csv_path} and {html_path}")
        
        # Create a symlink or copy to the best_model_dir for convenience
        best_model_dir = directories.get('best_model_dir', 'best_models')
        if not os.path.isabs(best_model_dir):
            best_model_dir = os.path.join(base_dir, best_model_dir)
            
        try:
            csv_link = os.path.join(best_model_dir, 'latest_search_results.csv')
            html_link = os.path.join(best_model_dir, 'latest_search_results.html')
            
            # Remove existing links/files if they exist
            if os.path.exists(csv_link):
                os.remove(csv_link)
            if os.path.exists(html_link):
                os.remove(html_link)
            
            # Create new copies
            shutil.copy2(csv_path, csv_link)
            shutil.copy2(html_path, html_link)
            
            logging.info(f"Results also available at:")
            logging.info(f"  - {csv_link}")
            logging.info(f"  - {html_link}")
        except Exception as e:
            logging.warning(f"Could not create convenience links: {e}")
        
        return html_path, csv_path
        
    except ImportError:
        logging.warning("pandas not installed - cannot create performance summary table")
        return None, None
    except Exception as e:
        logging.error(f"Failed to create performance summary: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None
        
def create_performance_summary(all_models, best_model_path, output_dir):
    """
    Create a summary table of all model performances from architecture search
    
    Args:
        all_models: List of tuples (model_path, map50, [optional additional metrics])
        best_model_path: Path to the best model
        output_dir: Directory to save the summary
    
    Returns:
        Path to the created summary file
    """
    import os
    import pandas as pd
    from datetime import datetime
    import re
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame to store results
    results = []
    
    # Extract trial number from model paths
    pattern = r'trial(\d+)'
    
    # Process each model
    for model_data in all_models:
        model_path = model_data[0]
        map50 = model_data[1]
        
        # Extract additional metrics if available
        precision = model_data[2] if len(model_data) > 2 else None
        recall = model_data[3] if len(model_data) > 3 else None
        
        # Get filename only
        filename = os.path.basename(model_path)
        
        # Extract trial number
        match = re.search(pattern, model_path)
        trial_num = int(match.group(1)) if match else -1
        
        # Check if this is the best model
        is_best = os.path.abspath(model_path) == os.path.abspath(best_model_path)
        
        # Add to results
        results.append({
            'Trial': trial_num,
            'Model Filename': filename,
            'mAP@50': round(map50, 4) if map50 is not None else None,
            'Precision': round(precision, 4) if precision is not None else None,
            'Recall': round(recall, 4) if recall is not None else None,
            'Is Best Model': '✓' if is_best else ''
        })
    
    # Convert to DataFrame and sort by mAP (descending)
    df = pd.DataFrame(results)
    df = df.sort_values(by=['mAP@50'], ascending=False)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'architecture_search_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as formatted HTML with styling
    html_path = os.path.join(output_dir, f'architecture_search_results_{timestamp}.html')
    
    # Create styled HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Architecture Search Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th {{ background-color: #333366; color: white; text-align: left; padding: 8px; }}
            td {{ border: 1px solid #ddd; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
            .best-model {{ background-color: #e6ffe6; font-weight: bold; }}
            .summary {{ margin-top: 30px; padding: 10px; background-color: #f8f8f8; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>YOLO Architecture Search Results</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Total models trained: {len(results)}</p>
            <p>Best model: {os.path.basename(best_model_path) if best_model_path else 'None'}</p>
            <p>Best mAP@50: {df['mAP@50'].max() if not df.empty else 'N/A'}</p>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <h2>Performance Table</h2>
    """
    
    # Convert DataFrame to HTML with custom formatting
    table_html = df.to_html(index=False, classes='dataframe')
    
    # Add highlighting for the best model
    for i, row in df.iterrows():
        if row['Is Best Model'] == '✓':
            # Replace the row with a version that has the "best-model" class
            row_html = f'<tr class="best-model">{" ".join([f"<td>{cell}</td>" for cell in row.values])}</tr>'
            table_row = f'<tr>{" ".join([f"<td>{cell}</td>" for cell in row.values])}</tr>'
            table_html = table_html.replace(table_row, row_html)
    
    html_content += table_html
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Performance summary saved to {csv_path} and {html_path}")
    
    return html_path, csv_path


# Then add this to the end of run_architecture_search, before returning the best model path:

def update_run_architecture_search_with_summary(config, best_model_dir, best_model_path, all_models):
    """
    Generate performance summary at the end of the search
    """
    try:
        # Ensure all models includes the metrics (map50, precision, recall)
        models_with_metrics = []
        for model_path, map50 in all_models:
            try:
                # If precision and recall not stored, re-evaluate the model
                metrics = evaluate_trial(model_path, dataset_path)
                models_with_metrics.append(
                    (model_path, map50, metrics.get("precision", None), metrics.get("recall", None))
                )
            except Exception as e:
                logging.warning(f"Could not re-evaluate model {model_path}: {e}")
                models_with_metrics.append((model_path, map50, None, None))
        
        # Get results directory from config
        directories = config.get('directories', {})
        results_dir = directories.get('results_dir', 'results')
        if not os.path.isabs(results_dir):
            base_dir = directories.get('base_dir', '.')
            results_dir = os.path.join(base_dir, results_dir)
        
        # Create the performance summary
        html_path, csv_path = create_performance_summary(
            models_with_metrics,
            best_model_path,
            results_dir
        )
        
        logging.info(f"Architecture search performance summary created:")
        logging.info(f"  - HTML: {html_path}")
        logging.info(f"  - CSV: {csv_path}")
        
        # Create a symlink or copy to the best_model_dir for convenience
        try:
            csv_link = os.path.join(best_model_dir, 'latest_search_results.csv')
            html_link = os.path.join(best_model_dir, 'latest_search_results.html')
            
            # Remove existing links/files if they exist
            if os.path.exists(csv_link):
                os.remove(csv_link)
            if os.path.exists(html_link):
                os.remove(html_link)
            
            # Create new links or copies
            shutil.copy2(csv_path, csv_link)
            shutil.copy2(html_path, html_link)
            
            logging.info(f"Results also available at:")
            logging.info(f"  - {csv_link}")
            logging.info(f"  - {html_link}")
        except Exception as e:
            logging.warning(f"Could not create convenience links: {e}")
    
    except Exception as e:
        logging.error(f"Failed to create performance summary: {e}")
        import traceback
        logging.error(traceback.format_exc())