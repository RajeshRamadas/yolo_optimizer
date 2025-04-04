"""
Utility functions for loading and using YAML configuration files for YOLO optimization.
"""

import os
import yaml
import json
import logging
from datetime import datetime


def load_yaml_config(config_path="configs/nas_config.yaml"):
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: The loaded configuration as a dictionary
    """
    try:
        # Get the absolute path if not already
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        logging.info(f"Loaded YAML configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading YAML configuration: {e}")
        raise


def get_search_bounds(config):
    """
    Convert the search space configuration to BayesianOptimization bounds format.

    Args:
        config: The loaded configuration dictionary

    Returns:
        dict: Dictionary with parameter bounds for BayesianOptimization
    """
    search_space = config.get('search_space', {})
    pbounds = {}
    for param_name, bounds in search_space.items():
        if isinstance(bounds, list) and len(bounds) == 2:
            pbounds[param_name] = (bounds[0], bounds[1])

    return pbounds


def get_directories(config):
    """
    Get directory paths from configuration and create them if they don't exist.

    Args:
        config: The loaded configuration dictionary

    Returns:
        dict: Dictionary with directory paths
    """
    dir_config = config.get('directories', {})
    base_dir = dir_config.get('base_dir', '.')

    # Get directory paths, making them absolute if needed
    dirs = {}
    for key, value in dir_config.items():
        if key != 'base_dir':
            if os.path.isabs(value):
                dirs[key] = value
            else:
                dirs[key] = os.path.join(base_dir, value)

    # Create directories if they don't exist
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    return dirs


def save_nas_results(best_params, best_score, optimizer, config, output_path=None):
    """
    Save the NAS results to a JSON file.

    Args:
        best_params: Best parameters found during optimization
        best_score: Best score achieved
        optimizer: The BayesianOptimization object
        config: The loaded configuration dictionary
        output_path: Path to save results (optional)

    Returns:
        str: Path to the saved configuration file
    """
    directories = get_directories(config)
    best_model_dir = directories.get('best_model_dir', 'best_model')

    # Extract optimization settings
    opt_config = config.get('optimization', {})

    # Create detailed NAS configuration
    nas_config = {
        "parameters": best_params,
        "performance": {
            "map50": best_score,
            "source": "bayesian_optimization"
        },
        "optimization": {
            "init_points": opt_config.get('init_points', 3),
            "n_iter": opt_config.get('n_iter', 15),
            "threshold": opt_config.get('performance_threshold', 0.85),
            "early_stopped": best_score >= opt_config.get('performance_threshold', 0.85)
        },
        "search_space": config.get('search_space', {}),
        "trials": len(optimizer.res) if optimizer else 0,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save NAS configuration
    if output_path:
        nas_config_path = output_path
    else:
        nas_config_path = os.path.join(best_model_dir, "nas_results.json")

    with open(nas_config_path, "w") as f:
        json.dump(nas_config, f, indent=4)
    logging.info(f"Saved NAS results to {nas_config_path}")

    # Also save the full optimization history if optimizer exists
    if optimizer:
        history_path = os.path.join(best_model_dir, "optimization_history.json")
        with open(history_path, "w") as f:
            history_data = [
                {
                    "iteration": i,
                    "parameters": {k: v for k, v in res["params"].items()},
                    "score": res["target"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                for i, res in enumerate(optimizer.res)
            ]
            json.dump(history_data, f, indent=4)
        logging.info(f"Saved full optimization history to {history_path}")

    # Save final best parameters in architecture format
    best_architecture = {}
    for key, value in best_params.items():
        if key == "resolution":
            best_architecture[key] = int(value)
        elif key in ["depth_mult", "width_mult"]:
            # Convert to standard YOLO naming convention
            new_key = key.replace("_mult", "_multiple")
            best_architecture[new_key] = float(value)
        elif key == "num_channels":
            best_architecture["base_channels"] = int(value)
        elif key == "kernel_size":
            best_architecture[key] = int(value)
        else:
            # For training hyperparameters
            best_architecture[key] = float(value) if isinstance(value, (int, float)) else value

    best_architecture["map50"] = best_score

    arch_path = os.path.join(best_model_dir, "best_architecture.json")
    with open(arch_path, "w") as f:
        json.dump(best_architecture, f, indent=4)
    logging.info(f"Saved best architecture to {arch_path}")

    return nas_config_path


def get_dataset_path(config):
    """
    Get the dataset path from the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        str: Dataset path
    """
    return config.get('dataset_path', 'data.yaml')


def get_initial_test_params(config):
    """
    Get the initial test parameters from the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        dict: Initial test parameters
    """
    return config.get('initial_test', {})


def get_optimization_params(config):
    """
    Get the optimization parameters from the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        dict: Optimization parameters
    """
    return config.get('optimization', {})


def get_training_params(config):
    """
    Get the training parameters from the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        dict: Training parameters
    """
    return config.get('training', {})


def get_pruning_params(config):
    """
    Get the pruning parameters from the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        dict: Pruning parameters
    """
    return config.get('pruning', {})