"""
Optimized data augmentation techniques for improving YOLO model performance.
Focus on fewer, more effective augmentations for quicker training.
"""

import os
import yaml
import logging
import torch
import numpy as np

# Fix for the import error with different Ultralytics versions
try:
    # Try the newer path first
    from ultralytics.utils.augment import Albumentations
except ImportError:
    try:
        # Try alternative paths
        from ultralytics.yolo.data.augment import Albumentations
    except ImportError:
        try:
            from ultralytics.data.augment import Albumentations
        except ImportError:
            logging.warning("Could not import Albumentations from Ultralytics. "
                           "Advanced augmentations will be disabled.")
            # Define a placeholder class to avoid errors
            class Albumentations:
                def __init__(self, transform=None):
                    self.transform = transform
                def __call__(self, *args, **kwargs):
                    return args[0]  # Return input unchanged


def get_augmentation_params(config):
    """
    Extract augmentation parameters from the configuration.
    
    Args:
        config: The loaded configuration dictionary
        
    Returns:
        dict: Augmentation parameters
    """
    return config.get('augmentation', {})


def create_augmentation_config(aug_params, best_model_dir):
    """
    Create an augmentation configuration YAML file.
    
    Args:
        aug_params: Augmentation parameters
        best_model_dir: Directory to save the augmentation config
        
    Returns:
        str: Path to the augmentation config file
    """
    # Ensure the directory exists
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Convert augmentation settings to a format YOLOv8 understands
    aug_config = {}
    
    # Handle Mosaic augmentation - reduced for faster convergence
    if aug_params.get('use_mosaic', True):
        aug_config['mosaic'] = aug_params.get('mosaic_prob', 0.8)  # Reduced from 1.0
    else:
        aug_config['mosaic'] = 0.0
    
    # Handle MixUp augmentation - disabled for faster convergence
    if aug_params.get('use_mixup', False):  # Default to False
        aug_config['mixup'] = aug_params.get('mixup_prob', 0.15)
    else:
        aug_config['mixup'] = 0.0
    
    # Handle Copy-Paste augmentation - disabled for faster convergence
    if aug_params.get('use_copy_paste', False):
        aug_config['copy_paste'] = aug_params.get('copy_paste_prob', 0.1)
    else:
        aug_config['copy_paste'] = 0.0

    # Set basic augmentations - reduced for faster convergence
    aug_config['hsv_h'] = aug_params.get('hsv_h', 0.015)  # HSV Hue augmentation
    aug_config['hsv_s'] = aug_params.get('hsv_s', 0.5)    # HSV Saturation augmentation (reduced)
    aug_config['hsv_v'] = aug_params.get('hsv_v', 0.3)    # HSV Value augmentation (reduced)
    aug_config['degrees'] = aug_params.get('degrees', 0.0)  # Rotation degrees (disabled)
    aug_config['translate'] = aug_params.get('translate', 0.1)  # Translation
    aug_config['scale'] = aug_params.get('scale', 0.4)  # Scaling (reduced)
    aug_config['shear'] = aug_params.get('shear', 0.0)  # Shear
    aug_config['perspective'] = aug_params.get('perspective', 0.0)  # Perspective
    aug_config['flipud'] = aug_params.get('flipud', 0