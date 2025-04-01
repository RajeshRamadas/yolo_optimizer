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
    aug_config['flipud'] = aug_params.get('flipud', 0.0)  # Vertical flip
    aug_config['fliplr'] = aug_params.get('fliplr', 0.5)  # Horizontal flip
    
    # Create config path
    config_path = os.path.join(best_model_dir, "augmentation_config.yaml")
    
    # Save the configuration
    try:
        with open(config_path, 'w') as f:
            yaml.dump(aug_config, f, default_flow_style=False)
        logging.info(f"Saved augmentation configuration to {config_path}")
        return config_path
    except Exception as e:
        logging.error(f"Error saving augmentation configuration: {e}")
        return None


def configure_additional_augmentations(aug_params):
    """
    Configure additional augmentations using Albumentations.
    
    Args:
        aug_params: Augmentation parameters
        
    Returns:
        Albumentations transforms object or None
    """
    if not aug_params.get('use_albumentations', False):
        return None
    
    try:
        import albumentations as A
        
        # Create a transform pipeline based on the parameters
        transform_list = []
        
        # Add transforms based on configuration settings
        if aug_params.get('blur', False):
            transform_list.append(A.Blur(p=aug_params.get('blur_prob', 0.1)))
        
        if aug_params.get('random_brightness_contrast', False):
            transform_list.append(A.RandomBrightnessContrast(
                brightness_limit=aug_params.get('brightness_limit', 0.2),
                contrast_limit=aug_params.get('contrast_limit', 0.2),
                p=aug_params.get('random_brightness_contrast_prob', 0.2)
            ))
            
        if aug_params.get('cutout', False):
            try:
                # Try newer version (CoarseDropout)
                transform_list.append(A.CoarseDropout(
                    max_holes=aug_params.get('cutout_holes', 4),
                    max_height=aug_params.get('cutout_height', 8),
                    max_width=aug_params.get('cutout_width', 8),
                    p=aug_params.get('cutout_prob', 0.3)
                ))
            except Exception:
                try:
                    # Try older version (Cutout)
                    transform_list.append(A.Cutout(
                        num_holes=aug_params.get('cutout_holes', 4),
                        max_h_size=aug_params.get('cutout_height', 8),
                        max_w_size=aug_params.get('cutout_width', 8),
                        p=aug_params.get('cutout_prob', 0.3)
                    ))
                except Exception as e:
                    logging.warning(f"Could not add cutout augmentation: {e}")
        
        # Skip if no transforms were added
        if not transform_list:
            return None
            
        # Create the transform pipeline
        transform = A.Compose(transform_list, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        # Create the Albumentations YOLO wrapper
        return Albumentations(transform=transform)
        
    except ImportError:
        logging.warning("Albumentations not installed. Advanced augmentations will not be used.")
        return None
    except Exception as e:
        logging.error(f"Error configuring Albumentations: {e}")
        return None


def apply_augmentation_config_to_training(train_args, aug_params, best_model_dir):
    """
    Apply augmentation configuration to training arguments.
    
    Args:
        train_args: Training arguments dictionary
        aug_params: Augmentation parameters
        best_model_dir: Directory to save the augmentation config
        
    Returns:
        dict: Updated training arguments
    """
    # Create augmentation config file
    aug_config_path = create_augmentation_config(aug_params, best_model_dir)
    
    # Update training arguments with augmentation settings
    if aug_config_path:
        train_args['augment'] = True
        
        # Apply custom augmentations
        albumentations_transform = configure_additional_augmentations(aug_params)
        if albumentations_transform:
            # Different versions of YOLOv8 might use different parameter names
            try:
                # Try the known parameter names
                if 'augment_transform' in train_args:
                    train_args['augment_transform'] = albumentations_transform
                elif 'transforms' in train_args:
                    train_args['transforms'] = albumentations_transform
                else:
                    # Fall back to adding it directly to the trainer later
                    logging.info("Will apply albumentations transforms during model initialization")
                    train_args['_albumentations_transform'] = albumentations_transform
            except Exception as e:
                logging.error(f"Error applying albumentations transforms: {e}")
    
    return train_args


def get_default_augmentation_config():
    """
    Get a default augmentation configuration.
    
    Returns:
        dict: Default augmentation configuration
    """
    return {
        # Basic YOLOv8 augmentations
        'use_mosaic': True,
        'mosaic_prob': 0.8,
        'use_mixup': False,
        'mixup_prob': 0.15,
        'use_copy_paste': False,
        'copy_paste_prob': 0.1,
        
        # General augmentations
        'hsv_h': 0.015,  # HSV Hue augmentation
        'hsv_s': 0.5,    # HSV Saturation augmentation
        'hsv_v': 0.3,    # HSV Value augmentation
        'degrees': 0.0,  # Rotation degrees (0-180)
        'translate': 0.1, # Translation (0-1)
        'scale': 0.4,    # Scaling (0-1)
        'shear': 0.0,    # Shear (0-10)
        'perspective': 0.0, # Perspective (0-0.001)
        'flipud': 0.0,   # Vertical flip (0-1)
        'fliplr': 0.5,   # Horizontal flip (0-1)
        
        # Albumentations-specific augmentations
        'use_albumentations': True,
        'random_brightness_contrast': True,
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'random_brightness_contrast_prob': 0.2,
        'cutout': True,
        'cutout_holes': 4,
        'cutout_height': 8,
        'cutout_width': 8,
        'cutout_prob': 0.3
    }