"""
Functions for creating and modifying YOLO model architectures for optimization.
"""

import os
import yaml
import logging
from ultralytics import YOLO


def modify_yolo_architecture(depth_mult, width_mult, kernel_size, num_channels, best_model_dir):
    """
    Create a custom YOLO architecture with specified parameters

    Args:
        depth_mult: Depth multiplier (scaling of layers)
        width_mult: Width multiplier (scaling of channels)
        kernel_size: Kernel size for specific layers
        num_channels: Base number of channels
        best_model_dir: Directory to save the model architecture

    Returns:
        Path to the custom YOLO YAML config
    """
    depth_mult = float(depth_mult)
    width_mult = float(width_mult)
    kernel_size = int(kernel_size)
    num_channels = int(num_channels)

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Use a standard YOLOv8 nano model as base but with custom parameters
    model_config = {
        'nc': 80,  # number of classes (COCO has 80 classes)
        'depth_multiple': depth_mult,
        'width_multiple': width_mult,

        # YOLOv8 backbone - simplified to avoid tensor size mismatches
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [num_channels, 3, 2, 1]],  # 0-P1/2  (Use fixed kernel_size=3 for initial layers)
            [-1, 1, 'Conv', [num_channels * 2, 3, 2, 1]],  # 1-P2/4
            [-1, 3, 'C2f', [num_channels * 2, True]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],  # 3-P3/8
            [-1, 6, 'C2f', [num_channels * 4, True]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],  # 5-P4/16
            [-1, 6, 'C2f', [num_channels * 8, True]],
            [-1, 1, 'Conv', [num_channels * 16, 3, 2, 1]],  # 7-P5/32
            [-1, 3, 'C2f', [num_channels * 16, True]],
        ],

        # YOLOv8 head - simplified to match standard YOLOv8 architecture
        'head': [
            [-1, 1, 'SPPF', [num_channels * 16, kernel_size]],  # 9
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 3, 'C2f', [num_channels * 8, False]],  # 12

            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 3, 'C2f', [num_channels * 4, False]],  # 15 (P3/8-small)

            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [[-1, 12], 1, 'Concat', [1]],  # cat head P4
            [-1, 3, 'C2f', [num_channels * 8, False]],  # 18 (P4/16-medium)

            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [[-1, 9], 1, 'Concat', [1]],  # cat head P5
            [-1, 3, 'C2f', [num_channels * 16, False]],  # 21 (P5/32-large)

            [[15, 18, 21], 1, 'Detect', ['nc']],  # Detect(P3, P4, P5)
        ]
    }

    custom_model_path = os.path.join(best_model_dir, "custom_yolo.yaml")
    with open(custom_model_path, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    return custom_model_path


def log_model_architecture(model_path):
    """
    Log the model architecture to the log file

    Args:
        model_path: Path to the YOLO model
    """
    try:
        model = YOLO(model_path, task='detect')
        # Just log the model summary from YOLO's built-in string representation
        logging.info(f"Modified YOLOv8 Architecture:")
        logging.info(str(model.model))

        # Optionally log model parameters count
        total_params = sum(p.numel() for p in model.model.parameters())
        logging.info(f"Total parameters: {total_params:,}")
    except Exception as e:
        logging.error(f"Failed to display model architecture: {e}")


def count_parameters(model):
    """
    Count the number of parameters in a model

    Args:
        model: YOLO model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)