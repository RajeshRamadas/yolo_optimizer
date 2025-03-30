"""
Standardized YOLO architecture based on optimal parameters found in trials.
This module provides a consistent, high-performing model architecture
that can be used directly without running neural architecture search.
"""

import os
import yaml
import logging
from ultralytics import YOLO
import torch
import torch.nn as nn

def create_standard_yolo_config(num_classes=80):
    """
    Create a standardized YOLO architecture that's proven to work well
    based on successful trial results.
    
    Args:
        num_classes: Number of classes to detect (default: 80 for COCO)
        
    Returns:
        Model configuration dictionary
    """
    # Use the proven parameters from successful trials
    depth_mult = 0.58
    width_mult = 0.60
    num_channels = 64
    kernel_size = 5
    dropout_rate = 0.29
    num_heads = 2
    bottleneck_ratio = 0.5
    input_size = 352
    
    model_config = {
        'nc': num_classes,
        'depth_multiple': depth_mult,
        'width_multiple': width_mult,
        'input_size': input_size,
        # Use SiLU instead of ReLU as it's YOLOv8's default and works reliably
        'activation': 'torch.nn.SiLU()',
        'dropout': dropout_rate,
        
        # Backbone from successful models
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [num_channels, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels]],
            [-1, 1, 'Conv', [num_channels * 2, 3, 2, 1]],
            [-1, 3, 'C2f', [num_channels * 2, True, int(num_channels * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels * 4]],
            [-1, 3, 'C2f', [num_channels * 4, True, int(num_channels * 2 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels * 8]],
            [-1, 3, 'C2f', [num_channels * 8, True, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 16, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels * 16]],
            [-1, 2, 'C2f', [num_channels * 16, True, int(num_channels * 8 * bottleneck_ratio)]],
        ],
        
        # Head from successful models
        'head': [
            [-1, 1, 'SPPF', [num_channels * 16, kernel_size]],
            [-1, 1, 'MultiHeadAttention', [num_channels * 16, num_heads]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 10], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 8, False, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 4, False, int(num_channels * 2 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 8, False, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [[-1, 1], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 16, False, int(num_channels * 8 * bottleneck_ratio)]],
            [[7, 10, 13], 1, 'Detect', ['nc']],
        ]
    }
    
    return model_config

def create_efficient_yolo_config(num_classes=80):
    """
    Create a more efficient YOLO architecture with fewer parameters
    but still good performance for resource-constrained environments.
    
    Args:
        num_classes: Number of classes to detect (default: 80 for COCO)
        
    Returns:
        Model configuration dictionary
    """
    # Use more efficient parameters
    depth_mult = 0.33
    width_mult = 0.50
    num_channels = 48
    kernel_size = 3
    dropout_rate = 0.2
    bottleneck_ratio = 0.5
    input_size = 320
    
    model_config = {
        'nc': num_classes,
        'depth_multiple': depth_mult,
        'width_multiple': width_mult,
        'input_size': input_size,
        'activation': 'torch.nn.SiLU()',
        'dropout': dropout_rate,
        
        # Simplified backbone
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [num_channels, 3, 2, 1]],
            [-1, 1, 'Conv', [num_channels * 2, 3, 2, 1]],
            [-1, 2, 'C2f', [num_channels * 2, True, int(num_channels * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [-1, 2, 'C2f', [num_channels * 4, True, int(num_channels * 2 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [-1, 2, 'C2f', [num_channels * 8, True, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 16, 3, 2, 1]],
            [-1, 1, 'C2f', [num_channels * 16, True, int(num_channels * 8 * bottleneck_ratio)]],
        ],
        
        # Simplified head
        'head': [
            [-1, 1, 'SPPF', [num_channels * 16, kernel_size]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 8, False, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 4, False, int(num_channels * 2 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [[-1, 3], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 8, False, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [[-1, 1], 1, 'Concat', [1]],
            [-1, 2, 'C2f', [num_channels * 16, False, int(num_channels * 8 * bottleneck_ratio)]],
            [[6, 9, 12], 1, 'Detect', ['nc']],
        ]
    }
    
    return model_config

def create_high_performance_yolo_config(num_classes=80):
    """
    Create a high-performance YOLO architecture with more parameters
    focused on maximizing detection accuracy.
    
    Args:
        num_classes: Number of classes to detect (default: 80 for COCO)
        
    Returns:
        Model configuration dictionary
    """
    # Use more powerful parameters
    depth_mult = 0.67
    width_mult = 0.75
    num_channels = 80
    kernel_size = 5
    dropout_rate = 0.2
    num_heads = 4
    bottleneck_ratio = 0.5
    input_size = 416
    
    model_config = {
        'nc': num_classes,
        'depth_multiple': depth_mult,
        'width_multiple': width_mult,
        'input_size': input_size,
        'activation': 'torch.nn.SiLU()',  # Using SiLU for max performance
        'dropout': dropout_rate,
        
        # Enhanced backbone
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [num_channels, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels]],
            [-1, 1, 'Conv', [num_channels * 2, 3, 2, 1]],
            [-1, 3, 'C2f', [num_channels * 2, True, int(num_channels * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels * 4]],
            [-1, 6, 'C2f', [num_channels * 4, True, int(num_channels * 2 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels * 8]],
            [-1, 6, 'C2f', [num_channels * 8, True, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 16, 3, 2, 1]],
            [-1, 1, 'CBAM', [num_channels * 16]],
            [-1, 3, 'C2f', [num_channels * 16, True, int(num_channels * 8 * bottleneck_ratio)]],
        ],
        
        # Enhanced head
        'head': [
            [-1, 1, 'SPPF', [num_channels * 16, kernel_size]],
            [-1, 1, 'MultiHeadAttention', [num_channels * 16, num_heads]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 10], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [num_channels * 8, False, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [num_channels * 4, False, int(num_channels * 2 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [num_channels * 8, False, int(num_channels * 4 * bottleneck_ratio)]],
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [[-1, 1], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [num_channels * 16, False, int(num_channels * 8 * bottleneck_ratio)]],
            [[7, 10, 13], 1, 'Detect', ['nc']],
        ]
    }
    
    return model_config

def save_standard_model_config(config_type="standard", output_dir="models", num_classes=80):
    """
    Save a standard model configuration to a YAML file.
    
    Args:
        config_type: Type of configuration to save ("standard", "efficient", or "high_performance")
        output_dir: Directory to save the configuration
        num_classes: Number of classes for detection
        
    Returns:
        Path to the saved configuration file
    """
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Select the appropriate configuration
    if config_type == "efficient":
        model_config = create_efficient_yolo_config(num_classes)
        filename = "yolo_efficient.yaml"
    elif config_type == "high_performance":
        model_config = create_high_performance_yolo_config(num_classes)
        filename = "yolo_high_performance.yaml"
    else:  # standard
        model_config = create_standard_yolo_config(num_classes)
        filename = "yolo_standard.yaml"
    
    # Save the configuration to YAML
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        logging.info(f"Saved {config_type} model configuration to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error saving model configuration: {e}")
        return None

def train_standard_model(config_type="standard", dataset_path=None, output_dir="models", 
                        epochs=100, batch_size=16, image_size=None, num_classes=80):
    """
    Train a standard model with the specified configuration.
    
    Args:
        config_type: Type of configuration to use ("standard", "efficient", or "high_performance")
        dataset_path: Path to the dataset YAML file
        output_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Image size for training (override config default)
        num_classes: Number of classes for detection
        
    Returns:
        Path to the trained model
    """
    # Save the model configuration
    config_path = save_standard_model_config(config_type, output_dir, num_classes)
    
    if not config_path or not dataset_path:
        logging.error("Failed to create model configuration or dataset path not provided")
        return None
    
    # Load the configuration to get the default image size if not specified
    if not image_size:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            image_size = config.get('input_size', 640)
        except:
            image_size = 640
    
    # Initialize the model
    try:
        model = YOLO(config_path)
        
        # Set up training parameters
        train_args = {
            'data': dataset_path,
            'epochs': epochs,
            'imgsz': image_size,
            'batch': batch_size,
            'optimizer': 'AdamW',
            'patience': 20,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'project': output_dir,
            'name': f"yolo_{config_type}",
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'warmup_epochs': 3,
            'lr0': 0.01,
            'lrf': 0.01,
            'use_ema': True
        }
        
        # Train the model
        results = model.train(**train_args)
        
        # Get the path to the best model
        best_model_path = os.path.join(output_dir, f"yolo_{config_type}", "weights", "best.pt")
        
        if os.path.exists(best_model_path):
            logging.info(f"Training completed successfully. Best model saved to: {best_model_path}")
            return best_model_path
        else:
            logging.warning(f"Training completed but best model not found at expected path: {best_model_path}")
            # Try to find any PT file in the weights directory
            weights_dir = os.path.dirname(best_model_path)
            if os.path.exists(weights_dir):
                pt_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                if pt_files:
                    alt_path = os.path.join(weights_dir, pt_files[0])
                    logging.info(f"Using alternative model: {alt_path}")
                    return alt_path
            
            return None
    
    except Exception as e:
        logging.error(f"Error training model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Standard YOLO Architecture Training")
    parser.add_argument("--type", type=str, default="standard", choices=["standard", "efficient", "high_performance"],
                        help="Type of model configuration to use")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset YAML file")
    parser.add_argument("--output", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--classes", type=int, default=80,
                        help="Number of classes for detection")
    
    args = parser.parse_args()
    
    # Train the selected model type
    model_path = train_standard_model(
        config_type=args.type,
        dataset_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        num_classes=args.classes
    )
    
    if model_path:
        logging.info(f"Model training completed. Best model: {model_path}")
    else:
        logging.error("Model training failed.")