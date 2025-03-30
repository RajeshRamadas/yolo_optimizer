"""
Advanced model architecture creation for YOLO models with complex configurations.
Optimized for faster learning and robustness.
"""

import os
import yaml
import logging
from ultralytics import YOLO
import torch.nn as nn

# Dictionary mapping activation function strings to their implementation
ACTIVATION_FUNCTIONS = {
    'SiLU': 'SiLU()',  # Default YOLOv8 activation
    'ReLU': 'ReLU()',
    'LeakyReLU': 'LeakyReLU(0.1)',
    # Removed Mish since it's not defined in your environment
    'Hardswish': 'Hardswish()',
    'PReLU': 'PReLU()'
}

def create_complex_yolo_config(depth_mult, width_mult, kernel_size, num_channels, 
                             activation='SiLU', use_cbam=False, dropout_rate=0.0,
                             use_gating=False, bottleneck_ratio=0.5, num_heads=4,
                             skip_connections='standard', use_eca=False):
    """
    Create a complex YOLO architecture configuration with advanced options
    
    Args:
        depth_mult: Depth multiplier (scaling of layers)
        width_mult: Width multiplier (scaling of channels)
        kernel_size: Kernel size for specific layers
        num_channels: Base number of channels
        activation: Activation function ('SiLU', 'ReLU', 'LeakyReLU', etc.)
        use_cbam: Whether to use CBAM attention modules
        dropout_rate: Dropout rate for regularization
        use_gating: Whether to use gated convolutions
        bottleneck_ratio: Channel reduction ratio in bottleneck blocks
        num_heads: Number of attention heads
        skip_connections: Type of skip connections ('standard', 'dense', 'residual')
        use_eca: Whether to use Efficient Channel Attention
        
    Returns:
        Model configuration dictionary
    """
    # Ensure parameters are in the right format
    depth_mult = float(depth_mult)
    width_mult = float(width_mult)
    kernel_size = int(kernel_size)
    num_channels = int(num_channels)
    dropout_rate = float(dropout_rate)
    bottleneck_ratio = float(bottleneck_ratio)
    num_heads = int(num_heads)
    
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Determine attention module to use
    attn_module = 'Identity'
    if use_cbam:
        attn_module = 'CBAM'
    elif use_eca:
        attn_module = 'ECA'
    
    # Define bottleneck channels for each stage
    bottleneck_ch = max(8, int(num_channels * bottleneck_ratio))
    
    # Choose skip connection type
    if skip_connections == 'dense':
        skip_type = 'C2f'  # Standard YOLOv8 connections as fallback
    elif skip_connections == 'residual':
        skip_type = 'C2f'  # Standard YOLOv8 connections as fallback
    else:
        skip_type = 'C2f'  # Standard YOLOv8 skip connections
    
    # Build the backbone configuration
    backbone = [
        # [from, number, module, args]
        [-1, 1, 'Conv', [num_channels, 3, 2, 1]],  # 0-P1/2 (stride 2)
    ]
    
    # Add attention if requested
    if attn_module != 'Identity':
        backbone.append([-1, 1, attn_module, [num_channels]])
    
    # Continue with the backbone
    backbone.extend([
        [-1, 1, 'Conv', [num_channels * 2, 3, 2, 1]],  # P2/4 (stride 2)
        [-1, 3, skip_type, [num_channels * 2, True, bottleneck_ch]],
        [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],  # P3/8 (stride 2)
    ])
    
    # Add attention if requested
    if attn_module != 'Identity':
        backbone.append([-1, 1, attn_module, [num_channels * 4]])
    
    # Continue with the backbone - using fewer repeats for faster training
    backbone.extend([
        [-1, 3, skip_type, [num_channels * 4, True, bottleneck_ch * 2]],  # Reduced from 6 to 3
        [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],  # P4/16 (stride 2)
    ])
    
    # Add attention if requested
    if attn_module != 'Identity':
        backbone.append([-1, 1, attn_module, [num_channels * 8]])
    
    # Continue with the backbone - using fewer repeats
    backbone.extend([
        [-1, 3, skip_type, [num_channels * 8, True, bottleneck_ch * 4]],  # Reduced from 6 to 3
        [-1, 1, 'Conv', [num_channels * 16, 3, 2, 1]],  # P5/32 (stride 2)
    ])
    
    # Add attention if requested
    if attn_module != 'Identity':
        backbone.append([-1, 1, attn_module, [num_channels * 16]])
    
    # Finish the backbone
    backbone.append([-1, 2, skip_type, [num_channels * 16, True, bottleneck_ch * 8]])  # Reduced from 3 to 2
    
    # Store the indices for use in the head
    backbone_p3_idx = 5 if attn_module == 'Identity' else 6
    backbone_p4_idx = 8 if attn_module == 'Identity' else 10
    backbone_p5_idx = 11 if attn_module == 'Identity' else 14
    
    # Build the head configuration
    head = [
        [-1, 1, 'SPPF', [num_channels * 16, kernel_size]],  # SPPF
    ]
    
    # Add multi-head attention if requested
    if use_gating:
        head.append([-1, 1, 'MultiHeadAttention', [num_channels * 16, num_heads]])
    
    # Continue with the head
    head.extend([
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, backbone_p4_idx], 1, 'Concat', [1]],  # Concat with P4
        [-1, 2, skip_type, [num_channels * 8, False, bottleneck_ch * 4]],  # Reduced from 3 to 2
        
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, backbone_p3_idx], 1, 'Concat', [1]],  # Concat with P3
        [-1, 2, skip_type, [num_channels * 4, False, bottleneck_ch * 2]],  # P3/8 output, reduced from 3 to 2
    ])
    
    # Store P3 output index for Detect layer
    p3_out_idx = len(head) - 1
    
    # P4 path
    head.extend([
        [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
        [[-1, p3_out_idx - 3], 1, 'Concat', [1]],  # Concat with P4 features
        [-1, 2, skip_type, [num_channels * 8, False, bottleneck_ch * 4]],  # P4/16 output, reduced from 3 to 2
    ])
    
    # Store P4 output index for Detect layer
    p4_out_idx = len(head) - 1
    
    # P5 path
    head.extend([
        [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
        [[-1, 1], 1, 'Concat', [1]],  # Concat with P5 features
        [-1, 2, skip_type, [num_channels * 16, False, bottleneck_ch * 8]],  # P5/32 output, reduced from 3 to 2
    ])
    
    # Store P5 output index for Detect layer
    p5_out_idx = len(head) - 1
    
    # Add Detect layer that uses outputs from P3, P4, P5
    head.append([[p3_out_idx, p4_out_idx, p5_out_idx], 1, 'Detect', ['nc']])
    
    # Create the full model configuration
    model_config = {
        'nc': 80,  # number of classes (COCO has 80 classes)
        'depth_multiple': depth_mult,
        'width_multiple': width_mult,
        'backbone': backbone,
        'head': head,
    }
    
    # Add dropout if specified
    if dropout_rate > 0:
        model_config['dropout'] = dropout_rate
    
    # Add activation function if not the default
    if activation != 'SiLU':
        model_config['activation'] = activation
    
    return model_config

def create_custom_yolo_config(depth_mult, width_mult, kernel_size, num_channels):
    """
    Create a standard YOLOv8-like architecture with specified parameters
    
    Args:
        depth_mult: Depth multiplier (scaling of layers)
        width_mult: Width multiplier (scaling of channels)
        kernel_size: Kernel size for specific layers
        num_channels: Base number of channels
        
    Returns:
        Model configuration dictionary
    """
    # Ensure parameters are in the right format
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
            [-1, 2, 'C2f', [num_channels * 2, True]],  # Reduced from 3 to 2 for faster training
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],  # 3-P3/8
            [-1, 3, 'C2f', [num_channels * 4, True]],  # Reduced from 6 to 3
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],  # 5-P4/16
            [-1, 3, 'C2f', [num_channels * 8, True]],  # Reduced from 6 to 3
            [-1, 1, 'Conv', [num_channels * 16, 3, 2, 1]],  # 7-P5/32
            [-1, 2, 'C2f', [num_channels * 16, True]],  # Reduced from 3 to 2
        ],
        
        # YOLOv8 head - simplified for faster training
        'head': [
            [-1, 1, 'SPPF', [num_channels * 16, kernel_size]],  # 9
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 2, 'C2f', [num_channels * 8, False]],  # 12 (reduced from 3 to 2)
            
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 2, 'C2f', [num_channels * 4, False]],  # 15 (P3/8-small) (reduced from 3 to 2)
            
            [-1, 1, 'Conv', [num_channels * 4, 3, 2, 1]],
            [[-1, 12], 1, 'Concat', [1]],  # cat head P4
            [-1, 2, 'C2f', [num_channels * 8, False]],  # 18 (P4/16-medium) (reduced from 3 to 2)
            
            [-1, 1, 'Conv', [num_channels * 8, 3, 2, 1]],
            [[-1, 9], 1, 'Concat', [1]],  # cat head P5
            [-1, 2, 'C2f', [num_channels * 16, False]],  # 21 (P5/32-large) (reduced from 3 to 2)
            
            [[15, 18, 21], 1, 'Detect', ['nc']],  # Detect(P3, P4, P5)
        ]
    }
    
    return model_config

def save_model_config(model_config, output_path):
    """
    Save a model configuration to a YAML file
    
    Args:
        model_config: Model configuration dictionary
        output_path: Path to save the YAML file
        
    Returns:
        Path to the saved configuration file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the configuration to YAML
    try:
        with open(output_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        logging.info(f"Saved model configuration to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error saving model configuration: {e}")
        return None

def modify_yolo_architecture(depth_mult, width_mult, kernel_size, num_channels, best_model_dir,
                           use_complex=True,  # Changed default to True
                           activation='ReLU',  # Changed default to ReLU
                           use_cbam=False, 
                           dropout_rate=0.2,  # Increased default dropout
                           use_gating=False,
                           bottleneck_ratio=0.5,
                           num_heads=4,
                           skip_connections='residual',  # Changed default to residual
                           use_eca=True):  # Changed default to True
    """
    Create a custom YOLO architecture with specified parameters
    
    Args:
        depth_mult: Depth multiplier (scaling of layers)
        width_mult: Width multiplier (scaling of channels)
        kernel_size: Kernel size for specific layers
        num_channels: Base number of channels
        best_model_dir: Directory to save the model architecture
        use_complex: Whether to use complex architecture with additional options
        activation: Activation function ('SiLU', 'ReLU', 'LeakyReLU', etc.)
        use_cbam: Whether to use attention modules
        dropout_rate: Dropout rate for regularization
        use_gating: Whether to use gated convolutions
        bottleneck_ratio: Channel reduction ratio in bottleneck blocks
        num_heads: Number of attention heads
        skip_connections: Type of skip connections ('standard', 'dense', 'residual')
        use_eca: Whether to use Efficient Channel Attention
        
    Returns:
        Path to the custom YOLO YAML config
    """
    # Create model configuration
    if use_complex:
        model_config = create_complex_yolo_config(
            depth_mult, width_mult, kernel_size, num_channels,
            activation=activation, use_cbam=use_cbam, dropout_rate=dropout_rate,
            use_gating=use_gating, bottleneck_ratio=bottleneck_ratio, num_heads=num_heads,
            skip_connections=skip_connections, use_eca=use_eca
        )
    else:
        model_config = create_custom_yolo_config(depth_mult, width_mult, kernel_size, num_channels)
    
    # Save the model configuration
    custom_model_path = os.path.join(best_model_dir, "custom_yolo.yaml")
    return save_model_config(model_config, custom_model_path)

def log_model_architecture(model_path):
    """
    Log the model architecture to the log file
    
    Args:
        model_path: Path to the YOLO model
    """
    try:
        # Load the model configuration from YAML
        if model_path.endswith('.yaml'):
            with open(model_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logging.info(f"Custom YOLOv8 Architecture Configuration:")
            logging.info(f"- Depth Multiple: {config.get('depth_multiple', 'N/A')}")
            logging.info(f"- Width Multiple: {config.get('width_multiple', 'N/A')}")
            logging.info(f"- Number of Classes: {config.get('nc', 'N/A')}")
            
            if 'activation' in config:
                logging.info(f"- Activation Function: {config.get('activation', 'SiLU')}")
            
            if 'dropout' in config:
                logging.info(f"- Dropout Rate: {config.get('dropout', 0.0)}")
            
            # Count the number of layers
            backbone_layers = len(config.get('backbone', []))
            head_layers = len(config.get('head', []))
            logging.info(f"- Total Layers: {backbone_layers + head_layers}")
            logging.info(f"  - Backbone Layers: {backbone_layers}")
            logging.info(f"  - Head Layers: {head_layers}")
        
        # If it's a model file, load it and print its summary
        else:
            model = YOLO(model_path, task='detect')
            # Log the model summary from YOLO's built-in string representation
            logging.info(f"Modified YOLOv8 Model Summary:")
            logging.info(str(model.model))
            
            # Optionally log model parameters count
            total_params = sum(p.numel() for p in model.model.parameters())
            logging.info(f"Total parameters: {total_params:,}")
            
    except Exception as e:
        logging.error(f"Failed to display model architecture: {e}")
        import traceback
        logging.error(traceback.format_exc())