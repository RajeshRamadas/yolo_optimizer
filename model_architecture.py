"""
Advanced model architecture creation for YOLO models with complex configurations.
Optimized for faster learning and robustness.
"""

import os
import yaml
import logging
from ultralytics import YOLO
import torch.nn as nn
import torch

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
            
            # NEW: Enhanced logging for model structure
            backbone = config.get('backbone', [])
            head = config.get('head', [])
            
            logging.info(f"- Detailed Backbone Structure:")
            for i, layer in enumerate(backbone):
                if len(layer) >= 4:  # Should have [from, number, module, args]
                    logging.info(f"  Layer {i}: {layer[2]} - Args: {layer[3]}")
            
            logging.info(f"- Detailed Head Structure:")
            for i, layer in enumerate(head):
                if isinstance(layer, list) and len(layer) >= 3:  # Should have structure
                    logging.info(f"  Layer {i}: {layer[2]} - Args: {layer[3] if len(layer) > 3 else 'N/A'}")
            
            # Calculate and log parameter counts (estimated)
            try:
                channels_base = int(backbone[0][3][0]) if backbone and len(backbone) > 0 and len(backbone[0]) > 3 else 32
                width_mult = float(config.get('width_multiple', 1.0))
                depth_mult = float(config.get('depth_multiple', 1.0))
                estimated_params = estimate_params(channels_base, width_mult, depth_mult, backbone_layers, head_layers)
                logging.info(f"- Estimated Parameters: {estimated_params:,}")
            except Exception as est_err:
                logging.warning(f"  Could not estimate parameters: {est_err}")
        
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

# NEW: Helper function to estimate parameters
def estimate_params(base_channels, width_mult, depth_mult, backbone_layers, head_layers):
    """Rough estimation of parameter count based on network structure"""
    # This is a very rough estimate - real count requires actual model
    base_params = 1000000  # Base YOLO nano has ~1M params
    # Scale based on width (quadratic effect on conv layers)
    width_scale = width_mult ** 2
    # Scale based on depth
    depth_scale = depth_mult
    # Scale based on layer count
    layer_scale = (backbone_layers + head_layers) / 20  # Standard YOLOv8n has ~20 layers
    
    return int(base_params * width_scale * depth_scale * layer_scale)

# NEW: Function to compare model structures
def compare_model_structures(model_path1, model_path2):
    """
    Compare two model architectures and log differences
    
    Args:
        model_path1: Path to first model YAML
        model_path2: Path to second model YAML
    """
    try:
        # Load both configurations
        with open(model_path1, 'r') as f1:
            config1 = yaml.safe_load(f1)
        
        with open(model_path2, 'r') as f2:
            config2 = yaml.safe_load(f2)
        
        # Compare basic parameters
        logging.info(f"Model Comparison - {os.path.basename(model_path1)} vs {os.path.basename(model_path2)}:")
        
        # Compare depth and width
        depth1 = config1.get('depth_multiple', 'N/A')
        depth2 = config2.get('depth_multiple', 'N/A')
        logging.info(f"- Depth Multiple: {depth1} vs {depth2} - {'SAME' if depth1 == depth2 else 'DIFFERENT'}")
        
        width1 = config1.get('width_multiple', 'N/A')
        width2 = config2.get('width_multiple', 'N/A')
        logging.info(f"- Width Multiple: {width1} vs {width2} - {'SAME' if width1 == width2 else 'DIFFERENT'}")
        
        # Compare activation and dropout
        act1 = config1.get('activation', 'SiLU')
        act2 = config2.get('activation', 'SiLU')
        logging.info(f"- Activation: {act1} vs {act2} - {'SAME' if act1 == act2 else 'DIFFERENT'}")
        
        drop1 = config1.get('dropout', 0.0)
        drop2 = config2.get('dropout', 0.0)
        logging.info(f"- Dropout: {drop1} vs {drop2} - {'SAME' if drop1 == drop2 else 'DIFFERENT'}")
        
        # Compare layer counts
        backbone1 = len(config1.get('backbone', []))
        backbone2 = len(config2.get('backbone', []))
        logging.info(f"- Backbone Layers: {backbone1} vs {backbone2} - {'SAME' if backbone1 == backbone2 else 'DIFFERENT'}")
        
        head1 = len(config1.get('head', []))
        head2 = len(config2.get('head', []))
        logging.info(f"- Head Layers: {head1} vs {head2} - {'SAME' if head1 == head2 else 'DIFFERENT'}")
        
        # Quick check of structure (e.g., first backbone and head layers)
        if config1.get('backbone') and config2.get('backbone'):
            first_layer1 = config1['backbone'][0] if config1['backbone'] else None
            first_layer2 = config2['backbone'][0] if config2['backbone'] else None
            logging.info(f"- First Backbone Layer: {'SAME' if first_layer1 == first_layer2 else 'DIFFERENT'}")
        
        if config1.get('head') and config2.get('head'):
            first_head1 = config1['head'][0] if config1['head'] else None
            first_head2 = config2['head'][0] if config2['head'] else None
            logging.info(f"- First Head Layer: {'SAME' if first_head1 == first_head2 else 'DIFFERENT'}")
        
        # Return a summary of differences
        differences = {
            'depth': depth1 != depth2,
            'width': width1 != width2,
            'activation': act1 != act2,
            'dropout': drop1 != drop2,
            'backbone_layers': backbone1 != backbone2,
            'head_layers': head1 != head2
        }
        
        total_diffs = sum(1 for diff in differences.values() if diff)
        logging.info(f"Summary: Found {total_diffs} significant differences between models")
        
        return differences, total_diffs > 0
        
    except Exception as e:
        logging.error(f"Error comparing model structures: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {}, False

# NEW: Function to check model initialization errors
def check_model_initialization(model_path):
    """
    Test if a model can be successfully initialized
    
    Args:
        model_path: Path to model YAML or PT file
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        model = YOLO(model_path)
        # Try to access some properties to ensure it's loaded properly
        if hasattr(model, 'model'):
            param_count = sum(p.numel() for p in model.model.parameters())
            logging.info(f"Model initialized successfully with {param_count:,} parameters")
            return True, None
        else:
            return False, "Model loaded but incomplete structure"
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Failed to initialize model from {model_path}: {error_msg}")
        return False, error_msg

# NEW: Function to generate a "fingerprint" of model architecture
def get_model_fingerprint(model_path):
    """
    Generate a unique fingerprint for a model architecture to detect duplicates
    
    Args:
        model_path: Path to model YAML
        
    Returns:
        Tuple of (success, fingerprint_str)
    """
    try:
        with open(model_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract key parameters
        depth = config.get('depth_multiple', 0)
        width = config.get('width_multiple', 0)
        act = config.get('activation', 'SiLU')
        drop = config.get('dropout', 0)
        
        # Count layers and channels
        backbone = config.get('backbone', [])
        head = config.get('head', [])
        
        # Count unique modules
        modules = set()
        for layer in backbone + head:
            if isinstance(layer, list) and len(layer) >= 3:
                module_name = layer[2]
                modules.add(module_name)
        
        # Create fingerprint string
        fingerprint = f"D{depth:.2f}_W{width:.2f}_A{act}_DR{drop:.2f}_M{','.join(sorted(modules))}"
        
        return True, fingerprint
    except Exception as e:
        logging.error(f"Failed to generate model fingerprint: {e}")
        return False, None