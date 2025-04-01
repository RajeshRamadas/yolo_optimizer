"""
Fine-tuning and Transfer Learning for YOLOv8 models
after Neural Architecture Search (NAS) or using pretrained models
"""

import os
import logging
import torch
import yaml
import json
import time
import shutil
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path

def ensure_pretrained_model(model_path="yolov8n.pt"):
    """
    Ensure the pretrained YOLOv8n model is available
    
    Args:
        model_path: Path to model (default is yolov8n.pt)
        
    Returns:
        Path to the model if successful, None otherwise
    """
    if not os.path.exists(model_path):
        try:
            # Download using ultralytics functionality
            from ultralytics import YOLO
            model = YOLO(model_path)  # This will download if not found
            logging.info(f"Downloaded pretrained model: {model_path}")
            return model_path
        except Exception as e:
            logging.error(f"Error ensuring pretrained model: {e}")
            return None
    return model_path

class YOLOFineTuner:
    """Class to handle fine-tuning and transfer learning for YOLOv8 models."""
    
    def __init__(self, config_path="fine_tuning_config.yaml", nas_model_path=None):
        """
        Initialize the fine-tuner.
        
        Args:
            config_path: Path to configuration file
            nas_model_path: Path to the best model from NAS (if available)
        """
        self.config_path = config_path
        self.nas_model_path = nas_model_path
        self.config = self._load_config()
        self.results_dir = self._setup_directories()
        self.models = None  # Will be populated by _get_available_models()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            logging.error("Please ensure the configuration file exists and is valid YAML.")
            raise
    
    def _setup_directories(self):
        """Set up directory structure for fine-tuning."""
        dirs = self.config.get('directories', {})
        base_dir = dirs.get('base_dir', 'yolo_fine_tuning')
        models_dir = os.path.join(base_dir, dirs.get('models_dir', 'models'))
        results_dir = os.path.join(base_dir, dirs.get('results_dir', 'results'))
        
        # Create directories
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        logging.info(f"Set up directories: {base_dir}, {models_dir}, {results_dir}")
        return results_dir

    def _get_available_models(self):
        """Get list of available models for fine-tuning."""
        if self.models is not None:
            return self.models
            
        models = []
        
        # Add NAS model if available
        if self.nas_model_path and os.path.exists(self.nas_model_path):
            models.append({
                'path': self.nas_model_path,
                'name': 'nas_best_model',
                'description': 'Best model from Neural Architecture Search',
                'type': 'nas'
            })
        
        # Add pre-trained models from configuration
        base_models = self.config.get('base_models', [])
        for model in base_models:
            model_name = model.get('name')
            if model_name:
                # Ensure the model exists/can be downloaded
                model_path = ensure_pretrained_model(model_name)
                if model_path:
                    models.append({
                        'path': model_path,
                        'name': os.path.splitext(model_name)[0],
                        'description': model.get('description', model_name),
                        'type': 'pretrained'
                    })
        
        self.models = models
        return models
    
    def freeze_layers(self, model, freeze_setting):
        """
        Freeze layers in the model based on the specified setting.
        
        Args:
            model: YOLO model
            freeze_setting: Can be 'backbone', a negative integer specifying 
                           how many layers from the end to keep trainable,
                           or None to train all layers
        """
        # Access the PyTorch model
        if not hasattr(model, 'model') or not hasattr(model.model, 'model'):
            logging.warning("Model structure not as expected, skipping layer freezing")
            return
        
        pt_model = model.model.model
        
        # Set all parameters to require gradients by default
        for param in pt_model.parameters():
            param.requires_grad = True
        
        if freeze_setting is None:
            logging.info("Training all layers (no freezing)")
            return
        
        elif freeze_setting == 'backbone':
            # Find the backbone layers - typically the first ~10 layers in YOLO models
            backbone_end = len(pt_model) // 2  # Approximate backbone size
            
            # Freeze backbone layers
            for i, layer in enumerate(pt_model):
                if i < backbone_end:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            logging.info(f"Froze backbone layers (0-{backbone_end-1})")
        
        elif isinstance(freeze_setting, int) and freeze_setting < 0:
            # Freeze all layers except the last |freeze_setting| layers
            trainable_start = len(pt_model) + freeze_setting
            
            for i, layer in enumerate(pt_model):
                if i < trainable_start:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            logging.info(f"Froze layers 0-{trainable_start-1}, keeping {-freeze_setting} layers trainable")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in pt_model.parameters())
        logging.info(f"Trainable parameters: {trainable_params:,} of {total_params:,} "
                    f"({trainable_params/total_params:.2%})")

    def fine_tune_model(self, model_info, method, dataset_path):
        """
        Fine-tune a model with the specified method.
        
        Args:
            model_info: Dictionary with model information
            method: Dictionary with fine-tuning method parameters
            dataset_path: Path to dataset
            
        Returns:
            Dictionary with results and paths to the fine-tuned model
        """
        # Create a unique name for this fine-tuning run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_info['name']}_{method['name']}_{timestamp}"
        
        # Create output directory
        output_dir = os.path.join(self.results_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        try:
            model = YOLO(model_info['path'])
            logging.info(f"Loaded model from {model_info['path']}")
        except Exception as e:
            logging.error(f"Error loading model {model_info['path']}: {e}")
            return {'success': False, 'error': str(e)}
        
        # Get training parameters
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 16)
        patience = training_config.get('patience', 5)
        optimizer = training_config.get('optimizer', 'AdamW')
        device = training_config.get('device', None)  # None = auto-detect
        image_size = training_config.get('image_size', 640)
        save_period = training_config.get('save_period', 5)
        
        # Set up results dictionary
        results = {
            'model': model_info['name'],
            'method': method['name'],
            'timestamp': timestamp,
            'phases': []
        }
        
        # Check if this is a progressive fine-tuning method
        if method.get('progressive', False):
            phases = method.get('phases', [])
            
            for i, phase in enumerate(phases):
                logging.info(f"Starting progressive fine-tuning phase {i+1}/{len(phases)}")
                
                # Apply freezing for this phase
                freeze_setting = phase.get('freeze_layers')
                self.freeze_layers(model, freeze_setting)
                
                # Set up training arguments
                train_args = {
                    'data': dataset_path,
                    'epochs': phase.get('epochs', 10),
                    'patience': patience,
                    'batch': batch_size,
                    'imgsz': image_size,
                    'optimizer': optimizer,
                    'lr0': phase.get('learning_rate', 0.001),
                    'device': device,
                    'project': output_dir,
                    'name': f"phase_{i+1}",
                    'exist_ok': True,
                    'save_period': save_period
                }
                
                # Train the model for this phase
                try:
                    phase_results = model.train(**train_args)
                    
                    # Record phase results
                    phase_info = {
                        'phase': i+1,
                        'freeze_setting': str(freeze_setting),
                        'epochs': phase.get('epochs', 10),
                        'learning_rate': phase.get('learning_rate', 0.001),
                        'metrics': self._extract_metrics(phase_results)
                    }
                    results['phases'].append(phase_info)
                    
                    logging.info(f"Completed phase {i+1} training")
                    
                except Exception as e:
                    logging.error(f"Error in phase {i+1} training: {e}")
                    results['phases'].append({
                        'phase': i+1,
                        'error': str(e)
                    })
                    break
            
            # Final evaluation after all phases
            try:
                eval_results = model.val(data=dataset_path)
                results['final_metrics'] = self._extract_metrics(eval_results)
            except Exception as e:
                logging.error(f"Error in final evaluation: {e}")
                results['final_metrics'] = {'error': str(e)}
            
        else:
            # Single-phase fine-tuning
            
            # Apply freezing
            freeze_setting = method.get('freeze_layers')
            self.freeze_layers(model, freeze_setting)
            
            # Set up training arguments
            train_args = {
                'data': dataset_path,
                'epochs': method.get('epochs', 20),
                'patience': patience,
                'batch': batch_size,
                'imgsz': image_size,
                'optimizer': optimizer,
                'lr0': method.get('learning_rate', 0.001),
                'device': device,
                'project': output_dir,
                'name': "training",
                'exist_ok': True,
                'save_period': save_period
            }
            
            # Train the model
            try:
                training_results = model.train(**train_args)
                results['metrics'] = self._extract_metrics(training_results)
                logging.info(f"Completed single-phase training")
            except Exception as e:
                logging.error(f"Error in training: {e}")
                results['metrics'] = {'error': str(e)}
        
        # Save final model
        try:
            final_model_path = os.path.join(output_dir, f"{run_name}_final.pt")
            model.model.save(final_model_path)
            results['final_model_path'] = final_model_path
            logging.info(f"Saved final model to {final_model_path}")
        except Exception as e:
            logging.error(f"Error saving final model: {e}")
            results['save_error'] = str(e)
        
        # Save results summary
        summary_path = os.path.join(output_dir, "results_summary.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info(f"Saved results summary to {summary_path}")
        except Exception as e:
            logging.error(f"Error saving results summary: {e}")
        
        # Return results dictionary with paths
        results['output_dir'] = output_dir
        results['summary_path'] = summary_path
        results['success'] = True
        
        return results

    def _extract_metrics(self, training_results):
        """Extract metrics from training results object."""
        metrics = {}
        
        try:
            # Different versions of YOLOv8 have different result structures
            if hasattr(training_results, 'box'):
                # Newer versions structure
                for metric in ['map50', 'map', 'p', 'r']:
                    if hasattr(training_results.box, metric):
                        value = getattr(training_results.box, metric)
                        # Convert to Python float if it's a tensor or numpy array
                        if hasattr(value, 'item'):
                            value = value.item()
                        elif isinstance(value, np.ndarray):
                            value = float(value.mean())
                        metrics[metric] = value
            else:
                # Older versions
                for metric in ['map50', 'map', 'precision', 'recall']:
                    if hasattr(training_results, metric):
                        value = getattr(training_results, metric)
                        # Convert to Python float if it's a tensor or numpy array
                        if hasattr(value, 'item'):
                            value = value.item()
                        elif isinstance(value, np.ndarray):
                            value = float(value.mean())
                        metrics[metric] = value
        
        except Exception as e:
            logging.warning(f"Error extracting metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def run_all_fine_tuning(self, dataset_path=None):
        """
        Run all fine-tuning experiments defined in the configuration.
        
        Args:
            dataset_path: Path to dataset (overrides config)
            
        Returns:
            Dictionary with all results
        """
        # Get dataset path
        if not dataset_path:
            dataset_path = self.config.get('dataset', {}).get('path', 'data.yaml')
        
        logging.info(f"Starting fine-tuning experiments with dataset: {dataset_path}")
        
        # Get available models
        models = self._get_available_models()
        logging.info(f"Found {len(models)} model(s) for fine-tuning")
        
        # Get fine-tuning methods
        methods = self.config.get('fine_tuning', {}).get('methods', [])
        logging.info(f"Found {len(methods)} fine-tuning method(s)")
        
        # Track all results
        all_results = []
        
        # Run fine-tuning for each model and method
        for model_info in models:
            logging.info(f"Processing model: {model_info['name']} ({model_info['description']})")
            
            for method in methods:
                method_name = method.get('name', 'unknown')
                logging.info(f"Applying fine-tuning method: {method_name}")
                
                # Fine-tune the model
                result = self.fine_tune_model(model_info, method, dataset_path)
                all_results.append(result)
                
                # Log success or failure
                if result.get('success', False):
                    logging.info(f"Successfully fine-tuned {model_info['name']} with {method_name}")
                else:
                    logging.warning(f"Failed to fine-tune {model_info['name']} with {method_name}")
        
        # Create a summary of all results
        summary = self._create_comparison_summary(all_results)
        
        return {
            'detailed_results': all_results,
            'summary': summary
        }

    def _create_comparison_summary(self, all_results):
        """Create a summary comparing all fine-tuning results."""
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models_count': len(set(r.get('model') for r in all_results if 'model' in r)),
            'methods_count': len(set(r.get('method') for r in all_results if 'method' in r)),
            'experiments_count': len(all_results),
            'successful_experiments': sum(1 for r in all_results if r.get('success', False)),
            'best_result': None,
            'results_table': []
        }
        
        # Extract metrics for comparison
        best_map50 = 0
        best_experiment = None
        
        for result in all_results:
            if not result.get('success', False):
                continue
            
            # Get model and method info
            model_name = result.get('model', 'unknown')
            method_name = result.get('method', 'unknown')
            
            # Get metrics - either from final_metrics or metrics
            metrics = result.get('final_metrics', result.get('metrics', {}))
            
            # Check for best result based on mAP@50
            map50 = metrics.get('map50', 0)
            if map50 > best_map50:
                best_map50 = map50
                best_experiment = {
                    'model': model_name,
                    'method': method_name,
                    'map50': map50,
                    'output_dir': result.get('output_dir'),
                    'final_model_path': result.get('final_model_path')
                }
            
            # Add to results table
            summary['results_table'].append({
                'model': model_name,
                'method': method_name,
                'map50': metrics.get('map50', 'N/A'),
                'map': metrics.get('map', 'N/A'),
                'precision': metrics.get('p', metrics.get('precision', 'N/A')),
                'recall': metrics.get('r', metrics.get('recall', 'N/A')),
                'final_model_path': result.get('final_model_path', 'N/A')
            })
        
        # Set best result
        summary['best_result'] = best_experiment
        
        # Sort results table by mAP@50
        summary['results_table'].sort(key=lambda x: 
            x['map50'] if isinstance(x['map50'], (int, float)) else 0, 
            reverse=True)
        
        # Save the summary
        summary_path = os.path.join(self.results_dir, f"fine_tuning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            logging.info(f"Saved fine-tuning comparison summary to {summary_path}")
            summary['summary_path'] = summary_path
        except Exception as e:
            logging.error(f"Error saving fine-tuning summary: {e}")
        
        return summary

# Module-level functions (outside the class)
def run_pretrained_fine_tuning(dataset_path, config_path="fine_tuning_config.yaml"):
    """
    Run fine-tuning specifically on YOLOv8n pretrained model
    
    Args:
        dataset_path: Path to dataset
        config_path: Path to fine-tuning configuration file
        
    Returns:
        Results dictionary
    """
    # Ensure pretrained model exists
    model_path = ensure_pretrained_model("yolov8n.pt")
    if not model_path:
        logging.error("Could not find or download YOLOv8n pretrained model")
        return None
    
    # Initialize fine-tuner with pretrained model
    fine_tuner = YOLOFineTuner(config_path=config_path, nas_model_path=None)
    
    # Override available models to only use YOLOv8n
    fine_tuner.models = [{
        'path': model_path,
        'name': 'yolov8n',
        'description': 'YOLOv8 Nano Pretrained Model',
        'type': 'pretrained'
    }]
    
    # Run fine-tuning
    results = fine_tuner.run_all_fine_tuning(dataset_path)
    return results

def create_fine_tuning_config(output_path="fine_tuning_config.yaml"):
    """
    Create a default fine-tuning configuration file.
    
    Args:
        output_path: Path to save the configuration file
        
    Returns:
        Path to the created configuration file
    """
    # Default configuration for YOLOv8n fine-tuning
    config = {
        'directories': {
            'base_dir': 'yolo_fine_tuning',
            'models_dir': 'models',
            'results_dir': 'results',
        },
        'dataset': {
            'path': 'data.yaml',
        },
        'base_models': [
            {'name': 'yolov8n.pt', 'description': 'YOLOv8 Nano Pretrained Model'},
        ],
        'fine_tuning': {
            'methods': [
                # Method 1: Fine-tune all layers
                {
                    'name': 'full_finetune',
                    'description': 'Fine-tune all layers of YOLOv8n',
                    'freeze_layers': None,
                    'learning_rate': 0.001,
                    'epochs': 20,
                },
                # Method 2: Freeze backbone and train only detection head
                {
                    'name': 'head_only',
                    'description': 'Transfer learning - train only detection head',
                    'freeze_layers': 'backbone',
                    'learning_rate': 0.01,
                    'epochs': 15,
                },
                # Method 3: Progressive unfreezing (gradual unfreezing)
                {
                    'name': 'progressive_unfreezing',
                    'description': 'Progressive transfer learning strategy',
                    'progressive': True,
                    'phases': [
                        {'freeze_layers': 'backbone', 'epochs': 5, 'learning_rate': 0.01},
                        {'freeze_layers': -10, 'epochs': 10, 'learning_rate': 0.001},
                        {'freeze_layers': None, 'epochs': 10, 'learning_rate': 0.0001},
                    ]
                },
                # Method 4: Domain adaptation (focus on final layers with higher learning rate)
                {
                    'name': 'domain_adaptation',
                    'description': 'Adapt YOLOv8n to new domain',
                    'freeze_layers': -5,
                    'learning_rate': 0.02,
                    'epochs': 10,
                }
            ]
        },
        'training': {
            'batch_size': 16,
            'patience': 5,
            'optimizer': 'AdamW',
            'device': None,  # Auto-detect
            'image_size': 640,
            'save_period': 5,
        },
    }
    
    # Save the configuration
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Created fine-tuning configuration at {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error creating fine-tuning configuration: {e}")
        return None

def main():
    """Main function to run fine-tuning."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 models")
    parser.add_argument("--config", type=str, default="fine_tuning_config.yaml",
                        help="Path to fine-tuning configuration file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model to fine-tune")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset (overrides config)")
    parser.add_argument("--create_config", action="store_true",
                        help="Create default configuration file and exit")
    parser.add_argument("--yolov8n_only", action="store_true",
                        help="Focus only on fine-tuning the YOLOv8n pretrained model")
    
    args = parser.parse_args()
    
    # Create configuration file if requested
    if args.create_config:
        create_fine_tuning_config(args.config)
        return
    
    # Special case for YOLOv8n-only mode
    if args.yolov8n_only:
        logging.info("Running in YOLOv8n-only mode")
        results = run_pretrained_fine_tuning(args.dataset, args.config)
        
        if results and results.get('summary', {}).get('best_result'):
            best = results['summary']['best_result']
            logging.info("Fine-tuning completed successfully")
            logging.info(f"Best fine-tuning method: {best['method']}")
            logging.info(f"Best mAP@50: {best.get('map50', 'N/A')}")
            logging.info(f"Best model saved at: {best.get('final_model_path', 'unknown')}")
        else:
            logging.warning("Fine-tuning completed but no best model was identified")
            
        return
    
    # Regular fine-tuning with specified model (NAS or custom)
    fine_tuner = YOLOFineTuner(args.config, args.model)
    
    # Run all fine-tuning experiments
    results = fine_tuner.run_all_fine_tuning(args.dataset)
    
    # Print summary
    summary = results['summary']
    logging.info("Fine-tuning experiments completed")
    logging.info(f"Ran {summary['experiments_count']} experiments, {summary['successful_experiments']} successful")
    
    if summary['best_result']:
        best = summary['best_result']
        logging.info(f"Best result: {best['model']} with {best['method']}, mAP@50: {best['map50']:.4f}")
        logging.info(f"Best model saved at: {best['final_model_path']}")
    
    logging.info(f"Full summary saved to: {summary.get('summary_path', 'unknown')}")

if __name__ == "__main__":
    main()