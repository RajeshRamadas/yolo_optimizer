#!/usr/bin/env python
"""
Complete integration script to run Neural Architecture Search followed by Fine-tuning
End-to-end pipeline for optimizing YOLO models
"""

import os
import logging
import yaml
import argparse
import time
import json
import sys
from datetime import datetime
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yolo_optimization.log"),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLO Neural Architecture Search and Fine-tuning")
    
    # NAS arguments
    parser.add_argument("--nas_config", type=str, default="configs/nas_config.yaml",
                        help="Path to NAS configuration file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset configuration (data.yaml)")
    parser.add_argument("--skip_nas", action="store_true",
                        help="Skip Neural Architecture Search and use existing model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model (when skipping NAS)")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--low_memory", action="store_true",
                       help="Use lower memory settings")
    
    # Fine-tuning arguments
    parser.add_argument("--ft_config", type=str, default="configs/fine_tuning_config.yaml",
                        help="Path to fine-tuning configuration file")
    parser.add_argument("--skip_ft", action="store_true",
                        help="Skip fine-tuning")
    parser.add_argument("--yolov8n_only", action="store_true",
                        help="Focus only on fine-tuning the YOLOv8n pretrained model")
    
    # General arguments
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    
    return parser.parse_args()
    

def create_tensorboard_writer():
    """Create a TensorBoard SummaryWriter if possible"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("runs", f"yolo_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return SummaryWriter(log_dir=log_dir)
    except ImportError:
        logging.warning("Could not import TensorBoard SummaryWriter. Continuing without TensorBoard logging.")
        return None
    except Exception as e:
        logging.warning(f"Error creating TensorBoard writer: {e}")
        return None

def load_and_validate_configs(nas_config_path, ft_config_path):
    """Load and validate both NAS and fine-tuning configurations"""
    nas_config = None
    ft_config = None
    
    # Import config_utils for loading NAS config
    try:
        from config_utils import load_yaml_config
        nas_config = load_yaml_config(nas_config_path)
        logging.info(f"Loaded NAS configuration from {nas_config_path}")
    except ImportError:
        logging.warning("Could not import config_utils module. NAS config will not be loaded.")
    except Exception as e:
        logging.error(f"Error loading NAS configuration: {e}")
    
    # Check if fine-tuning config exists, create if not
    if not os.path.exists(ft_config_path):
        try:
            from fine_tuning import create_fine_tuning_config
            create_fine_tuning_config(ft_config_path)
            logging.info(f"Created fine-tuning configuration at {ft_config_path}")
        except ImportError:
            logging.warning("Could not import fine_tuning module to create config.")
        except Exception as e:
            logging.error(f"Error creating fine-tuning config: {e}")
    
    # Load fine-tuning config
    try:
        with open(ft_config_path, 'r') as f:
            ft_config = yaml.safe_load(f)
        logging.info(f"Loaded fine-tuning configuration from {ft_config_path}")
    except Exception as e:
        logging.error(f"Error loading fine-tuning configuration: {e}")
    
    return nas_config, ft_config

def create_unified_summary(nas_result, ft_results, output_dir):
    """Create a unified summary of the entire optimization process"""
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'nas': {
            'best_model': nas_result.get('best_model_path') if nas_result else None,
            'best_map50': nas_result.get('best_map50', 'N/A') if nas_result else 'N/A',
            'iterations': nas_result.get('iterations', 'N/A') if nas_result else 'N/A'
        },
        'fine_tuning': ft_results.get('summary', {}) if ft_results else {},
        'overall_best_model': None
    }
    
    # Determine the overall best model
    if ft_results and ft_results.get('summary', {}).get('best_result', {}).get('final_model_path'):
        summary['overall_best_model'] = ft_results['summary']['best_result']['final_model_path']
    elif nas_result and nas_result.get('best_model_path'):
        summary['overall_best_model'] = nas_result['best_model_path']
    
    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"complete_optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logging.info(f"Saved unified optimization summary to {summary_path}")
    except Exception as e:
        logging.error(f"Error saving unified summary: {e}")
    
    return summary_path
    

def run_pipeline(args):
    """Run the complete optimization pipeline"""
    # Start timing
    start_time = time.time()
    
    # Special case for YOLOv8n-only mode
    if args.yolov8n_only:
        logging.info("Running in YOLOv8n-only mode (skipping NAS)")
        
        # Import fine-tuning module
        try:
            from fine_tuning import run_pretrained_fine_tuning
        except ImportError as e:
            logging.error(f"Error importing fine-tuning module: {e}")
            return
        
        # Run fine-tuning on YOLOv8n
        results = run_pretrained_fine_tuning(
            dataset_path=args.dataset,
            config_path=args.ft_config
        )
        
        if results and results.get('summary', {}).get('best_result'):
            best = results['summary']['best_result']
            logging.info(f"Best fine-tuning method: {best['method']}")
            logging.info(f"Best fine-tuned model: {best['final_model_path']}")
            logging.info(f"Best mAP@50: {best.get('map50', 'N/A')}")
        
        # Calculate and log total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        # Done with YOLOv8n-only mode
        return
    
    # Create TensorBoard writer
    writer = create_tensorboard_writer()
    
    # Directory for best models
    best_model_dir = args.output_dir if args.output_dir else "best_models"
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Path to the best model (will be set by NAS or args)
    best_model_path = args.model_path
    
    # Load configurations
    nas_config, ft_config = load_and_validate_configs(args.nas_config, args.ft_config)
    
    # Variable to store NAS results
    nas_result = None
    
    # Step 1: Neural Architecture Search (if not skipped)
    if not args.skip_nas:
        logging.info("Starting Neural Architecture Search")
        
        # Import required modules for NAS
        try:
            from model_architecture import (
                create_custom_yolo_config,
                create_complex_yolo_config,
                save_model_config
            )
            from optimization import run_architecture_search
            from config_utils import load_yaml_config, get_directories
        except ImportError as e:
            logging.error(f"Error importing NAS modules: {e}")
            logging.error("Make sure all required modules are in the Python path.")
            return
        
        # Check if NAS config was loaded successfully
        if not nas_config:
            logging.error("NAS configuration not available. Exiting.")
            return
        
        # Get directories
        try:
            from config_utils import get_directories
            directories = get_directories(nas_config)
            if args.output_dir:
                best_model_dir = args.output_dir
            else:
                best_model_dir = directories.get('best_model_dir', 'best_models')
        except Exception as e:
            logging.error(f"Error getting directories from config: {e}")
            if args.output_dir:
                best_model_dir = args.output_dir
            else:
                best_model_dir = "best_models"
        
        # Create args object for NAS
        class NasArgs:
            batch = args.batch
            low_memory = args.low_memory
        
        # Run architecture search
        performance_threshold = nas_config.get('optimization', {}).get('performance_threshold', 0.75)
        try:
            best_model_path = run_architecture_search(
                config=nas_config,
                dataset_path=args.dataset,
                best_model_dir=best_model_dir,
                args=NasArgs(),
                performance_threshold=performance_threshold,
                writer=writer
            )
            
            if best_model_path:
                logging.info(f"Neural Architecture Search completed successfully.")
                logging.info(f"Best model saved at: {best_model_path}")
                nas_result = {
                    'best_model_path': best_model_path,
                    'best_map50': performance_threshold,  # Approximate, will be refined later
                    'iterations': nas_config.get('optimization', {}).get('n_iter', 15)
                }
            else:
                logging.error("Neural Architecture Search failed to produce a valid model.")
                if not args.model_path:  # If no fallback model was provided
                    logging.error("No fallback model available. Exiting.")
                    return
        except Exception as e:
            logging.error(f"Error during NAS: {e}")
            logging.error("Neural Architecture Search failed with an exception.")
            import traceback
            logging.error(traceback.format_exc())
            if not args.model_path:
                logging.error("No fallback model available. Exiting.")
                return
               
# Variable to store fine-tuning results
    ft_results = None
    
    # Step 2: Fine-tuning (if not skipped)
    if not args.skip_ft:
        logging.info("Starting Fine-tuning")
        
        # Check if we have a model to fine-tune
        if not best_model_path and not args.model_path:
            logging.warning("No model available for fine-tuning. Will use YOLOv8n instead.")
            # Default to YOLOv8n in this case
            use_yolov8n = True
        else:
            model_to_finetune = best_model_path if best_model_path else args.model_path
            use_yolov8n = False
            logging.info(f"Using model for fine-tuning: {model_to_finetune}")
        
        # Run fine-tuning
        try:
            if use_yolov8n:
                # Import and run pretrained YOLOv8n fine-tuning
                from fine_tuning import run_pretrained_fine_tuning
                ft_results = run_pretrained_fine_tuning(
                    dataset_path=args.dataset,
                    config_path=args.ft_config
                )
            else:
                # Import normal fine-tuning class
                from fine_tuning import YOLOFineTuner
                fine_tuner = YOLOFineTuner(
                    config_path=args.ft_config,
                    nas_model_path=model_to_finetune
                )
                ft_results = fine_tuner.run_all_fine_tuning(args.dataset)
            
            # Log fine-tuning results
            if ft_results and ft_results.get('summary', {}).get('best_result'):
                best = ft_results['summary']['best_result']
                logging.info("Fine-tuning completed successfully")
                logging.info(f"Best fine-tuned model: {best['model']} with {best['method']}")
                logging.info(f"Best mAP@50: {best.get('map50', 'N/A')}")
                logging.info(f"Best model saved at: {best.get('final_model_path', 'unknown')}")
            else:
                logging.warning("Fine-tuning completed but no best model was identified")
            
        except Exception as e:
            logging.error(f"Error during fine-tuning: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Create unified summary
    if nas_result or ft_results:
        summary_path = create_unified_summary(nas_result, ft_results, best_model_dir)
        logging.info(f"Saved unified optimization summary to {summary_path}")
    
    # Log total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Display settings
    logging.info("Starting YOLO optimization pipeline")
    logging.info(f"Dataset: {args.dataset}")
    
    if args.yolov8n_only:
        logging.info("Mode: YOLOv8n fine-tuning only")
    else:
        logging.info(f"NAS config: {args.nas_config}")
        logging.info(f"Fine-tuning config: {args.ft_config}")
        logging.info(f"Skip NAS: {args.skip_nas}")
        logging.info(f"Skip fine-tuning: {args.skip_ft}")
        if args.model_path:
            logging.info(f"Using existing model: {args.model_path}")
    
    # Run the pipeline
    run_pipeline(args)
    
    logging.info("Pipeline completed")

if __name__ == "__main__":
    main()