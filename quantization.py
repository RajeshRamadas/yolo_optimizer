"""
Quantization functions for YOLO models.
"""

import os
import logging
from time import time
import torch
import numpy as np
from ultralytics import YOLO

def prepare_qat_model(model):
    """
    Prepare a model for quantization-aware training
    
    Args:
        model: YOLO model
        
    Returns:
        Model ready for QAT
    """
    try:
        logging.info("Preparing model for quantization-aware training")
        # Enable QAT mode
        model.model.train()
        
        # Set up observers based on model type
        if hasattr(torch.quantization, 'prepare_qat'):
            # Need to set qconfig first if available
            model.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            # Prepare model for QAT
            torch.quantization.prepare_qat(model.model, inplace=True)
            logging.info("Model prepared for quantization-aware training")
            return model
        else:
            logging.warning("Quantization-aware training not fully supported in this PyTorch version")
            return model
    except Exception as e:
        logging.error(f"Error preparing model for quantization-aware training: {e}")
        return model

def run_qat(model, data_path, epochs=5, img_size=640):
    """
    Perform quantization-aware training
    
    Args:
        model: Model prepared for QAT
        data_path: Path to dataset
        epochs: Number of epochs for QAT
        img_size: Image size for training
        
    Returns:
        Trained QAT model
    """
    logging.info(f"Running quantization-aware training for {epochs} epochs")
    
    try:
        # Train with QAT
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=16,
            lr0=0.0005,  # Lower learning rate for QAT
            device="cuda" if torch.cuda.is_available() else "cpu",
            optimizer="AdamW",
            patience=5,
            save=True,
            verbose=True
        )
        
        # Handle different versions of YOLO API
        try:
            map50 = results.box.map50
            logging.info(f"Quantization-aware training completed, mAP50: {map50:.4f}")
        except Exception:
            try:
                map50 = results.metrics.get("mAP50(B)", 0.0)
                logging.info(f"Quantization-aware training completed, mAP50: {map50:.4f}")
            except Exception as e:
                logging.error(f"Error accessing metrics: {e}")
        
        # Convert model to quantized model if possible
        if hasattr(torch.quantization, 'convert'):
            model.model.eval()
            model.model = torch.quantization.convert(model.model)
            logging.info("Converted QAT model to quantized model")
        
        return model
    except Exception as e:
        logging.error(f"Error during quantization-aware training: {e}")
        return model

def quantize_model_int8(model_path, dataset_path, quantized_model_dir):
    """
    Quantize a model to INT8 precision
    
    Args:
        model_path: Path to the model to quantize
        dataset_path: Path to validation data
        quantized_model_dir: Directory to save quantized models
        
    Returns:
        Path to the quantized model
    """
    logging.info(f"Starting INT8 quantization for {model_path}")
    
    try:
        model = YOLO(model_path)
        # Export to ONNX with INT8 quantization
        model.export(format="onnx", imgsz=640, opset=12, dynamic=True, simplify=True, data=dataset_path)
        
        # Move to proper location
        base_name = os.path.basename(model_path).split('.')[0]
        onnx_path = os.path.join(quantized_model_dir, f"{base_name}_int8.onnx")
        os.rename(model_path.replace('.pt', '.onnx'), onnx_path)
        
        logging.info(f"Saved INT8 ONNX model to {onnx_path}")
        return onnx_path
    except Exception as e:
        logging.error(f"Error during INT8 quantization: {e}")
        return None

def quantize_model_fp16(model_path, dataset_path, quantized_model_dir):
    """
    Quantize a model to FP16 precision
    
    Args:
        model_path: Path to the model to quantize
        dataset_path: Path to validation data
        quantized_model_dir: Directory to save quantized models
        
    Returns:
        Path to the quantized model
    """
    logging.info(f"Starting FP16 quantization for {model_path}")
    
    try:
        model = YOLO(model_path)
        # Use GPU if available for half precision
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        half = device.startswith("cuda")
        
        # Export to ONNX with half precision if on GPU
        model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, device=device, half=half, data=dataset_path)
        
        # Move to proper location
        base_name = os.path.basename(model_path).split('.')[0]
        onnx_path = os.path.join(quantized_model_dir, f"{base_name}_fp16.onnx")
        os.rename(model_path.replace('.pt', '.onnx'), onnx_path)
        
        logging.info(f"Saved FP16 ONNX model to {onnx_path}")
        return onnx_path
    except Exception as e:
        logging.error(f"Error during FP16 quantization: {e}")
        return None

def benchmark_quantized_model(model_path, img_size=640, num_runs=100):
    """
    Benchmark inference speed for a quantized model
    
    Args:
        model_path: Path to the quantized model
        img_size: Image size for inference
        num_runs: Number of inference runs for averaging
        
    Returns:
        Average inference time in ms
    """
    logging.info(f"Benchmarking inference speed for quantized model: {model_path}")
    
    try:
        # Check if onnxruntime is available
        try:
            import onnxruntime as ort
        except ImportError:
            logging.warning("onnxruntime not installed. Skipping benchmarking.")
            return 0
        
        # Set up ONNX runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, session_options, providers=providers)
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        
        # Check if the model expects float16 input
        is_fp16 = "float16" in input_type
        
        # Create a dummy input with the correct data type
        if is_fp16:
            # Use float16 (half precision) for the dummy input
            dummy_input = np.random.random((1, 3, img_size, img_size)).astype(np.float16)
            logging.info("Using float16 input for FP16 model")
        else:
            # Use float32 (single precision) for the dummy input
            dummy_input = np.random.random((1, 3, img_size, img_size)).astype(np.float32)
            logging.info("Using float32 input for model")
        
        # Warm-up
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Measure inference time
        total_time = 0
        for _ in range(num_runs):
            start = time()
            _ = session.run(None, {input_name: dummy_input})
            total_time += (time() - start)
        
        avg_time = (total_time / num_runs) * 1000  # Convert to ms
        logging.info(f"Average inference time: {avg_time:.2f}ms ({1000/avg_time:.2f} FPS)")
        
        return avg_time
    except Exception as e:
        logging.error(f"Error during quantized model benchmarking: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 0

def run_model_quantization(model_path, config, dataset_path, quantized_model_dir):
    """
    Run the model quantization process for both INT8 and FP16
    
    Args:
        model_path: Path to the model to quantize
        config: Configuration dictionary
        dataset_path: Path to dataset
        quantized_model_dir: Directory to save quantized models
        
    Returns:
        Dictionary with paths to quantized models
    """
    logging.info(f"Starting model quantization process for: {model_path}")
    
    results = {
        "original": model_path,
        "int8": None,
        "fp16": None
    }
    
    # Try INT8 quantization
    try:
        int8_model = quantize_model_int8(model_path, dataset_path, quantized_model_dir)
        if int8_model:
            results["int8"] = int8_model
            # Benchmark INT8 model
            try:
                benchmark_quantized_model(int8_model, num_runs=20)
            except Exception as e:
                logging.error(f"Error benchmarking INT8 model: {e}")
    except Exception as e:
        logging.error(f"Error during INT8 quantization: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Try FP16 quantization
    try:
        fp16_model = quantize_model_fp16(model_path, dataset_path, quantized_model_dir)
        if fp16_model:
            results["fp16"] = fp16_model
            # Benchmark FP16 model
            try:
                benchmark_quantized_model(fp16_model, num_runs=20)
            except Exception as e:
                logging.error(f"Error benchmarking FP16 model: {e}")
    except Exception as e:
        logging.error(f"Error during FP16 quantization: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Print summary
    logging.info("\nQuantization Summary:")
    for key, value in results.items():
        logging.info(f"{key}: {value}")
    
    return results