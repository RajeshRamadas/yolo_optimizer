"""
Quantization functions for YOLO models.
"""

import os
import logging
from time import time
import torch
import numpy as np
from ultralytics import YOLO

def is_torchscript_model(model_path):
    """
    Check if the model is a TorchScript model
    
    Args:
        model_path: Path to the model
        
    Returns:
        True if it's a TorchScript model, False otherwise
    """
    try:
        # Try to load as TorchScript
        model = torch.jit.load(model_path)
        return True
    except:
        return False

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
        # Check if this is a TorchScript model
        if is_torchscript_model(model_path):
            logging.info("Detected TorchScript model, using direct ONNX export")
            
            # For TorchScript models, use torch.jit.load and export directly to ONNX
            # Explicitly load model to CPU to avoid device mismatch issues
            model = torch.jit.load(model_path, map_location="cpu")
            
            # Make sure the output directory exists
            os.makedirs(quantized_model_dir, exist_ok=True)
            
            # Define input parameters and export - use CPU for export
            dummy_input = torch.randn(1, 3, 640, 640, device="cpu")
            base_name = os.path.basename(model_path).split('.')[0]
            onnx_path = os.path.join(quantized_model_dir, f"{base_name}_int8.onnx")
            
            # Export to ONNX with error handling
            try:
                torch.onnx.export(
                    model, 
                    dummy_input, 
                    onnx_path,
                    opset_version=12,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                logging.info(f"Saved INT8 ONNX model to {onnx_path}")
                
                # Try to apply INT8 quantization to the ONNX model if onnxruntime is available
                try:
                    import onnx
                    import onnxruntime as ort
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    
                    # Apply INT8 quantization
                    int8_path = os.path.join(quantized_model_dir, f"{base_name}_int8_quant.onnx")
                    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
                    logging.info(f"Applied INT8 quantization to ONNX model: {int8_path}")
                    return int8_path
                except ImportError:
                    logging.warning("onnxruntime-tools not available for INT8 quantization. Returning ONNX model.")
                    return onnx_path
                except Exception as e:
                    logging.error(f"Error during INT8 quantization of ONNX model: {e}")
                    return onnx_path
            except Exception as e:
                logging.error(f"ONNX export failed: {e}")
                logging.info("Trying alternative export approach...")
                
                # Try alternative export approach
                try:
                    from ultralytics.engine.exporter import Exporter
                    from ultralytics import YOLO
                    
                    # Load original model (not TorchScript)
                    original_model = YOLO(model_path.replace('.torchscript', '.pt'))
                    exporter = Exporter()
                    f = exporter(
                        model=original_model,
                        format='onnx',
                        imgsz=640, 
                        device='cpu'
                    )
                    if f and os.path.exists(f):
                        import shutil
                        shutil.copy(f, onnx_path)
                        logging.info(f"Saved ONNX model to {onnx_path} using alternative method")
                        
                        # Try to apply INT8 quantization
                        try:
                            import onnx
                            import onnxruntime as ort
                            from onnxruntime.quantization import quantize_dynamic, QuantType
                            
                            int8_path = os.path.join(quantized_model_dir, f"{base_name}_int8_quant.onnx")
                            quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
                            logging.info(f"Applied INT8 quantization to ONNX model: {int8_path}")
                            return int8_path
                        except ImportError:
                            return onnx_path
                        except Exception as e:
                            logging.error(f"Error during INT8 quantization: {e}")
                            return onnx_path
                except Exception as e2:
                    logging.error(f"Alternative export also failed: {e2}")
                    return None
        else:
            # If it's a regular YOLO model, use the standard approach
            model = YOLO(model_path)
            
            # Make sure the output directory exists
            os.makedirs(quantized_model_dir, exist_ok=True)
            
            # Force CPU for export to avoid device mismatch issues
            device = "cpu"
            
            # Export to ONNX
            try:
                export_path = model.export(
                    format="onnx", 
                    imgsz=640, 
                    dynamic=True, 
                    simplify=True, 
                    device=device,
                    data=dataset_path
                )
                
                # Move to proper location
                base_name = os.path.basename(model_path).split('.')[0]
                onnx_path = os.path.join(quantized_model_dir, f"{base_name}_int8.onnx")
                
                if os.path.exists(export_path):
                    # Use shutil.copy instead of os.rename to avoid cross-device link errors
                    import shutil
                    shutil.copy(export_path, onnx_path)
                    os.remove(export_path)  # Remove the original file
                    logging.info(f"Saved ONNX model to {onnx_path}")
                    
                    # Try to apply INT8 quantization
                    try:
                        import onnx
                        import onnxruntime as ort
                        from onnxruntime.quantization import quantize_dynamic, QuantType
                        
                        int8_path = os.path.join(quantized_model_dir, f"{base_name}_int8_quant.onnx")
                        quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
                        logging.info(f"Applied INT8 quantization to ONNX model: {int8_path}")
                        return int8_path
                    except ImportError:
                        logging.warning("onnxruntime-tools not available for INT8 quantization. Returning ONNX model.")
                        return onnx_path
                    except Exception as e:
                        logging.error(f"Error during INT8 quantization: {e}")
                        return onnx_path
                else:
                    logging.error(f"Exported file not found: {export_path}")
                    return None
            except Exception as e:
                logging.error(f"Error during ONNX export: {e}")
                
                # Try direct PyTorch export
                try:
                    base_name = os.path.basename(model_path).split('.')[0]
                    onnx_path = os.path.join(quantized_model_dir, f"{base_name}_int8.onnx")
                    
                    dummy_input = torch.randn(1, 3, 640, 640, device="cpu")
                    torch.onnx.export(
                        model.model, 
                        dummy_input, 
                        onnx_path,
                        opset_version=12,
                        input_names=['input'],
                        output_names=['output']
                    )
                    logging.info(f"Saved ONNX model to {onnx_path} using direct PyTorch export")
                    return onnx_path
                except Exception as e2:
                    logging.error(f"Alternative export also failed: {e2}")
                    return None
    except Exception as e:
        logging.error(f"Error during INT8 quantization: {e}")
        import traceback
        logging.error(traceback.format_exc())
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
        # Check if this is a TorchScript model
        if is_torchscript_model(model_path):
            logging.info("Detected TorchScript model, using direct ONNX export with half precision")
            
            # For TorchScript models, use torch.jit.load and export directly to ONNX
            # Explicitly load model to CPU first to avoid device mismatch issues
            model = torch.jit.load(model_path, map_location="cpu")
            
            # Make sure the output directory exists
            os.makedirs(quantized_model_dir, exist_ok=True)
            
            # Use CPU for export to avoid device mismatch issues
            device = torch.device("cpu")
            model = model.to(device)
            
            # Use half precision
            try:
                model = model.half()
                dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float16, device=device)
            except Exception as e:
                logging.warning(f"Could not convert to half precision: {e}. Using float32 instead.")
                dummy_input = torch.randn(1, 3, 640, 640, device=device)
                
            base_name = os.path.basename(model_path).split('.')[0]
            onnx_path = os.path.join(quantized_model_dir, f"{base_name}_fp16.onnx")
            
            # Export to ONNX with error handling
            try:
                torch.onnx.export(
                    model, 
                    dummy_input, 
                    onnx_path,
                    opset_version=12,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                logging.info(f"Saved FP16 ONNX model to {onnx_path}")
                return onnx_path
            except Exception as e:
                logging.error(f"ONNX export failed: {e}")
                logging.info("Trying alternative export approach...")
                
                # Try alternative export approach using Ultralytics export
                try:
                    from ultralytics.engine.exporter import Exporter
                    from ultralytics import YOLO
                    
                    # Load original model (not TorchScript)
                    original_model = YOLO(model_path.replace('.torchscript', '.pt'))
                    exporter = Exporter()
                    f = exporter(
                        model=original_model,
                        format='onnx',
                        imgsz=640, 
                        half=True,
                        device='cpu'
                    )
                    if f and os.path.exists(f):
                        import shutil
                        shutil.copy(f, onnx_path)
                        logging.info(f"Saved FP16 ONNX model to {onnx_path} using alternative method")
                        return onnx_path
                except Exception as e2:
                    logging.error(f"Alternative export also failed: {e2}")
                    return None
        else:
            # If it's a regular YOLO model, use the standard approach
            model = YOLO(model_path)
            
            # Make sure the output directory exists
            os.makedirs(quantized_model_dir, exist_ok=True)
            
            # Force CPU for export to avoid device mismatch issues
            device = "cpu"
            half = True  # Still use half precision
            
            # Export to ONNX with half precision
            try:
                export_path = model.export(
                    format="onnx", 
                    imgsz=640, 
                    dynamic=True, 
                    simplify=True, 
                    device=device, 
                    half=half,
                    data=dataset_path
                )
                
                # Move to proper location
                base_name = os.path.basename(model_path).split('.')[0]
                onnx_path = os.path.join(quantized_model_dir, f"{base_name}_fp16.onnx")
                
                if os.path.exists(export_path):
                    # Use shutil.copy instead of os.rename to avoid cross-device link errors
                    import shutil
                    shutil.copy(export_path, onnx_path)
                    os.remove(export_path)  # Remove the original file
                    logging.info(f"Saved FP16 ONNX model to {onnx_path}")
                    return onnx_path
                else:
                    logging.error(f"Exported file not found: {export_path}")
                    return None
            except Exception as e:
                logging.error(f"Error during ONNX export: {e}")
                
                # Try alternative approach
                try:
                    # Try direct PyTorch export
                    dummy_input = torch.randn(1, 3, 640, 640)
                    torch.onnx.export(
                        model.model, 
                        dummy_input, 
                        onnx_path,
                        opset_version=12,
                        input_names=['input'],
                        output_names=['output']
                    )
                    logging.info(f"Saved FP16 ONNX model to {onnx_path} using direct PyTorch export")
                    return onnx_path
                except Exception as e2:
                    logging.error(f"Alternative export also failed: {e2}")
                    return None
    except Exception as e:
        logging.error(f"Error during FP16 quantization: {e}")
        import traceback
        logging.error(traceback.format_exc())
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
        
        try:
            session = ort.InferenceSession(model_path, session_options, providers=providers)
        except Exception as e:
            logging.warning(f"Could not create inference session for {model_path}: {e}")
            logging.warning("This model may contain operations not supported by the current ONNX Runtime version.")
            logging.warning("Skipping benchmarking but the model may still be usable with other inference engines.")
            return 0
        
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
        
        try:
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
            logging.warning(f"Error during benchmarking execution: {e}")
            logging.warning("The model was loaded but failed during inference. It may need additional configuration.")
            return 0
            
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
    
    # Create output directory if it doesn't exist
    os.makedirs(quantized_model_dir, exist_ok=True)
    
    results = {
        "original": model_path,
        "int8": None,
        "fp16": None,
        "int8_benchmark_ms": None,
        "fp16_benchmark_ms": None
    }
    
    # Check if model exists
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return results
    
    # Try INT8 quantization
    try:
        int8_model = quantize_model_int8(model_path, dataset_path, quantized_model_dir)
        if int8_model and os.path.exists(int8_model):
            results["int8"] = int8_model
            # Benchmark INT8 model
            try:
                benchmark_time = benchmark_quantized_model(int8_model, num_runs=20)
                results["int8_benchmark_ms"] = benchmark_time
                if benchmark_time > 0:
                    logging.info(f"INT8 model benchmark successful: {benchmark_time:.2f}ms")
                else:
                    logging.warning("INT8 model benchmarking skipped or failed. The model may still be usable with appropriate tools.")
            except Exception as e:
                logging.warning(f"Error benchmarking INT8 model: {e}")
                logging.warning("Skipping INT8 benchmarking. The model is still exported and can be used with compatible inference engines.")
        else:
            logging.warning("INT8 quantization did not produce a valid model file.")
    except Exception as e:
        logging.error(f"Error during INT8 quantization: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Try FP16 quantization
    try:
        fp16_model = quantize_model_fp16(model_path, dataset_path, quantized_model_dir)
        if fp16_model and os.path.exists(fp16_model):
            results["fp16"] = fp16_model
            # Benchmark FP16 model
            try:
                benchmark_time = benchmark_quantized_model(fp16_model, num_runs=20)
                results["fp16_benchmark_ms"] = benchmark_time
                if benchmark_time > 0:
                    logging.info(f"FP16 model benchmark successful: {benchmark_time:.2f}ms")
                else:
                    logging.warning("FP16 model benchmarking skipped or failed. The model may still be usable with appropriate tools.")
            except Exception as e:
                logging.warning(f"Error benchmarking FP16 model: {e}")
                logging.warning("Skipping FP16 benchmarking. The model is still exported and can be used with compatible inference engines.")
        else:
            logging.warning("FP16 quantization did not produce a valid model file.")
    except Exception as e:
        logging.error(f"Error during FP16 quantization: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Print summary
    logging.info("\nQuantization Summary:")
    logging.info(f"Original model: {results['original']}")
    
    if results["int8"]:
        benchmark_info = f" (Benchmark: {results['int8_benchmark_ms']:.2f}ms)" if results["int8_benchmark_ms"] else " (Benchmarking skipped or unsupported)"
        logging.info(f"INT8 model: {results['int8']}{benchmark_info}")
    else:
        logging.info("INT8 model: Not created")
        
    if results["fp16"]:
        benchmark_info = f" (Benchmark: {results['fp16_benchmark_ms']:.2f}ms)" if results["fp16_benchmark_ms"] else " (Benchmarking skipped or unsupported)"
        logging.info(f"FP16 model: {results['fp16']}{benchmark_info}")
    else:
        logging.info("FP16 model: Not created")
    
    logging.info("Quantization process completed. Models can be used with compatible inference engines.")
    
    return results