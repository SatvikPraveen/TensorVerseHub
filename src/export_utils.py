# Location: /src/export_utils.py

"""
TensorFlow model export utilities for various deployment formats.
Supports SavedModel, TFLite, ONNX, TensorFlow.js, and other deployment formats.
"""

import tensorflow as tf
import numpy as np
import os
import json
import tempfile
from typing import Tuple, Optional, List, Dict, Any, Union
import zipfile


class SavedModelExporter:
    """Utilities for exporting to TensorFlow SavedModel format."""
    
    @staticmethod
    def export_savedmodel(model: tf.keras.Model,
                         export_path: str,
                         signature_name: str = 'serving_default',
                         include_optimizer: bool = False,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Export model to SavedModel format with metadata.
        
        Args:
            model: tf.keras model to export
            export_path: Path to save the model
            signature_name: Name for the serving signature
            include_optimizer: Whether to include optimizer state
            metadata: Additional metadata to save
        """
        # Ensure directory exists
        os.makedirs(export_path, exist_ok=True)
        
        # Save model in SavedModel format
        tf.saved_model.save(
            model, 
            export_path,
            signatures={signature_name: model.call.get_concrete_function(
                tf.TensorSpec(shape=model.input_shape, dtype=tf.float32)
            )}
        )
        
        # Save metadata
        if metadata is not None:
            metadata_path = os.path.join(export_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        # Save model configuration
        config_path = os.path.join(export_path, 'model_config.json')
        model_config = {
            'input_shape': list(model.input_shape),
            'output_shape': list(model.output_shape),
            'num_parameters': model.count_params(),
            'num_layers': len(model.layers),
            'optimizer': model.optimizer.get_config() if include_optimizer and model.optimizer else None,
            'loss': model.loss,
            'metrics': [m.name if hasattr(m, 'name') else str(m) for m in model.metrics] if model.metrics else []
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2, default=str)
        
        print(f"Model exported to SavedModel format at: {export_path}")
    
    @staticmethod
    def load_savedmodel_with_metadata(model_path: str) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Load SavedModel with metadata.
        
        Args:
            model_path: Path to SavedModel
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        # Load model
        model = tf.saved_model.load(model_path)
        
        # Load metadata
        metadata = {}
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load model config
        config = {}
        config_path = os.path.join(model_path, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        metadata['model_config'] = config
        
        return model, metadata


class TFLiteExporter:
    """Utilities for exporting to TensorFlow Lite format."""
    
    @staticmethod
    def export_tflite(model: tf.keras.Model,
                     export_path: str,
                     quantization_type: str = 'float32',
                     representative_dataset: Optional[tf.data.Dataset] = None,
                     target_ops: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export model to TFLite format with various optimizations.
        
        Args:
            model: tf.keras model to export
            export_path: Path to save TFLite model
            quantization_type: Type of quantization ('float32', 'float16', 'int8')
            representative_dataset: Dataset for calibration (needed for int8)
            target_ops: Target operations for optimization
            
        Returns:
            Dictionary with export statistics
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure quantization
        if quantization_type == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization_type == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            if representative_dataset is not None:
                def representative_data_gen():
                    for input_value in representative_dataset.take(100):
                        if isinstance(input_value, tuple):
                            yield [tf.cast(input_value[0], tf.float32)]
                        else:
                            yield [tf.cast(input_value, tf.float32)]
                
                converter.representative_dataset = representative_data_gen
            else:
                print("Warning: int8 quantization requested but no representative dataset provided")
        
        elif quantization_type == 'float32':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set target operations if specified
        if target_ops:
            op_sets = []
            for op in target_ops:
                if op == 'TFLITE_BUILTINS':
                    op_sets.append(tf.lite.OpsSet.TFLITE_BUILTINS)
                elif op == 'SELECT_TF_OPS':
                    op_sets.append(tf.lite.OpsSet.SELECT_TF_OPS)
            converter.target_spec.supported_ops = op_sets
        
        # Convert model
        try:
            tflite_model = converter.convert()
            
            # Save to file
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
            
            # Calculate statistics
            original_size = model.count_params() * 4  # Assuming float32
            tflite_size = len(tflite_model)
            compression_ratio = original_size / tflite_size if tflite_size > 0 else 0
            
            stats = {
                'original_size_bytes': original_size,
                'tflite_size_bytes': tflite_size,
                'original_size_mb': original_size / (1024 * 1024),
                'tflite_size_mb': tflite_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'quantization_type': quantization_type,
                'export_path': export_path
            }
            
            print(f"TFLite model exported to: {export_path}")
            print(f"Original size: {stats['original_size_mb']:.2f} MB")
            print(f"TFLite size: {stats['tflite_size_mb']:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            
            return stats
            
        except Exception as e:
            raise RuntimeError(f"TFLite conversion failed: {str(e)}")
    
    @staticmethod
    def benchmark_tflite_model(tflite_path: str,
                              test_input: np.ndarray,
                              num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark TFLite model performance.
        
        Args:
            tflite_path: Path to TFLite model
            test_input: Test input array
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        import time
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warm up
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        
        return {
            'avg_inference_time_ms': avg_time_ms,
            'total_time_s': total_time,
            'throughput_fps': num_runs / total_time if total_time > 0 else 0
        }


class ONNXExporter:
    """Utilities for exporting to ONNX format."""
    
    @staticmethod
    def export_onnx(model: tf.keras.Model,
                   export_path: str,
                   input_signature: Optional[List[tf.TensorSpec]] = None,
                   opset_version: int = 13) -> Dict[str, Any]:
        """
        Export model to ONNX format.
        
        Args:
            model: tf.keras model to export
            export_path: Path to save ONNX model
            input_signature: Input signature specification
            opset_version: ONNX opset version
            
        Returns:
            Export statistics
        """
        try:
            import tf2onnx
            
            # Create input signature if not provided
            if input_signature is None:
                input_signature = [tf.TensorSpec(shape=model.input_shape, dtype=tf.float32)]
            
            # Convert to ONNX
            with tempfile.TemporaryDirectory() as temp_dir:
                # First save as SavedModel
                saved_model_path = os.path.join(temp_dir, 'saved_model')
                model.save(saved_model_path, save_format='tf')
                
                # Convert SavedModel to ONNX
                onnx_model, _ = tf2onnx.convert.from_saved_model(
                    saved_model_path,
                    input_names=None,
                    output_names=None,
                    opset=opset_version
                )
                
                # Save ONNX model
                with open(export_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
            
            # Calculate file sizes
            original_size = model.count_params() * 4  # Assuming float32
            onnx_size = os.path.getsize(export_path)
            
            stats = {
                'original_size_bytes': original_size,
                'onnx_size_bytes': onnx_size,
                'original_size_mb': original_size / (1024 * 1024),
                'onnx_size_mb': onnx_size / (1024 * 1024),
                'size_ratio': onnx_size / original_size if original_size > 0 else 0,
                'opset_version': opset_version,
                'export_path': export_path
            }
            
            print(f"ONNX model exported to: {export_path}")
            print(f"ONNX size: {stats['onnx_size_mb']:.2f} MB")
            
            return stats
            
        except ImportError:
            raise ImportError("tf2onnx package required for ONNX export. Install with: pip install tf2onnx")
        except Exception as e:
            raise RuntimeError(f"ONNX conversion failed: {str(e)}")


class TensorFlowJSExporter:
    """Utilities for exporting to TensorFlow.js format."""
    
    @staticmethod
    def export_tfjs(model: tf.keras.Model,
                   export_path: str,
                   quantization_bytes: Optional[int] = None,
                   skip_op_check: bool = False,
                   strip_debug_ops: bool = True) -> Dict[str, Any]:
        """
        Export model to TensorFlow.js format.
        
        Args:
            model: tf.keras model to export
            export_path: Directory to save TensorFlow.js model
            quantization_bytes: Quantization precision (1 or 2 bytes)
            skip_op_check: Skip operation compatibility check
            strip_debug_ops: Remove debug operations
            
        Returns:
            Export statistics
        """
        try:
            import tensorflowjs as tfjs
            
            # Ensure export directory exists
            os.makedirs(export_path, exist_ok=True)
            
            # Configure conversion options
            conversion_options = {}
            if quantization_bytes:
                conversion_options['quantization_bytes'] = quantization_bytes
            if skip_op_check:
                conversion_options['skip_op_check'] = skip_op_check
            if strip_debug_ops:
                conversion_options['strip_debug_ops'] = strip_debug_ops
            
            # Convert model
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save as SavedModel first
                saved_model_path = os.path.join(temp_dir, 'saved_model')
                model.save(saved_model_path, save_format='tf')
                
                # Convert to TensorFlow.js
                tfjs.converters.convert_tf_saved_model(
                    saved_model_path,
                    export_path,
                    **conversion_options
                )
            
            # Calculate directory size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(export_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            original_size = model.count_params() * 4  # Assuming float32
            
            stats = {
                'original_size_bytes': original_size,
                'tfjs_size_bytes': total_size,
                'original_size_mb': original_size / (1024 * 1024),
                'tfjs_size_mb': total_size / (1024 * 1024),
                'size_ratio': total_size / original_size if original_size > 0 else 0,
                'quantization_bytes': quantization_bytes,
                'export_path': export_path
            }
            
            print(f"TensorFlow.js model exported to: {export_path}")
            print(f"TensorFlow.js size: {stats['tfjs_size_mb']:.2f} MB")
            
            return stats
            
        except ImportError:
            raise ImportError("tensorflowjs package required. Install with: pip install tensorflowjs")
        except Exception as e:
            raise RuntimeError(f"TensorFlow.js conversion failed: {str(e)}")


class CoreMLExporter:
    """Utilities for exporting to Core ML format (Apple platforms)."""
    
    @staticmethod
    def export_coreml(model: tf.keras.Model,
                     export_path: str,
                     input_names: Optional[List[str]] = None,
                     output_names: Optional[List[str]] = None,
                     class_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export model to Core ML format.
        
        Args:
            model: tf.keras model to export
            export_path: Path to save Core ML model
            input_names: Names for input tensors
            output_names: Names for output tensors
            class_labels: Class labels for classification models
            
        Returns:
            Export statistics
        """
        try:
            import coremltools as ct
            import tf2onnx
            
            # First convert to ONNX
            with tempfile.TemporaryDirectory() as temp_dir:
                onnx_path = os.path.join(temp_dir, 'model.onnx')
                
                # Save as SavedModel first
                saved_model_path = os.path.join(temp_dir, 'saved_model')
                model.save(saved_model_path, save_format='tf')
                
                # Convert to ONNX
                onnx_model, _ = tf2onnx.convert.from_saved_model(saved_model_path)
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                
                # Convert ONNX to Core ML
                coreml_model = ct.convert(
                    onnx_path,
                    source='onnx',
                    inputs=[ct.TensorType(shape=model.input_shape[1:])],  # Remove batch dimension
                    classifier_config=ct.ClassifierConfig(class_labels) if class_labels else None
                )
                
                # Save Core ML model
                coreml_model.save(export_path)
            
            # Calculate file size
            coreml_size = os.path.getsize(export_path)
            original_size = model.count_params() * 4  # Assuming float32
            
            stats = {
                'original_size_bytes': original_size,
                'coreml_size_bytes': coreml_size,
                'original_size_mb': original_size / (1024 * 1024),
                'coreml_size_mb': coreml_size / (1024 * 1024),
                'size_ratio': coreml_size / original_size if original_size > 0 else 0,
                'export_path': export_path
            }
            
            print(f"Core ML model exported to: {export_path}")
            print(f"Core ML size: {stats['coreml_size_mb']:.2f} MB")
            
            return stats
            
        except ImportError:
            raise ImportError("coremltools required for Core ML export. Install with: pip install coremltools")
        except Exception as e:
            raise RuntimeError(f"Core ML conversion failed: {str(e)}")


class MultiFormatExporter:
    """Unified exporter for multiple formats."""
    
    def __init__(self, model: tf.keras.Model, model_name: str = "model"):
        """
        Initialize multi-format exporter.
        
        Args:
            model: tf.keras model to export
            model_name: Base name for exported models
        """
        self.model = model
        self.model_name = model_name
        self.export_stats = {}
    
    def export_all_formats(self,
                          export_dir: str,
                          formats: List[str] = None,
                          representative_dataset: Optional[tf.data.Dataset] = None,
                          **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Export model to multiple formats.
        
        Args:
            export_dir: Base directory for exports
            formats: List of formats to export ('savedmodel', 'tflite', 'onnx', 'tfjs')
            representative_dataset: Dataset for calibration
            **kwargs: Additional arguments for specific exporters
            
        Returns:
            Dictionary with export statistics for each format
        """
        if formats is None:
            formats = ['savedmodel', 'tflite', 'onnx', 'tfjs']
        
        os.makedirs(export_dir, exist_ok=True)
        
        for format_type in formats:
            try:
                format_dir = os.path.join(export_dir, format_type)
                
                if format_type == 'savedmodel':
                    SavedModelExporter.export_savedmodel(
                        self.model,
                        format_dir,
                        metadata=kwargs.get('metadata', {})
                    )
                    self.export_stats['savedmodel'] = {'export_path': format_dir}
                    
                elif format_type == 'tflite':
                    tflite_path = os.path.join(export_dir, f"{self.model_name}.tflite")
                    stats = TFLiteExporter.export_tflite(
                        self.model,
                        tflite_path,
                        quantization_type=kwargs.get('tflite_quantization', 'float32'),
                        representative_dataset=representative_dataset
                    )
                    self.export_stats['tflite'] = stats
                    
                elif format_type == 'onnx':
                    onnx_path = os.path.join(export_dir, f"{self.model_name}.onnx")
                    stats = ONNXExporter.export_onnx(
                        self.model,
                        onnx_path,
                        opset_version=kwargs.get('onnx_opset', 13)
                    )
                    self.export_stats['onnx'] = stats
                    
                elif format_type == 'tfjs':
                    tfjs_dir = os.path.join(export_dir, 'tfjs')
                    stats = TensorFlowJSExporter.export_tfjs(
                        self.model,
                        tfjs_dir,
                        quantization_bytes=kwargs.get('tfjs_quantization', None)
                    )
                    self.export_stats['tfjs'] = stats
                    
                elif format_type == 'coreml':
                    coreml_path = os.path.join(export_dir, f"{self.model_name}.mlmodel")
                    stats = CoreMLExporter.export_coreml(
                        self.model,
                        coreml_path,
                        class_labels=kwargs.get('class_labels', None)
                    )
                    self.export_stats['coreml'] = stats
                    
            except Exception as e:
                print(f"Failed to export {format_type}: {str(e)}")
                self.export_stats[format_type] = {'error': str(e)}
        
        # Create summary report
        self._create_export_summary(export_dir)
        
        return self.export_stats
    
    def _create_export_summary(self, export_dir: str) -> None:
        """Create export summary report."""
        summary_path = os.path.join(export_dir, 'export_summary.json')
        
        summary = {
            'model_name': self.model_name,
            'original_model': {
                'parameters': self.model.count_params(),
                'layers': len(self.model.layers),
                'input_shape': list(self.model.input_shape),
                'output_shape': list(self.model.output_shape)
            },
            'export_stats': self.export_stats,
            'timestamp': tf.timestamp().numpy().item()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Export summary saved to: {summary_path}")


# Convenience functions
def quick_export(model: tf.keras.Model,
                export_dir: str,
                model_name: str = "model",
                formats: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Quick export of model to multiple formats.
    
    Args:
        model: tf.keras model to export
        export_dir: Directory to save exports
        model_name: Base name for exported files
        formats: List of formats to export
        
    Returns:
        Export statistics for each format
    """
    exporter = MultiFormatExporter(model, model_name)
    return exporter.export_all_formats(export_dir, formats)


def create_deployment_package(model: tf.keras.Model,
                            package_path: str,
                            model_name: str = "model",
                            include_metadata: bool = True) -> str:
    """
    Create a deployment package with model and metadata.
    
    Args:
        model: tf.keras model to package
        package_path: Path for the deployment package
        model_name: Name of the model
        include_metadata: Whether to include metadata
        
    Returns:
        Path to created package
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export model in multiple formats
        export_dir = os.path.join(temp_dir, 'exports')
        exporter = MultiFormatExporter(model, model_name)
        export_stats = exporter.export_all_formats(export_dir)
        
        # Create package metadata
        if include_metadata:
            package_metadata = {
                'model_name': model_name,
                'model_version': '1.0.0',
                'tensorflow_version': tf.__version__,
                'export_timestamp': tf.timestamp().numpy().item(),
                'model_architecture': {
                    'layers': len(model.layers),
                    'parameters': model.count_params(),
                    'input_shape': list(model.input_shape),
                    'output_shape': list(model.output_shape)
                },
                'export_formats': list(export_stats.keys()),
                'deployment_instructions': {
                    'savedmodel': 'Use tf.saved_model.load() to load the model',
                    'tflite': 'Use tf.lite.Interpreter() for mobile deployment',
                    'onnx': 'Use ONNX Runtime for cross-platform inference',
                    'tfjs': 'Load in browser using tf.loadLayersModel()'
                }
            }
            
            metadata_path = os.path.join(temp_dir, 'package_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(package_metadata, f, indent=2, default=str)
        
        # Create deployment package (zip file)
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arc_name)
    
    print(f"Deployment package created: {package_path}")
    return package_path