# Location: /scripts/convert_models.py

"""
Batch model conversion script for TensorVerseHub.
Converts Keras models to TensorFlow Lite, ONNX, and other formats for deployment.
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
import numpy as np
from datetime import datetime

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TF_AVAILABLE = False

# Optional imports for different conversion formats
try:
    import tf2onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Info: tf2onnx not available. ONNX conversion will be skipped.")

try:
    import onnx
    ONNX_VERIFY_AVAILABLE = True
except ImportError:
    ONNX_VERIFY_AVAILABLE = False
    print("Info: onnx not available for verification.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelConverter:
    """Comprehensive model conversion utility."""
    
    def __init__(self, models_dir: str, output_dir: str):
        """
        Initialize model converter.
        
        Args:
            models_dir: Directory containing source models
            output_dir: Directory for converted models
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.tflite_dir = self.output_dir / "tflite"
        self.onnx_dir = self.output_dir / "onnx" 
        self.savedmodel_dir = self.output_dir / "savedmodel"
        self.tensorrt_dir = self.output_dir / "tensorrt"
        
        for dir_path in [self.tflite_dir, self.onnx_dir, self.savedmodel_dir, self.tensorrt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Conversion results tracking
        self.conversion_results = {}
        
        # TFLite optimization strategies
        self.tflite_optimizations = {
            'default': [],
            'optimize_for_size': [tf.lite.Optimize.DEFAULT],
            'optimize_for_latency': [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY], 
            'full_integer': [tf.lite.Optimize.DEFAULT]
        }
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discover all convertible models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # H5 models
        for model_file in self.models_dir.rglob("*.h5"):
            models.append({
                'name': model_file.stem,
                'path': str(model_file),
                'format': 'h5',
                'size_mb': model_file.stat().st_size / (1024 * 1024)
            })
        
        # SavedModel format
        for saved_model_dir in self.models_dir.rglob("saved_model.pb"):
            parent_dir = saved_model_dir.parent
            models.append({
                'name': parent_dir.name,
                'path': str(parent_dir),
                'format': 'savedmodel',
                'size_mb': sum(f.stat().st_size for f in parent_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            })
        
        logger.info(f"Discovered {len(models)} models for conversion")
        return models
    
    def load_model(self, model_path: str, model_format: str) -> tf.keras.Model:
        """
        Load model from path.
        
        Args:
            model_path: Path to model
            model_format: Format of the model ('h5' or 'savedmodel')
            
        Returns:
            Loaded Keras model
        """
        try:
            if model_format == 'h5':
                model = tf.keras.models.load_model(model_path)
            elif model_format == 'savedmodel':
                model = tf.keras.models.load_model(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
            
            logger.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def convert_to_tflite(self, model: tf.keras.Model, model_name: str, 
                         optimization: str = 'default', 
                         representative_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Convert model to TensorFlow Lite format.
        
        Args:
            model: Keras model to convert
            model_name: Name for the converted model
            optimization: Optimization strategy
            representative_data: Representative dataset for quantization
            
        Returns:
            Conversion result dictionary
        """
        logger.info(f"Converting {model_name} to TFLite with {optimization} optimization")
        
        result = {
            'model_name': model_name,
            'optimization': optimization,
            'success': False,
            'error': None,
            'output_path': None,
            'original_size_mb': 0,
            'converted_size_mb': 0,
            'compression_ratio': 0
        }
        
        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Set optimization
            if optimization in self.tflite_optimizations:
                converter.optimizations = self.tflite_optimizations[optimization]
            
            # Handle quantization
            if optimization == 'full_integer' and representative_data is not None:
                def representative_dataset_gen():
                    for i in range(min(100, len(representative_data))):
                        yield [representative_data[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save converted model
            output_path = self.tflite_dir / f"{model_name}_{optimization}.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Calculate sizes and compression
            original_size = self._estimate_keras_model_size(model)
            converted_size = len(tflite_model) / (1024 * 1024)
            compression_ratio = original_size / converted_size if converted_size > 0 else 0
            
            result.update({
                'success': True,
                'output_path': str(output_path),
                'original_size_mb': original_size,
                'converted_size_mb': converted_size,
                'compression_ratio': compression_ratio
            })
            
            # Verify converted model
            verification_result = self._verify_tflite_model(tflite_model, model)
            result['verification'] = verification_result
            
            logger.info(f"TFLite conversion successful: {output_path}")
            logger.info(f"Size reduction: {original_size:.2f}MB -> {converted_size:.2f}MB "
                       f"({compression_ratio:.2f}x compression)")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"TFLite conversion failed for {model_name}: {e}")
        
        return result
    
    def convert_to_onnx(self, model: tf.keras.Model, model_name: str,
                       input_signature: Optional[List[tf.TensorSpec]] = None) -> Dict[str, Any]:
        """
        Convert model to ONNX format.
        
        Args:
            model: Keras model to convert
            model_name: Name for the converted model
            input_signature: Input signature specification
            
        Returns:
            Conversion result dictionary
        """
        if not ONNX_AVAILABLE:
            return {
                'model_name': model_name,
                'success': False,
                'error': 'tf2onnx not available'
            }
        
        logger.info(f"Converting {model_name} to ONNX format")
        
        result = {
            'model_name': model_name,
            'success': False,
            'error': None,
            'output_path': None,
            'original_size_mb': 0,
            'converted_size_mb': 0
        }
        
        try:
            # Save model as SavedModel first (required for tf2onnx)
            temp_savedmodel_path = self.output_dir / "temp_savedmodel" / model_name
            temp_savedmodel_path.mkdir(parents=True, exist_ok=True)
            
            model.save(temp_savedmodel_path, save_format='tf')
            
            # Convert to ONNX
            output_path = self.onnx_dir / f"{model_name}.onnx"
            
            # Determine input signature
            if input_signature is None:
                # Infer from model
                input_signature = [tf.TensorSpec(shape=model.input_shape, dtype=tf.float32)]
            
            # Convert using tf2onnx
            import subprocess
            cmd = [
                "python", "-m", "tf2onnx.convert",
                "--saved-model", str(temp_savedmodel_path),
                "--output", str(output_path),
                "--opset", "11"
            ]
            
            subprocess_result = subprocess.run(cmd, capture_output=True, text=True)
            
            if subprocess_result.returncode != 0:
                raise RuntimeError(f"tf2onnx conversion failed: {subprocess_result.stderr}")
            
            # Calculate sizes
            original_size = self._estimate_keras_model_size(model)
            converted_size = output_path.stat().st_size / (1024 * 1024)
            
            result.update({
                'success': True,
                'output_path': str(output_path),
                'original_size_mb': original_size,
                'converted_size_mb': converted_size
            })
            
            # Verify ONNX model
            if ONNX_VERIFY_AVAILABLE:
                verification_result = self._verify_onnx_model(str(output_path))
                result['verification'] = verification_result
            
            # Cleanup temp directory
            shutil.rmtree(temp_savedmodel_path)
            
            logger.info(f"ONNX conversion successful: {output_path}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"ONNX conversion failed for {model_name}: {e}")
        
        return result
    
    def convert_to_savedmodel(self, model: tf.keras.Model, model_name: str) -> Dict[str, Any]:
        """
        Convert model to SavedModel format.
        
        Args:
            model: Keras model to convert
            model_name: Name for the converted model
            
        Returns:
            Conversion result dictionary
        """
        logger.info(f"Converting {model_name} to SavedModel format")
        
        result = {
            'model_name': model_name,
            'success': False,
            'error': None,
            'output_path': None,
            'original_size_mb': 0,
            'converted_size_mb': 0
        }
        
        try:
            output_path = self.savedmodel_dir / model_name
            
            # Save as SavedModel
            model.save(output_path, save_format='tf')
            
            # Calculate sizes
            original_size = self._estimate_keras_model_size(model)
            converted_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
            
            result.update({
                'success': True,
                'output_path': str(output_path),
                'original_size_mb': original_size,
                'converted_size_mb': converted_size
            })
            
            logger.info(f"SavedModel conversion successful: {output_path}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"SavedModel conversion failed for {model_name}: {e}")
        
        return result
    
    def convert_to_tensorrt(self, savedmodel_path: str, model_name: str) -> Dict[str, Any]:
        """
        Convert SavedModel to TensorRT format (if available).
        
        Args:
            savedmodel_path: Path to SavedModel
            model_name: Name for the converted model
            
        Returns:
            Conversion result dictionary
        """
        result = {
            'model_name': model_name,
            'success': False,
            'error': None,
            'output_path': None
        }
        
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            
            logger.info(f"Converting {model_name} to TensorRT format")
            
            # Create TRT converter
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
            conversion_params = conversion_params._replace(
                precision_mode=trt.TrtPrecisionMode.FP16,
                max_workspace_size_bytes=1 << 30  # 1GB
            )
            
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=savedmodel_path,
                conversion_params=conversion_params
            )
            
            # Convert
            converter.convert()
            
            # Save converted model
            output_path = self.tensorrt_dir / model_name
            converter.save(str(output_path))
            
            result.update({
                'success': True,
                'output_path': str(output_path)
            })
            
            logger.info(f"TensorRT conversion successful: {output_path}")
            
        except ImportError:
            result['error'] = 'TensorRT not available'
            logger.warning(f"TensorRT conversion skipped for {model_name}: TensorRT not available")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"TensorRT conversion failed for {model_name}: {e}")
        
        return result
    
    def _estimate_keras_model_size(self, model: tf.keras.Model) -> float:
        """Estimate Keras model size in MB."""
        total_params = model.count_params()
        # Assume float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def _verify_tflite_model(self, tflite_model: bytes, original_model: tf.keras.Model) -> Dict[str, Any]:
        """
        Verify TFLite model by comparing outputs.
        
        Args:
            tflite_model: TFLite model bytes
            original_model: Original Keras model
            
        Returns:
            Verification result dictionary
        """
        verification = {
            'success': False,
            'max_diff': None,
            'mean_diff': None,
            'error': None
        }
        
        try:
            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Generate test input
            input_shape = input_details[0]['shape']
            test_input = np.random.random(input_shape).astype(np.float32)
            
            # Get original model prediction
            original_output = original_model.predict(test_input, verbose=0)
            
            # Get TFLite prediction
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]['index'])
            
            # Compare outputs
            diff = np.abs(original_output - tflite_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            verification.update({
                'success': True,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff)
            })
            
        except Exception as e:
            verification['error'] = str(e)
        
        return verification
    
    def _verify_onnx_model(self, onnx_path: str) -> Dict[str, Any]:
        """
        Verify ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Verification result dictionary
        """
        verification = {
            'success': False,
            'error': None
        }
        
        try:
            import onnx
            
            # Load and check model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            verification['success'] = True
            
        except Exception as e:
            verification['error'] = str(e)
        
        return verification
    
    def generate_representative_data(self, model: tf.keras.Model, 
                                   data_dir: str = None) -> Optional[np.ndarray]:
        """
        Generate representative data for quantization.
        
        Args:
            model: Keras model
            data_dir: Directory containing sample data
            
        Returns:
            Representative dataset or None
        """
        try:
            if data_dir and os.path.exists(data_dir):
                # Try to load actual data
                data_path = Path(data_dir)
                
                # Look for common data files
                for pattern in ["*.npy", "X_val.npy", "X_test.npy"]:
                    data_files = list(data_path.rglob(pattern))
                    if data_files:
                        data = np.load(data_files[0])
                        if len(data.shape) == len(model.input_shape):
                            return data[:100]  # Use first 100 samples
            
            # Generate synthetic data
            input_shape = model.input_shape
            if input_shape[0] is None:  # Batch dimension
                synthetic_shape = (100,) + input_shape[1:]
            else:
                synthetic_shape = input_shape
            
            # Generate appropriate data based on likely data type
            if len(input_shape) == 4:  # Image data
                representative_data = np.random.random(synthetic_shape).astype(np.float32)
            elif len(input_shape) == 2:  # Tabular data
                representative_data = np.random.randn(*synthetic_shape).astype(np.float32)
            else:  # Other data
                representative_data = np.random.random(synthetic_shape).astype(np.float32)
            
            logger.info(f"Generated synthetic representative data with shape {representative_data.shape}")
            return representative_data
            
        except Exception as e:
            logger.warning(f"Failed to generate representative data: {e}")
            return None
    
    def batch_convert_models(self, models: List[Dict[str, Any]], 
                           formats: List[str] = None,
                           tflite_optimizations: List[str] = None,
                           data_dir: str = None) -> Dict[str, Any]:
        """
        Convert multiple models to specified formats.
        
        Args:
            models: List of model information dictionaries
            formats: List of target formats ('tflite', 'onnx', 'savedmodel', 'tensorrt')
            tflite_optimizations: List of TFLite optimization strategies
            data_dir: Directory containing representative data
            
        Returns:
            Batch conversion results
        """
        if formats is None:
            formats = ['tflite', 'savedmodel']
        
        if tflite_optimizations is None:
            tflite_optimizations = ['default', 'optimize_for_size']
        
        logger.info(f"Starting batch conversion of {len(models)} models")
        logger.info(f"Target formats: {formats}")
        
        batch_results = {
            'total_models': len(models),
            'successful_conversions': 0,
            'failed_conversions': 0,
            'results_by_model': {},
            'summary_by_format': {}
        }
        
        # Initialize format summaries
        for fmt in formats:
            batch_results['summary_by_format'][fmt] = {
                'successful': 0,
                'failed': 0,
                'total_size_reduction_mb': 0
            }
        
        for model_info in models:
            model_name = model_info['name']
            model_path = model_info['path']
            model_format = model_info['format']
            
            logger.info(f"Processing model: {model_name}")
            
            model_results = {
                'model_info': model_info,
                'conversions': {}
            }
            
            try:
                # Load model
                model = self.load_model(model_path, model_format)
                
                # Generate representative data if needed
                representative_data = None
                if 'tflite' in formats and 'full_integer' in tflite_optimizations:
                    representative_data = self.generate_representative_data(model, data_dir)
                
                # Convert to each requested format
                if 'tflite' in formats:
                    for optimization in tflite_optimizations:
                        conversion_key = f"tflite_{optimization}"
                        result = self.convert_to_tflite(
                            model, model_name, optimization, representative_data
                        )
                        model_results['conversions'][conversion_key] = result
                        
                        # Update summary
                        if result['success']:
                            batch_results['summary_by_format']['tflite']['successful'] += 1
                            size_reduction = result['original_size_mb'] - result['converted_size_mb']
                            batch_results['summary_by_format']['tflite']['total_size_reduction_mb'] += size_reduction
                        else:
                            batch_results['summary_by_format']['tflite']['failed'] += 1
                
                if 'onnx' in formats:
                    result = self.convert_to_onnx(model, model_name)
                    model_results['conversions']['onnx'] = result
                    
                    # Update summary
                    if result['success']:
                        batch_results['summary_by_format']['onnx']['successful'] += 1
                    else:
                        batch_results['summary_by_format']['onnx']['failed'] += 1
                
                if 'savedmodel' in formats:
                    result = self.convert_to_savedmodel(model, model_name)
                    model_results['conversions']['savedmodel'] = result
                    
                    # Update summary
                    if result['success']:
                        batch_results['summary_by_format']['savedmodel']['successful'] += 1
                    else:
                        batch_results['summary_by_format']['savedmodel']['failed'] += 1
                    
                    # Convert to TensorRT if requested and SavedModel was successful
                    if 'tensorrt' in formats and result['success']:
                        tensorrt_result = self.convert_to_tensorrt(result['output_path'], model_name)
                        model_results['conversions']['tensorrt'] = tensorrt_result
                        
                        # Update summary
                        if tensorrt_result['success']:
                            batch_results['summary_by_format']['tensorrt']['successful'] += 1
                        else:
                            batch_results['summary_by_format']['tensorrt']['failed'] += 1
                
                # Check if any conversion was successful
                if any(conv['success'] for conv in model_results['conversions'].values()):
                    batch_results['successful_conversions'] += 1
                else:
                    batch_results['failed_conversions'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process model {model_name}: {e}")
                model_results['error'] = str(e)
                batch_results['failed_conversions'] += 1
            
            batch_results['results_by_model'][model_name] = model_results
        
        logger.info(f"Batch conversion completed: {batch_results['successful_conversions']}/{batch_results['total_models']} models converted successfully")
        return batch_results
    
    def generate_conversion_report(self, batch_results: Dict[str, Any]) -> str:
        """Generate HTML report for batch conversions."""
        report_path = self.output_dir / "conversion_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorVerseHub Model Conversion Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .model-section {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; font-weight: bold; }}
                .failed {{ color: red; font-weight: bold; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔄 TensorVerseHub Model Conversion Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total models processed: {batch_results['total_models']}</p>
                <p>Successful: {batch_results['successful_conversions']} | Failed: {batch_results['failed_conversions']}</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary by Format</h2>
                <table>
                    <tr>
                        <th>Format</th>
                        <th>Successful</th>
                        <th>Failed</th>
                        <th>Success Rate</th>
                        <th>Total Size Reduction (MB)</th>
                    </tr>
        """
        
        for format_name, summary in batch_results['summary_by_format'].items():
            total = summary['successful'] + summary['failed']
            success_rate = (summary['successful'] / total * 100) if total > 0 else 0
            size_reduction = summary.get('total_size_reduction_mb', 0)
            
            html_content += f"""
                <tr>
                    <td>{format_name.upper()}</td>
                    <td class="success">{summary['successful']}</td>
                    <td class="failed">{summary['failed']}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{size_reduction:.2f} MB</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Detailed Results</h2>
        """
        
        for model_name, model_results in batch_results['results_by_model'].items():
            html_content += f"""
                <div class="model-section">
                    <h3>📋 {model_name}</h3>
            """
            
            if 'error' in model_results:
                html_content += f'<p class="failed">Processing failed: {model_results["error"]}</p>'
            else:
                html_content += """
                    <table>
                        <tr>
                            <th>Format</th>
                            <th>Status</th>
                            <th>Size Change</th>
                            <th>Output Path</th>
                        </tr>
                """
                
                for conv_name, conv_result in model_results['conversions'].items():
                    status = "✅ SUCCESS" if conv_result['success'] else "❌ FAILED"
                    status_class = "success" if conv_result['success'] else "failed"
                    
                    size_info = "N/A"
                    if conv_result['success'] and 'original_size_mb' in conv_result:
                        orig_size = conv_result['original_size_mb']
                        conv_size = conv_result['converted_size_mb']
                        if 'compression_ratio' in conv_result:
                            size_info = f"{orig_size:.2f} → {conv_size:.2f} MB ({conv_result['compression_ratio']:.2f}x)"
                        else:
                            size_info = f"{orig_size:.2f} → {conv_size:.2f} MB"
                    
                    output_path = conv_result.get('output_path', 'N/A')
                    if output_path != 'N/A':
                        output_path = Path(output_path).name
                    
                    html_content += f"""
                        <tr>
                            <td>{conv_name.upper()}</td>
                            <td class="{status_class}">{status}</td>
                            <td>{size_info}</td>
                            <td>{output_path}</td>
                        </tr>
                    """
                
                html_content += "</table>"
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Conversion report saved to {report_path}")
        return str(report_path)
    
    def save_conversion_results(self, batch_results: Dict[str, Any]) -> None:
        """Save conversion results to JSON file."""
        results_path = self.output_dir / "conversion_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        logger.info(f"Conversion results saved to {results_path}")


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description="Convert TensorVerseHub models to various formats")
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing source models"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="converted_models",
        help="Output directory for converted models"
    )
    
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["tflite", "onnx", "savedmodel", "tensorrt"],
        default=["tflite", "savedmodel"],
        help="Target conversion formats"
    )
    
    parser.add_argument(
        "--tflite-optimizations",
        nargs="+",
        choices=["default", "optimize_for_size", "optimize_for_latency", "full_integer"],
        default=["default", "optimize_for_size"],
        help="TFLite optimization strategies"
    )
    
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Specific model names to convert (optional)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing representative data for quantization"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation"
    )
    
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Please install TensorFlow.")
        sys.exit(1)
    
    logger.info("🚀 Starting TensorVerseHub Model Conversion")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target formats: {args.formats}")
    
    # Initialize converter
    converter = ModelConverter(args.models_dir, args.output_dir)
    
    # Discover models
    models = converter.discover_models()
    
    if not models:
        logger.error("No models found for conversion!")
        sys.exit(1)
    
    # Filter models if specific names provided
    if args.model_names:
        models = [m for m in models if m['name'] in args.model_names]
    
    if not models:
        logger.error("No matching models found!")
        sys.exit(1)
    
    logger.info(f"Converting {len(models)} models: {[m['name'] for m in models]}")
    
    # Perform batch conversion
    batch_results = converter.batch_convert_models(
        models=models,
        formats=args.formats,
        tflite_optimizations=args.tflite_optimizations,
        data_dir=args.data_dir
    )
    
    # Save results
    converter.save_conversion_results(batch_results)
    
    # Generate report
    if not args.no_report:
        try:
            report_path = converter.generate_conversion_report(batch_results)
            logger.info(f"📄 Conversion report generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    # Summary
    logger.info("🎉 Model conversion completed!")
    logger.info(f"📊 Results: {batch_results['successful_conversions']}/{batch_results['total_models']} models converted successfully")
    
    # Format-specific summaries
    for format_name, summary in batch_results['summary_by_format'].items():
        if summary['successful'] > 0 or summary['failed'] > 0:
            logger.info(f"  {format_name.upper()}: {summary['successful']} success, {summary['failed']} failed")
    
    if batch_results['failed_conversions'] > 0:
        logger.warning(f"⚠️  {batch_results['failed_conversions']} models failed to convert")
        sys.exit(1)


if __name__ == "__main__":
    main()