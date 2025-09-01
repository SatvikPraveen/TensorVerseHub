# Location: /benchmarks/inference_benchmarks.py

"""
Model inference speed benchmarking for TensorVerseHub.
Comprehensive testing of inference performance across different models, batch sizes, and hardware configurations.
"""

import os
import sys
import json
import time
import argparse
import logging
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
import numpy as np
import pandas as pd

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_benchmarks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.timestamps = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()
        self.timestamps.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread."""
        while self.monitoring:
            timestamp = time.time()
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.timestamps.append(timestamp)
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_percent)
            
            # GPU monitoring (if available)
            gpu_percent = self._get_gpu_usage()
            self.gpu_usage.append(gpu_percent)
            
            time.sleep(0.1)  # Monitor every 100ms
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # This is a simplified approach - actual GPU monitoring
                # would require nvidia-ml-py or similar
                return 0.0  # Placeholder
            return 0.0
        except Exception:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if not self.cpu_usage:
            return {}
        
        return {
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
                'std': np.std(self.cpu_usage)
            },
            'memory': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage),
                'std': np.std(self.memory_usage)
            },
            'gpu': {
                'mean': np.mean(self.gpu_usage) if self.gpu_usage else 0,
                'max': np.max(self.gpu_usage) if self.gpu_usage else 0,
                'min': np.min(self.gpu_usage) if self.gpu_usage else 0,
                'std': np.std(self.gpu_usage) if self.gpu_usage else 0
            },
            'duration': self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0
        }


class InferenceBenchmarker:
    """Comprehensive inference performance benchmarking."""
    
    def __init__(self, models_dir: str, results_dir: str):
        """
        Initialize inference benchmarker.
        
        Args:
            models_dir: Directory containing models to benchmark
            results_dir: Directory to save benchmark results
        """
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # System information
        self.system_info = self._get_system_info()
        
        # Benchmark configuration
        self.batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
        
        # Results storage
        self.benchmark_results = {}
        
        # System monitor
        self.monitor = SystemMonitor()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'python_version': sys.version,
            'tensorflow_version': tf.__version__ if TF_AVAILABLE else 'not available'
        }
        
        # GPU information
        if TF_AVAILABLE:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            info['gpus'] = []
            for gpu in gpus:
                gpu_info = {'name': gpu.name, 'type': gpu.device_type}
                try:
                    # Get GPU memory info if available
                    details = tf.config.experimental.get_device_details(gpu)
                    if details:
                        gpu_info.update(details)
                except Exception:
                    pass
                info['gpus'].append(gpu_info)
        
        return info
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models for benchmarking."""
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
        
        # TFLite models
        for tflite_file in self.models_dir.rglob("*.tflite"):
            models.append({
                'name': tflite_file.stem,
                'path': str(tflite_file),
                'format': 'tflite',
                'size_mb': tflite_file.stat().st_size / (1024 * 1024)
            })
        
        logger.info(f"Discovered {len(models)} models for benchmarking")
        return models
    
    def load_model(self, model_info: Dict[str, Any]) -> Any:
        """Load model based on format."""
        model_path = model_info['path']
        model_format = model_info['format']
        
        try:
            if model_format in ['h5', 'savedmodel']:
                model = tf.keras.models.load_model(model_path)
                return model
            elif model_format == 'tflite':
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                return interpreter
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_info['name']}: {e}")
            raise
    
    def generate_test_data(self, model: Any, model_format: str, batch_size: int) -> np.ndarray:
        """Generate test data for the model."""
        if model_format == 'tflite':
            input_details = model.get_input_details()
            input_shape = input_details[0]['shape'].copy()
            input_shape[0] = batch_size  # Set batch size
            data_type = input_details[0]['dtype']
        else:
            input_shape = list(model.input_shape)
            input_shape[0] = batch_size  # Set batch size
            data_type = np.float32
        
        # Generate appropriate test data based on shape
        if len(input_shape) == 4:  # Image data (batch, height, width, channels)
            test_data = np.random.random(input_shape).astype(data_type)
        elif len(input_shape) == 2:  # Tabular data (batch, features)
            test_data = np.random.randn(*input_shape).astype(data_type)
        elif len(input_shape) == 3:  # Sequence data (batch, sequence, features)
            test_data = np.random.random(input_shape).astype(data_type)
        else:
            test_data = np.random.random(input_shape).astype(data_type)
        
        return test_data
    
    def benchmark_keras_model(self, model: tf.keras.Model, model_name: str, 
                            batch_size: int) -> Dict[str, Any]:
        """Benchmark Keras model inference."""
        test_data = self.generate_test_data(model, 'keras', batch_size)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = model.predict(test_data, verbose=0)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Benchmark
        times = []
        start_time = time.time()
        
        for _ in range(self.benchmark_iterations):
            iter_start = time.time()
            _ = model.predict(test_data, verbose=0)
            iter_end = time.time()
            times.append(iter_end - iter_start)
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        system_stats = self.monitor.get_stats()
        
        # Calculate metrics
        times = np.array(times)
        total_samples = self.benchmark_iterations * batch_size
        
        return {
            'model_name': model_name,
            'batch_size': batch_size,
            'model_format': 'keras',
            'iterations': self.benchmark_iterations,
            'total_time': total_time,
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times),
            'p50_latency': np.percentile(times, 50),
            'p90_latency': np.percentile(times, 90),
            'p95_latency': np.percentile(times, 95),
            'p99_latency': np.percentile(times, 99),
            'throughput': total_samples / total_time,
            'samples_per_second': batch_size / np.mean(times),
            'system_stats': system_stats
        }
    
    def benchmark_tflite_model(self, interpreter: tf.lite.Interpreter, model_name: str,
                             batch_size: int) -> Dict[str, Any]:
        """Benchmark TFLite model inference."""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        test_data = self.generate_test_data(interpreter, 'tflite', batch_size)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            interpreter.set_tensor(input_details[0]['index'], test_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Benchmark
        times = []
        start_time = time.time()
        
        for _ in range(self.benchmark_iterations):
            iter_start = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            iter_end = time.time()
            times.append(iter_end - iter_start)
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        system_stats = self.monitor.get_stats()
        
        # Calculate metrics
        times = np.array(times)
        total_samples = self.benchmark_iterations * batch_size
        
        return {
            'model_name': model_name,
            'batch_size': batch_size,
            'model_format': 'tflite',
            'iterations': self.benchmark_iterations,
            'total_time': total_time,
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times),
            'p50_latency': np.percentile(times, 50),
            'p90_latency': np.percentile(times, 90),
            'p95_latency': np.percentile(times, 95),
            'p99_latency': np.percentile(times, 99),
            'throughput': total_samples / total_time,
            'samples_per_second': batch_size / np.mean(times),
            'system_stats': system_stats
        }
    
    def benchmark_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single model across different batch sizes."""
        model_name = model_info['name']
        logger.info(f"Benchmarking model: {model_name}")
        
        try:
            model = self.load_model(model_info)
            model_results = {
                'model_info': model_info,
                'batch_results': {},
                'summary': {}
            }
            
            # Test different batch sizes
            valid_batch_sizes = []
            
            for batch_size in self.batch_sizes:
                try:
                    logger.info(f"  Testing batch size: {batch_size}")
                    
                    if model_info['format'] == 'tflite':
                        batch_result = self.benchmark_tflite_model(model, model_name, batch_size)
                    else:
                        batch_result = self.benchmark_keras_model(model, model_name, batch_size)
                    
                    model_results['batch_results'][batch_size] = batch_result
                    valid_batch_sizes.append(batch_size)
                    
                except Exception as e:
                    logger.warning(f"    Failed batch size {batch_size}: {e}")
                    continue
            
            # Calculate summary statistics
            if valid_batch_sizes:
                model_results['summary'] = self._calculate_model_summary(model_results['batch_results'])
                
            logger.info(f"  Completed benchmarking {model_name}")
            return model_results
            
        except Exception as e:
            logger.error(f"Failed to benchmark model {model_name}: {e}")
            return {
                'model_info': model_info,
                'error': str(e)
            }
    
    def _calculate_model_summary(self, batch_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across batch sizes."""
        throughputs = []
        latencies = []
        
        for batch_size, result in batch_results.items():
            throughputs.append(result['throughput'])
            latencies.append(result['mean_latency'])
        
        return {
            'best_throughput': max(throughputs),
            'best_throughput_batch_size': max(batch_results.keys(), key=lambda x: batch_results[x]['throughput']),
            'lowest_latency': min(latencies),
            'lowest_latency_batch_size': min(batch_results.keys(), key=lambda x: batch_results[x]['mean_latency']),
            'mean_throughput': np.mean(throughputs),
            'mean_latency': np.mean(latencies),
            'throughput_std': np.std(throughputs),
            'latency_std': np.std(latencies)
        }
    
    def run_benchmark_suite(self, models: List[Dict[str, Any]], 
                          model_names: List[str] = None) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("🚀 Starting inference benchmark suite")
        
        # Filter models if specific names provided
        if model_names:
            models = [m for m in models if m['name'] in model_names]
        
        if not models:
            logger.error("No models to benchmark!")
            return {}
        
        logger.info(f"Benchmarking {len(models)} models: {[m['name'] for m in models]}")
        
        benchmark_results = {
            'system_info': self.system_info,
            'benchmark_config': {
                'batch_sizes': self.batch_sizes,
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations
            },
            'models': {},
            'summary': {}
        }
        
        # Benchmark each model
        for model_info in models:
            model_name = model_info['name']
            model_result = self.benchmark_model(model_info)
            benchmark_results['models'][model_name] = model_result
        
        # Calculate global summary
        benchmark_results['summary'] = self._calculate_global_summary(benchmark_results['models'])
        
        logger.info("🎉 Benchmark suite completed")
        return benchmark_results
    
    def _calculate_global_summary(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary across all models."""
        successful_models = [name for name, result in model_results.items() if 'error' not in result]
        failed_models = [name for name, result in model_results.items() if 'error' in result]
        
        summary = {
            'total_models': len(model_results),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(model_results) * 100 if model_results else 0
        }
        
        if successful_models:
            # Find best performing models
            best_throughput_model = None
            best_throughput = 0
            best_latency_model = None
            best_latency = float('inf')
            
            for model_name in successful_models:
                model_result = model_results[model_name]
                if 'summary' in model_result:
                    model_summary = model_result['summary']
                    
                    if model_summary.get('best_throughput', 0) > best_throughput:
                        best_throughput = model_summary['best_throughput']
                        best_throughput_model = model_name
                    
                    if model_summary.get('lowest_latency', float('inf')) < best_latency:
                        best_latency = model_summary['lowest_latency']
                        best_latency_model = model_name
            
            summary.update({
                'best_throughput_model': best_throughput_model,
                'best_throughput_value': best_throughput,
                'best_latency_model': best_latency_model,
                'best_latency_value': best_latency
            })
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"inference_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return str(results_file)
    
    def generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV report for easy analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = self.results_dir / f"inference_report_{timestamp}.csv"
        
        # Flatten results for CSV
        rows = []
        
        for model_name, model_result in results['models'].items():
            if 'error' in model_result:
                rows.append({
                    'model_name': model_name,
                    'model_format': model_result['model_info'].get('format', 'unknown'),
                    'model_size_mb': model_result['model_info'].get('size_mb', 0),
                    'batch_size': 'N/A',
                    'error': model_result['error']
                })
                continue
            
            model_info = model_result['model_info']
            
            for batch_size, batch_result in model_result.get('batch_results', {}).items():
                rows.append({
                    'model_name': model_name,
                    'model_format': model_info.get('format', 'unknown'),
                    'model_size_mb': model_info.get('size_mb', 0),
                    'batch_size': batch_size,
                    'mean_latency_ms': batch_result['mean_latency'] * 1000,
                    'std_latency_ms': batch_result['std_latency'] * 1000,
                    'p50_latency_ms': batch_result['p50_latency'] * 1000,
                    'p95_latency_ms': batch_result['p95_latency'] * 1000,
                    'p99_latency_ms': batch_result['p99_latency'] * 1000,
                    'throughput': batch_result['throughput'],
                    'samples_per_second': batch_result['samples_per_second'],
                    'cpu_usage_mean': batch_result['system_stats'].get('cpu', {}).get('mean', 0),
                    'memory_usage_mean': batch_result['system_stats'].get('memory', {}).get('mean', 0)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"CSV report saved to {csv_file}")
        return str(csv_file)


def main():
    """Main benchmarking script."""
    parser = argparse.ArgumentParser(description="Benchmark model inference performance")
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing models to benchmark"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Specific model names to benchmark"
    )
    
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32],
        help="Batch sizes to test"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    
    parser.add_argument(
        "--output-csv",
        action="store_true",
        help="Generate CSV report"
    )
    
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Please install TensorFlow.")
        sys.exit(1)
    
    logger.info("🚀 Starting TensorVerseHub Inference Benchmarking")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    
    # Initialize benchmarker
    benchmarker = InferenceBenchmarker(args.models_dir, args.results_dir)
    
    # Configure benchmarker
    benchmarker.batch_sizes = args.batch_sizes
    benchmarker.benchmark_iterations = args.iterations
    benchmarker.warmup_iterations = args.warmup_iterations
    
    # Discover models
    models = benchmarker.discover_models()
    
    if not models:
        logger.error("No models found for benchmarking!")
        sys.exit(1)
    
    # Run benchmarks
    results = benchmarker.run_benchmark_suite(models, args.model_names)
    
    if not results:
        logger.error("Benchmarking failed!")
        sys.exit(1)
    
    # Save results
    results_file = benchmarker.save_results(results)
    
    # Generate CSV report if requested
    if args.output_csv:
        csv_file = benchmarker.generate_csv_report(results)
    
    # Display summary
    summary = results.get('summary', {})
    logger.info("📊 Benchmark Summary:")
    logger.info(f"  Total models: {summary.get('total_models', 0)}")
    logger.info(f"  Successful: {summary.get('successful_models', 0)}")
    logger.info(f"  Failed: {summary.get('failed_models', 0)}")
    
    if summary.get('best_throughput_model'):
        logger.info(f"  Best throughput: {summary['best_throughput_model']} "
                   f"({summary['best_throughput_value']:.2f} samples/sec)")
    
    if summary.get('best_latency_model'):
        logger.info(f"  Best latency: {summary['best_latency_model']} "
                   f"({summary['best_latency_value']*1000:.2f} ms)")
    
    logger.info(f"📄 Detailed results: {results_file}")


if __name__ == "__main__":
    main()