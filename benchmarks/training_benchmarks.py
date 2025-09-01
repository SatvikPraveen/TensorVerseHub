# Location: /benchmarks/training_benchmarks.py

"""
Training performance analysis for TensorVerseHub.
Comprehensive benchmarking of model training performance across different configurations.
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
from typing import Dict, List, Any, Tuple, Optional, Callable
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
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_benchmarks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training metrics and system resources."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset monitoring data."""
        self.epoch_times = []
        self.batch_times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.learning_rates = []
        self.start_time = None
        self.end_time = None
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        self.end_time = time.time()
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_system(self):
        """System monitoring loop."""
        while self.monitoring:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # GPU memory monitoring (simplified)
            try:
                if tf.config.experimental.list_physical_devices('GPU'):
                    # This would require nvidia-ml-py for accurate GPU monitoring
                    self.gpu_memory_usage.append(0.0)  # Placeholder
            except Exception:
                pass
            
            time.sleep(1.0)  # Monitor every second
    
    def log_epoch(self, epoch: int, logs: Dict[str, Any]):
        """Log epoch metrics."""
        if logs:
            self.training_loss.append(logs.get('loss', 0))
            self.training_accuracy.append(logs.get('accuracy', 0))
            self.validation_loss.append(logs.get('val_loss', 0))
            self.validation_accuracy.append(logs.get('val_accuracy', 0))
            
            # Learning rate
            if hasattr(logs, 'get') and 'lr' in logs:
                self.learning_rates.append(logs['lr'])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        total_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        summary = {
            'total_time': total_time,
            'epochs': len(self.training_loss),
            'avg_epoch_time': total_time / len(self.training_loss) if self.training_loss else 0
        }
        
        # System resources
        if self.cpu_usage:
            summary['cpu_stats'] = {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'std': np.std(self.cpu_usage)
            }
        
        if self.memory_usage:
            summary['memory_stats'] = {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'std': np.std(self.memory_usage)
            }
        
        # Training metrics
        if self.training_loss:
            summary['training_metrics'] = {
                'final_loss': self.training_loss[-1],
                'best_loss': min(self.training_loss),
                'final_accuracy': self.training_accuracy[-1] if self.training_accuracy else 0,
                'best_accuracy': max(self.training_accuracy) if self.training_accuracy else 0
            }
        
        if self.validation_loss:
            summary['validation_metrics'] = {
                'final_val_loss': self.validation_loss[-1],
                'best_val_loss': min(self.validation_loss),
                'final_val_accuracy': self.validation_accuracy[-1] if self.validation_accuracy else 0,
                'best_val_accuracy': max(self.validation_accuracy) if self.validation_accuracy else 0
            }
        
        return summary


class ModelFactory:
    """Factory for creating benchmark models."""
    
    @staticmethod
    def create_simple_cnn(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """Create a simple CNN for benchmarking."""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_deep_cnn(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """Create a deeper CNN for benchmarking."""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_mlp(input_shape: int, num_classes: int, hidden_layers: List[int] = None) -> tf.keras.Model:
        """Create an MLP for benchmarking."""
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        model = keras.Sequential()
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(input_shape,)))
        
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    
    @staticmethod
    def create_lstm(vocab_size: int, seq_length: int, num_classes: int) -> tf.keras.Model:
        """Create an LSTM for benchmarking."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, 128, input_length=seq_length),
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model


class DataGenerator:
    """Generate synthetic data for training benchmarks."""
    
    @staticmethod
    def generate_image_data(num_samples: int, input_shape: Tuple[int, ...], 
                          num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic image data."""
        X = np.random.random((num_samples,) + input_shape).astype(np.float32)
        y = np.random.randint(0, num_classes, num_samples)
        return X, y
    
    @staticmethod
    def generate_tabular_data(num_samples: int, num_features: int, 
                            num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic tabular data."""
        X = np.random.randn(num_samples, num_features).astype(np.float32)
        y = np.random.randint(0, num_classes, num_samples)
        return X, y
    
    @staticmethod
    def generate_sequence_data(num_samples: int, seq_length: int, vocab_size: int,
                             num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic sequence data."""
        X = np.random.randint(0, vocab_size, (num_samples, seq_length))
        y = np.random.randint(0, num_classes, num_samples)
        return X, y
    
    @staticmethod
    def create_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int,
                         shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.EXPERIMENTAL_AUTOTUNE)
        
        return dataset


class TrainingBenchmarker:
    """Comprehensive training performance benchmarking."""
    
    def __init__(self, results_dir: str):
        """
        Initialize training benchmarker.
        
        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # System information
        self.system_info = self._get_system_info()
        
        # Benchmark configurations
        self.model_configs = {
            'simple_cnn': {
                'model_factory': ModelFactory.create_simple_cnn,
                'data_generator': DataGenerator.generate_image_data,
                'input_shape': (32, 32, 3),
                'num_classes': 10,
                'data_args': {'input_shape': (32, 32, 3)}
            },
            'deep_cnn': {
                'model_factory': ModelFactory.create_deep_cnn,
                'data_generator': DataGenerator.generate_image_data,
                'input_shape': (64, 64, 3),
                'num_classes': 100,
                'data_args': {'input_shape': (64, 64, 3)}
            },
            'mlp_small': {
                'model_factory': lambda input_shape, num_classes: ModelFactory.create_mlp(input_shape, num_classes, [64, 32]),
                'data_generator': DataGenerator.generate_tabular_data,
                'input_shape': 50,
                'num_classes': 10,
                'data_args': {'num_features': 50}
            },
            'mlp_large': {
                'model_factory': lambda input_shape, num_classes: ModelFactory.create_mlp(input_shape, num_classes, [512, 256, 128]),
                'data_generator': DataGenerator.generate_tabular_data,
                'input_shape': 200,
                'num_classes': 20,
                'data_args': {'num_features': 200}
            },
            'lstm': {
                'model_factory': ModelFactory.create_lstm,
                'data_generator': DataGenerator.generate_sequence_data,
                'input_shape': (100, 1000, 50),  # (seq_length, vocab_size, num_classes)
                'num_classes': 50,
                'data_args': {'seq_length': 100, 'vocab_size': 1000}
            }
        }
        
        # Benchmark parameters
        self.batch_sizes = [16, 32, 64, 128]
        self.dataset_sizes = [1000, 5000, 10000]
        self.epochs = 10
        
        # Results storage
        self.benchmark_results = {}
        
        # Monitor
        self.monitor = TrainingMonitor()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'tensorflow_version': tf.__version__ if TF_AVAILABLE else 'not available'
        }
        
        # GPU information
        if TF_AVAILABLE:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            info['gpu_count'] = len(gpus)
            info['gpu_names'] = [gpu.name for gpu in gpus]
        
        return info
    
    def benchmark_model_config(self, config_name: str, config: Dict[str, Any],
                             batch_size: int, dataset_size: int) -> Dict[str, Any]:
        """Benchmark a specific model configuration."""
        logger.info(f"Benchmarking {config_name} (batch_size={batch_size}, dataset_size={dataset_size})")
        
        try:
            # Reset monitor
            self.monitor.reset()
            
            # Create model
            if config_name == 'lstm':
                seq_length, vocab_size, num_classes = config['input_shape']
                model = config['model_factory'](vocab_size, seq_length, num_classes)
            else:
                model = config['model_factory'](config['input_shape'], config['num_classes'])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Generate data
            if config_name == 'lstm':
                seq_length, vocab_size, num_classes = config['input_shape']
                X_train, y_train = config['data_generator'](
                    dataset_size, seq_length, vocab_size, num_classes
                )
                X_val, y_val = config['data_generator'](
                    dataset_size // 5, seq_length, vocab_size, num_classes
                )
            else:
                data_args = config['data_args'].copy()
                data_args['num_classes'] = config['num_classes']
                
                X_train, y_train = config['data_generator'](dataset_size, **data_args)
                X_val, y_val = config['data_generator'](dataset_size // 5, **data_args)
            
            # Create datasets
            train_dataset = DataGenerator.create_tf_dataset(X_train, y_train, batch_size)
            val_dataset = DataGenerator.create_tf_dataset(X_val, y_val, batch_size, shuffle=False)
            
            # Setup callbacks
            class BenchmarkCallback(keras.callbacks.Callback):
                def __init__(self, monitor):
                    super().__init__()
                    self.monitor = monitor
                
                def on_epoch_end(self, epoch, logs=None):
                    self.monitor.log_epoch(epoch, logs or {})
            
            callbacks = [
                BenchmarkCallback(self.monitor),
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Train model
            start_time = time.time()
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=0
            )
            training_time = time.time() - start_time
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Get model info
            model_params = model.count_params()
            model_size_mb = model_params * 4 / (1024 * 1024)  # Approximate size
            
            # Calculate metrics
            monitor_summary = self.monitor.get_summary()
            
            result = {
                'config_name': config_name,
                'batch_size': batch_size,
                'dataset_size': dataset_size,
                'model_params': int(model_params),
                'model_size_mb': model_size_mb,
                'training_time': training_time,
                'epochs_completed': len(history.history['loss']),
                'avg_epoch_time': training_time / len(history.history['loss']),
                'samples_per_second': (dataset_size * len(history.history['loss'])) / training_time,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history.get('accuracy', [0])[-1],
                'best_val_loss': min(history.history.get('val_loss', [float('inf')])),
                'best_val_accuracy': max(history.history.get('val_accuracy', [0])),
                'monitor_summary': monitor_summary,
                'history': {k: v for k, v in history.history.items()}
            }
            
            logger.info(f"  Completed in {training_time:.2f}s, "
                       f"final accuracy: {result['final_accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            return {
                'config_name': config_name,
                'batch_size': batch_size,
                'dataset_size': dataset_size,
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self, config_names: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive training benchmark."""
        logger.info("🚀 Starting comprehensive training benchmark")
        
        # Use all configs if none specified
        if config_names is None:
            config_names = list(self.model_configs.keys())
        
        # Filter configs
        configs_to_test = {name: config for name, config in self.model_configs.items() 
                          if name in config_names}
        
        if not configs_to_test:
            logger.error("No valid configurations to benchmark!")
            return {}
        
        logger.info(f"Benchmarking {len(configs_to_test)} configurations: {list(configs_to_test.keys())}")
        
        benchmark_results = {
            'system_info': self.system_info,
            'benchmark_config': {
                'batch_sizes': self.batch_sizes,
                'dataset_sizes': self.dataset_sizes,
                'epochs': self.epochs
            },
            'results': [],
            'summary': {}
        }
        
        # Run benchmarks
        total_tests = len(configs_to_test) * len(self.batch_sizes) * len(self.dataset_sizes)
        current_test = 0
        
        for config_name, config in configs_to_test.items():
            for batch_size in self.batch_sizes:
                for dataset_size in self.dataset_sizes:
                    current_test += 1
                    logger.info(f"Progress: {current_test}/{total_tests}")
                    
                    # Skip large combinations that might be too slow
                    if config_name == 'deep_cnn' and dataset_size > 5000 and batch_size < 32:
                        logger.info(f"  Skipping slow combination: {config_name}, {batch_size}, {dataset_size}")
                        continue
                    
                    result = self.benchmark_model_config(config_name, config, batch_size, dataset_size)
                    benchmark_results['results'].append(result)
        
        # Calculate summary
        benchmark_results['summary'] = self._calculate_summary(benchmark_results['results'])
        
        logger.info("🎉 Comprehensive benchmark completed")
        return benchmark_results
    
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        summary = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'failed_tests': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100 if results else 0
        }
        
        if successful_results:
            # Performance metrics
            training_times = [r['training_time'] for r in successful_results]
            throughputs = [r['samples_per_second'] for r in successful_results]
            accuracies = [r['best_val_accuracy'] for r in successful_results if r['best_val_accuracy'] > 0]
            
            summary.update({
                'avg_training_time': np.mean(training_times),
                'fastest_training_time': min(training_times),
                'slowest_training_time': max(training_times),
                'avg_throughput': np.mean(throughputs),
                'max_throughput': max(throughputs),
                'min_throughput': min(throughputs),
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'max_accuracy': max(accuracies) if accuracies else 0
            })
            
            # Best performing configurations
            best_throughput = max(successful_results, key=lambda x: x['samples_per_second'])
            best_accuracy = max(successful_results, key=lambda x: x['best_val_accuracy'])
            fastest_training = min(successful_results, key=lambda x: x['training_time'])
            
            summary.update({
                'best_throughput_config': {
                    'config': best_throughput['config_name'],
                    'batch_size': best_throughput['batch_size'],
                    'dataset_size': best_throughput['dataset_size'],
                    'throughput': best_throughput['samples_per_second']
                },
                'best_accuracy_config': {
                    'config': best_accuracy['config_name'],
                    'batch_size': best_accuracy['batch_size'],
                    'dataset_size': best_accuracy['dataset_size'],
                    'accuracy': best_accuracy['best_val_accuracy']
                },
                'fastest_training_config': {
                    'config': fastest_training['config_name'],
                    'batch_size': fastest_training['batch_size'],
                    'dataset_size': fastest_training['dataset_size'],
                    'training_time': fastest_training['training_time']
                }
            })
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return str(results_file)
    
    def generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV report for analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = self.results_dir / f"training_metrics_{timestamp}.csv"
        
        # Flatten results for CSV
        rows = []
        for result in results['results']:
            if 'error' not in result:
                rows.append({
                    'config_name': result['config_name'],
                    'batch_size': result['batch_size'],
                    'dataset_size': result['dataset_size'],
                    'model_params': result['model_params'],
                    'model_size_mb': result['model_size_mb'],
                    'training_time': result['training_time'],
                    'epochs_completed': result['epochs_completed'],
                    'avg_epoch_time': result['avg_epoch_time'],
                    'samples_per_second': result['samples_per_second'],
                    'final_loss': result['final_loss'],
                    'final_accuracy': result['final_accuracy'],
                    'best_val_loss': result['best_val_loss'],
                    'best_val_accuracy': result['best_val_accuracy'],
                    'avg_cpu_usage': result['monitor_summary'].get('cpu_stats', {}).get('mean', 0),
                    'max_cpu_usage': result['monitor_summary'].get('cpu_stats', {}).get('max', 0),
                    'avg_memory_usage': result['monitor_summary'].get('memory_stats', {}).get('mean', 0),
                    'max_memory_usage': result['monitor_summary'].get('memory_stats', {}).get('max', 0)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"CSV report saved to {csv_file}")
        return str(csv_file)


def main():
    """Main training benchmarking script."""
    parser = argparse.ArgumentParser(description="Benchmark training performance")
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["simple_cnn", "deep_cnn", "mlp_small", "mlp_large", "lstm"],
        help="Model configurations to benchmark"
    )
    
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64],
        help="Batch sizes to test"
    )
    
    parser.add_argument(
        "--dataset-sizes",
        nargs="+",
        type=int,
        default=[1000, 5000],
        help="Dataset sizes to test"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
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
    
    logger.info("🚀 Starting TensorVerseHub Training Benchmarking")
    logger.info(f"Results directory: {args.results_dir}")
    
    # Initialize benchmarker
    benchmarker = TrainingBenchmarker(args.results_dir)
    
    # Configure benchmarker
    benchmarker.batch_sizes = args.batch_sizes
    benchmarker.dataset_sizes = args.dataset_sizes
    benchmarker.epochs = args.epochs
    
    # Run benchmarks
    results = benchmarker.run_comprehensive_benchmark(args.configs)
    
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
    logger.info("📊 Training Benchmark Summary:")
    logger.info(f"  Total tests: {summary.get('total_tests', 0)}")
    logger.info(f"  Successful: {summary.get('successful_tests', 0)}")
    logger.info(f"  Failed: {summary.get('failed_tests', 0)}")
    logger.info(f"  Success rate: {summary.get('success_rate', 0):.1f}%")
    
    if 'best_throughput_config' in summary:
        best = summary['best_throughput_config']
        logger.info(f"  Best throughput: {best['config']} "
                   f"(batch={best['batch_size']}, samples={best['dataset_size']}) "
                   f"- {best['throughput']:.2f} samples/sec")
    
    if 'fastest_training_config' in summary:
        fastest = summary['fastest_training_config']
        logger.info(f"  Fastest training: {fastest['config']} "
                   f"(batch={fastest['batch_size']}, samples={fastest['dataset_size']}) "
                   f"- {fastest['training_time']:.2f}s")
    
    logger.info(f"📄 Detailed results: {results_file}")


if __name__ == "__main__":
    main()