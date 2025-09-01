# Location: /scripts/evaluate_models.py

"""
Model evaluation and benchmarking script for TensorVerseHub.
Automated evaluation of trained models with comprehensive metrics and reporting.
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow/sklearn not available: {e}")
    TF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and benchmarking."""
    
    def __init__(self, models_dir: str, data_dir: str, output_dir: str):
        """
        Initialize model evaluator.
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing test/validation data
            output_dir: Directory for evaluation outputs
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        self.benchmark_results = {}
        
        # Visualization setup
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        sns.set_palette("husl")
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discover all models in the models directory.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_file in self.models_dir.rglob("*.h5"):
            model_info = {
                'name': model_file.stem,
                'path': str(model_file),
                'type': self._infer_model_type(model_file),
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
            }
            models.append(model_info)
        
        # Also check for SavedModel format
        for saved_model_dir in self.models_dir.rglob("saved_model.pb"):
            parent_dir = saved_model_dir.parent
            model_info = {
                'name': parent_dir.name,
                'path': str(parent_dir),
                'type': 'savedmodel',
                'size_mb': sum(f.stat().st_size for f in parent_dir.rglob("*") if f.is_file()) / (1024 * 1024),
                'modified': datetime.fromtimestamp(saved_model_dir.stat().st_mtime)
            }
            models.append(model_info)
        
        logger.info(f"Discovered {len(models)} models for evaluation")
        return models
    
    def _infer_model_type(self, model_path: Path) -> str:
        """Infer model type from filename and path."""
        name = model_path.name.lower()
        
        if 'cnn' in name or 'conv' in name or 'resnet' in name or 'vgg' in name:
            return 'cnn'
        elif 'rnn' in name or 'lstm' in name or 'gru' in name:
            return 'rnn'
        elif 'mlp' in name or 'dense' in name or 'fc' in name:
            return 'mlp'
        elif 'transformer' in name or 'bert' in name or 'attention' in name:
            return 'transformer'
        else:
            return 'unknown'
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load model from path.
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            Loaded Keras model
        """
        try:
            if model_path.endswith('.h5'):
                model = tf.keras.models.load_model(model_path)
            else:
                # Assume SavedModel format
                model = tf.keras.models.load_model(model_path)
            
            logger.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def load_test_data(self, data_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data based on data type.
        
        Args:
            data_type: Type of data ('images', 'tabular', 'text')
            
        Returns:
            Tuple of (X_test, y_test)
        """
        if data_type == 'images':
            return self._load_image_test_data()
        elif data_type == 'tabular':
            return self._load_tabular_test_data()
        elif data_type == 'text':
            return self._load_text_test_data()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _load_image_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load image test data."""
        test_dir = self.data_dir / "images" / "test"
        
        if not test_dir.exists():
            # Use validation data if test data doesn't exist
            test_dir = self.data_dir / "images" / "val"
        
        if not test_dir.exists():
            raise FileNotFoundError(f"No image test data found in {self.data_dir}")
        
        # Load images and labels
        images = []
        labels = []
        
        class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = test_dir / class_name
            
            for image_file in class_dir.rglob("*.npy"):
                image = np.load(image_file)
                images.append(image)
                labels.append(class_idx)
        
        X_test = np.array(images).astype('float32') / 255.0
        y_test = np.array(labels)
        
        logger.info(f"Loaded {len(X_test)} image test samples")
        return X_test, y_test
    
    def _load_tabular_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load tabular test data."""
        tabular_dir = self.data_dir / "tabular"
        
        # Try different test file names
        test_files = [
            ("X_test.npy", "y_test.npy"),
            ("X_val.npy", "y_val.npy")
        ]
        
        for x_file, y_file in test_files:
            x_path = tabular_dir / x_file
            y_path = tabular_dir / y_file
            
            if x_path.exists() and y_path.exists():
                X_test = np.load(x_path)
                y_test = np.load(y_path)
                
                logger.info(f"Loaded {len(X_test)} tabular test samples from {x_file}")
                return X_test, y_test
        
        raise FileNotFoundError(f"No tabular test data found in {tabular_dir}")
    
    def _load_text_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load text test data."""
        text_dir = self.data_dir / "text"
        
        # Try different test file names
        test_files = [
            ("X_test.npy", "y_test.npy"),
            ("X_val.npy", "y_val.npy")
        ]
        
        for x_file, y_file in test_files:
            x_path = text_dir / x_file
            y_path = text_dir / y_file
            
            if x_path.exists() and y_path.exists():
                X_test = np.load(x_path)
                y_test = np.load(y_path)
                
                logger.info(f"Loaded {len(X_test)} text test samples from {x_file}")
                return X_test, y_test
        
        raise FileNotFoundError(f"No text test data found in {text_dir}")
    
    def evaluate_model(self, model: tf.keras.Model, model_name: str, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test)
        }
        
        # Basic evaluation
        start_time = time.time()
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        eval_time = time.time() - start_time
        
        results.update({
            'loss': float(loss),
            'accuracy': float(accuracy),
            'evaluation_time': eval_time
        })
        
        # Predictions
        start_time = time.time()
        y_pred_proba = model.predict(X_test, verbose=0)
        prediction_time = time.time() - start_time
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        results['prediction_time'] = prediction_time
        results['inference_speed'] = len(X_test) / prediction_time  # samples per second
        
        # Classification metrics
        num_classes = len(np.unique(y_test))
        
        if num_classes <= 10:  # Only compute detailed metrics for reasonable number of classes
            # Classification report
            class_report = classification_report(
                y_test, y_pred, 
                output_dict=True, 
                zero_division=0
            )
            results['classification_report'] = class_report
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Per-class metrics
            results['per_class_metrics'] = {
                'precision': class_report['macro avg']['precision'],
                'recall': class_report['macro avg']['recall'],
                'f1_score': class_report['macro avg']['f1-score']
            }
            
            # ROC curves for binary/multiclass
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                results['roc_auc'] = float(roc_auc)
                results['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            elif num_classes <= 5:
                # Multiclass ROC
                y_test_binarized = label_binarize(y_test, classes=range(num_classes))
                roc_aucs = []
                
                for i in range(num_classes):
                    if len(np.unique(y_test_binarized[:, i])) > 1:
                        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                        roc_aucs.append(auc(fpr, tpr))
                    else:
                        roc_aucs.append(0.0)
                
                results['roc_auc_multiclass'] = roc_aucs
                results['mean_roc_auc'] = float(np.mean(roc_aucs))
        
        # Model complexity metrics
        results['model_complexity'] = self._analyze_model_complexity(model)
        
        logger.info(f"Model {model_name} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return results
    
    def _analyze_model_complexity(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Analyze model complexity metrics."""
        complexity = {
            'total_params': int(model.count_params()),
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
            'non_trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
            'layers': len(model.layers)
        }
        
        # Layer type distribution
        layer_types = {}
        for layer in model.layers:
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        complexity['layer_distribution'] = layer_types
        
        return complexity
    
    def benchmark_model(self, model: tf.keras.Model, model_name: str, 
                       X_test: np.ndarray) -> Dict[str, Any]:
        """
        Benchmark model performance (speed, memory usage).
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test data for benchmarking
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking model: {model_name}")
        
        benchmark = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Warm up the model
        _ = model.predict(X_test[:min(10, len(X_test))], verbose=0)
        
        # Batch size benchmarking
        batch_sizes = [1, 8, 16, 32, 64, 128]
        batch_benchmarks = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                continue
            
            batch_data = X_test[:batch_size]
            
            # Multiple runs for accuracy
            times = []
            for _ in range(5):
                start_time = time.time()
                _ = model.predict(batch_data, verbose=0)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            batch_benchmarks[f"batch_{batch_size}"] = {
                'avg_time': float(avg_time),
                'std_time': float(std_time),
                'throughput': float(batch_size / avg_time),  # samples per second
                'latency_per_sample': float(avg_time / batch_size) * 1000  # ms per sample
            }
        
        benchmark['batch_performance'] = batch_benchmarks
        
        # Memory usage estimation (approximate)
        try:
            # Get model memory footprint
            model_size_mb = self._estimate_model_memory(model)
            benchmark['estimated_memory_mb'] = model_size_mb
        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")
            benchmark['estimated_memory_mb'] = None
        
        return benchmark
    
    def _estimate_model_memory(self, model: tf.keras.Model) -> float:
        """Estimate model memory usage in MB."""
        # This is a rough estimation
        total_params = model.count_params()
        
        # Assume float32 (4 bytes per parameter) + overhead
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        # Add estimated overhead for activations and gradients
        memory_mb *= 1.5
        
        return memory_mb
    
    def create_evaluation_plots(self, results: Dict[str, Any]) -> None:
        """Create visualization plots for evaluation results."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        model_names = list(results.keys())
        
        # Accuracy comparison
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Loss comparison
        losses = [results[name]['loss'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, losses)
        plt.title('Model Loss Comparison')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance metrics
        if len(model_names) > 1:
            metrics_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracies,
                'Loss': losses,
                'Inference Speed': [results[name]['inference_speed'] for name in model_names],
                'Parameters': [results[name]['model_complexity']['total_params'] for name in model_names]
            })
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy vs Parameters
            axes[0, 0].scatter(metrics_df['Parameters'], metrics_df['Accuracy'])
            axes[0, 0].set_xlabel('Parameters')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy vs Model Size')
            
            # Loss vs Parameters
            axes[0, 1].scatter(metrics_df['Parameters'], metrics_df['Loss'])
            axes[0, 1].set_xlabel('Parameters')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Loss vs Model Size')
            
            # Inference Speed vs Parameters
            axes[1, 0].scatter(metrics_df['Parameters'], metrics_df['Inference Speed'])
            axes[1, 0].set_xlabel('Parameters')
            axes[1, 0].set_ylabel('Inference Speed (samples/sec)')
            axes[1, 0].set_title('Speed vs Model Size')
            
            # Accuracy vs Inference Speed
            axes[1, 1].scatter(metrics_df['Inference Speed'], metrics_df['Accuracy'])
            axes[1, 1].set_xlabel('Inference Speed (samples/sec)')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Accuracy vs Speed Trade-off')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Confusion matrices
        for model_name in model_names:
            if 'confusion_matrix' in results[model_name]:
                cm = np.array(results[model_name]['confusion_matrix'])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(plots_dir / f'confusion_matrix_{model_name}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Evaluation plots saved to {plots_dir}")
    
    def create_benchmark_plots(self, benchmarks: Dict[str, Any]) -> None:
        """Create benchmark visualization plots."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Throughput comparison across batch sizes
        model_names = list(benchmarks.keys())
        batch_sizes = [1, 8, 16, 32, 64, 128]
        
        plt.figure(figsize=(12, 8))
        
        for model_name in model_names:
            batch_perf = benchmarks[model_name]['batch_performance']
            throughputs = []
            valid_batch_sizes = []
            
            for batch_size in batch_sizes:
                key = f"batch_{batch_size}"
                if key in batch_perf:
                    throughputs.append(batch_perf[key]['throughput'])
                    valid_batch_sizes.append(batch_size)
            
            plt.plot(valid_batch_sizes, throughputs, marker='o', label=model_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Model Throughput vs Batch Size')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Latency comparison
        plt.figure(figsize=(12, 8))
        
        for model_name in model_names:
            batch_perf = benchmarks[model_name]['batch_performance']
            latencies = []
            valid_batch_sizes = []
            
            for batch_size in batch_sizes:
                key = f"batch_{batch_size}"
                if key in batch_perf:
                    latencies.append(batch_perf[key]['latency_per_sample'])
                    valid_batch_sizes.append(batch_size)
            
            plt.plot(valid_batch_sizes, latencies, marker='s', label=model_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Latency per Sample (ms)')
        plt.title('Model Latency vs Batch Size')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Benchmark plots saved to {plots_dir}")
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       benchmark_results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / "evaluation_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorVerseHub Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .model-section {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
                .best {{ background-color: #d4edda !important; }}
                .plot {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 TensorVerseHub Model Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total models evaluated: {len(evaluation_results)}</p>
            </div>
        """
        
        # Summary section
        html_content += """
            <div class="section">
                <h2>📊 Summary</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Loss</th>
                        <th>Parameters</th>
                        <th>Inference Speed (samples/sec)</th>
                        <th>Memory (MB)</th>
                    </tr>
        """
        
        # Find best models
        best_accuracy = max(evaluation_results.values(), key=lambda x: x['accuracy'])
        best_speed = max(benchmark_results.values(), key=lambda x: list(x['batch_performance'].values())[0]['throughput'])
        
        for model_name in evaluation_results.keys():
            eval_result = evaluation_results[model_name]
            benchmark_result = benchmark_results.get(model_name, {})
            
            is_best_acc = eval_result['accuracy'] == best_accuracy['accuracy']
            is_best_speed = (model_name == best_speed['model_name'])
            
            row_class = "best" if is_best_acc or is_best_speed else ""
            
            speed = "N/A"
            memory = "N/A"
            
            if benchmark_result:
                first_batch = list(benchmark_result['batch_performance'].values())[0]
                speed = f"{first_batch['throughput']:.2f}"
                memory = f"{benchmark_result.get('estimated_memory_mb', 0):.2f}"
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{model_name}</td>
                    <td>{eval_result['accuracy']:.4f}</td>
                    <td>{eval_result['loss']:.4f}</td>
                    <td>{eval_result['model_complexity']['total_params']:,}</td>
                    <td>{speed}</td>
                    <td>{memory}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Individual model sections
        html_content += '<div class="section"><h2>🔍 Detailed Results</h2>'
        
        for model_name, result in evaluation_results.items():
            html_content += f"""
                <div class="model-section">
                    <h3>📋 {model_name}</h3>
                    
                    <h4>Performance Metrics</h4>
                    <div class="metric">Accuracy: <strong>{result['accuracy']:.4f}</strong></div>
                    <div class="metric">Loss: <strong>{result['loss']:.4f}</strong></div>
                    <div class="metric">Test Samples: <strong>{result['test_samples']:,}</strong></div>
                    <div class="metric">Inference Speed: <strong>{result['inference_speed']:.2f} samples/sec</strong></div>
                    
                    <h4>Model Complexity</h4>
                    <div class="metric">Total Parameters: <strong>{result['model_complexity']['total_params']:,}</strong></div>
                    <div class="metric">Trainable Parameters: <strong>{result['model_complexity']['trainable_params']:,}</strong></div>
                    <div class="metric">Layers: <strong>{result['model_complexity']['layers']}</strong></div>
            """
            
            # Add classification metrics if available
            if 'per_class_metrics' in result:
                metrics = result['per_class_metrics']
                html_content += f"""
                    <h4>Classification Metrics</h4>
                    <div class="metric">Precision: <strong>{metrics['precision']:.4f}</strong></div>
                    <div class="metric">Recall: <strong>{metrics['recall']:.4f}</strong></div>
                    <div class="metric">F1-Score: <strong>{metrics['f1_score']:.4f}</strong></div>
                """
                
                if 'roc_auc' in result:
                    html_content += f'<div class="metric">ROC AUC: <strong>{result["roc_auc"]:.4f}</strong></div>'
            
            html_content += '</div>'
        
        html_content += '</div>'
        
        # Add plots section
        plots_dir = self.output_dir / "plots"
        if plots_dir.exists():
            html_content += """
                <div class="section">
                    <h2>📈 Visualizations</h2>
                    <div class="plot">
                        <h3>Model Accuracy Comparison</h3>
                        <img src="plots/accuracy_comparison.png" alt="Accuracy Comparison" style="max-width: 100%;">
                    </div>
                    
                    <div class="plot">
                        <h3>Performance Analysis</h3>
                        <img src="plots/performance_analysis.png" alt="Performance Analysis" style="max-width: 100%;">
                    </div>
                    
                    <div class="plot">
                        <h3>Throughput Comparison</h3>
                        <img src="plots/throughput_comparison.png" alt="Throughput Comparison" style="max-width: 100%;">
                    </div>
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return str(report_path)
    
    def save_results(self, evaluation_results: Dict[str, Any], 
                    benchmark_results: Dict[str, Any]) -> None:
        """Save results to JSON files."""
        # Save evaluation results
        eval_json_path = self.output_dir / "evaluation_results.json"
        with open(eval_json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Save benchmark results
        benchmark_json_path = self.output_dir / "benchmark_results.json"
        with open(benchmark_json_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {eval_json_path} and {benchmark_json_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate TensorVerseHub models")
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing test data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Specific model names to evaluate (optional)"
    )
    
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=["images", "tabular", "text"],
        default=["images"],
        help="Types of data to evaluate on"
    )
    
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip performance benchmarking"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Please install TensorFlow to run evaluations.")
        sys.exit(1)
    
    logger.info("🚀 Starting TensorVerseHub Model Evaluation")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.models_dir, args.data_dir, args.output_dir)
    
    # Discover models
    models = evaluator.discover_models()
    
    if not models:
        logger.error("No models found for evaluation!")
        sys.exit(1)
    
    # Filter models if specific names provided
    if args.model_names:
        models = [m for m in models if m['name'] in args.model_names]
        
    if not models:
        logger.error("No matching models found!")
        sys.exit(1)
    
    logger.info(f"Evaluating {len(models)} models: {[m['name'] for m in models]}")
    
    # Results storage
    evaluation_results = {}
    benchmark_results = {}
    
    # Evaluate each model
    for model_info in models:
        model_name = model_info['name']
        model_path = model_info['path']
        
        try:
            logger.info(f"Processing model: {model_name}")
            
            # Load model
            model = evaluator.load_model(model_path)
            
            # Determine data type based on model type
            if model_info['type'] == 'cnn':
                data_type = 'images'
            elif model_info['type'] == 'rnn':
                data_type = 'text'
            else:
                data_type = 'tabular'
            
            # Override if specified
            if data_type not in args.data_types:
                data_type = args.data_types[0]
            
            # Load test data
            try:
                X_test, y_test = evaluator.load_test_data(data_type)
            except FileNotFoundError as e:
                logger.warning(f"Test data not found for {data_type}: {e}")
                continue
            
            # Evaluate model
            eval_result = evaluator.evaluate_model(model, model_name, X_test, y_test)
            evaluation_results[model_name] = eval_result
            
            # Benchmark model
            if not args.skip_benchmarks:
                benchmark_result = evaluator.benchmark_model(model, model_name, X_test)
                benchmark_results[model_name] = benchmark_result
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}")
            continue
    
    if not evaluation_results:
        logger.error("No models were successfully evaluated!")
        sys.exit(1)
    
    # Save results
    evaluator.save_results(evaluation_results, benchmark_results)
    
    # Generate plots
    if not args.no_plots:
        try:
            evaluator.create_evaluation_plots(evaluation_results)
            if benchmark_results:
                evaluator.create_benchmark_plots(benchmark_results)
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    # Generate report
    try:
        report_path = evaluator.generate_report(evaluation_results, benchmark_results)
        logger.info(f"📄 Evaluation report generated: {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
    
    # Summary
    logger.info("🎉 Evaluation completed successfully!")
    logger.info(f"📊 Models evaluated: {len(evaluation_results)}")
    
    if evaluation_results:
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"🏆 Best performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")


if __name__ == "__main__":
    main()