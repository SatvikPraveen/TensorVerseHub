# Location: /benchmarks/memory_profiling.py

"""
Memory usage profiling for TensorVerseHub.
Comprehensive analysis of memory consumption during model training and inference.
"""

import os
import sys
import json
import time
import argparse
import logging
import psutil
import threading
import tracemalloc
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
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

# Memory profiling imports
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Info: memory_profiler not available. Install with: pip install memory-profiler")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_profiling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available memory
    gpu_memory_mb: float = 0.0  # GPU memory usage


class MemoryTracker:
    """Track memory usage over time."""
    
    def __init__(self, sample_interval: float = 0.5):
        """
        Initialize memory tracker.
        
        Args:
            sample_interval: Sampling interval in seconds
        """
        self.sample_interval = sample_interval
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
        self.tracking = False
        self.track_thread = None
        
        # Peak usage tracking
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self.peak_gpu_mb = 0.0
        
        # Baseline memory
        self.baseline_rss_mb = 0.0
        self.baseline_vms_mb = 0.0
    
    def start_tracking(self):
        """Start memory tracking."""
        # Record baseline
        memory_info = self.process.memory_info()
        self.baseline_rss_mb = memory_info.rss / (1024 * 1024)
        self.baseline_vms_mb = memory_info.vms / (1024 * 1024)
        
        self.tracking = True
        self.snapshots.clear()
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self.peak_gpu_mb = 0.0
        
        self.track_thread = threading.Thread(target=self._tracking_loop)
        self.track_thread.start()
    
    def stop_tracking(self):
        """Stop memory tracking."""
        self.tracking = False
        if self.track_thread:
            self.track_thread.join()
    
    def _tracking_loop(self):
        """Memory tracking loop."""
        while self.tracking:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            
            # Update peaks
            self.peak_rss_mb = max(self.peak_rss_mb, snapshot.rss_mb)
            self.peak_vms_mb = max(self.peak_vms_mb, snapshot.vms_mb)
            self.peak_gpu_mb = max(self.peak_gpu_mb, snapshot.gpu_memory_mb)
            
            time.sleep(self.sample_interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=self.process.memory_percent(),
            available_mb=system_memory.available / (1024 * 1024),
            gpu_memory_mb=self._get_gpu_memory()
        )
        
        return snapshot
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB."""
        try:
            if TF_AVAILABLE:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # This is simplified - real GPU memory tracking would need nvidia-ml-py
                    return 0.0  # Placeholder
            return 0.0
        except Exception:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        vms_values = [s.vms_mb for s in self.snapshots]
        
        return {
            'baseline_rss_mb': self.baseline_rss_mb,
            'baseline_vms_mb': self.baseline_vms_mb,
            'peak_rss_mb': self.peak_rss_mb,
            'peak_vms_mb': self.peak_vms_mb,
            'peak_gpu_mb': self.peak_gpu_mb,
            'avg_rss_mb': np.mean(rss_values),
            'avg_vms_mb': np.mean(vms_values),
            'min_rss_mb': np.min(rss_values),
            'min_vms_mb': np.min(vms_values),
            'max_rss_mb': np.max(rss_values),
            'max_vms_mb': np.max(vms_values),
            'std_rss_mb': np.std(rss_values),
            'std_vms_mb': np.std(vms_values),
            'memory_growth_mb': self.peak_rss_mb - self.baseline_rss_mb,
            'samples_count': len(self.snapshots),
            'tracking_duration': self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        }


class ModelMemoryProfiler:
    """Profile memory usage of ML models."""
    
    def __init__(self, results_dir: str):
        """
        Initialize memory profiler.
        
        Args:
            results_dir: Directory to save profiling results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_tracker = MemoryTracker()
        self.profiling_results = {}
        
        # System info
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        
        return {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'python_version': sys.version,
            'tensorflow_version': tf.__version__ if TF_AVAILABLE else 'not available',
            'timestamp': datetime.now().isoformat()
        }
    
    def profile_model_creation(self, model_factory: Callable, model_name: str, 
                             *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage during model creation."""
        logger.info(f"Profiling model creation: {model_name}")
        
        # Start tracking
        tracemalloc.start()
        self.memory_tracker.start_tracking()
        
        # Force garbage collection
        gc.collect()
        
        try:
            # Create model
            start_time = time.time()
            model = model_factory(*args, **kwargs)
            creation_time = time.time() - start_time
            
            # Get model info
            model_params = model.count_params() if hasattr(model, 'count_params') else 0
            model_size_mb = model_params * 4 / (1024 * 1024)  # Approximate
            
            # Stop tracking
            self.memory_tracker.stop_tracking()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get memory stats
            memory_stats = self.memory_tracker.get_stats()
            
            result = {
                'model_name': model_name,
                'creation_time': creation_time,
                'model_params': int(model_params),
                'model_size_mb': model_size_mb,
                'tracemalloc_current_mb': current / (1024 * 1024),
                'tracemalloc_peak_mb': peak / (1024 * 1024),
                'memory_stats': memory_stats
            }
            
            logger.info(f"  Created in {creation_time:.2f}s, "
                       f"peak memory: {memory_stats.get('peak_rss_mb', 0):.2f}MB")
            
            return result, model
            
        except Exception as e:
            self.memory_tracker.stop_tracking()
            tracemalloc.stop()
            logger.error(f"Model creation failed: {e}")
            return {'model_name': model_name, 'error': str(e)}, None
    
    def profile_training(self, model: tf.keras.Model, train_data: tf.data.Dataset,
                        val_data: tf.data.Dataset, epochs: int, 
                        model_name: str) -> Dict[str, Any]:
        """Profile memory usage during training."""
        logger.info(f"Profiling training: {model_name}")
        
        # Start tracking
        tracemalloc.start()
        self.memory_tracker.start_tracking()
        
        # Force garbage collection
        gc.collect()
        
        try:
            # Setup memory callback
            class MemoryCallback(keras.callbacks.Callback):
                def __init__(self):
                    super().__init__()
                    self.epoch_memories = []
                
                def on_epoch_end(self, epoch, logs=None):
                    current, _ = tracemalloc.get_traced_memory()
                    memory_info = psutil.Process().memory_info()
                    self.epoch_memories.append({
                        'epoch': epoch,
                        'tracemalloc_mb': current / (1024 * 1024),
                        'rss_mb': memory_info.rss / (1024 * 1024)
                    })
            
            memory_callback = MemoryCallback()
            
            # Train model
            start_time = time.time()
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=[memory_callback],
                verbose=0
            )
            training_time = time.time() - start_time
            
            # Stop tracking
            self.memory_tracker.stop_tracking()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get memory stats
            memory_stats = self.memory_tracker.get_stats()
            
            result = {
                'model_name': model_name,
                'training_time': training_time,
                'epochs_completed': len(history.history['loss']),
                'tracemalloc_current_mb': current / (1024 * 1024),
                'tracemalloc_peak_mb': peak / (1024 * 1024),
                'epoch_memories': memory_callback.epoch_memories,
                'memory_stats': memory_stats,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history.get('accuracy', [0])[-1]
            }
            
            logger.info(f"  Training completed in {training_time:.2f}s, "
                       f"peak memory: {memory_stats.get('peak_rss_mb', 0):.2f}MB")
            
            return result
            
        except Exception as e:
            self.memory_tracker.stop_tracking()
            tracemalloc.stop()
            logger.error(f"Training profiling failed: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    def profile_inference(self, model: tf.keras.Model, test_data: np.ndarray,
                         batch_sizes: List[int], model_name: str) -> Dict[str, Any]:
        """Profile memory usage during inference."""
        logger.info(f"Profiling inference: {model_name}")
        
        inference_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")
            
            # Prepare batch data
            if len(test_data) < batch_size:
                continue
            
            batch_data = test_data[:batch_size]
            
            # Start tracking
            tracemalloc.start()
            self.memory_tracker.start_tracking()
            
            # Force garbage collection
            gc.collect()
            
            try:
                # Warmup
                for _ in range(5):
                    _ = model.predict(batch_data, verbose=0)
                
                # Inference benchmark
                start_time = time.time()
                for _ in range(20):
                    _ = model.predict(batch_data, verbose=0)
                inference_time = time.time() - start_time
                
                # Stop tracking
                self.memory_tracker.stop_tracking()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Get memory stats
                memory_stats = self.memory_tracker.get_stats()
                
                inference_results[batch_size] = {
                    'batch_size': batch_size,
                    'inference_time': inference_time,
                    'avg_inference_time': inference_time / 20,
                    'tracemalloc_current_mb': current / (1024 * 1024),
                    'tracemalloc_peak_mb': peak / (1024 * 1024),
                    'memory_stats': memory_stats
                }
                
            except Exception as e:
                self.memory_tracker.stop_tracking()
                tracemalloc.stop()
                logger.warning(f"  Inference profiling failed for batch size {batch_size}: {e}")
                inference_results[batch_size] = {'batch_size': batch_size, 'error': str(e)}
        
        return {
            'model_name': model_name,
            'batch_results': inference_results
        }
    
    def profile_model_lifecycle(self, model_factory: Callable, model_name: str,
                               train_data: tf.data.Dataset, val_data: tf.data.Dataset,
                               test_data: np.ndarray, epochs: int = 5,
                               batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Profile complete model lifecycle."""
        logger.info(f"Profiling complete lifecycle: {model_name}")
        
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        
        lifecycle_results = {
            'model_name': model_name,
            'system_info': self.system_info,
            'stages': {}
        }
        
        # Stage 1: Model Creation
        creation_result, model = self.profile_model_creation(model_factory, model_name)
        lifecycle_results['stages']['creation'] = creation_result
        
        if model is None:
            return lifecycle_results
        
        # Compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Stage 2: Training
        training_result = self.profile_training(model, train_data, val_data, epochs, model_name)
        lifecycle_results['stages']['training'] = training_result
        
        # Stage 3: Inference
        inference_result = self.profile_inference(model, test_data, batch_sizes, model_name)
        lifecycle_results['stages']['inference'] = inference_result
        
        # Overall summary
        lifecycle_results['summary'] = self._calculate_lifecycle_summary(lifecycle_results)
        
        return lifecycle_results
    
    def _calculate_lifecycle_summary(self, lifecycle_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary of lifecycle profiling."""
        summary = {}
        
        # Creation summary
        if 'creation' in lifecycle_results['stages'] and 'error' not in lifecycle_results['stages']['creation']:
            creation = lifecycle_results['stages']['creation']
            summary['creation'] = {
                'time': creation.get('creation_time', 0),
                'peak_memory_mb': creation.get('memory_stats', {}).get('peak_rss_mb', 0),
                'model_size_mb': creation.get('model_size_mb', 0)
            }
        
        # Training summary
        if 'training' in lifecycle_results['stages'] and 'error' not in lifecycle_results['stages']['training']:
            training = lifecycle_results['stages']['training']
            summary['training'] = {
                'time': training.get('training_time', 0),
                'peak_memory_mb': training.get('memory_stats', {}).get('peak_rss_mb', 0),
                'memory_growth_mb': training.get('memory_stats', {}).get('memory_growth_mb', 0),
                'final_accuracy': training.get('final_accuracy', 0)
            }
        
        # Inference summary
        if 'inference' in lifecycle_results['stages']:
            inference = lifecycle_results['stages']['inference']
            batch_results = inference.get('batch_results', {})
            
            if batch_results:
                # Find best performing batch size
                valid_results = {k: v for k, v in batch_results.items() if 'error' not in v}
                
                if valid_results:
                    best_batch = min(valid_results.items(), 
                                   key=lambda x: x[1].get('avg_inference_time', float('inf')))
                    
                    summary['inference'] = {
                        'best_batch_size': best_batch[0],
                        'best_inference_time': best_batch[1].get('avg_inference_time', 0),
                        'peak_memory_mb': best_batch[1].get('memory_stats', {}).get('peak_rss_mb', 0)
                    }
        
        return summary
    
    def run_comprehensive_profiling(self, model_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive memory profiling on multiple models."""
        logger.info("🚀 Starting comprehensive memory profiling")
        
        profiling_results = {
            'system_info': self.system_info,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'summary': {}
        }
        
        for config_name, config in model_configs.items():
            try:
                logger.info(f"Profiling configuration: {config_name}")
                
                # Create data
                train_data = config['train_data']
                val_data = config['val_data'] 
                test_data = config['test_data']
                model_factory = config['model_factory']
                
                # Profile model
                lifecycle_result = self.profile_model_lifecycle(
                    model_factory, config_name, train_data, val_data, test_data
                )
                
                profiling_results['models'][config_name] = lifecycle_result
                
            except Exception as e:
                logger.error(f"Failed to profile {config_name}: {e}")
                profiling_results['models'][config_name] = {
                    'model_name': config_name,
                    'error': str(e)
                }
        
        # Calculate overall summary
        profiling_results['summary'] = self._calculate_overall_summary(profiling_results['models'])
        
        logger.info("🎉 Comprehensive profiling completed")
        return profiling_results
    
    def _calculate_overall_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall profiling summary."""
        successful_models = []
        failed_models = []
        
        for model_name, result in model_results.items():
            if 'error' in result:
                failed_models.append(model_name)
            else:
                successful_models.append(model_name)
        
        summary = {
            'total_models': len(model_results),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(model_results) * 100 if model_results else 0
        }
        
        if successful_models:
            # Memory efficiency analysis
            memory_efficiencies = []
            training_peaks = []
            model_sizes = []
            
            for model_name in successful_models:
                result = model_results[model_name]
                
                if 'summary' in result:
                    model_summary = result['summary']
                    
                    if 'creation' in model_summary:
                        model_sizes.append(model_summary['creation'].get('model_size_mb', 0))
                    
                    if 'training' in model_summary:
                        training_peaks.append(model_summary['training'].get('peak_memory_mb', 0))
                        
                        # Memory efficiency = accuracy / peak_memory
                        accuracy = model_summary['training'].get('final_accuracy', 0)
                        peak_memory = model_summary['training'].get('peak_memory_mb', 1)
                        if peak_memory > 0 and accuracy > 0:
                            memory_efficiencies.append(accuracy / peak_memory)
            
            if memory_efficiencies:
                most_efficient_idx = np.argmax(memory_efficiencies)
                summary['most_memory_efficient_model'] = successful_models[most_efficient_idx]
                summary['highest_memory_efficiency'] = memory_efficiencies[most_efficient_idx]
            
            if training_peaks:
                summary['avg_training_peak_mb'] = np.mean(training_peaks)
                summary['max_training_peak_mb'] = np.max(training_peaks)
                summary['min_training_peak_mb'] = np.min(training_peaks)
            
            if model_sizes:
                summary['avg_model_size_mb'] = np.mean(model_sizes)
                summary['largest_model_mb'] = np.max(model_sizes)
                summary['smallest_model_mb'] = np.min(model_sizes)
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save profiling results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"memory_profile_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Memory profiling results saved to {results_file}")
        return str(results_file)
    
    def generate_memory_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML memory profiling report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"memory_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorVerseHub Memory Profiling Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .model-section {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }}
                .stage-section {{ background-color: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 3px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 TensorVerseHub Memory Profiling Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>System Memory: {results['system_info'].get('total_memory_gb', 0):.2f} GB</p>
            </div>
        """
        
        # Summary section
        summary = results.get('summary', {})
        html_content += f"""
            <div class="model-section">
                <h2>📊 Summary</h2>
                <div class="metric">Total Models: <strong>{summary.get('total_models', 0)}</strong></div>
                <div class="metric">Successful: <strong class="success">{summary.get('successful_models', 0)}</strong></div>
                <div class="metric">Failed: <strong class="error">{summary.get('failed_models', 0)}</strong></div>
                <div class="metric">Success Rate: <strong>{summary.get('success_rate', 0):.1f}%</strong></div>
        """
        
        if 'most_memory_efficient_model' in summary:
            html_content += f"""
                <div class="metric">Most Efficient: <strong>{summary['most_memory_efficient_model']}</strong></div>
                <div class="metric">Avg Training Peak: <strong>{summary.get('avg_training_peak_mb', 0):.2f} MB</strong></div>
            """
        
        html_content += "</div>"
        
        # Model details
        for model_name, model_result in results.get('models', {}).items():
            if 'error' in model_result:
                html_content += f"""
                    <div class="model-section">
                        <h3 class="error">❌ {model_name}</h3>
                        <p class="error">Error: {model_result['error']}</p>
                    </div>
                """
                continue
            
            html_content += f"""
                <div class="model-section">
                    <h3>🤖 {model_name}</h3>
            """
            
            # Stages
            for stage_name, stage_result in model_result.get('stages', {}).items():
                if 'error' in stage_result:
                    html_content += f"""
                        <div class="stage-section">
                            <h4 class="error">{stage_name.title()} - Failed</h4>
                            <p class="error">{stage_result['error']}</p>
                        </div>
                    """
                    continue
                
                html_content += f"""
                    <div class="stage-section">
                        <h4>{stage_name.title()}</h4>
                """
                
                if stage_name == 'creation':
                    html_content += f"""
                        <div class="metric">Time: {stage_result.get('creation_time', 0):.2f}s</div>
                        <div class="metric">Parameters: {stage_result.get('model_params', 0):,}</div>
                        <div class="metric">Model Size: {stage_result.get('model_size_mb', 0):.2f} MB</div>
                        <div class="metric">Peak Memory: {stage_result.get('memory_stats', {}).get('peak_rss_mb', 0):.2f} MB</div>
                    """
                
                elif stage_name == 'training':
                    html_content += f"""
                        <div class="metric">Time: {stage_result.get('training_time', 0):.2f}s</div>
                        <div class="metric">Epochs: {stage_result.get('epochs_completed', 0)}</div>
                        <div class="metric">Peak Memory: {stage_result.get('memory_stats', {}).get('peak_rss_mb', 0):.2f} MB</div>
                        <div class="metric">Memory Growth: {stage_result.get('memory_stats', {}).get('memory_growth_mb', 0):.2f} MB</div>
                        <div class="metric">Final Accuracy: {stage_result.get('final_accuracy', 0):.4f}</div>
                    """
                
                elif stage_name == 'inference':
                    batch_results = stage_result.get('batch_results', {})
                    if batch_results:
                        html_content += "<table><tr><th>Batch Size</th><th>Avg Time (ms)</th><th>Peak Memory (MB)</th></tr>"
                        
                        for batch_size, batch_result in batch_results.items():
                            if 'error' not in batch_result:
                                avg_time_ms = batch_result.get('avg_inference_time', 0) * 1000
                                peak_memory = batch_result.get('memory_stats', {}).get('peak_rss_mb', 0)
                                
                                html_content += f"""
                                    <tr>
                                        <td>{batch_size}</td>
                                        <td>{avg_time_ms:.2f}</td>
                                        <td>{peak_memory:.2f}</td>
                                    </tr>
                                """
                        
                        html_content += "</table>"
                
                html_content += "</div>"
            
            # Summary for this model
            if 'summary' in model_result:
                model_summary = model_result['summary']
                html_content += f"""
                    <div class="stage-section">
                        <h4>Summary</h4>
                        <div class="metric">Total Creation Time: {model_summary.get('creation', {}).get('time', 0):.2f}s</div>
                        <div class="metric">Total Training Time: {model_summary.get('training', {}).get('time', 0):.2f}s</div>
                        <div class="metric">Peak Training Memory: {model_summary.get('training', {}).get('peak_memory_mb', 0):.2f} MB</div>
                    </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Memory report saved to {report_file}")
        return str(report_file)


def create_sample_data():
    """Create sample data for profiling."""
    # Image data
    X_img_train = np.random.random((1000, 32, 32, 3)).astype(np.float32)
    y_img_train = np.random.randint(0, 10, 1000)
    X_img_val = np.random.random((200, 32, 32, 3)).astype(np.float32)
    y_img_val = np.random.randint(0, 10, 200)
    X_img_test = np.random.random((100, 32, 32, 3)).astype(np.float32)
    
    # Tabular data
    X_tab_train = np.random.randn(1000, 50).astype(np.float32)
    y_tab_train = np.random.randint(0, 5, 1000)
    X_tab_val = np.random.randn(200, 50).astype(np.float32)
    y_tab_val = np.random.randint(0, 5, 200)
    X_tab_test = np.random.randn(100, 50).astype(np.float32)
    
    return {
        'image': {
            'train_data': tf.data.Dataset.from_tensor_slices((X_img_train, y_img_train)).batch(32),
            'val_data': tf.data.Dataset.from_tensor_slices((X_img_val, y_img_val)).batch(32),
            'test_data': X_img_test
        },
        'tabular': {
            'train_data': tf.data.Dataset.from_tensor_slices((X_tab_train, y_tab_train)).batch(32),
            'val_data': tf.data.Dataset.from_tensor_slices((X_tab_val, y_tab_val)).batch(32),
            'test_data': X_tab_test
        }
    }


def main():
    """Main memory profiling script."""
    parser = argparse.ArgumentParser(description="Profile model memory usage")
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save profiling results"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["simple_cnn", "mlp"],
        default=["simple_cnn", "mlp"],
        help="Model types to profile"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML report"
    )
    
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Please install TensorFlow.")
        sys.exit(1)
    
    logger.info("🧠 Starting TensorVerseHub Memory Profiling")
    logger.info(f"Results directory: {args.results_dir}")
    
    # Initialize profiler
    profiler = ModelMemoryProfiler(args.results_dir)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Define model configurations
    model_configs = {}
    
    if "simple_cnn" in args.models:
        model_configs["simple_cnn"] = {
            'model_factory': lambda: keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation='softmax')
            ]),
            'train_data': sample_data['image']['train_data'],
            'val_data': sample_data['image']['val_data'],
            'test_data': sample_data['image']['test_data']
        }
    
    if "mlp" in args.models:
        model_configs["mlp"] = {
            'model_factory': lambda: keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(50,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(5, activation='softmax')
            ]),
            'train_data': sample_data['tabular']['train_data'],
            'val_data': sample_data['tabular']['val_data'],
            'test_data': sample_data['tabular']['test_data']
        }
    
    if not model_configs:
        logger.error("No valid model configurations!")
        sys.exit(1)
    
    # Run profiling
    results = profiler.run_comprehensive_profiling(model_configs)
    
    # Save results
    results_file = profiler.save_results(results)
    
    # Generate report
    if args.generate_report:
        report_file = profiler.generate_memory_report(results)
        logger.info(f"📄 Memory report: {report_file}")
    
    # Display summary
    summary = results.get('summary', {})
    logger.info("📊 Memory Profiling Summary:")
    logger.info(f"  Total models: {summary.get('total_models', 0)}")
    logger.info(f"  Successful: {summary.get('successful_models', 0)}")
    logger.info(f"  Failed: {summary.get('failed_models', 0)}")
    
    if 'most_memory_efficient_model' in summary:
        logger.info(f"  Most efficient: {summary['most_memory_efficient_model']}")
    
    if 'avg_training_peak_mb' in summary:
        logger.info(f"  Avg training peak: {summary['avg_training_peak_mb']:.2f} MB")
    
    logger.info(f"📄 Detailed results: {results_file}")


if __name__ == "__main__":
    main()