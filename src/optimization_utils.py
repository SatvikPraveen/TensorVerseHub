# Location: /src/optimization_utils.py

"""
TensorFlow model optimization utilities.
Provides quantization, pruning, distillation, and other optimization techniques.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable, Any
import tempfile
import os


class ModelQuantization:
    """Model quantization utilities for TensorFlow models."""
    
    @staticmethod
    def quantize_model_post_training(model: tf.keras.Model,
                                   representative_dataset: Optional[tf.data.Dataset] = None,
                                   optimization_type: str = 'default') -> bytes:
        """
        Apply post-training quantization to a model.
        
        Args:
            model: tf.keras model to quantize
            representative_dataset: Dataset for representative quantization
            optimization_type: Type of optimization ('default', 'int8', 'float16')
            
        Returns:
            Quantized model as TFLite bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if optimization_type == 'default':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif optimization_type == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            if representative_dataset:
                def representative_data_gen():
                    for input_value in representative_dataset.take(100):
                        if isinstance(input_value, tuple):
                            yield [input_value[0]]
                        else:
                            yield [input_value]
                
                converter.representative_dataset = representative_data_gen
            
        elif optimization_type == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        return tflite_model
    
    @staticmethod
    def quantize_model_qat(model: tf.keras.Model,
                          train_dataset: tf.data.Dataset,
                          validation_dataset: tf.data.Dataset,
                          epochs: int = 10) -> tf.keras.Model:
        """
        Apply Quantization Aware Training (QAT) to a model.
        
        Args:
            model: Pre-trained tf.keras model
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of training epochs
            
        Returns:
            Quantization-aware trained model
        """
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        
        # Compile the quantization aware model
        q_aware_model.compile(
            optimizer='adam',
            loss=model.loss,
            metrics=model.metrics
        )
        
        # Train the quantization aware model
        q_aware_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        
        return q_aware_model
    
    @staticmethod
    def convert_qat_to_tflite(qat_model: tf.keras.Model) -> bytes:
        """
        Convert QAT model to TFLite format.
        
        Args:
            qat_model: Quantization-aware trained model
            
        Returns:
            TFLite model as bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        return tflite_model


class ModelPruning:
    """Model pruning utilities for reducing model size."""
    
    @staticmethod
    def create_pruned_model(model: tf.keras.Model,
                           pruning_schedule: tfmot.sparsity.keras.PruningSchedule,
                           target_layers: Optional[List[str]] = None) -> tf.keras.Model:
        """
        Create a pruned version of the model.
        
        Args:
            model: Original tf.keras model
            pruning_schedule: Pruning schedule
            target_layers: List of layer names to prune (None for all)
            
        Returns:
            Model configured for pruning
        """
        if target_layers is None:
            # Prune the entire model
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                model, pruning_schedule=pruning_schedule
            )
        else:
            # Prune specific layers
            def apply_pruning_to_layer(layer):
                if layer.name in target_layers:
                    return tfmot.sparsity.keras.prune_low_magnitude(
                        layer, pruning_schedule=pruning_schedule
                    )
                return layer
            
            pruned_model = tf.keras.models.clone_model(
                model, clone_function=apply_pruning_to_layer
            )
        
        return pruned_model
    
    @staticmethod
    def create_pruning_schedule(initial_sparsity: float = 0.0,
                               final_sparsity: float = 0.5,
                               begin_step: int = 1000,
                               end_step: int = 10000,
                               frequency: int = 100) -> tfmot.sparsity.keras.PruningSchedule:
        """
        Create a polynomial decay pruning schedule.
        
        Args:
            initial_sparsity: Starting sparsity level
            final_sparsity: Target sparsity level
            begin_step: Step to begin pruning
            end_step: Step to end pruning
            frequency: Pruning frequency in steps
            
        Returns:
            Pruning schedule
        """
        return tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
            frequency=frequency
        )
    
    @staticmethod
    def train_pruned_model(pruned_model: tf.keras.Model,
                          train_dataset: tf.data.Dataset,
                          validation_dataset: tf.data.Dataset,
                          epochs: int = 20) -> tf.keras.Model:
        """
        Train a pruned model with pruning callbacks.
        
        Args:
            pruned_model: Model configured for pruning
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of training epochs
            
        Returns:
            Trained pruned model
        """
        # Compile the pruned model
        pruned_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir='logs/pruning'),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        # Train the model
        pruned_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        
        return pruned_model
    
    @staticmethod
    def finalize_pruned_model(pruned_model: tf.keras.Model) -> tf.keras.Model:
        """
        Remove pruning wrappers and finalize the pruned model.
        
        Args:
            pruned_model: Trained pruned model
            
        Returns:
            Final pruned model without pruning wrappers
        """
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        return final_model


class KnowledgeDistillation:
    """Knowledge distillation for creating smaller student models."""
    
    def __init__(self, teacher_model: tf.keras.Model, alpha: float = 0.7, temperature: float = 3.0):
        """
        Initialize knowledge distillation.
        
        Args:
            teacher_model: Pre-trained teacher model
            alpha: Weight for distillation loss
            temperature: Temperature for softmax distillation
        """
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
    
    def create_distillation_loss(self) -> Callable:
        """
        Create distillation loss function.
        
        Returns:
            Distillation loss function
        """
        def distillation_loss(y_true, y_pred, teacher_pred):
            # Hard target loss
            hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            
            # Soft target loss (knowledge distillation)
            teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
            student_soft = tf.nn.softmax(y_pred / self.temperature)
            
            soft_loss = tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft)
            soft_loss *= (self.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
            return total_loss
        
        return distillation_loss
    
    def create_student_model(self, student_architecture: str,
                           input_shape: Tuple[int, ...],
                           num_classes: int) -> tf.keras.Model:
        """
        Create a student model architecture.
        
        Args:
            student_architecture: Type of student architecture
            input_shape: Input shape
            num_classes: Number of output classes
            
        Returns:
            Student model
        """
        if student_architecture == 'simple_cnn':
            inputs = tf.keras.layers.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(num_classes)(x)  # No activation for distillation
            
        elif student_architecture == 'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = True  # Fine-tune
            
            inputs = tf.keras.layers.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(num_classes)(x)
        
        student_model = tf.keras.Model(inputs, outputs)
        return student_model
    
    def train_student_model(self, student_model: tf.keras.Model,
                           train_dataset: tf.data.Dataset,
                           validation_dataset: tf.data.Dataset,
                           epochs: int = 20) -> tf.keras.Model:
        """
        Train student model using knowledge distillation.
        
        Args:
            student_model: Student model architecture
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of training epochs
            
        Returns:
            Trained student model
        """
        # Custom training loop for knowledge distillation
        optimizer = tf.keras.optimizers.Adam()
        
        @tf.function
        def train_step(x_batch, y_batch):
            # Get teacher predictions
            teacher_pred = self.teacher_model(x_batch, training=False)
            
            with tf.GradientTape() as tape:
                # Get student predictions
                student_pred = student_model(x_batch, training=True)
                
                # Calculate distillation loss
                loss_fn = self.create_distillation_loss()
                loss = loss_fn(y_batch, student_pred, teacher_pred)
            
            # Apply gradients
            gradients = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
            
            return loss
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = []
            
            for x_batch, y_batch in train_dataset:
                loss = train_step(x_batch, y_batch)
                epoch_loss.append(loss.numpy())
            
            # Validation
            val_loss = []
            val_accuracy = []
            
            for x_val, y_val in validation_dataset:
                teacher_pred = self.teacher_model(x_val, training=False)
                student_pred = student_model(x_val, training=False)
                
                loss_fn = self.create_distillation_loss()
                v_loss = loss_fn(y_val, student_pred, teacher_pred)
                val_loss.append(v_loss.numpy())
                
                # Calculate accuracy
                accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_val, student_pred)
                val_accuracy.append(tf.reduce_mean(accuracy).numpy())
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Loss: {np.mean(epoch_loss):.4f} - Val Loss: {np.mean(val_loss):.4f} - Val Acc: {np.mean(val_accuracy):.4f}")
        
        return student_model


class MixedPrecisionOptimization:
    """Mixed precision training utilities."""
    
    @staticmethod
    def enable_mixed_precision() -> None:
        """Enable mixed precision training policy."""
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled: using float16 for computations, float32 for variables")
    
    @staticmethod
    def create_mixed_precision_model(model: tf.keras.Model) -> tf.keras.Model:
        """
        Modify model for mixed precision training.
        
        Args:
            model: Original model
            
        Returns:
            Model configured for mixed precision
        """
        # Clone model with mixed precision considerations
        def modify_layer(layer):
            # Ensure final dense layers use float32 dtype for numerical stability
            if isinstance(layer, tf.keras.layers.Dense) and layer == model.layers[-1]:
                config = layer.get_config()
                config['dtype'] = 'float32'
                return tf.keras.layers.Dense.from_config(config)
            return layer
        
        mixed_precision_model = tf.keras.models.clone_model(model, clone_function=modify_layer)
        mixed_precision_model.set_weights(model.get_weights())
        
        return mixed_precision_model
    
    @staticmethod
    def compile_mixed_precision_model(model: tf.keras.Model,
                                    optimizer: str = 'adam',
                                    loss: str = 'sparse_categorical_crossentropy') -> tf.keras.Model:
        """
        Compile model for mixed precision training with loss scaling.
        
        Args:
            model: Model to compile
            optimizer: Optimizer name
            loss: Loss function name
            
        Returns:
            Compiled model
        """
        # Create optimizer with loss scaling
        opt = tf.keras.optimizers.get(optimizer)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model


class ModelCompression:
    """Comprehensive model compression utilities."""
    
    @staticmethod
    def apply_magnitude_pruning_and_quantization(model: tf.keras.Model,
                                               train_dataset: tf.data.Dataset,
                                               validation_dataset: tf.data.Dataset,
                                               target_sparsity: float = 0.5,
                                               epochs: int = 10) -> Tuple[tf.keras.Model, bytes]:
        """
        Apply both pruning and quantization to a model.
        
        Args:
            model: Original tf.keras model
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            target_sparsity: Target sparsity level
            epochs: Training epochs for pruning
            
        Returns:
            Tuple of (pruned_model, quantized_tflite_bytes)
        """
        # Step 1: Apply magnitude-based pruning
        pruning_schedule = ModelPruning.create_pruning_schedule(
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=epochs * len(train_dataset)
        )
        
        pruned_model = ModelPruning.create_pruned_model(model, pruning_schedule)
        pruned_model = ModelPruning.train_pruned_model(
            pruned_model, train_dataset, validation_dataset, epochs
        )
        pruned_model = ModelPruning.finalize_pruned_model(pruned_model)
        
        # Step 2: Apply post-training quantization
        quantized_tflite = ModelQuantization.quantize_model_post_training(
            pruned_model, train_dataset.take(100), 'int8'
        )
        
        return pruned_model, quantized_tflite
    
    @staticmethod
    def analyze_compression_ratio(original_model: tf.keras.Model,
                                compressed_model: tf.keras.Model,
                                tflite_model: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Analyze compression ratios and model sizes.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            tflite_model: Optional TFLite model bytes
            
        Returns:
            Dictionary with compression analysis
        """
        original_params = original_model.count_params()
        compressed_params = compressed_model.count_params()
        
        analysis = {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'parameter_reduction_ratio': original_params / compressed_params if compressed_params > 0 else 0,
            'parameter_reduction_percentage': (1 - compressed_params / original_params) * 100 if original_params > 0 else 0
        }
        
        # Estimate model sizes (assuming float32)
        original_size_mb = original_params * 4 / (1024 * 1024)
        compressed_size_mb = compressed_params * 4 / (1024 * 1024)
        
        analysis.update({
            'original_size_mb': original_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'size_reduction_ratio': original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
        })
        
        if tflite_model:
            tflite_size_mb = len(tflite_model) / (1024 * 1024)
            analysis.update({
                'tflite_size_mb': tflite_size_mb,
                'tflite_compression_ratio': original_size_mb / tflite_size_mb if tflite_size_mb > 0 else 0
            })
        
        return analysis


# Convenience functions for common optimization workflows
def optimize_for_mobile(model: tf.keras.Model,
                       representative_dataset: tf.data.Dataset,
                       target_size_mb: float = 5.0) -> Tuple[bytes, Dict[str, Any]]:
    """
    Optimize model for mobile deployment.
    
    Args:
        model: tf.keras model to optimize
        representative_dataset: Dataset for calibration
        target_size_mb: Target model size in MB
        
    Returns:
        Tuple of (optimized_tflite_model, optimization_report)
    """
    # Try different optimization strategies
    strategies = ['default', 'float16', 'int8']
    results = {}
    
    for strategy in strategies:
        try:
            tflite_model = ModelQuantization.quantize_model_post_training(
                model, representative_dataset.take(100), strategy
            )
            
            size_mb = len(tflite_model) / (1024 * 1024)
            results[strategy] = {
                'model': tflite_model,
                'size_mb': size_mb,
                'meets_target': size_mb <= target_size_mb
            }
            
        except Exception as e:
            results[strategy] = {'error': str(e)}
    
    # Select best strategy that meets target size
    best_strategy = None
    for strategy in ['int8', 'float16', 'default']:
        if strategy in results and 'model' in results[strategy]:
            if results[strategy]['meets_target'] or best_strategy is None:
                best_strategy = strategy
                break
    
    if best_strategy:
        return results[best_strategy]['model'], results
    else:
        raise RuntimeError("Could not optimize model for mobile deployment")


def create_inference_optimized_model(model: tf.keras.Model,
                                   input_shape: Tuple[int, ...],
                                   optimization_level: str = 'aggressive') -> tf.keras.Model:
    """
    Create inference-optimized version of model.
    
    Args:
        model: Original tf.keras model
        input_shape: Model input shape
        optimization_level: Optimization level ('conservative', 'moderate', 'aggressive')
        
    Returns:
        Inference-optimized model
    """
    # Apply different optimizations based on level
    if optimization_level == 'conservative':
        # Basic optimizations
        optimized_model = tf.keras.models.clone_model(model)
        optimized_model.set_weights(model.get_weights())
        
    elif optimization_level == 'moderate':
        # Apply layer fusion and other moderate optimizations
        optimized_model = tf.keras.models.clone_model(model)
        optimized_model.set_weights(model.get_weights())
        
        # Convert to concrete function for graph optimization
        full_model = tf.function(lambda x: optimized_model(x))
        concrete_func = full_model.get_concrete_function(
            tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32)
        )
        
        # This would typically involve TensorRT or other optimization libraries
        
    elif optimization_level == 'aggressive':
        # Apply all available optimizations
        optimized_model = tf.keras.models.clone_model(model)
        optimized_model.set_weights(model.get_weights())
        
        # Enable XLA compilation
        optimized_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
            jit_compile=True  # Enable XLA
        )
    
    return optimized_model


def benchmark_model_performance(original_model: tf.keras.Model,
                              optimized_model: tf.keras.Model,
                              test_input: tf.Tensor,
                              num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark performance comparison between original and optimized models.
    
    Args:
        original_model: Original model
        optimized_model: Optimized model
        test_input: Test input tensor
        num_runs: Number of benchmark runs
        
    Returns:
        Performance comparison results
    """
    import time
    
    # Warm up
    _ = original_model(test_input)
    _ = optimized_model(test_input)
    
    # Benchmark original model
    start_time = time.time()
    for _ in range(num_runs):
        _ = original_model(test_input)
    original_time = time.time() - start_time
    
    # Benchmark optimized model
    start_time = time.time()
    for _ in range(num_runs):
        _ = optimized_model(test_input)
    optimized_time = time.time() - start_time
    
    speedup = original_time / optimized_time if optimized_time > 0 else 0
    
    return {
        'original_avg_time_ms': (original_time / num_runs) * 1000,
        'optimized_avg_time_ms': (optimized_time / num_runs) * 1000,
        'speedup_ratio': speedup,
        'performance_improvement_percent': (speedup - 1) * 100 if speedup > 1 else 0
    }


class TensorRTOptimization:
    """TensorRT optimization utilities (requires TensorRT installation)."""
    
    @staticmethod
    def convert_to_tensorrt(model: tf.keras.Model,
                          input_shape: Tuple[int, ...],
                          precision: str = 'FP16',
                          max_batch_size: int = 32) -> Any:
        """
        Convert model to TensorRT format (requires TensorRT).
        
        Args:
            model: tf.keras model
            input_shape: Input shape
            precision: Precision mode ('FP32', 'FP16', 'INT8')
            max_batch_size: Maximum batch size
            
        Returns:
            TensorRT optimized model
        """
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            
            # Convert model to SavedModel format first
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = os.path.join(temp_dir, 'saved_model')
                model.save(saved_model_path, save_format='tf')
                
                # Configure TensorRT conversion
                conversion_params = trt.TrtConversionParams(
                    precision_mode=precision,
                    max_batch_size=max_batch_size,
                    use_calibration=precision == 'INT8'
                )
                
                # Convert to TensorRT
                converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=saved_model_path,
                    conversion_params=conversion_params
                )
                
                converter.convert()
                
                # Save TensorRT model
                trt_model_path = os.path.join(temp_dir, 'trt_model')
                converter.save(trt_model_path)
                
                # Load and return TensorRT model
                trt_model = tf.saved_model.load(trt_model_path)
                return trt_model
                
        except ImportError:
            print("TensorRT not available. Please install TensorRT for GPU optimization.")
            return model
        except Exception as e:
            print(f"TensorRT conversion failed: {e}")
            return model


def create_optimization_report(original_model: tf.keras.Model,
                             optimized_components: Dict[str, Any],
                             performance_metrics: Dict[str, float]) -> str:
    """
    Create comprehensive optimization report.
    
    Args:
        original_model: Original model
        optimized_components: Dictionary of optimized components
        performance_metrics: Performance benchmark results
        
    Returns:
        Formatted optimization report
    """
    report = f"""
# Model Optimization Report

## Original Model
- **Parameters**: {original_model.count_params():,}
- **Layers**: {len(original_model.layers)}
- **Model Size**: {(original_model.count_params() * 4) / (1024 * 1024):.2f} MB

## Optimization Results
"""
    
    if 'pruned_model' in optimized_components:
        pruned_params = optimized_components['pruned_model'].count_params()
        report += f"""
### Pruning
- **Pruned Parameters**: {pruned_params:,}
- **Parameter Reduction**: {((original_model.count_params() - pruned_params) / original_model.count_params() * 100):.1f}%
"""
    
    if 'quantized_model_size' in optimized_components:
        report += f"""
### Quantization
- **Quantized Model Size**: {optimized_components['quantized_model_size']:.2f} MB
- **Size Reduction**: {((original_model.count_params() * 4 / (1024 * 1024) - optimized_components['quantized_model_size']) / (original_model.count_params() * 4 / (1024 * 1024)) * 100):.1f}%
"""
    
    if performance_metrics:
        report += f"""
## Performance Metrics
- **Original Inference Time**: {performance_metrics.get('original_avg_time_ms', 0):.2f} ms
- **Optimized Inference Time**: {performance_metrics.get('optimized_avg_time_ms', 0):.2f} ms
- **Speedup**: {performance_metrics.get('speedup_ratio', 1):.2f}x
- **Performance Improvement**: {performance_metrics.get('performance_improvement_percent', 0):.1f}%

## Recommendations
"""
        
        if performance_metrics.get('speedup_ratio', 1) > 1.5:
            report += "✅ Excellent optimization results achieved\n"
        elif performance_metrics.get('speedup_ratio', 1) > 1.2:
            report += "✅ Good optimization results achieved\n"
        else:
            report += "⚠️ Consider additional optimization techniques\n"
    
    return report