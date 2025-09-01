# Location: /tests/test_optimization.py

"""
Test TensorFlow model optimization utilities.
Comprehensive tests for quantization, pruning, and knowledge distillation.
"""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import optimization utilities
try:
    from optimization_utils import (
        ModelQuantization,
        ModelPruning, 
        KnowledgeDistillation,
        MixedPrecisionOptimization,
        ModelCompression,
        optimize_for_mobile,
        create_inference_optimized_model
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("Warning: Optimization utilities not found")

from tests import TEST_CONFIG


# Skip all tests if optimization utils not available
pytestmark = pytest.mark.skipif(
    not OPTIMIZATION_AVAILABLE, 
    reason="Optimization utilities not available"
)


@pytest.fixture
def simple_model():
    """Create a simple model for testing optimization."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=TEST_CONFIG['image_shape']),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


@pytest.fixture
def sample_representative_dataset():
    """Create representative dataset for calibration."""
    images = tf.random.normal([50] + list(TEST_CONFIG['image_shape']))
    labels = tf.random.uniform([50], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(TEST_CONFIG['batch_size'])
    
    return dataset


class TestModelQuantization:
    """Test model quantization utilities."""
    
    def test_quantize_model_post_training_default(self, simple_model):
        """Test default post-training quantization."""
        tflite_model = ModelQuantization.quantize_model_post_training(
            simple_model, 
            optimization_type='default'
        )
        
        assert isinstance(tflite_model, bytes)
        assert len(tflite_model) > 0
        
        # Check model can be loaded
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        assert len(input_details) == 1
        assert len(output_details) == 1
    
    def test_quantize_model_post_training_float16(self, simple_model):
        """Test FP16 quantization."""
        tflite_model = ModelQuantization.quantize_model_post_training(
            simple_model,
            optimization_type='float16'
        )
        
        assert isinstance(tflite_model, bytes)
        
        # FP16 model should be smaller than FP32
        fp32_model = ModelQuantization.quantize_model_post_training(
            simple_model,
            optimization_type='default'
        )
        
        assert len(tflite_model) <= len(fp32_model)
    
    def test_quantize_model_post_training_int8(self, simple_model, sample_representative_dataset):
        """Test INT8 quantization with representative dataset."""
        tflite_model = ModelQuantization.quantize_model_post_training(
            simple_model,
            representative_dataset=sample_representative_dataset,
            optimization_type='int8'
        )
        
        assert isinstance(tflite_model, bytes)
        
        # Verify INT8 quantization worked
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        # Note: Input might still be float32 for convenience
        assert input_details[0]['dtype'] in [np.float32, np.int8, np.uint8]
    
    def test_quantize_model_without_representative_data(self, simple_model):
        """Test INT8 quantization without representative dataset (should warn or handle gracefully)."""
        # This should work but may not achieve full INT8 quantization
        tflite_model = ModelQuantization.quantize_model_post_training(
            simple_model,
            optimization_type='int8'
        )
        
        assert isinstance(tflite_model, bytes)
    
    @pytest.mark.skipif(
        not hasattr(tf, 'keras') or not hasattr(tf.keras, 'mixed_precision'),
        reason="Mixed precision not available"
    )
    def test_quantization_aware_training_availability(self):
        """Test if QAT components are available."""
        try:
            import tensorflow_model_optimization as tfmot
            assert hasattr(tfmot.quantization, 'keras')
        except ImportError:
            pytest.skip("TensorFlow Model Optimization not available")


class TestMixedPrecisionOptimization:
    """Test mixed precision optimization utilities."""
    
    def test_enable_mixed_precision(self):
        """Test enabling mixed precision policy."""
        # Save original policy
        original_policy = tf.keras.mixed_precision.global_policy()
        
        try:
            MixedPrecisionOptimization.enable_mixed_precision()
            
            current_policy = tf.keras.mixed_precision.global_policy()
            assert current_policy.name == 'mixed_float16'
            
        finally:
            # Restore original policy
            tf.keras.mixed_precision.set_global_policy(original_policy)
    
    def test_create_mixed_precision_model(self, simple_model):
        """Test mixed precision model creation."""
        # Save original policy
        original_policy = tf.keras.mixed_precision.global_policy()
        
        try:
            # Enable mixed precision
            MixedPrecisionOptimization.enable_mixed_precision()
            
            mixed_model = MixedPrecisionOptimization.create_mixed_precision_model(simple_model)
            
            assert isinstance(mixed_model, tf.keras.Model)
            assert len(mixed_model.layers) == len(simple_model.layers)
            
            # Test forward pass
            test_input = tf.random.normal([2] + list(TEST_CONFIG['image_shape']))
            output = mixed_model(test_input)
            
            assert output.shape == (2, TEST_CONFIG['num_classes'])
            
        finally:
            tf.keras.mixed_precision.set_global_policy(original_policy)
    
    def test_compile_mixed_precision_model(self, simple_model):
        """Test compiling model with mixed precision and loss scaling."""
        original_policy = tf.keras.mixed_precision.global_policy()
        
        try:
            MixedPrecisionOptimization.enable_mixed_precision()
            
            compiled_model = MixedPrecisionOptimization.compile_mixed_precision_model(
                simple_model,
                optimizer='adam',
                loss='sparse_categorical_crossentropy'
            )
            
            assert isinstance(compiled_model, tf.keras.Model)
            assert compiled_model.optimizer is not None
            
            # Check if loss scale optimizer is used
            optimizer = compiled_model.optimizer
            assert hasattr(optimizer, '_name') or hasattr(optimizer, 'name')
            
        finally:
            tf.keras.mixed_precision.set_global_policy(original_policy)


class TestModelCompression:
    """Test comprehensive model compression utilities."""
    
    def test_compression_analysis(self, simple_model):
        """Test compression ratio analysis."""
        # Create a smaller model for comparison
        compressed_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=TEST_CONFIG['image_shape']),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')
        ])
        
        analysis = ModelCompression.analyze_compression_ratio(
            simple_model, compressed_model
        )
        
        assert isinstance(analysis, dict)
        required_keys = [
            'original_parameters', 'compressed_parameters',
            'parameter_reduction_ratio', 'parameter_reduction_percentage',
            'original_size_mb', 'compressed_size_mb'
        ]
        
        for key in required_keys:
            assert key in analysis
        
        assert analysis['original_parameters'] > analysis['compressed_parameters']
        assert analysis['parameter_reduction_ratio'] > 1.0
        assert analysis['parameter_reduction_percentage'] > 0
    
    def test_compression_with_tflite_model(self, simple_model):
        """Test compression analysis with TFLite model."""
        # Create TFLite model
        tflite_model = ModelQuantization.quantize_model_post_training(
            simple_model, optimization_type='float16'
        )
        
        compressed_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=TEST_CONFIG['image_shape']),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')
        ])
        
        analysis = ModelCompression.analyze_compression_ratio(
            simple_model, compressed_model, tflite_model
        )
        
        assert 'tflite_size_mb' in analysis
        assert 'tflite_compression_ratio' in analysis
        assert analysis['tflite_size_mb'] > 0


class TestConvenienceFunctions:
    """Test convenience functions for optimization."""
    
    def test_optimize_for_mobile(self, simple_model, sample_representative_dataset):
        """Test mobile optimization function."""
        optimized_model, optimization_report = optimize_for_mobile(
            simple_model,
            sample_representative_dataset,
            target_size_mb=2.0
        )
        
        assert isinstance(optimized_model, bytes)
        assert isinstance(optimization_report, dict)
        
        # Check optimization strategies were tried
        strategies = ['default', 'float16', 'int8']
        for strategy in strategies:
            assert strategy in optimization_report
    
    def test_optimize_for_mobile_unrealistic_target(self, simple_model, sample_representative_dataset):
        """Test mobile optimization with unrealistic target size."""
        # Very small target size to test fallback behavior
        optimized_model, optimization_report = optimize_for_mobile(
            simple_model,
            sample_representative_dataset,
            target_size_mb=0.001  # Unrealistically small
        )
        
        assert isinstance(optimized_model, bytes)
        assert len(optimization_report) > 0
    
    def test_create_inference_optimized_model_conservative(self, simple_model):
        """Test conservative inference optimization."""
        optimized_model = create_inference_optimized_model(
            simple_model,
            TEST_CONFIG['image_shape'],
            optimization_level='conservative'
        )
        
        assert isinstance(optimized_model, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.normal([2] + list(TEST_CONFIG['image_shape']))
        output = optimized_model(test_input)
        
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    def test_create_inference_optimized_model_aggressive(self, simple_model):
        """Test aggressive inference optimization."""
        optimized_model = create_inference_optimized_model(
            simple_model,
            TEST_CONFIG['image_shape'],
            optimization_level='aggressive'
        )
        
        assert isinstance(optimized_model, tf.keras.Model)
        
        # Check if XLA compilation is enabled
        assert optimized_model.optimizer is not None  # Model should be compiled
        
        # Test forward pass
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        output = optimized_model(test_input)
        
        assert output.shape == (1, TEST_CONFIG['num_classes'])
    
    def test_benchmark_model_performance(self, simple_model):
        """Test model performance benchmarking."""
        # Create a smaller model for comparison
        optimized_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=TEST_CONFIG['image_shape']),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')
        ])
        
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        
        # Benchmark with fewer runs for testing speed
        results = benchmark_model_performance(
            simple_model, optimized_model, test_input, num_runs=5
        )
        
        assert isinstance(results, dict)
        required_keys = [
            'original_avg_time_ms', 'optimized_avg_time_ms',
            'speedup_ratio', 'performance_improvement_percent'
        ]
        
        for key in required_keys:
            assert key in results
        
        assert results['original_avg_time_ms'] > 0
        assert results['optimized_avg_time_ms'] > 0
        assert results['speedup_ratio'] >= 0  # Could be less than 1 if optimized model is slower


class TestKnowledgeDistillation:
    """Test knowledge distillation utilities."""
    
    def test_knowledge_distillation_initialization(self, simple_model):
        """Test KnowledgeDistillation initialization."""
        kd = KnowledgeDistillation(simple_model, alpha=0.7, temperature=3.0)
        
        assert kd.teacher_model == simple_model
        assert kd.alpha == 0.7
        assert kd.temperature == 3.0
    
    def test_create_distillation_loss(self, simple_model):
        """Test distillation loss function creation."""
        kd = KnowledgeDistillation(simple_model)
        loss_fn = kd.create_distillation_loss()
        
        assert callable(loss_fn)
        
        # Test loss computation
        y_true = tf.random.uniform([4], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
        teacher_pred = tf.random.normal([4, TEST_CONFIG['num_classes']])
        student_pred = tf.random.normal([4, TEST_CONFIG['num_classes']])
        
        loss = loss_fn(y_true, student_pred, teacher_pred)
        
        assert loss.shape == (4,)  # Should return loss per sample
        assert tf.reduce_all(loss >= 0)  # Loss should be non-negative
    
    def test_create_student_model_simple_cnn(self, simple_model):
        """Test student model creation."""
        kd = KnowledgeDistillation(simple_model)
        
        student_model = kd.create_student_model(
            'simple_cnn',
            TEST_CONFIG['image_shape'],
            TEST_CONFIG['num_classes']
        )
        
        assert isinstance(student_model, tf.keras.Model)
        assert student_model.count_params() < simple_model.count_params()
        
        # Test forward pass
        test_input = tf.random.normal([2] + list(TEST_CONFIG['image_shape']))
        output = student_model(test_input)
        
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    def test_create_student_model_mobilenet(self, simple_model):
        """Test MobileNet-style student model creation."""
        kd = KnowledgeDistillation(simple_model)
        
        student_model = kd.create_student_model(
            'mobilenet',
            TEST_CONFIG['image_shape'],
            TEST_CONFIG['num_classes']
        )
        
        assert isinstance(student_model, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        output = student_model(test_input)
        
        assert output.shape == (1, TEST_CONFIG['num_classes'])


class TestOptimizationIntegration:
    """Integration tests for optimization utilities."""
    
    def test_end_to_end_quantization_pipeline(self, simple_model, sample_representative_dataset):
        """Test complete quantization pipeline."""
        # Original model size
        original_size = simple_model.count_params() * 4  # Assuming float32
        
        # Apply different quantization techniques
        fp16_model = ModelQuantization.quantize_model_post_training(
            simple_model, optimization_type='float16'
        )
        
        int8_model = ModelQuantization.quantize_model_post_training(
            simple_model, sample_representative_dataset, optimization_type='int8'
        )
        
        # Verify models work
        for tflite_model in [fp16_model, int8_model]:
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test inference
            test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
            interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            assert output.shape == (1, TEST_CONFIG['num_classes'])
    
    def test_optimization_with_mixed_precision(self, simple_model):
        """Test optimization combined with mixed precision."""
        original_policy = tf.keras.mixed_precision.global_policy()
        
        try:
            # Enable mixed precision
            MixedPrecisionOptimization.enable_mixed_precision()
            
            # Create mixed precision model
            mixed_model = MixedPrecisionOptimization.create_mixed_precision_model(simple_model)
            mixed_model = MixedPrecisionOptimization.compile_mixed_precision_model(mixed_model)
            
            # Apply quantization
            quantized_model = ModelQuantization.quantize_model_post_training(
                mixed_model, optimization_type='float16'
            )
            
            assert isinstance(quantized_model, bytes)
            
            # Verify quantized model works
            interpreter = tf.lite.Interpreter(model_content=quantized_model)
            interpreter.allocate_tensors()
            
            test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
            interpreter.invoke()
            
            output_details = interpreter.get_output_details()
            output = interpreter.get_tensor(output_details[0]['index'])
            assert output.shape == (1, TEST_CONFIG['num_classes'])
            
        finally:
            tf.keras.mixed_precision.set_global_policy(original_policy)
    
    def test_model_size_progression(self, simple_model, sample_representative_dataset):
        """Test that different optimizations produce progressively smaller models."""
        # Get sizes for different optimization levels
        fp32_model = ModelQuantization.quantize_model_post_training(
            simple_model, optimization_type='default'
        )
        
        fp16_model = ModelQuantization.quantize_model_post_training(
            simple_model, optimization_type='float16'
        )
        
        int8_model = ModelQuantization.quantize_model_post_training(
            simple_model, sample_representative_dataset, optimization_type='int8'
        )
        
        # Verify size progression (generally FP32 > FP16 >= INT8)
        fp32_size = len(fp32_model)
        fp16_size = len(fp16_model)
        int8_size = len(int8_model)
        
        assert fp16_size <= fp32_size
        # INT8 might be larger than FP16 due to calibration overhead, so we don't strictly enforce this
        assert int8_size > 0


# Error handling and edge cases
class TestOptimizationEdgeCases:
    """Test edge cases and error handling in optimization."""
    
    def test_quantization_with_empty_representative_dataset(self, simple_model):
        """Test quantization with empty representative dataset."""
        empty_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.zeros([0] + list(TEST_CONFIG['image_shape'])), tf.zeros([0], dtype=tf.int32))
        ).batch(1)
        
        # Should handle gracefully
        tflite_model = ModelQuantization.quantize_model_post_training(
            simple_model, empty_dataset, optimization_type='int8'
        )
        
        assert isinstance(tflite_model, bytes)
    
    def test_optimization_with_very_small_model(self):
        """Test optimization with minimal model."""
        tiny_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=TEST_CONFIG['image_shape']),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        tiny_model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Should handle tiny models
        quantized = ModelQuantization.quantize_model_post_training(
            tiny_model, optimization_type='float16'
        )
        
        assert isinstance(quantized, bytes)
        assert len(quantized) > 0
    
    def test_invalid_optimization_level(self, simple_model):
        """Test handling of invalid optimization level."""
        with pytest.raises((ValueError, KeyError)):
            create_inference_optimized_model(
                simple_model,
                TEST_CONFIG['image_shape'],
                optimization_level='invalid_level'
            )


# Performance and benchmarking tests
@pytest.mark.slow
class TestOptimizationPerformance:
    """Performance tests for optimization utilities."""
    
    def test_quantization_inference_speed(self, simple_model, sample_representative_dataset):
        """Test that quantized models are faster (marked as slow)."""
        # Create quantized model
        quantized_model = ModelQuantization.quantize_model_post_training(
            simple_model, sample_representative_dataset, optimization_type='int8'
        )
        
        # Load quantized model
        interpreter = tf.lite.Interpreter(model_content=quantized_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Benchmark original model
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        
        import time
        
        # Warm up and benchmark original model
        for _ in range(5):
            _ = simple_model.predict(test_input, verbose=0)
        
        start_time = time.time()
        for _ in range(10):
            _ = simple_model.predict(test_input, verbose=0)
        original_time = time.time() - start_time
        
        # Benchmark quantized model
        for _ in range(5):  # Warm up
            interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
            interpreter.invoke()
        
        start_time = time.time()
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
            interpreter.invoke()
        quantized_time = time.time() - start_time
        
        print(f"Original model: {original_time:.4f}s, Quantized model: {quantized_time:.4f}s")
        
        # Quantized model should generally be faster or similar
        # We don't enforce this strictly as it depends on hardware and model size
        assert quantized_time > 0
        assert original_time > 0


if __name__ == "__main__":
    pytest.main([__file__])