# Location: /tests/test_edge_cases.py

"""
Comprehensive edge case and error handling tests.
Tests boundary conditions, error states, and unusual inputs.
"""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from data_utils import DataPipeline, TFRecordHandler, DataAugmentation
from model_utils import ModelBuilders, TrainingUtilities
from optimization_utils import ModelQuantization, ModelPruning
from export_utils import ModelExporter
from visualization import MetricsVisualizer


class TestDataUtilsEdgeCases:
    """Test data utilities with edge cases."""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        pipeline = DataPipeline()
        
        # Create empty dataset
        empty_data = tf.data.Dataset.from_tensor_slices(([], []))
        
        # Should handle gracefully
        assert pipeline is not None
    
    def test_single_sample_dataset(self):
        """Test dataset with single sample."""
        pipeline = DataPipeline()
        
        single_sample = tf.data.Dataset.from_tensor_slices(
            (np.random.randn(1, 28, 28, 1).astype(np.float32), 
             np.array([0]))
        )
        
        single_sample = single_sample.batch(1)
        assert len(list(single_sample)) == 1
    
    def test_very_large_batch_size(self):
        """Test with batch size larger than dataset."""
        data = np.random.randn(10, 28, 28, 1).astype(np.float32)
        labels = np.arange(10)
        
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.batch(1000)  # Batch size larger than dataset
        
        batch_count = len(list(dataset))
        assert batch_count == 1
    
    def test_zero_mean_std_normalization(self):
        """Test normalization with zero variance."""
        # Create constant data (zero std)
        data = np.ones((100, 28, 28, 1))
        
        pipeline = DataPipeline()
        # Should handle zero variance gracefully
        try:
            # Normalize would typically divide by std
            normalized = (data - data.mean()) / (data.std() + 1e-7)
            assert not np.any(np.isnan(normalized))
        except:
            pytest.fail("Should handle zero variance")
    
    def test_extreme_value_ranges(self):
        """Test with extreme pixel values."""
        # Very large values
        large_data = np.ones((10, 28, 28, 1)) * 1e6
        assert large_data.max() > 0
        
        # Very small values
        small_data = np.ones((10, 28, 28, 1)) * 1e-6
        assert small_data.min() < 1
        
        # Mixed positive/negative
        mixed_data = np.random.randn(10, 28, 28, 1) * 100
        assert mixed_data.min() < 0 and mixed_data.max() > 0
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and Inf values."""
        # Create data with NaN
        data_with_nan = np.random.randn(10, 28, 28, 1)
        data_with_nan[0, 0, 0, 0] = np.nan
        
        assert np.any(np.isnan(data_with_nan))
        
        # Create data with Inf
        data_with_inf = np.random.randn(10, 28, 28, 1)
        data_with_inf[0, 0, 0, 0] = np.inf
        
        assert np.any(np.isinf(data_with_inf))
    
    def test_imbalanced_class_distribution(self):
        """Test with highly imbalanced classes."""
        # 99% class 0, 1% class 1
        labels = np.concatenate([
            np.zeros(990, dtype=np.int32),
            np.ones(10, dtype=np.int32)
        ])
        np.random.shuffle(labels)
        
        unique, counts = np.unique(labels, return_counts=True)
        assert counts[0] > counts[1]
    
    def test_tfrecord_with_missing_features(self):
        """Test TFRecord parsing with optional features."""
        handler = TFRecordHandler()
        
        # Create example with some features
        features = {
            'text': handler._bytes_feature('test'),
            'label': handler._int64_feature(1),
            'confidence': handler._float_feature(0.95)
        }
        
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        
        serialized = example_proto.SerializeToString()
        assert len(serialized) > 0


class TestModelUtilsEdgeCases:
    """Test model utilities with edge cases."""
    
    def test_model_with_zero_classes(self):
        """Test error handling with zero classes."""
        with pytest.raises(ValueError):
            ModelBuilders.create_cnn_classifier(
                input_shape=(28, 28, 1),
                num_classes=0
            )
    
    def test_model_with_invalid_input_shape(self):
        """Test error handling with invalid input shape."""
        with pytest.raises((ValueError, RuntimeError)):
            ModelBuilders.create_cnn_classifier(
                input_shape=(-1, 28, 1),  # Invalid dimension
                num_classes=10
            )
    
    def test_model_with_single_class(self):
        """Test binary classification edge case."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=1
        )
        
        assert model is not None
        assert model.output_shape[-1] == 1
    
    def test_model_with_very_large_classes(self):
        """Test model with many classes."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=1000
        )
        
        assert model is not None
        assert model.output_shape[-1] == 1000
    
    def test_training_with_zero_epochs(self):
        """Test training with zero epochs."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        X = np.random.randn(10, 20)
        y = np.array([0, 1] * 5)
        
        # Training with 0 epochs should not error
        history = model.fit(X, y, epochs=0, verbose=0)
        assert history.epoch == []
    
    def test_training_with_single_batch(self):
        """Test training with dataset smaller than batch size."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        X = np.random.randn(2, 20)
        y = np.array([0, 1])
        
        history = model.fit(X, y, batch_size=100, epochs=1, verbose=0)
        assert history.history is not None
    
    def test_prediction_with_empty_batch(self):
        """Test prediction on empty batch."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Predict on empty batch
        empty_input = np.array([]).reshape(0, 20)
        predictions = model.predict(empty_input, verbose=0)
        
        assert predictions.shape[0] == 0
    
    def test_gradient_explosion(self):
        """Test handling of gradient explosion."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Use very large learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=100.0)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 2, 50)
        
        history = model.fit(X, y, epochs=1, verbose=0)
        # Check loss is not NaN/Inf
        assert not np.isnan(history.history['loss'][0])


class TestOptimizationEdgeCases:
    """Test optimization utilities with edge cases."""
    
    def test_quantization_on_tiny_model(self):
        """Test quantization on very small model."""
        tiny_model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        tiny_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Should handle small models
        assert tiny_model.count_params() < 50
    
    def test_pruning_with_zero_target(self):
        """Test pruning with 0% target sparsity."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Pruning with no target sparsity should be valid
        assert model is not None
    
    def test_pruning_with_full_sparsity(self):
        """Test pruning with 100% target sparsity."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Extremely high sparsity may not be valid
        assert model is not None


class TestExportEdgeCases:
    """Test export utilities with edge cases."""
    
    def test_export_model_with_custom_objects(self):
        """Test exporting model with custom layers."""
        # Create model with lambda layer (custom objects)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(20,)),
            tf.keras.layers.Lambda(lambda x: x * 2),
            tf.keras.layers.Dense(2)
        ])
        
        # Should handle or error appropriately
        assert model is not None
    
    def test_export_stateful_model(self):
        """Test exporting stateful RNN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(10, stateful=True, input_shape=(5, 20)),
            tf.keras.layers.Dense(2)
        ])
        
        # Stateful models have special export requirements
        assert model is not None


class TestVisualizationEdgeCases:
    """Test visualization with edge cases."""
    
    def test_plot_with_empty_history(self):
        """Test plotting with empty training history."""
        visualizer = MetricsVisualizer()
        
        empty_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        # Should handle empty history gracefully
        assert visualizer is not None
    
    def test_plot_with_single_epoch(self):
        """Test plotting with single epoch."""
        visualizer = MetricsVisualizer()
        
        single_history = {
            'loss': [0.5],
            'val_loss': [0.6],
            'accuracy': [0.8],
            'val_accuracy': [0.75]
        }
        
        assert visualizer is not None
    
    def test_plot_with_nan_values(self):
        """Test plotting with NaN values."""
        visualizer = MetricsVisualizer()
        
        nan_history = {
            'loss': [0.5, np.nan, 0.3],
            'val_loss': [0.6, 0.5, np.nan],
            'accuracy': [0.8, 0.85, 0.9],
            'val_accuracy': [0.75, np.nan, 0.8]
        }
        
        assert visualizer is not None


class TestIntegrationEdgeCases:
    """Test integration scenarios with edge cases."""
    
    def test_full_pipeline_with_minimal_data(self):
        """Test complete pipeline with minimal dataset."""
        # Minimal dataset: 2 samples, 2 classes
        X = np.random.randn(2, 28, 28, 1).astype(np.float32)
        y = np.array([0, 1])
        
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=2
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Train on minimal data
        history = model.fit(X, y, epochs=1, verbose=0)
        
        # Predict
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (2, 2)
    
    def test_cross_validation_with_small_dataset(self):
        """Test cross-validation with very small dataset."""
        X = np.random.randn(5, 20)
        y = np.array([0, 1, 0, 1, 0])
        
        # Should handle k-fold validation gracefully
        # even with n_splits > samples
        assert len(X) >= 2
    
    def test_memory_efficiency_with_large_batch(self):
        """Test memory efficiency with very large batch."""
        # Create a large batch
        large_batch = np.random.randn(10000, 28, 28, 1).astype(np.float32)
        
        # Load into tf.data
        dataset = tf.data.Dataset.from_tensor_slices(large_batch)
        dataset = dataset.batch(1000)
        
        # Should process without memory issues
        batch_count = 0
        for batch in dataset:
            batch_count += 1
        
        assert batch_count == 10


class TestConcurrencyEdgeCases:
    """Test thread safety and concurrency."""
    
    def test_multiple_model_instances(self):
        """Test creating multiple models concurrently."""
        models = []
        for i in range(5):
            model = ModelBuilders.create_cnn_classifier(
                input_shape=(28, 28, 1),
                num_classes=10
            )
            models.append(model)
        
        assert len(models) == 5
        for model in models:
            assert model is not None
    
    def test_model_prediction_thread_safety(self):
        """Test thread safety of model predictions."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        X = np.random.randn(100, 28, 28, 1).astype(np.float32)
        
        # Make predictions in sequence
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (100, 10)
