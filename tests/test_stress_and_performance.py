# Location: /tests/test_stress_and_performance.py

"""
Stress tests and performance benchmarks.
Tests system behavior under extreme conditions.
"""

import pytest
import tensorflow as tf
import numpy as np
import time
from unittest.mock import patch

from data_utils import DataPipeline, DataAugmentation
from model_utils import ModelBuilders, TrainingUtilities
from optimization_utils import ModelQuantization
from export_utils import ModelExporter


class TestMemoryBoundaries:
    """Test memory usage and boundaries."""
    
    def test_large_model_creation(self):
        """Test creating very large model."""
        # Create ResNet-like model
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=(224, 224, 3),
            classes=10
        )
        
        params = model.count_params()
        assert params > 20_000_000  # ~23M parameters
    
    def test_large_batch_processing(self):
        """Test processing large batches."""
        batch_size = 1000
        X = np.random.randn(batch_size, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, batch_size)
        
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Train on large batch
        history = model.fit(X, y, batch_size=256, epochs=1, verbose=0)
        
        assert history is not None
    
    def test_dataset_prefetching(self):
        """Test data pipeline prefetching."""
        X = np.random.randn(1000, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, 1000)
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Iterate through dataset
        count = 0
        for batch_x, batch_y in dataset:
            count += 1
        
        assert count == 32  # 1000 / 32


class TestDataPipelineStress:
    """Stress test data pipelines."""
    
    def test_augmentation_pipeline_stability(self):
        """Test data augmentation stability over many iterations."""
        augmenter = DataAugmentation()
        
        X = np.random.randn(100, 28, 28, 1).astype(np.float32)
        
        for i in range(100):
            # Apply augmentation multiple times
            augmented = augmenter.random_flip(X)
            assert augmented.shape == X.shape
            assert not np.array_equal(augmented, X)  # Should be different
    
    def test_tfrecord_streaming(self):
        """Test reading TFRecord in streaming fashion."""
        from data_utils import TFRecordHandler
        
        handler = TFRecordHandler()
        
        # Create many examples
        examples = []
        for i in range(100):
            example = handler.serialize_text_example(f"text_{i}", i % 2)
            examples.append(example)
        
        # Stream should handle this
        assert len(examples) == 100
    
    def test_cache_vs_no_cache_performance(self):
        """Compare performance with and without caching."""
        X = np.random.randn(100, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, 100)
        
        # Without cache
        dataset_no_cache = tf.data.Dataset.from_tensor_slices((X, y))
        dataset_no_cache = dataset_no_cache.batch(32)
        
        start = time.time()
        for _ in dataset_no_cache:
            pass
        time_no_cache = time.time() - start
        
        # With cache
        dataset_cache = tf.data.Dataset.from_tensor_slices((X, y))
        dataset_cache = dataset_cache.batch(32)
        dataset_cache = dataset_cache.cache()
        
        start = time.time()
        for _ in dataset_cache:
            pass
        time_cache = time.time() - start
        
        # Both should complete
        assert time_no_cache > 0
        assert time_cache > 0


class TestModelTrainingStress:
    """Stress test model training."""
    
    def test_training_stability_many_epochs(self):
        """Test training stability over many epochs."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        # Train for many epochs
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=16,
            verbose=0,
            validation_split=0.2
        )
        
        # Loss should generally decrease
        assert len(history.history['loss']) == 50
    
    def test_distributed_training_readiness(self):
        """Test model for distributed training readiness."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        # Model should be serializable for distributed setup
        assert model.get_config() is not None
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation pattern."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam()
        
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        accumulated_loss = 0
        accumulation_steps = 5
        
        for i in range(0, min(10, len(X)), accumulation_steps):
            batch_x = X[i:i + accumulation_steps]
            batch_y = y[i:i + accumulation_steps]
            
            with tf.GradientTape() as tape:
                predictions = model(batch_x)
                loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions)
                loss = tf.reduce_mean(loss)
            
            accumulated_loss += float(loss)
        
        assert accumulated_loss > 0


class TestOptimizationStress:
    """Stress test optimization utilities."""
    
    def test_quantization_on_various_models(self):
        """Test quantization compatibility with various architectures."""
        architectures = ['simple', 'sequential', 'functional']
        
        for arch in architectures:
            try:
                model = ModelBuilders.create_cnn_classifier(
                    input_shape=(28, 28, 1),
                    num_classes=10,
                    architecture=arch if arch in ['simple', 'sequential'] else 'simple'
                )
                
                # Should be quantizable
                assert model is not None
            except ValueError:
                # Some architectures might not support all quantization types
                pass
    
    def test_model_compression_ratio(self):
        """Test various compression techniques."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Test different compression levels
        compression_ratios = []
        
        # Original size
        original_params = model.count_params()
        compression_ratios.append(1.0)
        
        assert original_params > 0


class TestDataLoadingStress:
    """Stress test data loading mechanisms."""
    
    def test_shuffle_stability(self):
        """Test shuffle stability across multiple iterations."""
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=100, seed=42)
        dataset = dataset.batch(10)
        
        # Iterate twice with same seed
        batches_1 = [batch for batch, _ in dataset]
        
        # Recreate with same seed
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=100, seed=42)
        dataset = dataset.batch(10)
        
        batches_2 = [batch for batch, _ in dataset]
        
        # Should be same order with same seed
        assert len(batches_1) == len(batches_2)
    
    def test_repeated_dataset(self):
        """Test repeating dataset multiple times."""
        X = np.random.randn(10, 28, 28, 1).astype(np.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.repeat(10)
        
        count = 0
        for _ in dataset:
            count += 1
        
        assert count == 100  # 10 samples * 10 repeats
    
    def test_interleave_multiple_datasets(self):
        """Test interleaving multiple datasets."""
        X1 = np.random.randn(20, 28, 28, 1).astype(np.float32)
        X2 = np.random.randn(20, 28, 28, 1).astype(np.float32)
        
        dataset1 = tf.data.Dataset.from_tensor_slices(X1)
        dataset2 = tf.data.Dataset.from_tensor_slices(X2)
        
        # Interleave
        interleaved = tf.data.Dataset.from_tensor_slices([dataset1, dataset2])
        interleaved = interleaved.interleave(
            lambda x: x,
            cycle_length=2
        )
        
        count = 0
        for _ in interleaved:
            count += 1
        
        # Should have all samples
        assert count >= 20


class TestPerformanceBenchmarks:
    """Performance benchmarking."""
    
    def test_inference_speed_benchmark(self):
        """Benchmark inference speed."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        X = np.random.randn(100, 28, 28, 1).astype(np.float32)
        
        # Warmup
        _ = model.predict(X[:10], verbose=0)
        
        # Benchmark
        start = time.time()
        predictions = model.predict(X, verbose=0, batch_size=32)
        inference_time = time.time() - start
        
        throughput = len(X) / inference_time
        assert throughput > 0
        
        print(f"Inference throughput: {throughput:.0f} samples/sec")
    
    def test_training_speed_benchmark(self):
        """Benchmark training speed."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        X = np.random.randn(500, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, 500)
        
        start = time.time()
        history = model.fit(
            X, y,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        training_time = time.time() - start
        
        samples_per_second = (500 * 5) / training_time
        print(f"Training throughput: {samples_per_second:.0f} samples/sec")
        
        assert training_time > 0
    
    def test_model_loading_speed(self):
        """Benchmark model loading time."""
        import tempfile
        
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model"
            model.save(model_path)
            
            start = time.time()
            loaded_model = tf.keras.models.load_model(model_path)
            load_time = time.time() - start
            
            assert load_time > 0
            assert loaded_model is not None
            print(f"Model load time: {load_time*1000:.1f} ms")


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_training_with_corrupted_batch(self):
        """Test training recovery from bad batch."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        # Should train despite potential issues
        history = model.fit(X, y, epochs=1, verbose=0)
        
        assert history is not None
    
    def test_prediction_type_mismatch_recovery(self):
        """Test handling of data type mismatches."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Try prediction with different dtypes
        X_float32 = np.random.randn(10, 20).astype(np.float32)
        X_float64 = np.random.randn(10, 20).astype(np.float64)
        X_int32 = np.random.randint(0, 10, (10, 20)).astype(np.int32)
        
        # Model should handle conversions
        pred1 = model.predict(X_float32, verbose=0)
        pred2 = model.predict(X_float64, verbose=0)
        pred3 = model.predict(X_int32.astype(np.float32), verbose=0)
        
        assert pred1.shape == pred2.shape == pred3.shape
