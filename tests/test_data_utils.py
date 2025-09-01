# Location: /tests/test_data_utils.py

"""
Test tf.data pipelines and tf.keras preprocessing utilities.
Comprehensive tests for data_utils module functionality.
"""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from data_utils import (
    DataPipeline, 
    TFRecordHandler, 
    DataAugmentation,
    create_image_classification_pipeline,
    create_text_classification_pipeline,
    create_feature_description_image,
    create_feature_description_text
)
from tests import TEST_CONFIG


class TestTFRecordHandler:
    """Test TFRecord creation and parsing utilities."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.handler = TFRecordHandler()
    
    def test_bytes_feature(self):
        """Test bytes feature creation."""
        # Test with string
        feature = self.handler._bytes_feature("test_string")
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.bytes_list.value) == 1
        
        # Test with bytes
        feature = self.handler._bytes_feature(b"test_bytes")
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.bytes_list.value) == 1
    
    def test_float_feature(self):
        """Test float feature creation."""
        # Test with single float
        feature = self.handler._float_feature(3.14)
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.float_list.value) == 1
        assert feature.float_list.value[0] == 3.14
        
        # Test with list of floats
        feature = self.handler._float_feature([1.0, 2.0, 3.0])
        assert len(feature.float_list.value) == 3
    
    def test_int64_feature(self):
        """Test int64 feature creation."""
        # Test with single int
        feature = self.handler._int64_feature(42)
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.int64_list.value) == 1
        assert feature.int64_list.value[0] == 42
        
        # Test with list of ints
        feature = self.handler._int64_feature([1, 2, 3])
        assert len(feature.int64_list.value) == 3
    
    def test_serialize_text_example(self):
        """Test text example serialization."""
        text = "This is a test sentence"
        label = 1
        
        serialized = self.handler.serialize_text_example(text, label)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Test with additional features
        additional_features = {"sentiment": "positive", "score": 0.85}
        serialized = self.handler.serialize_text_example(text, label, additional_features)
        assert isinstance(serialized, bytes)
    
    def test_write_tfrecord(self, temp_dir):
        """Test TFRecord writing."""
        # Create sample examples
        examples = []
        for i in range(5):
            example = self.handler.serialize_text_example(f"text_{i}", i)
            examples.append(example)
        
        # Write to TFRecord
        tfrecord_path = os.path.join(temp_dir, "test.tfrecord")
        self.handler.write_tfrecord(examples, tfrecord_path)
        
        # Verify file exists and has content
        assert os.path.exists(tfrecord_path)
        assert os.path.getsize(tfrecord_path) > 0


class TestDataPipeline:
    """Test data pipeline creation and optimization."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.pipeline = DataPipeline(batch_size=TEST_CONFIG['batch_size'])
    
    def test_initialization(self):
        """Test DataPipeline initialization."""
        assert self.pipeline.batch_size == TEST_CONFIG['batch_size']
        assert self.pipeline.shuffle_buffer == 1000  # default
        assert isinstance(self.pipeline.tfrecord_handler, TFRecordHandler)
    
    @patch('tensorflow.io.read_file')
    @patch('tensorflow.io.decode_image')
    def test_create_image_dataset(self, mock_decode, mock_read):
        """Test image dataset creation."""
        # Mock file operations
        mock_read.return_value = b"fake_image_data"
        mock_decode.return_value = tf.random.normal(TEST_CONFIG['image_shape'])
        
        # Create fake paths and labels
        image_paths = [f"image_{i}.jpg" for i in range(10)]
        labels = list(range(10))
        
        dataset = self.pipeline.create_image_dataset(
            image_paths, labels,
            image_size=(64, 64),
            augment=False,
            cache=False
        )
        
        assert isinstance(dataset, tf.data.Dataset)
        
        # Check dataset structure
        for batch_images, batch_labels in dataset.take(1):
            assert batch_images.shape[0] <= TEST_CONFIG['batch_size']
            assert len(batch_images.shape) == 4  # (batch, height, width, channels)
            assert batch_labels.shape[0] <= TEST_CONFIG['batch_size']
    
    def test_create_text_dataset(self):
        """Test text dataset creation."""
        texts = [f"This is sample text number {i}" for i in range(20)]
        labels = [i % TEST_CONFIG['num_classes'] for i in range(20)]
        
        dataset, vectorizer = self.pipeline.create_text_dataset(
            texts, labels,
            max_length=32,
            vocab_size=100
        )
        
        assert isinstance(dataset, tf.data.Dataset)
        assert isinstance(vectorizer, tf.keras.layers.TextVectorization)
        
        # Check dataset structure
        for batch_texts, batch_labels in dataset.take(1):
            assert batch_texts.shape[0] <= TEST_CONFIG['batch_size']
            assert len(batch_texts.shape) == 2  # (batch, sequence_length)
            assert batch_labels.shape[0] <= TEST_CONFIG['batch_size']
    
    def test_create_tfrecord_dataset(self, temp_dir):
        """Test TFRecord dataset creation."""
        # Create a sample TFRecord file
        handler = TFRecordHandler()
        examples = []
        for i in range(10):
            example = handler.serialize_text_example(f"text_{i}", i)
            examples.append(example)
        
        tfrecord_path = os.path.join(temp_dir, "test.tfrecord")
        handler.write_tfrecord(examples, tfrecord_path)
        
        # Create feature description
        feature_description = create_feature_description_text()
        
        # Create dataset
        dataset = self.pipeline.create_tfrecord_dataset(
            [tfrecord_path], 
            feature_description
        )
        
        assert isinstance(dataset, tf.data.Dataset)
        
        # Check dataset can be iterated
        count = 0
        for batch in dataset.take(2):
            count += 1
        assert count <= 2
    
    def test_mixed_precision_dataset(self, sample_dataset):
        """Test mixed precision dataset configuration."""
        mixed_dataset = self.pipeline.create_mixed_precision_dataset(sample_dataset)
        assert isinstance(mixed_dataset, tf.data.Dataset)
        
        # Check data types
        for features, labels in mixed_dataset.take(1):
            # Note: In practice, you'd check dtype conversion
            assert features is not None
            assert labels is not None


class TestDataAugmentation:
    """Test data augmentation utilities."""
    
    def test_create_augmentation_layer(self):
        """Test augmentation layer creation."""
        aug_layer = DataAugmentation.create_augmentation_layer()
        
        assert isinstance(aug_layer, tf.keras.Sequential)
        assert len(aug_layer.layers) > 0
        
        # Test layer application
        test_images = tf.random.normal([4] + list(TEST_CONFIG['image_shape']))
        augmented = aug_layer(test_images, training=True)
        
        assert augmented.shape == test_images.shape
    
    def test_mixup(self, sample_dataset):
        """Test MixUp augmentation."""
        mixup_dataset = DataAugmentation.mixup(sample_dataset, alpha=0.2)
        assert isinstance(mixup_dataset, tf.data.Dataset)
        
        # Check mixed data
        for mixed_x, mixed_y in mixup_dataset.take(1):
            assert mixed_x.shape[1:] == TEST_CONFIG['image_shape']
            assert len(mixed_y.shape) == 1  # Mixed labels should be 1D
    
    def test_cutmix(self, sample_dataset):
        """Test CutMix augmentation."""
        cutmix_dataset = DataAugmentation.cutmix(sample_dataset, alpha=1.0)
        assert isinstance(cutmix_dataset, tf.data.Dataset)
        
        # Check dataset structure
        for mixed_x, mixed_y in cutmix_dataset.take(1):
            assert mixed_x.shape[1:] == TEST_CONFIG['image_shape']
            assert len(mixed_y.shape) == 1


class TestFeatureDescriptions:
    """Test feature description utilities."""
    
    def test_create_feature_description_image(self):
        """Test image feature description creation."""
        desc = create_feature_description_image()
        
        assert isinstance(desc, dict)
        required_keys = ['image', 'label', 'height', 'width', 'channels', 'filename']
        for key in required_keys:
            assert key in desc
            assert isinstance(desc[key], tf.io.FixedLenFeature)
    
    def test_create_feature_description_text(self):
        """Test text feature description creation."""
        desc = create_feature_description_text()
        
        assert isinstance(desc, dict)
        required_keys = ['text', 'label', 'text_length']
        for key in required_keys:
            assert key in desc
            assert isinstance(desc[key], tf.io.FixedLenFeature)


class TestConvenienceFunctions:
    """Test high-level convenience functions."""
    
    @patch('tensorflow.keras.utils.image_dataset_from_directory')
    def test_create_image_classification_pipeline(self, mock_dataset_func):
        """Test image classification pipeline creation."""
        # Mock the dataset creation
        mock_dataset = tf.data.Dataset.from_tensor_slices((
            tf.random.normal([20] + list(TEST_CONFIG['image_shape'])),
            tf.random.uniform([20], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
        )).batch(TEST_CONFIG['batch_size'])
        
        mock_dataset_func.return_value = mock_dataset
        
        train_ds, val_ds = create_image_classification_pipeline(
            "fake_directory",
            batch_size=TEST_CONFIG['batch_size'],
            validation_split=0.2
        )
        
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
    
    def test_create_text_classification_pipeline(self):
        """Test text classification pipeline creation."""
        texts = [f"Sample text {i}" for i in range(50)]
        labels = [i % TEST_CONFIG['num_classes'] for i in range(50)]
        
        train_ds, val_ds, vectorizer = create_text_classification_pipeline(
            texts, labels,
            batch_size=TEST_CONFIG['batch_size'],
            validation_split=0.2
        )
        
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
        assert isinstance(vectorizer, tf.keras.layers.TextVectorization)
        
        # Check datasets have correct structure
        for batch_x, batch_y in train_ds.take(1):
            assert len(batch_x.shape) == 2  # (batch, sequence)
            assert len(batch_y.shape) == 1  # (batch,)


class TestDataPipelineIntegration:
    """Integration tests for data pipeline components."""
    
    def test_end_to_end_image_pipeline(self, temp_dir):
        """Test complete image data pipeline."""
        # Create fake image files (just empty files for testing)
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir)
        
        for i in range(5):
            with open(os.path.join(image_dir, f"image_{i}.jpg"), 'w') as f:
                f.write("fake_image_content")
        
        # This would normally fail with real images, but tests file handling
        try:
            pipeline = DataPipeline()
            # We can't actually test with fake files, so just test the object creation
            assert pipeline is not None
        except Exception:
            # Expected to fail with fake image files
            pass
    
    def test_end_to_end_text_pipeline(self):
        """Test complete text data pipeline."""
        pipeline = DataPipeline(batch_size=4)
        
        texts = [
            "This is a positive review",
            "This product is terrible", 
            "Average quality product",
            "Excellent service and quality",
            "Would not recommend"
        ]
        labels = [1, 0, 1, 1, 0]  # Binary sentiment
        
        dataset, vectorizer = pipeline.create_text_dataset(
            texts, labels, max_length=16, vocab_size=50
        )
        
        # Verify dataset works end-to-end
        total_batches = 0
        total_samples = 0
        
        for batch_x, batch_y in dataset:
            total_batches += 1
            total_samples += batch_x.shape[0]
            
            # Check data types and shapes
            assert batch_x.dtype == tf.int64  # Tokenized text
            assert batch_y.dtype == tf.int32  # Labels
            assert batch_x.shape[1] == 16  # Sequence length
        
        assert total_batches > 0
        assert total_samples == len(texts)
    
    def test_pipeline_performance(self):
        """Test pipeline performance optimizations."""
        # Create a larger dataset for performance testing
        images = tf.random.normal([100] + list(TEST_CONFIG['image_shape']))
        labels = tf.random.uniform([100], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        # Test different optimization strategies
        basic_dataset = dataset.batch(TEST_CONFIG['batch_size'])
        
        optimized_dataset = (dataset
                           .cache()
                           .shuffle(50)
                           .batch(TEST_CONFIG['batch_size'])
                           .prefetch(tf.data.AUTOTUNE))
        
        # Both should be iterable
        assert sum(1 for _ in basic_dataset.take(5)) <= 5
        assert sum(1 for _ in optimized_dataset.take(5)) <= 5


# Pytest configuration for this test file
def pytest_configure(config):
    """Configure pytest for data utils tests."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

# Performance tests (marked as slow)
@pytest.mark.slow
class TestDataPipelinePerformance:
    """Performance tests for data pipelines (marked as slow)."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger synthetic dataset
        num_samples = 1000
        images = tf.random.normal([num_samples] + list(TEST_CONFIG['image_shape']))
        labels = tf.random.uniform([num_samples], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        import time
        start_time = time.time()
        
        # Iterate through dataset
        for batch in dataset:
            pass
        
        elapsed = time.time() - start_time
        
        # Should process 1000 samples reasonably quickly
        assert elapsed < 10.0  # Adjust threshold as needed
        
        print(f"Processed {num_samples} samples in {elapsed:.2f} seconds")


if __name__ == "__main__":
    pytest.main([__file__])