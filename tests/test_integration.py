# Location: /tests/test_integration.py

"""
End-to-end integration tests for TensorVerseHub.
Tests complete workflows, model training, data pipelines, and system integration.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import subprocess

# TensorFlow imports
try:
    import tensorflow as tf
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available for integration testing")

# Test configuration
from tests import TEST_CONFIG


# Skip all tests if TensorFlow not available
pytestmark = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow not available for integration testing"
)


class IntegrationTestEnvironment:
    """Setup and manage integration test environment."""
    
    def __init__(self, base_dir: str = None):
        """Initialize integration test environment."""
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="tensorversehub_integration_")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.checkpoints_dir = os.path.join(self.base_dir, "checkpoints")
        
        # Create directories
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.checkpoints_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Test results tracking
        self.test_results = {
            'environment_setup': False,
            'data_generation': False,
            'model_training': False,
            'model_evaluation': False,
            'pipeline_execution': False,
            'cleanup': False,
            'errors': []
        }
    
    def setup_test_data(self) -> bool:
        """Generate test datasets for integration testing."""
        try:
            # Generate synthetic image classification dataset
            self._create_image_dataset()
            
            # Generate synthetic tabular dataset
            self._create_tabular_dataset()
            
            # Generate synthetic text dataset
            self._create_text_dataset()
            
            # Generate TFRecords
            self._create_tfrecords()
            
            self.test_results['data_generation'] = True
            return True
            
        except Exception as e:
            self.test_results['errors'].append(f"Data setup failed: {str(e)}")
            return False
    
    def _create_image_dataset(self):
        """Create synthetic image dataset for testing."""
        # Create directory structure
        train_dir = os.path.join(self.data_dir, "images", "train")
        val_dir = os.path.join(self.data_dir, "images", "val")
        
        for class_name in ["class_a", "class_b", "class_c"]:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Generate synthetic images
        for split, split_dir in [("train", train_dir), ("val", val_dir)]:
            samples_per_class = 50 if split == "train" else 20
            
            for class_idx, class_name in enumerate(["class_a", "class_b", "class_c"]):
                for i in range(samples_per_class):
                    # Create synthetic 32x32x3 image
                    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                    
                    # Add some class-specific patterns
                    if class_idx == 0:  # class_a - add red tint
                        image[:, :, 0] = np.clip(image[:, :, 0] + 50, 0, 255)
                    elif class_idx == 1:  # class_b - add green tint
                        image[:, :, 1] = np.clip(image[:, :, 1] + 50, 0, 255)
                    else:  # class_c - add blue tint
                        image[:, :, 2] = np.clip(image[:, :, 2] + 50, 0, 255)
                    
                    # Save as numpy array (in real scenario, would be actual images)
                    image_path = os.path.join(split_dir, class_name, f"image_{i:04d}.npy")
                    np.save(image_path, image)
        
        # Create dataset metadata
        metadata = {
            'dataset_type': 'image_classification',
            'num_classes': 3,
            'class_names': ['class_a', 'class_b', 'class_c'],
            'image_shape': [32, 32, 3],
            'train_samples': 150,
            'val_samples': 60
        }
        
        with open(os.path.join(self.data_dir, "images", "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_tabular_dataset(self):
        """Create synthetic tabular dataset for testing."""
        # Generate synthetic tabular data
        n_train = 1000
        n_val = 300
        n_features = 10
        
        # Training data
        X_train = np.random.randn(n_train, n_features)
        # Create target with some relationship to features
        y_train = (X_train[:, 0] + X_train[:, 1] * 0.5 + np.random.randn(n_train) * 0.1) > 0
        y_train = y_train.astype(int)
        
        # Validation data
        X_val = np.random.randn(n_val, n_features)
        y_val = (X_val[:, 0] + X_val[:, 1] * 0.5 + np.random.randn(n_val) * 0.1) > 0
        y_val = y_val.astype(int)
        
        # Save datasets
        tabular_dir = os.path.join(self.data_dir, "tabular")
        os.makedirs(tabular_dir, exist_ok=True)
        
        np.save(os.path.join(tabular_dir, "X_train.npy"), X_train)
        np.save(os.path.join(tabular_dir, "y_train.npy"), y_train)
        np.save(os.path.join(tabular_dir, "X_val.npy"), X_val)
        np.save(os.path.join(tabular_dir, "y_val.npy"), y_val)
        
        # Create metadata
        metadata = {
            'dataset_type': 'binary_classification',
            'num_features': n_features,
            'num_classes': 2,
            'train_samples': n_train,
            'val_samples': n_val,
            'feature_names': [f'feature_{i}' for i in range(n_features)]
        }
        
        with open(os.path.join(tabular_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_text_dataset(self):
        """Create synthetic text dataset for testing."""
        # Generate synthetic text sequences
        vocab_size = 1000
        seq_length = 50
        n_train = 500
        n_val = 150
        
        # Training data
        X_train = np.random.randint(0, vocab_size, (n_train, seq_length))
        y_train = np.random.randint(0, 2, n_train)  # Binary classification
        
        # Validation data
        X_val = np.random.randint(0, vocab_size, (n_val, seq_length))
        y_val = np.random.randint(0, 2, n_val)
        
        # Save datasets
        text_dir = os.path.join(self.data_dir, "text")
        os.makedirs(text_dir, exist_ok=True)
        
        np.save(os.path.join(text_dir, "X_train.npy"), X_train)
        np.save(os.path.join(text_dir, "y_train.npy"), y_train)
        np.save(os.path.join(text_dir, "X_val.npy"), X_val)
        np.save(os.path.join(text_dir, "y_val.npy"), y_val)
        
        # Create metadata
        metadata = {
            'dataset_type': 'text_classification',
            'vocab_size': vocab_size,
            'sequence_length': seq_length,
            'num_classes': 2,
            'train_samples': n_train,
            'val_samples': n_val
        }
        
        with open(os.path.join(text_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_tfrecords(self):
        """Create TFRecord files for testing."""
        tfrecord_dir = os.path.join(self.data_dir, "tfrecords")
        os.makedirs(tfrecord_dir, exist_ok=True)
        
        # Create a simple TFRecord with image-like data
        train_tfrecord = os.path.join(tfrecord_dir, "train.tfrecord")
        val_tfrecord = os.path.join(tfrecord_dir, "val.tfrecord")
        
        def _create_example(image, label):
            """Create a TFRecord example."""
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[32])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[32])),
                'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[3]))
            }
            return tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Write training TFRecords
        with tf.io.TFRecordWriter(train_tfrecord) as writer:
            for i in range(100):
                image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                label = i % 3
                example = _create_example(image, label)
                writer.write(example.SerializeToString())
        
        # Write validation TFRecords
        with tf.io.TFRecordWriter(val_tfrecord) as writer:
            for i in range(30):
                image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                label = i % 3
                example = _create_example(image, label)
                writer.write(example.SerializeToString())
        
        # Create metadata
        metadata = {
            'dataset_type': 'tfrecord',
            'train_file': 'train.tfrecord',
            'val_file': 'val.tfrecord',
            'train_samples': 100,
            'val_samples': 30,
            'image_shape': [32, 32, 3],
            'num_classes': 3
        }
        
        with open(os.path.join(tfrecord_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            if os.path.exists(self.base_dir):
                shutil.rmtree(self.base_dir)
            self.test_results['cleanup'] = True
        except Exception as e:
            self.test_results['errors'].append(f"Cleanup failed: {str(e)}")


class ModelTestSuite:
    """Test suite for model training and evaluation."""
    
    def __init__(self, environment: IntegrationTestEnvironment):
        self.env = environment
        self.models = {}
        self.training_histories = {}
    
    def create_simple_cnn(self, input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
        """Create a simple CNN for testing."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_simple_mlp(self, input_shape: int, num_classes: int) -> tf.keras.Model:
        """Create a simple MLP for testing."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_simple_rnn(self, vocab_size: int, seq_length: int, num_classes: int) -> tf.keras.Model:
        """Create a simple RNN for testing."""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64, input_length=seq_length),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_image_classifier(self) -> Dict[str, Any]:
        """Train image classification model."""
        try:
            # Load synthetic image data
            train_dir = os.path.join(self.env.data_dir, "images", "train")
            val_dir = os.path.join(self.env.data_dir, "images", "val")
            
            # Load training data
            X_train, y_train = self._load_image_data(train_dir)
            X_val, y_val = self._load_image_data(val_dir)
            
            # Normalize pixel values
            X_train = X_train.astype('float32') / 255.0
            X_val = X_val.astype('float32') / 255.0
            
            # Create and train model
            model = self.create_simple_cnn((32, 32, 3), 3)
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.env.checkpoints_dir, "cnn_best.h5"),
                    save_best_only=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=5,  # Short training for testing
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save model and history
            self.models['cnn'] = model
            self.training_histories['cnn'] = history
            
            model.save(os.path.join(self.env.models_dir, "cnn_model.h5"))
            
            return {
                'success': True,
                'final_accuracy': history.history['val_accuracy'][-1],
                'final_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_tabular_classifier(self) -> Dict[str, Any]:
        """Train tabular classification model."""
        try:
            # Load tabular data
            tabular_dir = os.path.join(self.env.data_dir, "tabular")
            
            X_train = np.load(os.path.join(tabular_dir, "X_train.npy"))
            y_train = np.load(os.path.join(tabular_dir, "y_train.npy"))
            X_val = np.load(os.path.join(tabular_dir, "X_val.npy"))
            y_val = np.load(os.path.join(tabular_dir, "y_val.npy"))
            
            # Create and train model
            model = self.create_simple_mlp(X_train.shape[1], 2)
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save model and history
            self.models['mlp'] = model
            self.training_histories['mlp'] = history
            
            model.save(os.path.join(self.env.models_dir, "mlp_model.h5"))
            
            return {
                'success': True,
                'final_accuracy': history.history['val_accuracy'][-1],
                'final_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_text_classifier(self) -> Dict[str, Any]:
        """Train text classification model."""
        try:
            # Load text data
            text_dir = os.path.join(self.env.data_dir, "text")
            
            X_train = np.load(os.path.join(text_dir, "X_train.npy"))
            y_train = np.load(os.path.join(text_dir, "y_train.npy"))
            X_val = np.load(os.path.join(text_dir, "X_val.npy"))
            y_val = np.load(os.path.join(text_dir, "y_val.npy"))
            
            # Load metadata
            with open(os.path.join(text_dir, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            # Create and train model
            model = self.create_simple_rnn(
                metadata['vocab_size'], 
                metadata['sequence_length'], 
                metadata['num_classes']
            )
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=5,  # Short training for testing
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save model and history
            self.models['rnn'] = model
            self.training_histories['rnn'] = history
            
            model.save(os.path.join(self.env.models_dir, "rnn_model.h5"))
            
            return {
                'success': True,
                'final_accuracy': history.history['val_accuracy'][-1],
                'final_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_image_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image data from directory."""
        images = []
        labels = []
        
        class_names = sorted(os.listdir(data_dir))
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    image_path = os.path.join(class_dir, filename)
                    image = np.load(image_path)
                    
                    images.append(image)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels)


class PipelineTestSuite:
    """Test suite for data pipelines and preprocessing."""
    
    def __init__(self, environment: IntegrationTestEnvironment):
        self.env = environment
    
    def test_tfrecord_pipeline(self) -> Dict[str, Any]:
        """Test TFRecord data pipeline."""
        try:
            tfrecord_dir = os.path.join(self.env.data_dir, "tfrecords")
            train_tfrecord = os.path.join(tfrecord_dir, "train.tfrecord")
            
            # Define parsing function
            def parse_example(example_proto):
                feature_description = {
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'channels': tf.io.FixedLenFeature([], tf.int64),
                }
                
                parsed_features = tf.io.parse_single_example(example_proto, feature_description)
                
                # Decode image
                image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
                image = tf.reshape(image, [32, 32, 3])
                image = tf.cast(image, tf.float32) / 255.0
                
                label = tf.cast(parsed_features['label'], tf.int32)
                
                return image, label
            
            # Create dataset
            dataset = tf.data.TFRecordDataset(train_tfrecord)
            dataset = dataset.map(parse_example)
            dataset = dataset.batch(16)
            dataset = dataset.prefetch(tf.data.EXPERIMENTAL_AUTOTUNE)
            
            # Test pipeline by iterating through a few batches
            batches_processed = 0
            total_samples = 0
            
            for images, labels in dataset.take(5):
                batch_size = images.shape[0]
                total_samples += batch_size
                batches_processed += 1
                
                # Verify batch dimensions
                assert images.shape[1:] == (32, 32, 3)
                assert len(labels.shape) == 1
                assert labels.shape[0] == batch_size
            
            return {
                'success': True,
                'batches_processed': batches_processed,
                'total_samples': total_samples,
                'pipeline_functional': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'pipeline_functional': False
            }
    
    def test_data_augmentation_pipeline(self) -> Dict[str, Any]:
        """Test data augmentation pipeline."""
        try:
            # Create a simple dataset
            images = np.random.rand(50, 32, 32, 3).astype(np.float32)
            labels = np.random.randint(0, 3, 50)
            
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            
            # Define augmentation function
            def augment(image, label):
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                return image, label
            
            # Apply augmentations
            augmented_dataset = dataset.map(augment)
            augmented_dataset = augmented_dataset.batch(16)
            
            # Test pipeline
            batches_processed = 0
            for aug_images, aug_labels in augmented_dataset.take(3):
                batches_processed += 1
                
                # Verify augmented images are still in valid range
                assert tf.reduce_min(aug_images) >= -0.2  # Allow for some brightness reduction
                assert tf.reduce_max(aug_images) <= 1.2   # Allow for some brightness increase
            
            return {
                'success': True,
                'batches_processed': batches_processed,
                'augmentation_functional': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'augmentation_functional': False
            }
    
    def test_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Test preprocessing pipeline with normalization and resizing."""
        try:
            # Create synthetic data with different sizes
            images = [
                np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8),
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
                np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8),
            ]
            
            def preprocess_image(image):
                # Convert to float and normalize
                image = tf.cast(image, tf.float32) / 255.0
                
                # Resize to standard size
                image = tf.image.resize(image, [32, 32])
                
                # Additional preprocessing
                image = tf.image.per_image_standardization(image)
                
                return image
            
            # Test preprocessing on each image
            processed_images = []
            for img in images:
                processed = preprocess_image(img)
                processed_images.append(processed)
                
                # Verify output shape
                assert processed.shape == (32, 32, 3)
            
            # Test batch preprocessing
            batch_images = tf.stack(processed_images)
            assert batch_images.shape == (3, 32, 32, 3)
            
            return {
                'success': True,
                'preprocessing_functional': True,
                'output_shape_consistent': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'preprocessing_functional': False
            }


# Test Classes
class TestEnvironmentSetup:
    """Test integration environment setup."""
    
    @pytest.fixture(scope="class")
    def integration_env(self):
        """Create integration test environment."""
        env = IntegrationTestEnvironment()
        yield env
        env.cleanup()
    
    def test_environment_creation(self, integration_env):
        """Test that environment is created correctly."""
        assert os.path.exists(integration_env.base_dir)
        assert os.path.exists(integration_env.data_dir)
        assert os.path.exists(integration_env.models_dir)
        assert os.path.exists(integration_env.logs_dir)
        assert os.path.exists(integration_env.checkpoints_dir)
        
        integration_env.test_results['environment_setup'] = True
    
    def test_data_generation(self, integration_env):
        """Test synthetic data generation."""
        success = integration_env.setup_test_data()
        
        assert success, f"Data generation failed: {integration_env.test_results['errors']}"
        
        # Verify image data
        assert os.path.exists(os.path.join(integration_env.data_dir, "images"))
        assert os.path.exists(os.path.join(integration_env.data_dir, "images", "metadata.json"))
        
        # Verify tabular data
        assert os.path.exists(os.path.join(integration_env.data_dir, "tabular"))
        assert os.path.exists(os.path.join(integration_env.data_dir, "tabular", "X_train.npy"))
        
        # Verify text data
        assert os.path.exists(os.path.join(integration_env.data_dir, "text"))
        assert os.path.exists(os.path.join(integration_env.data_dir, "text", "X_train.npy"))
        
        # Verify TFRecords
        assert os.path.exists(os.path.join(integration_env.data_dir, "tfrecords"))
        assert os.path.exists(os.path.join(integration_env.data_dir, "tfrecords", "train.tfrecord"))


class TestModelTraining:
    """Test model training workflows."""
    
    @pytest.fixture(scope="class")
    def setup_environment(self):
        """Setup environment and data for model training tests."""
        env = IntegrationTestEnvironment()
        env.setup_test_data()
        model_suite = ModelTestSuite(env)
        
        yield env, model_suite
        
        env.cleanup()
    
    def test_cnn_training(self, setup_environment):
        """Test CNN model training."""
        env, model_suite = setup_environment
        
        result = model_suite.train_image_classifier()
        
        assert result['success'], f"CNN training failed: {result.get('error', 'Unknown error')}"
        assert result['final_accuracy'] > 0.1  # Sanity check
        assert result['epochs_trained'] > 0
        
        # Verify model was saved
        assert os.path.exists(os.path.join(env.models_dir, "cnn_model.h5"))
        
        # Test model loading
        loaded_model = tf.keras.models.load_model(os.path.join(env.models_dir, "cnn_model.h5"))
        assert loaded_model is not None
        
        env.test_results['model_training'] = True
    
    def test_mlp_training(self, setup_environment):
        """Test MLP model training."""
        env, model_suite = setup_environment
        
        result = model_suite.train_tabular_classifier()
        
        assert result['success'], f"MLP training failed: {result.get('error', 'Unknown error')}"
        assert result['final_accuracy'] > 0.4  # Should be reasonable for synthetic data
        assert result['epochs_trained'] > 0
        
        # Verify model was saved
        assert os.path.exists(os.path.join(env.models_dir, "mlp_model.h5"))
    
    def test_rnn_training(self, setup_environment):
        """Test RNN model training."""
        env, model_suite = setup_environment
        
        result = model_suite.train_text_classifier()
        
        assert result['success'], f"RNN training failed: {result.get('error', 'Unknown error')}"
        assert result['final_accuracy'] > 0.3  # Should be reasonable for synthetic data
        assert result['epochs_trained'] > 0
        
        # Verify model was saved
        assert os.path.exists(os.path.join(env.models_dir, "rnn_model.h5"))
    
    def test_training_callbacks(self, setup_environment):
        """Test that training callbacks work correctly."""
        env, model_suite = setup_environment
        
        # Train a simple model with callbacks
        result = model_suite.train_image_classifier()
        
        assert result['success']
        
        # Check if checkpoint was created
        checkpoint_path = os.path.join(env.checkpoints_dir, "cnn_best.h5")
        assert os.path.exists(checkpoint_path)
        
        # Verify checkpoint can be loaded
        checkpoint_model = tf.keras.models.load_model(checkpoint_path)
        assert checkpoint_model is not None


class TestDataPipelines:
    """Test data pipeline functionality."""
    
    @pytest.fixture(scope="class")
    def setup_pipeline_environment(self):
        """Setup environment for pipeline testing."""
        env = IntegrationTestEnvironment()
        env.setup_test_data()
        pipeline_suite = PipelineTestSuite(env)
        
        yield env, pipeline_suite
        
        env.cleanup()
    
    def test_tfrecord_pipeline(self, setup_pipeline_environment):
        """Test TFRecord data pipeline."""
        env, pipeline_suite = setup_pipeline_environment
        
        result = pipeline_suite.test_tfrecord_pipeline()
        
        assert result['success'], f"TFRecord pipeline failed: {result.get('error', 'Unknown error')}"
        assert result['pipeline_functional']
        assert result['batches_processed'] > 0
        assert result['total_samples'] > 0
        
        env.test_results['pipeline_execution'] = True
    
    def test_augmentation_pipeline(self, setup_pipeline_environment):
        """Test data augmentation pipeline."""
        env, pipeline_suite = setup_pipeline_environment
        
        result = pipeline_suite.test_data_augmentation_pipeline()
        
        assert result['success'], f"Augmentation pipeline failed: {result.get('error', 'Unknown error')}"
        assert result['augmentation_functional']
        assert result['batches_processed'] > 0
    
    def test_preprocessing_pipeline(self, setup_pipeline_environment):
        """Test preprocessing pipeline."""
        env, pipeline_suite = setup_pipeline_environment
        
        result = pipeline_suite.test_preprocessing_pipeline()
        
        assert result['success'], f"Preprocessing pipeline failed: {result.get('error', 'Unknown error')}"
        assert result['preprocessing_functional']
        assert result['output_shape_consistent']


class TestModelEvaluation:
    """Test model evaluation and inference."""
    
    @pytest.fixture(scope="class")
    def trained_models_environment(self):
        """Setup environment with trained models."""
        env = IntegrationTestEnvironment()
        env.setup_test_data()
        model_suite = ModelTestSuite(env)
        
        # Train all models
        cnn_result = model_suite.train_image_classifier()
        mlp_result = model_suite.train_tabular_classifier()
        rnn_result = model_suite.train_text_classifier()
        
        yield env, model_suite, {
            'cnn': cnn_result,
            'mlp': mlp_result,
            'rnn': rnn_result
        }
        
        env.cleanup()
    
    def test_model_inference(self, trained_models_environment):
        """Test model inference functionality."""
        env, model_suite, training_results = trained_models_environment
        
        # Test CNN inference
        if training_results['cnn']['success']:
            cnn_model = model_suite.models['cnn']
            
            # Create test input
            test_input = np.random.rand(1, 32, 32, 3)
            prediction = cnn_model.predict(test_input, verbose=0)
            
            assert prediction.shape == (1, 3)  # 1 sample, 3 classes
            assert np.isclose(np.sum(prediction), 1.0, atol=1e-6)  # Probabilities sum to 1
        
        # Test MLP inference
        if training_results['mlp']['success']:
            mlp_model = model_suite.models['mlp']
            
            # Create test input
            test_input = np.random.randn(1, 10)
            prediction = mlp_model.predict(test_input, verbose=0)
            
            assert prediction.shape == (1, 2)  # 1 sample, 2 classes
            assert np.isclose(np.sum(prediction), 1.0, atol=1e-6)
        
        env.test_results['model_evaluation'] = True
    
    def test_model_metrics_evaluation(self, trained_models_environment):
        """Test comprehensive model evaluation with metrics."""
        env, model_suite, training_results = trained_models_environment
        
        if training_results['mlp']['success']:
            # Load validation data
            tabular_dir = os.path.join(env.data_dir, "tabular")
            X_val = np.load(os.path.join(tabular_dir, "X_val.npy"))
            y_val = np.load(os.path.join(tabular_dir, "y_val.npy"))
            
            # Get model
            mlp_model = model_suite.models['mlp']
            
            # Evaluate model
            eval_results = mlp_model.evaluate(X_val, y_val, verbose=0)
            
            # eval_results contains [loss, accuracy]
            assert len(eval_results) == 2
            assert eval_results[0] > 0  # Loss should be positive
            assert 0 <= eval_results[1] <= 1  # Accuracy should be between 0 and 1
    
    def test_batch_prediction(self, trained_models_environment):
        """Test batch prediction functionality."""
        env, model_suite, training_results = trained_models_environment
        
        if training_results['cnn']['success']:
            cnn_model = model_suite.models['cnn']
            
            # Create batch test input
            batch_size = 10
            test_batch = np.random.rand(batch_size, 32, 32, 3)
            
            # Batch prediction
            predictions = cnn_model.predict(test_batch, verbose=0)
            
            assert predictions.shape == (batch_size, 3)
            
            # Each prediction should sum to 1 (softmax output)
            for i in range(batch_size):
                assert np.isclose(np.sum(predictions[i]), 1.0, atol=1e-6)


class TestSystemIntegration:
    """Test complete system integration scenarios."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end machine learning workflow."""
        # Setup environment
        env = IntegrationTestEnvironment()
        
        try:
            # Step 1: Data generation
            data_success = env.setup_test_data()
            assert data_success, "Data generation failed"
            
            # Step 2: Pipeline testing
            pipeline_suite = PipelineTestSuite(env)
            pipeline_result = pipeline_suite.test_tfrecord_pipeline()
            assert pipeline_result['success'], "Pipeline testing failed"
            
            # Step 3: Model training
            model_suite = ModelTestSuite(env)
            training_result = model_suite.train_image_classifier()
            assert training_result['success'], "Model training failed"
            
            # Step 4: Model evaluation
            if 'cnn' in model_suite.models:
                model = model_suite.models['cnn']
                
                # Load test data
                val_dir = os.path.join(env.data_dir, "images", "val")
                X_val, y_val = model_suite._load_image_data(val_dir)
                X_val = X_val.astype('float32') / 255.0
                
                # Evaluate
                eval_results = model.evaluate(X_val, y_val, verbose=0)
                assert len(eval_results) == 2
                
            # Step 5: Verify all components worked
            assert env.test_results['data_generation']
            
            # Mark overall success
            env.test_results['model_training'] = True
            env.test_results['model_evaluation'] = True
            env.test_results['pipeline_execution'] = True
            
        finally:
            env.cleanup()
    
    def test_multi_model_workflow(self):
        """Test workflow with multiple model types."""
        env = IntegrationTestEnvironment()
        
        try:
            # Setup data
            env.setup_test_data()
            
            # Train multiple models
            model_suite = ModelTestSuite(env)
            
            results = {}
            results['cnn'] = model_suite.train_image_classifier()
            results['mlp'] = model_suite.train_tabular_classifier()
            results['rnn'] = model_suite.train_text_classifier()
            
            # Verify at least 2 models trained successfully
            successful_models = sum(1 for result in results.values() if result['success'])
            assert successful_models >= 2, f"Only {successful_models} models trained successfully"
            
            # Test model persistence and loading
            for model_type, result in results.items():
                if result['success']:
                    model_path = os.path.join(env.models_dir, f"{model_type}_model.h5")
                    assert os.path.exists(model_path)
                    
                    # Test loading
                    loaded_model = tf.keras.models.load_model(model_path)
                    assert loaded_model is not None
            
        finally:
            env.cleanup()
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks and resource usage."""
        env = IntegrationTestEnvironment()
        
        try:
            # Setup data
            env.setup_test_data()
            
            # Measure training time
            model_suite = ModelTestSuite(env)
            
            start_time = time.time()
            result = model_suite.train_tabular_classifier()
            training_time = time.time() - start_time
            
            if result['success']:
                # Training should complete in reasonable time (< 60 seconds for test)
                assert training_time < 60, f"Training took too long: {training_time:.2f} seconds"
                
                # Model should achieve reasonable performance
                assert result['final_accuracy'] > 0.3, f"Model accuracy too low: {result['final_accuracy']:.3f}"
            
        finally:
            env.cleanup()


# Utility functions for CI/CD integration
def run_integration_tests(test_types: List[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Run integration tests programmatically.
    
    Args:
        test_types: List of test types to run ('environment', 'models', 'pipelines', 'evaluation', 'system')
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with test results
    """
    if test_types is None:
        test_types = ['environment', 'models', 'pipelines', 'evaluation', 'system']
    
    results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'test_details': {},
        'overall_success': False
    }
    
    # Environment setup test
    if 'environment' in test_types:
        if verbose:
            print("🔧 Testing environment setup...")
        
        try:
            env = IntegrationTestEnvironment()
            env.setup_test_data()
            
            results['test_details']['environment'] = {
                'success': True,
                'data_generation': env.test_results['data_generation']
            }
            results['passed_tests'] += 1
            
            env.cleanup()
            
        except Exception as e:
            results['test_details']['environment'] = {
                'success': False,
                'error': str(e)
            }
            results['failed_tests'] += 1
        
        results['total_tests'] += 1
    
    # Model training tests
    if 'models' in test_types:
        if verbose:
            print("🧠 Testing model training...")
        
        try:
            env = IntegrationTestEnvironment()
            env.setup_test_data()
            model_suite = ModelTestSuite(env)
            
            # Test one model type
            cnn_result = model_suite.train_image_classifier()
            
            results['test_details']['models'] = {
                'success': cnn_result['success'],
                'cnn_accuracy': cnn_result.get('final_accuracy', 0)
            }
            
            if cnn_result['success']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1
            
            env.cleanup()
            
        except Exception as e:
            results['test_details']['models'] = {
                'success': False,
                'error': str(e)
            }
            results['failed_tests'] += 1
        
        results['total_tests'] += 1
    
    # Calculate overall success
    results['overall_success'] = results['failed_tests'] == 0
    
    return results


if __name__ == "__main__":
    """Run integration tests when executed as script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TensorVerseHub integration tests")
    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=['environment', 'models', 'pipelines', 'evaluation', 'system'],
        default=['environment', 'models'],
        help="Types of tests to run"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting TensorVerseHub Integration Tests")
    print(f"📋 Running tests: {', '.join(args.test_types)}")
    
    results = run_integration_tests(args.test_types, args.verbose)
    
    print(f"\n📊 Test Results:")
    print(f"  Total: {results['total_tests']}")
    print(f"  Passed: {results['passed_tests']}")
    print(f"  Failed: {results['failed_tests']}")
    
    for test_type, details in results['test_details'].items():
        status = "✅ PASS" if details['success'] else "❌ FAIL"
        print(f"  {test_type}: {status}")
        
        if not details['success'] and 'error' in details:
            print(f"    Error: {details['error']}")
    
    if results['overall_success']:
        print("\n🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some integration tests failed!")
        sys.exit(1)