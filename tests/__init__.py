# Location: /tests/__init__.py

"""
TensorVerseHub Test Suite
Comprehensive testing framework for TensorFlow 2.15+ with tf.keras integration.
"""

import os
import sys
import pytest
import tensorflow as tf

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration
TEST_CONFIG = {
    'tensorflow_version': tf.__version__,
    'test_data_dir': os.path.join(os.path.dirname(__file__), 'test_data'),
    'temp_dir': os.path.join(os.path.dirname(__file__), 'temp'),
    'batch_size': 8,  # Small batch size for testing
    'num_samples': 100,  # Small dataset for testing
    'image_shape': (32, 32, 3),  # Small images for faster testing
    'num_classes': 5,
    'epochs': 2,  # Few epochs for testing
    'verbose': 0  # Quiet mode for tests
}

# Setup test environment
def setup_test_environment():
    """Setup TensorFlow for testing."""
    # Set memory growth for GPU testing
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Memory growth must be set before GPUs have been initialized
    
    # Set random seeds for reproducible tests
    tf.random.set_seed(42)
    
    # Create test directories
    os.makedirs(TEST_CONFIG['test_data_dir'], exist_ok=True)
    os.makedirs(TEST_CONFIG['temp_dir'], exist_ok=True)

# Common test fixtures
@pytest.fixture
def sample_image_data():
    """Generate sample image data for testing."""
    images = tf.random.normal([TEST_CONFIG['num_samples']] + list(TEST_CONFIG['image_shape']))
    labels = tf.random.uniform([TEST_CONFIG['num_samples']], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
    return images, labels

@pytest.fixture
def sample_text_data():
    """Generate sample text data for testing."""
    vocab_size = 1000
    seq_length = 32
    
    sequences = tf.random.uniform([TEST_CONFIG['num_samples'], seq_length], 0, vocab_size, dtype=tf.int32)
    labels = tf.random.uniform([TEST_CONFIG['num_samples']], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
    return sequences, labels

@pytest.fixture
def sample_dataset():
    """Create a sample tf.data.Dataset for testing."""
    images = tf.random.normal([TEST_CONFIG['num_samples']] + list(TEST_CONFIG['image_shape']))
    labels = tf.random.uniform([TEST_CONFIG['num_samples']], 0, TEST_CONFIG['num_classes'], dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(TEST_CONFIG['batch_size'])
    return dataset

@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    import tempfile
    import shutil
    
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

# Setup test environment on import
setup_test_environment()

# Export test utilities
__all__ = [
    'TEST_CONFIG',
    'sample_image_data',
    'sample_text_data', 
    'sample_dataset',
    'temp_dir',
    'setup_test_environment'
]