# Location: /tests/test_model_utils.py

"""
Test tf.keras model creation and training utilities.
Comprehensive tests for model_utils module functionality.
"""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from model_utils import (
    ModelBuilders,
    CustomLayers, 
    TrainingUtilities,
    ModelAnalysis,
    create_classification_model,
    create_transfer_learning_model,
    save_model_with_metadata,
    load_model_with_metadata
)
from tests import TEST_CONFIG


class TestCustomLayers:
    """Test custom tf.keras layers."""
    
    def test_multi_head_attention(self):
        """Test MultiHeadAttention layer."""
        d_model = 64
        num_heads = 8
        seq_len = 10
        batch_size = 2
        
        attention_layer = CustomLayers.MultiHeadAttention(d_model, num_heads)
        
        # Test input
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        
        # Apply attention (self-attention)
        output = attention_layer(inputs, inputs, inputs)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert isinstance(output, tf.Tensor)
    
    def test_multi_head_attention_with_mask(self):
        """Test MultiHeadAttention with mask."""
        d_model = 32
        num_heads = 4
        seq_len = 5
        batch_size = 1
        
        attention_layer = CustomLayers.MultiHeadAttention(d_model, num_heads)
        
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        mask = tf.random.uniform([batch_size, num_heads, seq_len, seq_len]) > 0.5
        
        output = attention_layer(inputs, inputs, inputs, mask=mask)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding(self):
        """Test PositionalEncoding layer."""
        position = 50
        d_model = 64
        batch_size = 2
        seq_len = 20
        
        pos_encoding = CustomLayers.PositionalEncoding(position, d_model)
        
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        output = pos_encoding(inputs)
        
        assert output.shape == inputs.shape
        assert not tf.reduce_all(tf.equal(output, inputs))  # Should be different


class TestModelBuilders:
    """Test model building utilities."""
    
    def test_create_cnn_classifier_simple(self):
        """Test simple CNN classifier creation."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='simple'
        )
        
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None,) + TEST_CONFIG['image_shape']
        assert model.output_shape == (None, TEST_CONFIG['num_classes'])
        
        # Test model can make predictions
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        output = model(test_input)
        assert output.shape == (1, TEST_CONFIG['num_classes'])
    
    def test_create_cnn_classifier_vgg(self):
        """Test VGG-like CNN classifier."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='vgg'
        )
        
        assert isinstance(model, tf.keras.Model)
        assert model.count_params() > 10000  # VGG should have many parameters
        
        # Test forward pass
        test_input = tf.random.normal([2] + list(TEST_CONFIG['image_shape']))
        output = model(test_input)
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    def test_create_cnn_classifier_resnet(self):
        """Test ResNet-like CNN classifier."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='resnet'
        )
        
        assert isinstance(model, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        output = model(test_input)
        assert output.shape == (1, TEST_CONFIG['num_classes'])
    
    def test_create_text_classifier_lstm(self):
        """Test LSTM text classifier creation."""
        vocab_size = 1000
        embedding_dim = 64
        max_length = 32
        
        model = ModelBuilders.create_text_classifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length,
            num_classes=TEST_CONFIG['num_classes'],
            architecture='lstm'
        )
        
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, max_length)
        assert model.output_shape == (None, TEST_CONFIG['num_classes'])
        
        # Test forward pass
        test_input = tf.random.uniform([2, max_length], 0, vocab_size, dtype=tf.int32)
        output = model(test_input)
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    def test_create_text_classifier_gru(self):
        """Test GRU text classifier creation."""
        model = ModelBuilders.create_text_classifier(
            vocab_size=500,
            embedding_dim=32,
            max_length=16,
            num_classes=3,
            architecture='gru'
        )
        
        assert isinstance(model, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.uniform([1, 16], 0, 500, dtype=tf.int32)
        output = model(test_input)
        assert output.shape == (1, 3)
    
    def test_create_text_classifier_transformer(self):
        """Test transformer text classifier creation."""
        model = ModelBuilders.create_text_classifier(
            vocab_size=1000,
            embedding_dim=64,
            max_length=32,
            num_classes=TEST_CONFIG['num_classes'],
            architecture='transformer'
        )
        
        assert isinstance(model, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.uniform([2, 32], 0, 1000, dtype=tf.int32)
        output = model(test_input)
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    def test_create_autoencoder_dense(self):
        """Test dense autoencoder creation."""
        input_shape = (784,)  # Flattened 28x28 image
        encoding_dim = 64
        
        autoencoder, encoder, decoder = ModelBuilders.create_autoencoder(
            input_shape=input_shape,
            encoding_dim=encoding_dim,
            architecture='dense'
        )
        
        assert isinstance(autoencoder, tf.keras.Model)
        assert isinstance(encoder, tf.keras.Model)
        assert isinstance(decoder, tf.keras.Model)
        
        # Test shapes
        test_input = tf.random.normal([2] + list(input_shape))
        
        encoded = encoder(test_input)
        assert encoded.shape == (2, encoding_dim)
        
        decoded = decoder(encoded)
        assert decoded.shape == test_input.shape
        
        reconstructed = autoencoder(test_input)
        assert reconstructed.shape == test_input.shape
    
    def test_create_autoencoder_conv(self):
        """Test convolutional autoencoder creation."""
        autoencoder, encoder, decoder = ModelBuilders.create_autoencoder(
            input_shape=TEST_CONFIG['image_shape'],
            encoding_dim=128,
            architecture='conv'
        )
        
        assert isinstance(autoencoder, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        reconstructed = autoencoder(test_input)
        assert reconstructed.shape == test_input.shape
    
    def test_create_gan(self):
        """Test GAN creation."""
        latent_dim = 100
        output_shape = (28, 28, 1)
        
        gan, generator, discriminator = ModelBuilders.create_gan(
            latent_dim=latent_dim,
            output_shape=output_shape,
            generator_architecture='dense',
            discriminator_architecture='dense'
        )
        
        assert isinstance(gan, tf.keras.Model)
        assert isinstance(generator, tf.keras.Model)
        assert isinstance(discriminator, tf.keras.Model)
        
        # Test generator
        noise = tf.random.normal([2, latent_dim])
        generated = generator(noise)
        assert generated.shape == (2,) + output_shape
        
        # Test discriminator
        validity = discriminator(generated)
        assert validity.shape == (2, 1)
        
        # Test GAN
        gan_output = gan(noise)
        assert gan_output.shape == (2, 1)


class TestTrainingUtilities:
    """Test training utilities and callbacks."""
    
    def test_create_callbacks(self, temp_dir):
        """Test callback creation."""
        callbacks = TrainingUtilities.create_callbacks(
            model_name="test_model",
            patience=5
        )
        
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0
        
        # Check callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'ModelCheckpoint' in callback_types
        assert 'EarlyStopping' in callback_types
    
    def test_custom_training_step(self):
        """Test custom training step creation."""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()
        
        train_step = TrainingUtilities.create_custom_training_step(
            model, loss_fn, optimizer
        )
        
        # Test training step
        x = tf.random.normal([4, 5])
        y = tf.random.normal([4, 1])
        
        result = train_step(x, y)
        
        assert 'loss' in result
        assert 'accuracy' in result
        assert isinstance(result['loss'], tf.Tensor)
    
    def test_custom_callback(self, sample_dataset):
        """Test custom callback functionality."""
        callback = TrainingUtilities.CustomCallback(
            validation_data=sample_dataset,
            log_freq=2
        )
        
        assert isinstance(callback, tf.keras.callbacks.Callback)
        assert callback.log_freq == 2
        assert callback.validation_data is not None
        
        # Test callback methods exist
        assert hasattr(callback, 'on_epoch_end')


class TestModelAnalysis:
    """Test model analysis utilities."""
    
    def test_analyze_model_architecture(self):
        """Test model architecture analysis."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(10,)),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(1)
        ])
        
        analysis = ModelAnalysis.analyze_model_architecture(model)
        
        assert isinstance(analysis, dict)
        required_keys = [
            'total_parameters', 'trainable_parameters', 
            'non_trainable_parameters', 'total_layers',
            'model_size_mb', 'layer_details'
        ]
        
        for key in required_keys:
            assert key in analysis
        
        assert analysis['total_parameters'] > 0
        assert len(analysis['layer_details']) == len(model.layers)
    
    def test_compute_model_flops(self):
        """Test FLOP computation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(10,)),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(1)
        ])
        
        flops = ModelAnalysis.compute_model_flops(model, (10,))
        
        assert isinstance(flops, int)
        assert flops > 0
    
    def test_create_model_summary_report(self, temp_dir):
        """Test model summary report creation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(20,)),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(10)
        ])
        
        report_path = os.path.join(temp_dir, "model_report.md")
        report = ModelAnalysis.create_model_summary_report(
            model, (20,), save_path=report_path
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Model Analysis Report" in report
        
        # Check file was saved
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            saved_report = f.read()
        
        assert saved_report == report


class TestConvenienceFunctions:
    """Test convenience functions for model creation."""
    
    def test_create_classification_model_cnn(self):
        """Test CNN classification model creation."""
        model = create_classification_model(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='cnn'
        )
        
        assert isinstance(model, tf.keras.Model)
        assert model.compiled_loss is not None  # Should be compiled
        assert model.compiled_metrics is not None
        
        # Test prediction
        test_input = tf.random.normal([1] + list(TEST_CONFIG['image_shape']))
        output = model(test_input)
        assert output.shape == (1, TEST_CONFIG['num_classes'])
    
    def test_create_classification_model_text(self):
        """Test text classification model creation."""
        model = create_classification_model(
            input_shape=(32,),  # Sequence length
            num_classes=TEST_CONFIG['num_classes'],
            architecture='lstm'
        )
        
        assert isinstance(model, tf.keras.Model)
        
        # Test prediction
        test_input = tf.random.uniform([2, 32], 0, 1000, dtype=tf.int32)
        output = model(test_input)
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    @patch('tensorflow.keras.applications.ResNet50')
    def test_create_transfer_learning_model(self, mock_resnet):
        """Test transfer learning model creation."""
        # Mock the base model
        mock_base_model = MagicMock()
        mock_base_model.input_shape = (None, 224, 224, 3)
        mock_base_model.trainable = True
        mock_base_model.layers = [MagicMock() for _ in range(50)]  # Simulate 50 layers
        mock_resnet.return_value = mock_base_model
        
        # Mock the model call
        def mock_call(inputs, training=False):
            return tf.random.normal([tf.shape(inputs)[0], 7, 7, 2048])
        mock_base_model.side_effect = mock_call
        
        model = create_transfer_learning_model(
            base_model_name='ResNet50',
            input_shape=(224, 224, 3),
            num_classes=TEST_CONFIG['num_classes'],
            fine_tune=True,
            fine_tune_at=40
        )
        
        assert isinstance(model, tf.keras.Model)
        mock_resnet.assert_called_once()
    
    def test_save_and_load_model_with_metadata(self, temp_dir):
        """Test model saving and loading with metadata."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(10,)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Metadata
        metadata = {
            'model_version': '1.0',
            'training_dataset': 'synthetic',
            'performance': {'accuracy': 0.95}
        }
        
        # Save model with metadata
        save_path = os.path.join(temp_dir, 'test_model')
        save_model_with_metadata(model, save_path, metadata)
        
        # Check files exist
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, 'metadata.json'))
        
        # Load model with metadata
        loaded_model, loaded_metadata = load_model_with_metadata(save_path)
        
        assert isinstance(loaded_model, tf.keras.Model)
        assert loaded_metadata['model_version'] == '1.0'
        assert loaded_metadata['performance']['accuracy'] == 0.95


class TestModelIntegration:
    """Integration tests for model utilities."""
    
    def test_end_to_end_model_training(self, sample_dataset):
        """Test complete model training pipeline."""
        # Create model
        model = ModelBuilders.create_cnn_classifier(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='simple'
        )
        
        # Create callbacks
        callbacks = TrainingUtilities.create_callbacks(
            model_name="integration_test",
            patience=2,
            tensorboard=False  # Disable for testing
        )
        
        # Train model
        history = model.fit(
            sample_dataset,
            epochs=TEST_CONFIG['epochs'],
            callbacks=callbacks,
            verbose=0
        )
        
        assert hasattr(history, 'history')
        assert 'loss' in history.history
        assert len(history.history['loss']) <= TEST_CONFIG['epochs']
        
        # Analyze trained model
        analysis = ModelAnalysis.analyze_model_architecture(model)
        assert analysis['total_parameters'] > 0
    
    def test_model_compilation_variations(self):
        """Test different model compilation options."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='simple'
        )
        
        # Test different optimizers
        optimizers = ['adam', 'sgd', 'rmsprop']
        losses = ['sparse_categorical_crossentropy', 'categorical_crossentropy']
        metrics = [['accuracy'], ['accuracy', 'top_k_categorical_accuracy']]
        
        for opt in optimizers[:1]:  # Test only first one for speed
            for loss in losses[:1]:
                for metric in metrics[:1]:
                    model.compile(optimizer=opt, loss=loss, metrics=metric)
                    
                    # Verify compilation
                    assert model.optimizer is not None
                    assert model.compiled_loss is not None
    
    def test_model_saving_formats(self, temp_dir):
        """Test different model saving formats."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, input_shape=(8,)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Test SavedModel format
        savedmodel_path = os.path.join(temp_dir, 'test_savedmodel')
        model.save(savedmodel_path, save_format='tf')
        assert os.path.exists(savedmodel_path)
        
        loaded_model = tf.keras.models.load_model(savedmodel_path)
        assert isinstance(loaded_model, tf.keras.Model)
        
        # Test H5 format
        h5_path = os.path.join(temp_dir, 'test_model.h5')
        model.save(h5_path, save_format='h5')
        assert os.path.exists(h5_path)
        
        loaded_h5_model = tf.keras.models.load_model(h5_path)
        assert isinstance(loaded_h5_model, tf.keras.Model)


# Performance tests
@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for model operations."""
    
    def test_large_model_creation(self):
        """Test creation of larger models."""
        # Create a larger model
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(224, 224, 3),
            num_classes=1000,  # ImageNet-like
            architecture='resnet'
        )
        
        assert model.count_params() > 100000  # Should be substantial
        
        # Test forward pass with larger input
        test_input = tf.random.normal([4, 224, 224, 3])
        output = model(test_input)
        assert output.shape == (4, 1000)
    
    def test_model_training_performance(self, sample_dataset):
        """Test training performance with timing."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=TEST_CONFIG['image_shape'],
            num_classes=TEST_CONFIG['num_classes'],
            architecture='simple'
        )
        
        import time
        start_time = time.time()
        
        # Train for a few steps
        history = model.fit(
            sample_dataset,
            epochs=2,
            verbose=0
        )
        
        elapsed = time.time() - start_time
        
        # Should complete reasonably quickly
        assert elapsed < 30.0  # Adjust threshold as needed
        assert len(history.history['loss']) == 2
        
        print(f"Training completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    pytest.main([__file__])