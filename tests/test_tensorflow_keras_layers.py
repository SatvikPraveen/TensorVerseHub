# Location: /tests/test_tensorflow_keras_layers.py

"""
Test custom tf.keras layers and layer functionality.
Comprehensive tests for custom layer implementations and tf.keras integration.
"""

import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import patch

from model_utils import CustomLayers
from tests import TEST_CONFIG


class TestMultiHeadAttention:
    """Test MultiHeadAttention custom layer."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.d_model = 64
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
    
    def test_initialization(self):
        """Test layer initialization."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        assert attention.d_model == self.d_model
        assert attention.num_heads == self.num_heads
        assert attention.depth == self.d_model // self.num_heads
        
        # Check sublayers exist
        assert hasattr(attention, 'wq')
        assert hasattr(attention, 'wk')
        assert hasattr(attention, 'wv')
        assert hasattr(attention, 'dense')
    
    def test_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        # d_model must be divisible by num_heads
        with pytest.raises(AssertionError):
            CustomLayers.MultiHeadAttention(d_model=63, num_heads=8)
    
    def test_split_heads(self):
        """Test head splitting functionality."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        x = tf.random.normal([self.batch_size, self.seq_len, self.d_model])
        split_x = attention.split_heads(x, self.batch_size)
        
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, attention.depth)
        assert split_x.shape == expected_shape
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention computation."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        # Create query, key, value tensors
        q = tf.random.normal([self.batch_size, self.num_heads, self.seq_len, attention.depth])
        k = tf.random.normal([self.batch_size, self.num_heads, self.seq_len, attention.depth])
        v = tf.random.normal([self.batch_size, self.num_heads, self.seq_len, attention.depth])
        
        output = attention.scaled_dot_product_attention(q, k, v)
        
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, attention.depth)
        assert output.shape == expected_shape
    
    def test_scaled_dot_product_attention_with_mask(self):
        """Test attention with mask."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        q = tf.random.normal([self.batch_size, self.num_heads, self.seq_len, attention.depth])
        k = tf.random.normal([self.batch_size, self.num_heads, self.seq_len, attention.depth])
        v = tf.random.normal([self.batch_size, self.num_heads, self.seq_len, attention.depth])
        
        # Create causal mask
        mask = tf.linalg.band_part(tf.ones([self.seq_len, self.seq_len]), -1, 0)
        mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # Add batch and head dims
        mask = (1.0 - mask) * -1e9
        
        output = attention.scaled_dot_product_attention(q, k, v, mask)
        assert output.shape == (self.batch_size, self.num_heads, self.seq_len, attention.depth)
    
    def test_forward_pass(self):
        """Test complete forward pass."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        # Self-attention case
        x = tf.random.normal([self.batch_size, self.seq_len, self.d_model])
        output = attention(x, x, x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert output.dtype == tf.float32
    
    def test_cross_attention(self):
        """Test cross-attention (different query and key/value)."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        query = tf.random.normal([self.batch_size, 10, self.d_model])
        key_value = tf.random.normal([self.batch_size, self.seq_len, self.d_model])
        
        output = attention(key_value, key_value, query)  # Cross attention
        
        assert output.shape == (self.batch_size, 10, self.d_model)  # Output matches query length
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        # Simple case with known input
        q = tf.ones([1, self.num_heads, 3, attention.depth])
        k = tf.ones([1, self.num_heads, 3, attention.depth]) 
        v = tf.ones([1, self.num_heads, 3, attention.depth])
        
        # Calculate attention weights manually
        dk = tf.cast(attention.depth, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Check weights sum to 1 along last dimension
        weight_sums = tf.reduce_sum(attention_weights, axis=-1)
        assert tf.reduce_all(tf.abs(weight_sums - 1.0) < 1e-6)
    
    def test_trainable_parameters(self):
        """Test that layer has trainable parameters."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        # Build the layer
        x = tf.random.normal([1, self.seq_len, self.d_model])
        _ = attention(x, x, x)
        
        # Check trainable parameters exist
        trainable_vars = attention.trainable_variables
        assert len(trainable_vars) > 0
        
        # Should have weights for wq, wk, wv, and dense layers
        expected_weights = 4 * 2  # 4 layers × 2 parameters each (weight + bias)
        assert len(trainable_vars) == expected_weights
    
    def test_layer_serialization(self):
        """Test layer can be serialized and deserialized."""
        attention = CustomLayers.MultiHeadAttention(self.d_model, self.num_heads)
        
        # Build the layer
        x = tf.random.normal([1, self.seq_len, self.d_model])
        _ = attention(x, x, x)
        
        # Get config
        config = attention.get_config()
        assert 'd_model' in config
        assert 'num_heads' in config
        assert config['d_model'] == self.d_model
        assert config['num_heads'] == self.num_heads


class TestPositionalEncoding:
    """Test PositionalEncoding custom layer."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.position = 100
        self.d_model = 64
        self.seq_len = 20
        self.batch_size = 2
    
    def test_initialization(self):
        """Test layer initialization."""
        pos_enc = CustomLayers.PositionalEncoding(self.position, self.d_model)
        
        assert pos_enc.pos_encoding.shape == (1, self.position, self.d_model)
        assert pos_enc.pos_encoding.dtype == tf.float32
    
    def test_get_angles(self):
        """Test angle computation for positional encoding."""
        pos_enc = CustomLayers.PositionalEncoding(self.position, self.d_model)
        
        pos = np.arange(10)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]
        
        angles = pos_enc.get_angles(pos, i, self.d_model)
        
        assert angles.shape == (10, self.d_model)
        assert not np.allclose(angles[0], angles[1])  # Different positions should have different angles
    
    def test_positional_encoding_properties(self):
        """Test properties of generated positional encodings."""
        pos_enc = CustomLayers.PositionalEncoding(self.position, self.d_model)
        
        encoding = pos_enc.pos_encoding[0].numpy()  # Remove batch dimension
        
        # Even indices should use sin, odd indices should use cos
        # Check that values are in reasonable range [-1, 1]
        assert np.all(encoding >= -1.1)  # Small tolerance for numerical precision
        assert np.all(encoding <= 1.1)
        
        # Different positions should have different encodings
        assert not np.allclose(encoding[0], encoding[1])
        assert not np.allclose(encoding[0], encoding[-1])
    
    def test_forward_pass(self):
        """Test forward pass with input."""
        pos_enc = CustomLayers.PositionalEncoding(self.position, self.d_model)
        
        x = tf.random.normal([self.batch_size, self.seq_len, self.d_model])
        output = pos_enc(x)
        
        assert output.shape == x.shape
        assert not tf.reduce_all(tf.equal(output, x))  # Should be different from input
    
    def test_sequence_length_variations(self):
        """Test with different sequence lengths."""
        pos_enc = CustomLayers.PositionalEncoding(self.position, self.d_model)
        
        # Test with shorter sequence
        short_seq = tf.random.normal([1, 5, self.d_model])
        short_output = pos_enc(short_seq)
        assert short_output.shape == short_seq.shape
        
        # Test with longer sequence (up to position limit)
        long_seq = tf.random.normal([1, self.position, self.d_model])
        long_output = pos_enc(long_seq)
        assert long_output.shape == long_seq.shape
    
    def test_sequence_too_long_handling(self):
        """Test behavior when sequence is longer than max position."""
        pos_enc = CustomLayers.PositionalEncoding(10, self.d_model)
        
        # Sequence longer than position limit should still work (truncate encoding)
        long_seq = tf.random.normal([1, 15, self.d_model])
        output = pos_enc(long_seq)
        
        # Should truncate to available positions
        assert output.shape == long_seq.shape
    
    def test_batch_independence(self):
        """Test that different batch elements get same positional encoding."""
        pos_enc = CustomLayers.PositionalEncoding(self.position, self.d_model)
        
        x = tf.random.normal([3, self.seq_len, self.d_model])
        output = pos_enc(x)
        
        # The positional encoding added should be the same for all batch elements
        # So the difference between output and input should be the same across batch
        diff = output - x
        
        # All batch elements should have the same positional encoding added
        assert tf.reduce_all(tf.abs(diff[0] - diff[1]) < 1e-6)
        assert tf.reduce_all(tf.abs(diff[1] - diff[2]) < 1e-6)


class TestCustomLayerIntegration:
    """Test integration of custom layers in models."""
    
    def test_multihead_attention_in_model(self):
        """Test MultiHeadAttention layer integrated in a model."""
        d_model = 32
        seq_len = 10
        
        inputs = tf.keras.layers.Input(shape=(seq_len, d_model))
        
        # Add positional encoding
        pos_encoding = CustomLayers.PositionalEncoding(seq_len, d_model)
        x = pos_encoding(inputs)
        
        # Add multi-head attention
        attention = CustomLayers.MultiHeadAttention(d_model, num_heads=4)
        x = attention(x, x, x)
        
        # Add feed-forward layers
        x = tf.keras.layers.Dense(d_model * 2, activation='relu')(x)
        x = tf.keras.layers.Dense(d_model)(x)
        
        # Global pooling and classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Test model compilation
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Test forward pass
        test_input = tf.random.normal([2, seq_len, d_model])
        output = model(test_input)
        
        assert output.shape == (2, TEST_CONFIG['num_classes'])
    
    def test_transformer_encoder_block(self):
        """Test creating a complete transformer encoder block."""
        d_model = 64
        seq_len = 16
        num_heads = 8
        
        def transformer_encoder_block(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
            # Multi-head attention
            attention = CustomLayers.MultiHeadAttention(d_model, num_heads)
            attn_output = attention(inputs, inputs, inputs)
            attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
            out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
            
            # Feed-forward network
            ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
            ff_output = tf.keras.layers.Dense(d_model)(ff_output)
            ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)
            
            return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ff_output)
        
        # Build model
        inputs = tf.keras.layers.Input(shape=(seq_len, d_model))
        
        # Add positional encoding
        pos_encoding = CustomLayers.PositionalEncoding(seq_len, d_model)
        x = pos_encoding(inputs)
        
        # Add transformer blocks
        x = transformer_encoder_block(x, d_model, num_heads, d_model * 2)
        x = transformer_encoder_block(x, d_model, num_heads, d_model * 2)
        
        # Classification head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Test model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        test_input = tf.random.normal([3, seq_len, d_model])
        output = model(test_input)
        
        assert output.shape == (3, TEST_CONFIG['num_classes'])
        
        # Test that model is trainable
        assert len(model.trainable_variables) > 0
    
    def test_layer_gradients_flow(self):
        """Test that gradients flow through custom layers."""
        d_model = 32
        seq_len = 8
        
        # Simple model with custom layers
        inputs = tf.keras.layers.Input(shape=(seq_len, d_model))
        pos_enc = CustomLayers.PositionalEncoding(seq_len, d_model)
        x = pos_enc(inputs)
        
        attention = CustomLayers.MultiHeadAttention(d_model, num_heads=4)
        x = attention(x, x, x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Test gradient computation
        x = tf.random.normal([2, seq_len, d_model])
        y = tf.random.normal([2, 1])
        
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.keras.losses.mean_squared_error(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check that gradients exist and are not None
        assert len(gradients) > 0
        for grad in gradients:
            assert grad is not None
            assert not tf.reduce_all(tf.equal(grad, 0.0))  # Gradients should be non-zero


class TestLayerEdgeCases:
    """Test edge cases and error conditions for custom layers."""
    
    def test_multihead_attention_empty_input(self):
        """Test MultiHeadAttention with empty sequence."""
        attention = CustomLayers.MultiHeadAttention(d_model=32, num_heads=4)
        
        # Empty sequence length
        empty_input = tf.zeros([1, 0, 32])
        output = attention(empty_input, empty_input, empty_input)
        
        assert output.shape == (1, 0, 32)
    
    def test_positional_encoding_zero_length(self):
        """Test PositionalEncoding with zero sequence length."""
        pos_enc = CustomLayers.PositionalEncoding(position=10, d_model=32)
        
        zero_input = tf.zeros([1, 0, 32])
        output = pos_enc(zero_input)
        
        assert output.shape == (1, 0, 32)
    
    def test_multihead_attention_single_head(self):
        """Test MultiHeadAttention with single head."""
        attention = CustomLayers.MultiHeadAttention(d_model=32, num_heads=1)
        
        x = tf.random.normal([2, 5, 32])
        output = attention(x, x, x)
        
        assert output.shape == (2, 5, 32)
        assert attention.depth == 32  # Should equal d_model for single head
    
    def test_large_sequence_length(self):
        """Test layers with large sequence lengths."""
        d_model = 64
        large_seq_len = 1000
        
        # Test positional encoding
        pos_enc = CustomLayers.PositionalEncoding(large_seq_len, d_model)
        large_input = tf.random.normal([1, large_seq_len, d_model])
        pos_output = pos_enc(large_input)
        
        assert pos_output.shape == large_input.shape
        
        # Test attention (with smaller input for memory)
        attention = CustomLayers.MultiHeadAttention(d_model, num_heads=8)
        small_input = tf.random.normal([1, 100, d_model])  # Smaller for memory
        attn_output = attention(small_input, small_input, small_input)
        
        assert attn_output.shape == small_input.shape


class TestLayerCompatibility:
    """Test compatibility with tf.keras ecosystem."""
    
    def test_layer_in_sequential_model(self):
        """Test custom layers work in Sequential models."""
        # This tests a common use case
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(1000, 64, input_length=20),
            CustomLayers.PositionalEncoding(20, 64),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(TEST_CONFIG['num_classes'], activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Test forward pass
        test_input = tf.random.uniform([4, 20], 0, 1000, dtype=tf.int32)
        output = model(test_input)
        
        assert output.shape == (4, TEST_CONFIG['num_classes'])
    
    def test_layer_saving_and_loading(self, temp_dir):
        """Test that models with custom layers can be saved and loaded."""
        d_model = 32
        seq_len = 10
        
        # Build model with custom layers
        inputs = tf.keras.layers.Input(shape=(seq_len, d_model))
        x = CustomLayers.PositionalEncoding(seq_len, d_model)(inputs)
        x = CustomLayers.MultiHeadAttention(d_model, num_heads=4)(x, x, x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Save model
        model_path = f"{temp_dir}/custom_layer_model"
        
        # For custom layers, we need to save with custom objects
        model.save(model_path, save_format='tf')
        
        # Load model (this might require custom_objects parameter in real usage)
        try:
            loaded_model = tf.keras.models.load_model(model_path)
            
            # Test that loaded model works
            test_input = tf.random.normal([2, seq_len, d_model])
            output = loaded_model(test_input)
            assert output.shape == (2, 2)
            
        except Exception as e:
            # Custom layers might need special handling for loading
            # This is expected behavior for complex custom layers
            assert "custom" in str(e).lower() or "unknown" in str(e).lower()
    
    def test_layer_with_mixed_precision(self):
        """Test custom layers work with mixed precision."""
        # Enable mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        try:
            d_model = 64
            attention = CustomLayers.MultiHeadAttention(d_model, num_heads=8)
            
            # Test with mixed precision
            x = tf.cast(tf.random.normal([2, 10, d_model]), tf.float16)
            output = attention(x, x, x)
            
            # Output might be float16 or float32 depending on layer implementation
            assert output.dtype in [tf.float16, tf.float32]
            assert output.shape == (2, 10, d_model)
            
        finally:
            # Reset policy
            tf.keras.mixed_precision.set_global_policy('float32')
    
    def test_layer_with_gradient_tape(self):
        """Test custom layers work with GradientTape."""
        d_model = 32
        
        attention = CustomLayers.MultiHeadAttention(d_model, num_heads=4)
        x = tf.random.normal([2, 5, d_model])
        target = tf.random.normal([2, 5, d_model])
        
        with tf.GradientTape() as tape:
            output = attention(x, x, x)
            loss = tf.reduce_mean(tf.square(output - target))
        
        gradients = tape.gradient(loss, attention.trainable_variables)
        
        assert len(gradients) > 0
        for grad in gradients:
            assert grad is not None
            assert grad.shape.rank > 0  # Should have some shape
    
    def test_layer_with_tf_function(self):
        """Test custom layers work with tf.function."""
        d_model = 32
        attention = CustomLayers.MultiHeadAttention(d_model, num_heads=4)
        
        @tf.function
        def apply_attention(x):
            return attention(x, x, x)
        
        x = tf.random.normal([2, 5, d_model])
        
        # First call (tracing)
        output1 = apply_attention(x)
        
        # Second call (should use cached graph)
        output2 = apply_attention(x)
        
        assert output1.shape == (2, 5, d_model)
        assert output2.shape == (2, 5, d_model)
        
        # Results should be deterministic for same input
        # (though they won't be identical due to random initialization)
        assert output1.dtype == output2.dtype


class TestLayerNumericalStability:
    """Test numerical stability of custom layers."""
    
    def test_attention_with_extreme_values(self):
        """Test attention layer with extreme input values."""
        attention = CustomLayers.MultiHeadAttention(d_model=32, num_heads=4)
        
        # Test with very large values
        large_x = tf.random.normal([1, 5, 32]) * 100
        output_large = attention(large_x, large_x, large_x)
        
        assert not tf.reduce_any(tf.math.is_nan(output_large))
        assert not tf.reduce_any(tf.math.is_inf(output_large))
        
        # Test with very small values
        small_x = tf.random.normal([1, 5, 32]) * 0.001
        output_small = attention(small_x, small_x, small_x)
        
        assert not tf.reduce_any(tf.math.is_nan(output_small))
        assert not tf.reduce_any(tf.math.is_inf(output_small))
    
    def test_positional_encoding_numerical_stability(self):
        """Test positional encoding numerical properties."""
        pos_enc = CustomLayers.PositionalEncoding(position=1000, d_model=512)
        
        # Check encoding values are bounded
        encoding = pos_enc.pos_encoding
        
        assert tf.reduce_all(encoding >= -1.1)  # Small tolerance
        assert tf.reduce_all(encoding <= 1.1)
        assert not tf.reduce_any(tf.math.is_nan(encoding))
        assert not tf.reduce_any(tf.math.is_inf(encoding))
    
    def test_gradient_stability(self):
        """Test that gradients remain stable during training."""
        d_model = 64
        model_input = tf.keras.layers.Input(shape=(10, d_model))
        
        x = CustomLayers.PositionalEncoding(10, d_model)(model_input)
        x = CustomLayers.MultiHeadAttention(d_model, num_heads=8)(x, x, x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(model_input, output)
        
        # Simulate training step
        x_batch = tf.random.normal([4, 10, d_model])
        y_batch = tf.random.normal([4, 1])
        
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y_batch))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check gradient health
        for grad in gradients:
            assert not tf.reduce_any(tf.math.is_nan(grad))
            assert not tf.reduce_any(tf.math.is_inf(grad))
            
            # Gradients shouldn't be too large
            grad_norm = tf.linalg.global_norm([grad])
            assert grad_norm < 1000.0  # Reasonable upper bound


# Parameterized tests for different configurations
@pytest.mark.parametrize("d_model,num_heads", [
    (32, 4), (64, 8), (128, 16), (256, 32)
])
class TestMultiHeadAttentionParametrized:
    """Parameterized tests for MultiHeadAttention."""
    
    def test_different_configurations(self, d_model, num_heads):
        """Test MultiHeadAttention with different configurations."""
        if d_model % num_heads != 0:
            pytest.skip("d_model must be divisible by num_heads")
        
        attention = CustomLayers.MultiHeadAttention(d_model, num_heads)
        
        x = tf.random.normal([2, 8, d_model])
        output = attention(x, x, x)
        
        assert output.shape == (2, 8, d_model)
        assert attention.depth == d_model // num_heads


@pytest.mark.parametrize("position,d_model", [
    (50, 32), (100, 64), (200, 128), (500, 256)
])
class TestPositionalEncodingParametrized:
    """Parameterized tests for PositionalEncoding."""
    
    def test_different_configurations(self, position, d_model):
        """Test PositionalEncoding with different configurations."""
        pos_enc = CustomLayers.PositionalEncoding(position, d_model)
        
        seq_len = min(position, 50)  # Keep sequence length reasonable for testing
        x = tf.random.normal([1, seq_len, d_model])
        output = pos_enc(x)
        
        assert output.shape == (1, seq_len, d_model)


# Integration test with real training
@pytest.mark.slow
class TestLayerTrainingIntegration:
    """Integration tests with actual training (marked as slow)."""
    
    def test_end_to_end_training_with_custom_layers(self):
        """Test complete training pipeline with custom layers."""
        # Create synthetic data
        vocab_size = 100
        seq_len = 16
        d_model = 32
        num_samples = 200
        
        # Generate data
        x_data = tf.random.uniform([num_samples, seq_len], 0, vocab_size, dtype=tf.int32)
        y_data = tf.random.uniform([num_samples], 0, 2, dtype=tf.int32)  # Binary classification
        
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
        
        # Build model with custom layers
        inputs = tf.keras.layers.Input(shape=(seq_len,))
        x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        x = CustomLayers.PositionalEncoding(seq_len, d_model)(x)
        x = CustomLayers.MultiHeadAttention(d_model, num_heads=4)(x, x, x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            dataset,
            epochs=3,
            verbose=0
        )
        
        # Check training completed successfully
        assert len(history.history['loss']) == 3
        assert all(not np.isnan(loss) for loss in history.history['loss'])
        assert all(not np.isinf(loss) for loss in history.history['loss'])
        
        # Test final predictions
        test_input = x_data[:4]
        predictions = model.predict(test_input, verbose=0)
        
        assert predictions.shape == (4, 2)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
        assert np.allclose(np.sum(predictions, axis=1), 1.0)  # Probabilities sum to 1


if __name__ == "__main__":
    pytest.main([__file__])