# Location: /src/model_utils.py

"""
TensorFlow model utilities with tf.keras integration.
Provides model builders, custom layers, training utilities, and model analysis tools.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable, Union, Any
import json
import os


class CustomLayers:
    """Custom tf.keras layers for advanced architectures."""
    
    class MultiHeadAttention(tf.keras.layers.Layer):
        """Multi-head attention implementation."""
        
        def __init__(self, d_model: int, num_heads: int, **kwargs):
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.d_model = d_model
            
            assert d_model % self.num_heads == 0
            
            self.depth = d_model // self.num_heads
            
            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)
            
            self.dense = tf.keras.layers.Dense(d_model)
        
        def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
            """Split the last dimension into (num_heads, depth)."""
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        def call(self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, 
                 mask: Optional[tf.Tensor] = None) -> tf.Tensor:
            batch_size = tf.shape(q)[0]
            
            q = self.wq(q)
            k = self.wk(k)
            v = self.wv(v)
            
            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)
            
            # Scaled dot-product attention
            scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)
            
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            
            concat_attention = tf.reshape(scaled_attention, 
                                        (batch_size, -1, self.d_model))
            
            output = self.dense(concat_attention)
            return output
        
        def scaled_dot_product_attention(self, q: tf.Tensor, k: tf.Tensor, 
                                       v: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
            """Calculate the attention weights."""
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            
            # Scale matmul_qk
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            
            # Add the mask
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)
            
            # Softmax is normalized on the last axis
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            
            output = tf.matmul(attention_weights, v)
            return output
    
    class PositionalEncoding(tf.keras.layers.Layer):
        """Positional encoding for transformer models."""
        
        def __init__(self, position: int, d_model: int, **kwargs):
            super().__init__(**kwargs)
            self.pos_encoding = self.positional_encoding(position, d_model)
        
        def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates
        
        def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
            angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                       np.arange(d_model)[np.newaxis, :],
                                       d_model)
            
            # Apply sin to even indices
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            
            # Apply cos to odd indices
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
            pos_encoding = angle_rads[np.newaxis, ...]
            
            return tf.cast(pos_encoding, dtype=tf.float32)
        
        def call(self, x: tf.Tensor) -> tf.Tensor:
            seq_len = tf.shape(x)[1]
            return x + self.pos_encoding[:, :seq_len, :]


class ModelBuilders:
    """Pre-built model architectures using tf.keras."""
    
    @staticmethod
    def create_cnn_classifier(input_shape: Tuple[int, int, int],
                             num_classes: int,
                             architecture: str = 'simple',
                             dropout_rate: float = 0.5) -> tf.keras.Model:
        """
        Create a CNN classifier.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            architecture: Architecture type ('simple', 'vgg', 'resnet')
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled tf.keras.Model
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        if architecture == 'simple':
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
            
        elif architecture == 'vgg':
            # VGG-like architecture
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            
        elif architecture == 'resnet':
            # Simple ResNet-like architecture
            def residual_block(x, filters, kernel_size=3, stride=1):
                shortcut = x
                
                x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                
                x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                
                if stride != 1:
                    shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
                
                x = tf.keras.layers.Add()([x, shortcut])
                x = tf.keras.layers.ReLU()(x)
                return x
            
            x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
            
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            x = residual_block(x, 128, stride=2)
            x = residual_block(x, 128)
            x = residual_block(x, 256, stride=2)
            x = residual_block(x, 256)
        
        # Global average pooling and classifier
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    @staticmethod
    def create_text_classifier(vocab_size: int,
                              embedding_dim: int,
                              max_length: int,
                              num_classes: int,
                              architecture: str = 'lstm') -> tf.keras.Model:
        """
        Create a text classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            num_classes: Number of output classes
            architecture: Architecture type ('lstm', 'gru', 'transformer')
            
        Returns:
            tf.keras.Model
        """
        inputs = tf.keras.layers.Input(shape=(max_length,))
        
        # Embedding layer
        x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
        
        if architecture == 'lstm':
            x = tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5)(x)
            
        elif architecture == 'gru':
            x = tf.keras.layers.GRU(128, dropout=0.5, recurrent_dropout=0.5)(x)
            
        elif architecture == 'transformer':
            # Simple transformer encoder
            attention = CustomLayers.MultiHeadAttention(embedding_dim, 8)
            x = attention(x, x, x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    @staticmethod
    def create_autoencoder(input_shape: Tuple[int, ...],
                          encoding_dim: int,
                          architecture: str = 'dense') -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """
        Create an autoencoder with encoder and decoder components.
        
        Args:
            input_shape: Input data shape
            encoding_dim: Dimension of encoded representation
            architecture: Architecture type ('dense', 'conv')
            
        Returns:
            Tuple of (autoencoder, encoder, decoder)
        """
        # Encoder
        encoder_input = tf.keras.layers.Input(shape=input_shape)
        
        if architecture == 'dense':
            x = tf.keras.layers.Flatten()(encoder_input)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
            
        elif architecture == 'conv':
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Flatten()(x)
            encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
        
        encoder = tf.keras.Model(encoder_input, encoded)
        
        # Decoder
        decoder_input = tf.keras.layers.Input(shape=(encoding_dim,))
        
        if architecture == 'dense':
            x = tf.keras.layers.Dense(256, activation='relu')(decoder_input)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
            decoded = tf.keras.layers.Reshape(input_shape)(x)
            
        elif architecture == 'conv':
            # Calculate the shape after flattening in encoder
            temp_shape = (input_shape[0] // 4, input_shape[1] // 4, 32)
            x = tf.keras.layers.Dense(np.prod(temp_shape), activation='relu')(decoder_input)
            x = tf.keras.layers.Reshape(temp_shape)(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            decoded = tf.keras.layers.Conv2D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        
        decoder = tf.keras.Model(decoder_input, decoded)
        
        # Autoencoder
        autoencoder_input = tf.keras.layers.Input(shape=input_shape)
        encoded_repr = encoder(autoencoder_input)
        decoded_repr = decoder(encoded_repr)
        autoencoder = tf.keras.Model(autoencoder_input, decoded_repr)
        
        return autoencoder, encoder, decoder
    
    @staticmethod
    def create_gan(latent_dim: int,
                   output_shape: Tuple[int, int, int],
                   generator_architecture: str = 'dense',
                   discriminator_architecture: str = 'dense') -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """
        Create a GAN with generator and discriminator.
        
        Args:
            latent_dim: Dimension of latent space
            output_shape: Shape of generated images (height, width, channels)
            generator_architecture: Generator architecture ('dense', 'conv')
            discriminator_architecture: Discriminator architecture ('dense', 'conv')
            
        Returns:
            Tuple of (gan, generator, discriminator)
        """
        # Generator
        generator_input = tf.keras.layers.Input(shape=(latent_dim,))
        
        if generator_architecture == 'dense':
            x = tf.keras.layers.Dense(256, activation='relu')(generator_input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(np.prod(output_shape), activation='tanh')(x)
            generated = tf.keras.layers.Reshape(output_shape)(x)
            
        elif generator_architecture == 'conv':
            x = tf.keras.layers.Dense(7 * 7 * 256, activation='relu')(generator_input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Reshape((7, 7, 256))(x)
            
            x = tf.keras.layers.Conv2DTranspose(128, 5, strides=1, padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            x = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            generated = tf.keras.layers.Conv2DTranspose(output_shape[-1], 5, strides=2, padding='same', activation='tanh')(x)
        
        generator = tf.keras.Model(generator_input, generated)
        
        # Discriminator
        discriminator_input = tf.keras.layers.Input(shape=output_shape)
        
        if discriminator_architecture == 'dense':
            x = tf.keras.layers.Flatten()(discriminator_input)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            validity = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
        elif discriminator_architecture == 'conv':
            x = tf.keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='relu')(discriminator_input)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            x = tf.keras.layers.Conv2D(128, 5, strides=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            validity = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        discriminator = tf.keras.Model(discriminator_input, validity)
        
        # GAN
        discriminator.trainable = False
        gan_input = tf.keras.layers.Input(shape=(latent_dim,))
        generated_image = generator(gan_input)
        validity = discriminator(generated_image)
        gan = tf.keras.Model(gan_input, validity)
        
        return gan, generator, discriminator


class TrainingUtilities:
    """Advanced training utilities and callbacks."""
    
    @staticmethod
    def create_callbacks(model_name: str,
                        patience: int = 10,
                        reduce_lr: bool = True,
                        tensorboard: bool = True) -> List[tf.keras.callbacks.Callback]:
        """
        Create standard training callbacks.
        
        Args:
            model_name: Name for saving checkpoints and logs
            patience: Patience for early stopping
            reduce_lr: Whether to include learning rate reduction
            tensorboard: Whether to include TensorBoard logging
            
        Returns:
            List of configured callbacks
        """
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/checkpoints/{model_name}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate reduction
        if reduce_lr:
            lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(lr_callback)
        
        # TensorBoard logging
        if tensorboard:
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/tensorboard/{model_name}',
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callbacks.append(tb_callback)
        
        return callbacks
    
    @staticmethod
    def create_custom_training_step(model: tf.keras.Model,
                                   loss_fn: tf.keras.losses.Loss,
                                   optimizer: tf.keras.optimizers.Optimizer) -> Callable:
        """
        Create a custom training step function.
        
        Args:
            model: tf.keras model
            loss_fn: Loss function
            optimizer: Optimizer
            
        Returns:
            Custom training step function
        """
        @tf.function
        def train_step(x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = loss_fn(y, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Calculate metrics
            accuracy = tf.keras.metrics.categorical_accuracy(y, predictions)
            
            return {
                'loss': loss,
                'accuracy': tf.reduce_mean(accuracy)
            }
        
        return train_step
    
    class CustomCallback(tf.keras.callbacks.Callback):
        """Custom callback with advanced monitoring."""
        
        def __init__(self, validation_data: tf.data.Dataset, log_freq: int = 10):
            super().__init__()
            self.validation_data = validation_data
            self.log_freq = log_freq
            self.epoch_losses = []
            self.epoch_accuracies = []
        
        def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
            if logs:
                self.epoch_losses.append(logs.get('loss', 0))
                self.epoch_accuracies.append(logs.get('accuracy', 0))
            
            if epoch % self.log_freq == 0:
                print(f"Epoch {epoch}: Custom monitoring active")
                
                # Custom validation metrics
                val_loss = 0
                val_acc = 0
                num_batches = 0
                
                for x_val, y_val in self.validation_data:
                    predictions = self.model(x_val, training=False)
                    batch_loss = tf.keras.losses.sparse_categorical_crossentropy(y_val, predictions)
                    batch_acc = tf.keras.metrics.sparse_categorical_accuracy(y_val, predictions)
                    
                    val_loss += tf.reduce_mean(batch_loss)
                    val_acc += tf.reduce_mean(batch_acc)
                    num_batches += 1
                
                avg_val_loss = val_loss / num_batches
                avg_val_acc = val_acc / num_batches
                
                print(f"Custom validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")


class ModelAnalysis:
    """Model analysis and interpretation utilities."""
    
    @staticmethod
    def analyze_model_architecture(model: tf.keras.Model) -> Dict[str, Any]:
        """
        Analyze model architecture and return statistics.
        
        Args:
            model: tf.keras model to analyze
            
        Returns:
            Dictionary with model statistics
        """
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        layer_info = []
        for i, layer in enumerate(model.layers):
            layer_info.append({
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'output_shape': str(layer.output_shape),
                'params': layer.count_params()
            })
        
        analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'total_layers': len(model.layers),
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'layer_details': layer_info
        }
        
        return analysis
    
    @staticmethod
    def compute_model_flops(model: tf.keras.Model, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate FLOPs (Floating Point Operations) for model inference.
        
        Args:
            model: tf.keras model
            input_shape: Input shape for computation
            
        Returns:
            Estimated FLOPs
        """
        # This is a simplified FLOP estimation
        # For accurate FLOPS, consider using tf.profiler
        
        dummy_input = tf.random.normal((1,) + input_shape)
        
        # Use tf.profiler for accurate FLOP counting
        with tf.profiler.experimental.Profile('logdir'):
            _ = model(dummy_input)
        
        # Simplified estimation based on layer types
        total_flops = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                total_flops += 2 * layer.units * np.prod(layer.input_shape[1:])
            elif isinstance(layer, tf.keras.layers.Conv2D):
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
                output_elements = np.prod(layer.output_shape[1:])
                total_flops += kernel_size * layer.input_shape[-1] * output_elements * 2
        
        return total_flops
    
    @staticmethod
    def create_model_summary_report(model: tf.keras.Model, 
                                   input_shape: Tuple[int, ...],
                                   save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive model summary report.
        
        Args:
            model: tf.keras model
            input_shape: Input shape for analysis
            save_path: Optional path to save report
            
        Returns:
            Summary report as string
        """
        analysis = ModelAnalysis.analyze_model_architecture(model)
        flops = ModelAnalysis.compute_model_flops(model, input_shape)
        
        report = f"""
# Model Analysis Report

## Architecture Overview
- **Total Parameters**: {analysis['total_parameters']:,}
- **Trainable Parameters**: {analysis['trainable_parameters']:,}
- **Non-trainable Parameters**: {analysis['non_trainable_parameters']:,}
- **Total Layers**: {analysis['total_layers']}
- **Model Size**: {analysis['model_size_mb']:.2f} MB
- **Estimated FLOPs**: {flops:,}

## Layer Details
"""
        
        for layer in analysis['layer_details']:
            report += f"- **{layer['name']}** ({layer['type']}): {layer['output_shape']} - {layer['params']:,} params\n"
        
        report += f"""
## Model Summary
```
{model.summary()}
```

## Performance Considerations
- Memory usage scales with batch size and model size
- FLOPs indicate computational complexity
- Consider model optimization for deployment
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Model analysis report saved to {save_path}")
        
        return report


# Convenience functions for common model creation patterns
def create_classification_model(input_shape: Tuple[int, ...],
                               num_classes: int,
                               architecture: str = 'cnn',
                               compile_model: bool = True) -> tf.keras.Model:
    """
    Create and optionally compile a classification model.
    
    Args:
        input_shape: Input data shape
        num_classes: Number of output classes
        architecture: Model architecture type
        compile_model: Whether to compile the model
        
    Returns:
        tf.keras.Model
    """
    if len(input_shape) == 3:  # Image data
        model = ModelBuilders.create_cnn_classifier(input_shape, num_classes, architecture)
    elif len(input_shape) == 1:  # Sequential data
        model = ModelBuilders.create_text_classifier(
            vocab_size=10000,  # Default vocab size
            embedding_dim=128,
            max_length=input_shape[0],
            num_classes=num_classes,
            architecture='lstm'
        )
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")
    
    if compile_model:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model


def create_transfer_learning_model(base_model_name: str,
                                  input_shape: Tuple[int, int, int],
                                  num_classes: int,
                                  fine_tune: bool = False,
                                  fine_tune_at: int = 100) -> tf.keras.Model:
    """
    Create a transfer learning model using tf.keras.applications.
    
    Args:
        base_model_name: Name of base model ('ResNet50', 'VGG16', 'MobileNetV2')
        input_shape: Input image shape
        num_classes: Number of output classes
        fine_tune: Whether to fine-tune base model
        fine_tune_at: Layer index to start fine-tuning from
        
    Returns:
        Transfer learning model
    """
    # Map model names to tf.keras.applications models
    model_map = {
        'ResNet50': tf.keras.applications.ResNet50,
        'VGG16': tf.keras.applications.VGG16,
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'InceptionV3': tf.keras.applications.InceptionV3
    }
    
    if base_model_name not in model_map:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Create base model
    base_model = model_map[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Fine-tuning setup
    if fine_tune:
        base_model.trainable = True
        
        # Fine-tune from specified layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    return model


def save_model_with_metadata(model: tf.keras.Model,
                            save_path: str,
                            metadata: Optional[Dict] = None) -> None:
    """
    Save model with additional metadata.
    
    Args:
        model: tf.keras model to save
        save_path: Path to save model
        metadata: Additional metadata to save
    """
    # Save model in SavedModel format
    model.save(save_path, save_format='tf')
    
    # Save metadata
    if metadata:
        metadata_path = os.path.join(save_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {save_path} with metadata")


def load_model_with_metadata(model_path: str) -> Tuple[tf.keras.Model, Dict]:
    """
    Load model with metadata.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load metadata
    metadata = {}
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata