# Practical Examples and Use Cases

## Quick Start Examples

### Example 1: Image Classification with Transfer Learning

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Add custom layers for your task
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation='softmax')(x)  # 10 classes

model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Example training
# X_train, y_train = load_your_images()
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Fine-tune (unfreeze some layers)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# model.fit(X_train, y_train, epochs=5)
```

### Example 2: Text Classification with LSTM

```python
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, 
    Bidirectional, TextVectorization
)
from tensorflow.keras.optimizers import Adam

# Create text vectorization layer
text_vectorizer = TextVectorization(
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=100
)

# Build model
model = tf.keras.Sequential([
    text_vectorizer,
    Embedding(input_dim=5001, output_dim=128, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Dropout(0.2),
    Bidirectional(LSTM(32, dropout=0.2)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 sentiment classes
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Example usage
sample_texts = ["I love this movie!", "This is terrible.", "It's okay."]
# model.fit(sample_texts, labels, epochs=10)
```

### Example 3: Time Series Forecasting with Transformer

```python
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, MultiHeadAttention, 
    LayerNormalization, Dropout
)
import numpy as np

# Build Transformer encoder
def build_transformer_encoder(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=128,
    dropout=0.1,
    num_transformer_blocks=4
):
    inputs = Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed forward
        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dropout(dropout)(ff_output)
        ff_output = Dense(x.shape[-1])(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='linear')(x)  # Regression output
    
    return tf.keras.Model(inputs, outputs)

# Example usage
model = build_transformer_encoder(
    input_shape=(100, 10),  # 100 time steps, 10 features
    num_heads=4,
    head_size=256,
    ff_dim=128,
    num_transformer_blocks=2
)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

### Example 4: Generative Model - Simple VAE

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dim=20, input_shape=(28, 28, 1)):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape_ = input_shape
        
        # Encoder
        self.encoder_input = layers.Input(shape=input_shape)
        x = layers.Flatten()(self.encoder_input)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.z_sampling = layers.Lambda(self._sample_z)
        
        # Decoder
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(128, activation='relu')(latent_inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
        outputs = layers.Reshape(input_shape)(x)
        
        self.encoder = tf.keras.Model(self.encoder_input, 
                                     [self.z_mean(self.encoder_input),
                                      self.z_log_var(self.encoder_input)])
        self.decoder = tf.keras.Model(latent_inputs, outputs)
    
    def _sample_z(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.z_sampling([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss / 784)  # Normalize by input size
        
        return reconstructed
    
    def generate(self, num_samples=10):
        """Generate new samples"""
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decoder.predict(z)

# Example usage
vae = VAE(latent_dim=20)
vae.compile(optimizer='adam', loss='binary_crossentropy')
# vae.fit(X_train, epochs=30)
# generated_images = vae.generate(10)
```

### Example 5: Model Quantization for Mobile

```python
import tensorflow as tf
import numpy as np

# Load a trained model
model = tf.keras.models.load_model('trained_model.h5')

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Provide representative dataset for better accuracy
def representative_dataset():
    # Load sample images
    for _ in range(100):
        data = np.random.randn(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quantized_model = converter.convert()

# Save quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)

print(f"Original model size: {len(open('trained_model.h5', 'rb').read()) / 1024 / 1024:.1f} MB")
print(f"Quantized model size: {len(tflite_quantized_model) / 1024 / 1024:.1f} MB")
```

### Example 6: Custom Training Loop with Gradient Tape

```python
import tensorflow as tf
import numpy as np

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define loss and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

# Custom training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        loss += sum(model.losses)  # Add regularization losses
    
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    train_loss.update_state(loss)
    train_acc.update_state(y, predictions)
    
    return loss

# Training loop
X_train = np.random.randn(1000, 20).astype(np.float32)
y_train = np.random.randint(0, 2, 1000).astype(np.float32)

batch_size = 32
epochs = 5

for epoch in range(epochs):
    train_loss.reset_state()
    train_acc.reset_state()
    
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        
        loss = train_step(x_batch, y_batch)
    
    print(f"Epoch {epoch+1}: Loss={train_loss.result():.4f}, "
          f"Accuracy={train_acc.result():.4f}")
```

### Example 7: Data Pipeline Optimization

```python
import tensorflow as tf
import numpy as np

# Create dataset
def create_dataset(x, y, batch_size=32, prefetch_size=tf.data.AUTOTUNE):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Optimization techniques
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    dataset = dataset.cache()  # Cache in memory
    
    return dataset

# Data augmentation
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

# Apply augmentation
def create_augmented_dataset(x, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Example usage
X = np.random.randn(1000, 28, 28, 1).astype(np.float32)
y = np.random.randint(0, 10, 1000)

train_dataset = create_augmented_dataset(X, y)
```

### Example 8: Multi-GPU Training

```python
import tensorflow as tf

# Automatic GPU detection
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs: {len(gpus)}")

# Strategy for multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Build and compile model within strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Training with strategy
X = np.random.randn(1000, 20).astype(np.float32)
y = np.random.randint(0, 10, 1000)

# model.fit(X, y, batch_size=32, epochs=10)
```

### Example 9: Model Checkpointing and Early Stopping

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Custom callback for logging
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# X_train, y_train, X_val, y_val = ...
# model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=100,
#     batch_size=32,
#     callbacks=callbacks
# )
```

### Example 10: Ensemble Models

```python
import tensorflow as tf
import numpy as np

# Create multiple models
models = []

for i in range(3):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    models.append(model)

# Ensemble prediction
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X, verbose=0)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# Example usage
# X_test = np.random.randn(100, 20).astype(np.float32)
# ensemble_predictions = ensemble_predict(models, X_test)
# ensemble_classes = np.argmax(ensemble_predictions, axis=1)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Overfitting
**Problem**: Model performs well on training data but poorly on test data

**Solution**:
```python
# Use regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', 
                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                         input_shape=(20,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Use early stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]
```

### Pitfall 2: Learning Rate Too High
**Problem**: Training loss oscillates and doesn't converge

**Solution**:
```python
# Use learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Or use adaptive optimizers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Lower learning rate
    loss='categorical_crossentropy'
)
```

### Pitfall 3: Unbalanced Classes
**Problem**: Model is biased toward majority class

**Solution**:
```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))

# Use in training
model.fit(
    X_train, y_train,
    class_weight=class_weight_dict,
    epochs=10
)
```

### Pitfall 4: Not Normalizing Input Data
**Problem**: Model converges slowly or doesn't train well

**Solution**:
```python
# Normalize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Or use normalization layer
normalization_layer = tf.keras.layers.Normalization()
normalization_layer.adapt(X_train)

model = tf.keras.Sequential([
    normalization_layer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## Performance Optimization Checklist

- [ ] Use `tf.data` for efficient data loading
- [ ] Enable `mixed_precision` for faster training on GPUs
- [ ] Use `@tf.function` decorator for custom training loops
- [ ] Batch normalization for faster convergence
- [ ] Use appropriate batch size (32-256 for most tasks)
- [ ] Profile code to find bottlenecks
- [ ] Use `eager_execution=False` for production
- [ ] Quantize models for mobile/edge deployment
- [ ] Prune models to reduce size
- [ ] Use knowledge distillation for smaller models

---

## Deployment Checklist

- [ ] Export model in SavedModel format
- [ ] Convert to TFLite for mobile
- [ ] Create Docker container
- [ ] Set up health check endpoints
- [ ] Implement request/response logging
- [ ] Add rate limiting
- [ ] Set up monitoring and alerting
- [ ] Create API documentation
- [ ] Test with real data
- [ ] Plan rollback strategy
