# Location: /docs/QUICK_REFERENCE.md

# TensorFlow + tf.keras Quick Reference

> **Fast reference for TensorFlow 2.x and tf.keras essentials**

## 🚀 Essential Imports

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
```

## 📊 Data Operations

### Data Types & Tensors

```python
# Create tensors
tf.constant([1, 2, 3])                    # Constant tensor
tf.Variable([1, 2, 3])                    # Variable tensor
tf.zeros((3, 4))                          # Zero tensor
tf.ones((2, 3))                           # Ones tensor
tf.random.normal((2, 3))                  # Random normal
tf.random.uniform((2, 3), 0, 1)          # Random uniform

# Tensor operations
a + b                                     # Element-wise addition
tf.matmul(a, b)                          # Matrix multiplication
tf.reduce_mean(tensor)                    # Mean reduction
tf.reduce_sum(tensor, axis=1)            # Sum along axis
tf.reshape(tensor, (2, -1))              # Reshape tensor
```

### Data Pipeline (tf.data)

```python
# From arrays
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# From files
dataset = tf.data.TextLineDataset(filenames)

# Common operations
dataset = dataset.batch(32)               # Batch data
dataset = dataset.shuffle(1000)           # Shuffle buffer
dataset = dataset.map(preprocess_fn)      # Apply function
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch
dataset = dataset.repeat(epochs)          # Repeat epochs
dataset = dataset.take(100)               # Take first 100
```

## 🧠 Model Building

### Sequential API

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
```

### Functional API

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Subclassing API

```python
class MyModel(keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)
```

## 🏗️ Common Layer Types

### Dense Layers

```python
layers.Dense(units, activation='relu')
layers.Dense(64, activation='relu', kernel_regularizer='l2')
```

### Convolutional Layers

```python
layers.Conv2D(32, (3, 3), activation='relu')
layers.MaxPooling2D((2, 2))
layers.GlobalAveragePooling2D()
layers.Conv2DTranspose(32, (3, 3), strides=2)  # Deconvolution
```

### Recurrent Layers

```python
layers.LSTM(64, return_sequences=True)
layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)
layers.SimpleRNN(32)
layers.Bidirectional(layers.LSTM(64))
```

### Normalization & Regularization

```python
layers.BatchNormalization()
layers.LayerNormalization()
layers.Dropout(0.5)
layers.SpatialDropout2D(0.25)
```

### Embedding & Attention

```python
layers.Embedding(vocab_size, embedding_dim)
layers.MultiHeadAttention(num_heads=8, key_dim=64)
```

## ⚙️ Model Compilation

```python
model.compile(
    optimizer='adam',                     # or optimizers.Adam(learning_rate=0.001)
    loss='sparse_categorical_crossentropy',  # or losses.SparseCategoricalCrossentropy()
    metrics=['accuracy']                  # or metrics.Accuracy()
)

# Custom optimizer
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=[metrics.Precision(), metrics.Recall()]
)
```

## 🎯 Training

### Basic Training

```python
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)
```

### Training with tf.data

```python
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint]
)
```

## 📈 Callbacks

```python
# Early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model checkpoint
checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# Learning rate scheduler
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)

# TensorBoard
tensorboard = callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

# Custom callback
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: loss = {logs['loss']:.4f}")
```

## 🔄 Model Evaluation & Prediction

```python
# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Predict
predictions = model.predict(X_new)

# Predict single sample
single_pred = model(tf.expand_dims(sample, 0))
```

## 💾 Model Saving & Loading

```python
# Save entire model
model.save('my_model.h5')                # HDF5 format
model.save('my_model')                   # SavedModel format

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save/load weights only
model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')

# Save architecture
with open('model_config.json', 'w') as json_file:
    json_file.write(model.to_json())
```

## 🎨 Visualization

```python
# Model summary
model.summary()

# Plot model
keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

## 🛠️ Custom Components

### Custom Loss Function

```python
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Or subclass
class CustomLoss(losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
```

### Custom Metric

```python
class F1Score(metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
```

### Custom Layer

```python
class CustomDense(layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        return self.activation(x)
```

## 🔧 Debugging & Optimization

### Enable Mixed Precision

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### GPU Configuration

```python
# List GPUs
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Debugging

```python
# Enable eager execution debugging
tf.config.run_functions_eagerly(True)

# Print tensor values
tf.print("Tensor value:", tensor)

# Check for NaN/Inf
tf.debugging.check_numerics(tensor, "NaN/Inf check")
```

## 📋 Common Patterns

### Image Classification

```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Text Classification

```python
model = keras.Sequential([
    layers.Embedding(vocab_size, 16, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

### Time Series

```python
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(50),
    layers.Dropout(0.2),
    layers.Dense(1)
])
```

## ⚡ Performance Tips

- Use `tf.data` for data pipelines
- Enable mixed precision for faster training
- Use `tf.function` for performance-critical code
- Batch operations when possible
- Prefetch data with `dataset.prefetch(tf.data.AUTOTUNE)`
- Use appropriate data types (float16 vs float32)
- Profile with TensorBoard Profiler

## 🔍 Quick Troubleshooting

| Issue              | Solution                                         |
| ------------------ | ------------------------------------------------ |
| Out of memory      | Reduce batch size, use gradient accumulation     |
| Training too slow  | Check data pipeline, use mixed precision         |
| Model not learning | Check learning rate, data preprocessing          |
| Overfitting        | Add dropout, regularization, early stopping      |
| Underfitting       | Increase model complexity, reduce regularization |

---

**💡 Pro Tip**: Always check your data shapes and types first when debugging!
