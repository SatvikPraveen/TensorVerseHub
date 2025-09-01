# Location: /docs/TROUBLESHOOTING.md

# TensorFlow Troubleshooting Guide

> **Common TensorFlow issues, solutions, and debugging strategies**

## 🚨 Memory Issues

### Out of Memory (OOM) Errors

#### GPU Memory Exhaustion

```bash
# Error message
ResourceExhaustedError: OOM when allocating tensor with shape [32, 224, 224, 3]
```

**Solutions:**

```python
# 1. Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# 2. Reduce batch size
model.fit(dataset.batch(16), ...)  # Instead of batch(64)

# 3. Use mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 4. Clear session between runs
tf.keras.backend.clear_session()

# 5. Use gradient accumulation for large batches
def gradient_accumulation_step(model, optimizer, inputs, labels, accumulate_grad_batches=4):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions) / accumulate_grad_batches

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### CPU Memory Issues

```python
# Use tf.data for large datasets
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Instead of loading everything into memory
# X = np.load('huge_dataset.npy')  # Avoid this
```

### Memory Leaks

```python
# ✅ GOOD - Clear resources properly
def train_model():
    model = create_model()
    # ... training code
    del model  # Explicitly delete
    tf.keras.backend.clear_session()

# ❌ BAD - Accumulating models in memory
models = []
for i in range(100):
    model = create_model()
    models.append(model)  # Memory leak
```

## 🐛 Training Issues

### Model Not Learning

#### Loss Not Decreasing

```python
# Check 1: Learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Try different values

# Check 2: Data preprocessing
def debug_data(dataset):
    for batch in dataset.take(1):
        X, y = batch
        print(f"Input range: {tf.reduce_min(X):.3f} to {tf.reduce_max(X):.3f}")
        print(f"Label distribution: {tf.reduce_sum(tf.cast(y, tf.int32), axis=0)}")

# Check 3: Model capacity
model.summary()  # Ensure sufficient parameters

# Check 4: Loss function
# For sparse labels: use sparse_categorical_crossentropy
# For one-hot labels: use categorical_crossentropy
```

#### Exploding Gradients

```python
# Solution 1: Gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Solution 2: Manual gradient clipping
def train_step_with_clipping(model, inputs, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### Vanishing Gradients

```python
# Solution 1: Batch normalization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Solution 2: Residual connections
def residual_block(x, filters):
    residual = x
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])  # Skip connection
    return tf.keras.layers.ReLU()(x)

# Solution 3: Better weight initialization
initializer = tf.keras.initializers.HeNormal()
layer = tf.keras.layers.Dense(64, kernel_initializer=initializer)
```

### Overfitting

#### High Training Accuracy, Low Validation Accuracy

```python
# Solution 1: Regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Solution 2: Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Solution 3: Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Solution 4: Reduce model complexity
# Use fewer parameters, layers, or units
```

## 🔧 Data Pipeline Issues

### Slow Training Due to Data Loading

#### tf.data Optimization

```python
# ❌ SLOW - Inefficient pipeline
def slow_pipeline(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(load_and_preprocess)  # Sequential
    dataset = dataset.batch(32)
    return dataset

# ✅ FAST - Optimized pipeline
def fast_pipeline(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        parse_function,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

### Data Format Issues

#### Shape Mismatches

```python
# Debug data shapes
def debug_shapes(dataset):
    for batch in dataset.take(1):
        if isinstance(batch, tuple):
            inputs, labels = batch
            print(f"Input shape: {inputs.shape}")
            print(f"Label shape: {labels.shape}")
            print(f"Input dtype: {inputs.dtype}")
            print(f"Label dtype: {labels.dtype}")
        break

# Common fixes
def fix_shapes(dataset):
    def preprocess(inputs, labels):
        # Ensure 4D input for CNN
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=0)

        # Ensure proper label format
        if labels.dtype != tf.int32:
            labels = tf.cast(labels, tf.int32)

        return inputs, labels

    return dataset.map(preprocess)
```

#### Image Loading Issues

```python
def robust_image_loading(image_path):
    """Robust image loading with error handling"""
    try:
        # Load image
        image = tf.io.read_file(image_path)

        # Decode with fallback
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except tf.errors.InvalidArgumentError:
            try:
                image = tf.image.decode_png(image, channels=3)
            except tf.errors.InvalidArgumentError:
                # Return default image
                image = tf.zeros((224, 224, 3), dtype=tf.uint8)

        # Ensure proper shape
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0

        return image
    except Exception:
        # Return default image on any error
        return tf.zeros((224, 224, 3), dtype=tf.float32)
```

## ⚡ Performance Issues

### Slow Inference

#### Model Optimization

```python
# 1. Use TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 2. Batch predictions
def batch_predict(model, inputs, batch_size=32):
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.append(pred)
    return np.concatenate(predictions)

# 3. Use @tf.function
@tf.function
def fast_predict(model, inputs):
    return model(inputs, training=False)
```

### Slow Training

#### Training Optimization

```python
# 1. Profile training
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    profile_batch='500,520'  # Profile specific batches
)

# 2. Check data loading
import time

def time_data_loading(dataset):
    start_time = time.time()
    for i, batch in enumerate(dataset.take(10)):
        batch_time = time.time()
        print(f"Batch {i}: {batch_time - start_time:.3f}s")
        start_time = batch_time

# 3. Use mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## 🌐 Distributed Training Issues

### Multi-GPU Training Problems

#### Strategy Setup

```python
# Proper strategy setup
def setup_distributed_training():
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
        return strategy
    except Exception as e:
        print(f"Distributed training not available: {e}")
        return None

# Use strategy scope
strategy = setup_distributed_training()
if strategy:
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

#### Batch Size Scaling

```python
# Scale batch size with number of GPUs
GLOBAL_BATCH_SIZE = 64
strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync

dataset = dataset.batch(BATCH_SIZE_PER_REPLICA)
```

## 🔍 Debugging Techniques

### Enable Eager Execution for Debugging

```python
# Enable eager execution
tf.config.run_functions_eagerly(True)

# Debug model layers
def debug_model(model, inputs):
    x = inputs
    for i, layer in enumerate(model.layers):
        x = layer(x)
        print(f"Layer {i} ({layer.name}): {x.shape}, min={tf.reduce_min(x):.3f}, max={tf.reduce_max(x):.3f}")

        # Check for NaN or Inf
        if tf.reduce_any(tf.math.is_nan(x)):
            print(f"⚠️ NaN detected in layer {i}")
        if tf.reduce_any(tf.math.is_inf(x)):
            print(f"⚠️ Inf detected in layer {i}")
```

### Custom Training Loop for Debugging

```python
def debug_training_step(model, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)

    # Check loss value
    print(f"Loss: {loss.numpy():.6f}")

    gradients = tape.gradient(loss, model.trainable_variables)

    # Check gradients
    for i, grad in enumerate(gradients):
        if grad is not None:
            grad_norm = tf.norm(grad)
            print(f"Gradient {i}: norm={grad_norm:.6f}")

            if tf.reduce_any(tf.math.is_nan(grad)):
                print(f"⚠️ NaN gradient detected in layer {i}")

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### TensorBoard for Visual Debugging

```python
# Comprehensive TensorBoard logging
def create_tensorboard_callback():
    return tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,          # Log weight histograms
        write_graph=True,          # Log model graph
        write_images=True,         # Log model weights as images
        update_freq='epoch',       # Update frequency
        profile_batch='500,520',   # Profile specific batches
        embeddings_freq=1          # Log embeddings
    )
```

## 🛠️ Common Error Messages & Solutions

### Import Errors

#### TensorFlow Version Issues

```bash
# Error: No module named 'tensorflow'
pip install tensorflow

# Error: TF version compatibility
pip install tensorflow==2.13.0  # Use specific version

# Check TF version
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### CUDA Issues

```bash
# Error: CUDA out of memory
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# Check CUDA compatibility
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Model Loading Issues

#### SavedModel Format Issues

```python
# Try different loading methods
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e1:
    try:
        model = tf.saved_model.load('model')
    except Exception as e2:
        print(f"Loading failed: {e1}, {e2}")

# Check model format
import os
if os.path.isdir('model') and 'saved_model.pb' in os.listdir('model'):
    print("SavedModel format detected")
elif 'model.h5' in os.listdir('.'):
    print("HDF5 format detected")
```

### Runtime Errors

#### Shape Incompatibility

```python
def fix_shape_issues(model, inputs):
    """Debug and fix shape issues"""
    print(f"Model expects: {model.input_shape}")
    print(f"Input provided: {inputs.shape}")

    # Auto-fix common issues
    if len(inputs.shape) == 3 and len(model.input_shape) == 4:
        inputs = tf.expand_dims(inputs, axis=0)
        print("Added batch dimension")

    if inputs.shape[-1] != model.input_shape[-1]:
        print(f"Channel mismatch: {inputs.shape[-1]} vs {model.input_shape[-1]}")
        # Resize or pad as needed

    return inputs
```

## 📊 Performance Monitoring

### System Resource Monitoring

```python
import psutil
import GPUtil

def monitor_resources():
    """Monitor system resources during training"""

    def log_resources():
        # CPU usage
        cpu_percent = psutil.cpu_percent()

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
            gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
        except Exception:
            gpu_usage = 0
            gpu_memory = 0

        print(f"CPU: {cpu_percent}%, RAM: {memory_percent}%, GPU: {gpu_usage}%, GPU Mem: {gpu_memory}%")

    return log_resources

# Use as callback
class ResourceMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        self.monitor_fn = monitor_resources()

    def on_epoch_end(self, epoch, logs=None):
        self.monitor_fn()
```

## 🚀 Quick Fixes Checklist

### Before Training

- [ ] Check data shapes and types
- [ ] Verify label format (sparse vs one-hot)
- [ ] Test data pipeline with one batch
- [ ] Validate model architecture
- [ ] Check GPU memory availability

### During Training

- [ ] Monitor loss curves in TensorBoard
- [ ] Check for NaN/Inf values
- [ ] Verify learning rate schedule
- [ ] Monitor resource usage
- [ ] Use callbacks for early intervention

### Performance Issues

- [ ] Profile data loading speed
- [ ] Use tf.data optimization
- [ ] Enable mixed precision
- [ ] Batch predictions efficiently
- [ ] Use @tf.function for custom loops

### Debugging

- [ ] Enable eager execution
- [ ] Add print statements strategically
- [ ] Use tf.debugging assertions
- [ ] Check intermediate layer outputs
- [ ] Validate gradients

## 📞 Getting Help

### Community Resources

- [TensorFlow GitHub Issues](https://github.com/tensorflow/tensorflow/issues)
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow TensorFlow Tag](https://stackoverflow.com/questions/tagged/tensorflow)
- [TensorFlow Discord](https://discord.gg/tensorflow)

### Debugging Information to Include

```python
def collect_debug_info():
    """Collect system and TensorFlow info for bug reports"""
    info = {
        'tensorflow_version': tf.__version__,
        'python_version': sys.version,
        'system': platform.system(),
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'cuda_version': tf.sysconfig.get_build_info().get('cuda_version', 'N/A'),
        'cudnn_version': tf.sysconfig.get_build_info().get('cudnn_version', 'N/A')
    }

    for key, value in info.items():
        print(f"{key}: {value}")

    return info
```

---

**💡 Pro Tips:**

- Always test with a small dataset first
- Use TensorBoard for visual debugging
- Start simple and add complexity gradually
- Keep backups of working configurations
