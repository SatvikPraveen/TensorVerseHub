# Location: /docs/TENSORFLOW_KERAS_BEST_PRACTICES.md

# TensorFlow & tf.keras Best Practices

> **Production-ready coding standards and patterns for TensorFlow 2.x**

## 🎯 Core Principles

### 1. **Prefer tf.keras over low-level TensorFlow**

```python
# ✅ GOOD - Use high-level tf.keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# ❌ AVOID - Low-level operations for simple models
W = tf.Variable(tf.random.normal([784, 64]))
b = tf.Variable(tf.zeros([64]))
```

### 2. **Use tf.data for Data Pipelines**

```python
# ✅ GOOD - Efficient data pipeline
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# ❌ AVOID - Loading all data into memory
X_all = np.load('huge_dataset.npy')  # Memory issues
```

### 3. **Enable Eager Execution for Development**

```python
# ✅ GOOD - Keep eager execution for debugging
tf.config.run_functions_eagerly(True)  # During development

# Switch to graph mode for production
tf.config.run_functions_eagerly(False)  # For performance
```

## 🏗️ Model Architecture

### Layer Organization

```python
# ✅ GOOD - Clear, readable layer structure
class ImageClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=None):
        features = self.feature_extractor(inputs)
        return self.classifier(features, training=training)

# ❌ AVOID - Monolithic model definition
def create_model():
    return tf.keras.Sequential([
        # 50 layers defined inline...
    ])
```

### Input Validation

```python
# ✅ GOOD - Validate inputs with explicit shapes
def create_model(input_shape, num_classes):
    if len(input_shape) != 3:
        raise ValueError("Input shape must be 3D (height, width, channels)")

    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    # ... model definition
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# ❌ AVOID - Implicit input shapes
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),  # No input_shape specified
    # ...
])
```

### Layer Naming

```python
# ✅ GOOD - Meaningful layer names
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', name='feature_extractor'),
    tf.keras.layers.Dropout(0.5, name='regularization'),
    tf.keras.layers.Dense(10, activation='softmax', name='classifier')
])

# ❌ AVOID - Generic layer names
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),  # dense_1, dense_2, etc.
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## 📊 Data Preprocessing

### Preprocessing Layers

```python
# ✅ GOOD - Use preprocessing layers in model
preprocessing = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1)
])

model = tf.keras.Sequential([
    preprocessing,
    # ... rest of model
])

# ❌ AVOID - Preprocessing outside model
def preprocess_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    # Data augmentation logic...
    return image
```

### Data Validation

```python
# ✅ GOOD - Validate data shapes and types
def validate_dataset(dataset):
    for batch in dataset.take(1):
        if isinstance(batch, tuple):
            inputs, labels = batch
            tf.debugging.assert_rank(inputs, 4, "Images must be 4D")
            tf.debugging.assert_type(labels, tf.int32, "Labels must be int32")
        break

# ❌ AVOID - Assuming data format
model.fit(dataset, epochs=10)  # Hope for the best
```

### Efficient Data Loading

```python
# ✅ GOOD - Optimized data pipeline
def create_dataset(file_pattern, batch_size):
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ❌ AVOID - Sequential data loading
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_function)
dataset = dataset.batch(batch_size)
```

## 🎯 Training Best Practices

### Model Compilation

```python
# ✅ GOOD - Explicit compilation with proper metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)

# ❌ AVOID - String-based compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Callback Configuration

```python
# ✅ GOOD - Comprehensive callback setup
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        profile_batch='500,520'
    )
]

# ❌ AVOID - Training without callbacks
model.fit(train_dataset, validation_data=val_dataset, epochs=100)
```

### Training Loop Structure

```python
# ✅ GOOD - Structured training with error handling
def train_model(model, train_dataset, val_dataset, epochs):
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return None
    except tf.errors.ResourceExhaustedError:
        print("Out of memory. Try reducing batch size.")
        return None

# ❌ AVOID - Bare training call
history = model.fit(train_dataset, epochs=100)
```

## 🔧 Performance Optimization

### Mixed Precision Training

```python
# ✅ GOOD - Enable mixed precision properly
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# For models with softmax, use float32 for numerical stability
outputs = tf.keras.layers.Dense(num_classes, dtype='float32', name='predictions')(x)

# ❌ AVOID - Mixed precision without consideration
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# No dtype considerations
```

### GPU Memory Management

```python
# ✅ GOOD - Configure GPU memory growth
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

configure_gpu()

# ❌ AVOID - No GPU configuration
# TensorFlow allocates all GPU memory by default
```

### @tf.function Decoration

```python
# ✅ GOOD - Use @tf.function for performance-critical code
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# ❌ AVOID - Eager execution for training loops
def train_step(x, y):  # No @tf.function
    # Same code - but slower
```

## 🧪 Testing & Validation

### Unit Testing Models

```python
# ✅ GOOD - Test model components
import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = create_model(input_shape=(224, 224, 3), num_classes=10)

    def test_model_output_shape(self):
        dummy_input = tf.random.normal((1, 224, 224, 3))
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 10))

    def test_model_trainable_params(self):
        param_count = self.model.count_params()
        self.assertGreater(param_count, 0)

    def test_model_inference(self):
        dummy_input = tf.random.normal((4, 224, 224, 3))
        output = self.model(dummy_input, training=False)
        self.assertTrue(tf.reduce_all(output >= 0))  # Check softmax output

# ❌ AVOID - No model testing
```

### Input Shape Validation

```python
# ✅ GOOD - Validate inputs during model building
def build_model(input_shape):
    # Validate input shape
    if len(input_shape) != 3:
        raise ValueError(f"Expected 3D input shape, got {input_shape}")

    height, width, channels = input_shape
    if height < 32 or width < 32:
        raise ValueError("Input images must be at least 32x32")

    # Build model...

# ❌ AVOID - No input validation
def build_model(input_shape):
    # Just build without checking
```

## 📁 Code Organization

### Project Structure

```
# ✅ GOOD - Organized structure
project/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── classifier.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   └── training/
│       ├── __init__.py
│       ├── trainer.py
│       └── callbacks.py
├── configs/
│   └── model_config.py
└── main.py

# ❌ AVOID - Everything in one file
project/
└── main.py  # 1000+ lines
```

### Configuration Management

```python
# ✅ GOOD - External configuration
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('configs/model_config.yaml')
model = create_model(**config['model_params'])

# ❌ AVOID - Hardcoded values
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),  # Magic numbers
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## 🔍 Debugging & Monitoring

### Logging Best Practices

```python
# ✅ GOOD - Proper logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting model training")
    try:
        # Training code
        logger.info(f"Training completed. Final loss: {final_loss:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")

# ❌ AVOID - Print statements
def train_model():
    print("Starting training")
    # Training code
    print("Done")
```

### Model Monitoring

```python
# ✅ GOOD - Custom monitoring callback
class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Custom validation metrics
        val_loss = logs.get('val_loss', 0)
        val_accuracy = logs.get('val_accuracy', 0)

        if val_loss > 10.0:  # Divergence detection
            logger.warning(f"High validation loss detected: {val_loss}")
            self.model.stop_training = True

# ❌ AVOID - No monitoring
```

## 🚀 Deployment Considerations

### Model Versioning

```python
# ✅ GOOD - Version your models
def save_model_with_metadata(model, model_path, metadata):
    model.save(model_path)

    metadata_path = f"{model_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

metadata = {
    'version': '1.0.0',
    'training_date': datetime.now().isoformat(),
    'accuracy': float(best_accuracy),
    'input_shape': input_shape,
    'num_classes': num_classes
}
save_model_with_metadata(model, 'model_v1', metadata)

# ❌ AVOID - No versioning
model.save('model.h5')  # Overwritten every time
```

### Model Optimization for Deployment

```python
# ✅ GOOD - Optimize for deployment
def optimize_for_inference(model):
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    return tflite_model

# For mobile deployment
tflite_model = optimize_for_inference(model)
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# ❌ AVOID - No optimization
model.save('model_for_mobile.h5')  # Too large for mobile
```

## ⚡ Common Anti-Patterns to Avoid

### 1. **Data Leakage**

```python
# ❌ BAD - Preprocessing before split
X_normalized = (X - X.mean()) / X.std()
X_train, X_test = train_test_split(X_normalized)

# ✅ GOOD - Preprocessing after split
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. **Memory Leaks**

```python
# ❌ BAD - Accumulating gradients
for epoch in range(epochs):
    for batch in dataset:
        with tf.GradientTape() as tape:
            # ... computation
        grads = tape.gradient(loss, variables)
        # Not clearing tape - memory leak

# ✅ GOOD - Proper gradient tape usage
for epoch in range(epochs):
    for batch in dataset:
        with tf.GradientTape() as tape:
            # ... computation
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        # Tape automatically cleared
```

### 3. **Inefficient Data Loading**

```python
# ❌ BAD - Loading data in training loop
for epoch in range(epochs):
    data = load_data()  # Reloading every epoch
    for batch in data:
        # Training

# ✅ GOOD - Load once, use multiple times
dataset = load_and_preprocess_data()
for epoch in range(epochs):
    for batch in dataset:
        # Training
```

## 📚 Additional Resources

- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/performance)
- [tf.keras Functional API](https://www.tensorflow.org/guide/keras/functional)
- [TensorFlow Data Service](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
- [Mixed Precision Training](https://www.tensorflow.org/guide/mixed_precision)

---

**🎯 Remember**: Consistency, readability, and maintainability are key to successful ML projects!
