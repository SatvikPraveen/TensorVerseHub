# Location: /docs/MODEL_OPTIMIZATION_GUIDE.md

# TensorFlow Model Optimization Guide

> **Complete guide to optimizing TensorFlow models for performance, size, and efficiency**

## 🎯 Optimization Overview

### Why Optimize Models?

- **Performance**: Faster inference for real-time applications
- **Size**: Smaller models for mobile and edge deployment
- **Efficiency**: Lower memory and power consumption
- **Cost**: Reduced computational resources in production

### Optimization Strategies

1. **Architecture Optimization**: Efficient model designs
2. **Quantization**: Reduce precision without losing accuracy
3. **Pruning**: Remove unnecessary weights and connections
4. **Knowledge Distillation**: Transfer knowledge to smaller models
5. **Graph Optimization**: Optimize computational graphs
6. **Hardware-Specific**: Target specific hardware accelerators

## 🏗️ Architecture Optimization

### Efficient Model Architectures

#### MobileNets for Mobile Deployment

```python
import tensorflow as tf

# Standard convolution vs Depthwise Separable
def standard_conv_block(x, filters, kernel_size=3, strides=1):
    """Standard convolution - expensive"""
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def depthwise_separable_conv_block(x, filters, kernel_size=3, strides=1):
    """MobileNet building block - efficient"""
    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Pointwise convolution
    x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

# Usage comparison
def create_efficient_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Use depthwise separable convolutions
    x = depthwise_separable_conv_block(inputs, 32, strides=2)
    x = depthwise_separable_conv_block(x, 64)
    x = depthwise_separable_conv_block(x, 128, strides=2)
    x = depthwise_separable_conv_block(x, 128)
    x = depthwise_separable_conv_block(x, 256, strides=2)
    x = depthwise_separable_conv_block(x, 256)

    # Global average pooling instead of flatten
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)
```

#### EfficientNet Architecture

```python
def create_efficientnet_block(x, filters, kernel_size, strides, expand_ratio, se_ratio):
    """EfficientNet MBConv block"""
    input_filters = x.shape[-1]
    expanded_filters = input_filters * expand_ratio

    # Expansion phase
    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(expanded_filters, 1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)

    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size, strides=strides, padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)

    # Squeeze and excitation
    if se_ratio > 0:
        se_filters = max(1, int(input_filters * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        se = tf.keras.layers.Conv2D(se_filters, 1, activation='swish')(se)
        se = tf.keras.layers.Conv2D(expanded_filters, 1, activation='sigmoid')(se)
        x = tf.keras.layers.Multiply()([x, se])

    # Output projection
    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x
```

### Layer-Level Optimizations

#### Replace Dense Layers with Global Average Pooling

```python
# ❌ INEFFICIENT - Dense layers after flatten
def inefficient_classifier(x, num_classes):
    x = tf.keras.layers.Flatten()(x)  # Creates many parameters
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return x

# ✅ EFFICIENT - Global average pooling
def efficient_classifier(x, num_classes):
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Reduces parameters dramatically
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return x
```

## 🎲 Quantization

### Post-Training Quantization

```python
def quantize_model(model, representative_dataset=None):
    """Convert model to TensorFlow Lite with quantization"""

    # Basic quantization (float16)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    return quantized_model

def full_integer_quantization(model, representative_dataset):
    """Full integer quantization for maximum compression"""

    def representative_dataset_gen():
        for sample in representative_dataset.take(100):
            yield [tf.cast(sample, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quantized_model = converter.convert()
    return quantized_model
```

### Quantization-Aware Training (QAT)

```python
import tensorflow_model_optimization as tfmot

def create_qat_model(base_model):
    """Apply quantization-aware training"""

    # Quantize the entire model
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(base_model)

    # Compile with lower learning rate
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return qat_model

# Usage
base_model = create_model()
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train base model first
base_model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Apply QAT and fine-tune
qat_model = create_qat_model(base_model)
qat_model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# Convert to quantized TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
```

## ✂️ Model Pruning

### Magnitude-Based Pruning

```python
import tensorflow_model_optimization as tfmot

def create_pruned_model(model, target_sparsity=0.5):
    """Apply magnitude-based pruning"""

    # Define pruning schedule
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=target_sparsity,
        begin_step=1000,  # Start pruning after 1000 steps
        end_step=5000     # Reach target sparsity at 5000 steps
    )

    # Apply pruning to dense and conv layers
    def apply_pruning_to_dense(layer):
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule=pruning_schedule)
        return layer

    # Clone model with pruning
    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense,
    )

    return pruned_model

# Training with pruning
def train_pruned_model(model, train_dataset, val_dataset):
    # Compile pruned model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add pruning callbacks
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
    ]

    # Train
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=callbacks
    )

    # Strip pruning wrappers
    final_model = tfmot.sparsity.keras.strip_pruning(model)
    return final_model
```

### Structured Pruning

```python
def structured_pruning_example():
    """Example of structured pruning - removing entire filters"""

    def prune_filters_by_importance(layer_weights, prune_ratio=0.3):
        """Prune filters based on L2 norm importance"""
        if len(layer_weights.shape) == 4:  # Conv2D weights
            # Calculate L2 norm for each filter
            filter_norms = tf.norm(layer_weights, axis=(0, 1, 2))

            # Determine filters to keep
            num_filters = filter_norms.shape[0]
            num_keep = int(num_filters * (1 - prune_ratio))

            # Get indices of top filters
            top_indices = tf.nn.top_k(filter_norms, k=num_keep).indices

            # Keep only important filters
            pruned_weights = tf.gather(layer_weights, top_indices, axis=3)
            return pruned_weights

        return layer_weights

    return prune_filters_by_importance
```

## 📚 Knowledge Distillation

### Teacher-Student Framework

```python
class DistillationModel(tf.keras.Model):
    def __init__(self, teacher, student, alpha=0.7, temperature=3):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature

    def compile(self, optimizer, metrics=None):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

    def call(self, inputs, training=None):
        return self.student(inputs, training=training)

    def train_step(self, data):
        x, y = data

        # Teacher predictions (no gradient)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Student predictions
            student_predictions = self.student(x, training=True)

            # Student loss (regular)
            student_loss = self.student_loss_fn(y, student_predictions)

            # Distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )

            # Combined loss
            total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss

        # Update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        return {
            "total_loss": total_loss,
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
            **{m.name: m.result() for m in self.metrics}
        }

# Usage
def knowledge_distillation_training(teacher_model, student_model, train_dataset):
    # Create distillation model
    distiller = DistillationModel(teacher=teacher_model, student=student_model)

    # Compile
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Train student with teacher guidance
    distiller.fit(train_dataset, epochs=20)

    return distiller.student
```

## ⚡ Runtime Optimization

### Mixed Precision Training

```python
def enable_mixed_precision():
    """Enable mixed precision for faster training"""

    # Set global policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # For numerical stability, keep final layer in float32
    def create_mixed_precision_model(num_classes):
        inputs = tf.keras.Input(shape=(224, 224, 3))

        # Regular layers use mixed precision automatically
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Final layer in float32 for stability
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation='softmax',
            dtype='float32',  # Override to float32
            name='predictions'
        )(x)

        return tf.keras.Model(inputs, outputs)

    return create_mixed_precision_model
```

### Graph Optimization with TensorRT

```python
def optimize_with_tensorrt(saved_model_path, precision_mode='FP16'):
    """Optimize model with TensorRT (NVIDIA GPUs)"""

    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    # TensorRT conversion parameters
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=getattr(trt.TrtPrecisionMode, precision_mode),
        max_workspace_size_bytes=1 << 30,  # 1GB
        use_calibration=precision_mode == 'INT8'
    )

    # Create converter
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_path,
        conversion_params=conversion_params
    )

    # Convert
    converter.convert()

    # Save optimized model
    optimized_path = f"{saved_model_path}_tensorrt_{precision_mode.lower()}"
    converter.save(optimized_path)

    return optimized_path
```

### XLA Compilation

```python
def enable_xla_compilation():
    """Enable XLA (Accelerated Linear Algebra) compilation"""

    # Enable XLA globally
    tf.config.optimizer.set_jit(True)

    # Or enable for specific functions
    @tf.function(experimental_compile=True)
    def optimized_train_step(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return optimized_train_step
```

## 📊 Optimization Evaluation

### Model Size Comparison

```python
def compare_model_sizes(models_dict):
    """Compare sizes of different model variants"""

    results = {}

    for name, model in models_dict.items():
        # Parameter count
        params = model.count_params()

        # Save model to get file size
        temp_path = f"/tmp/{name}_temp.h5"
        model.save(temp_path)
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)

        results[name] = {
            'parameters': params,
            'size_mb': file_size_mb
        }

    # Display comparison
    print(f"{'Model':<20} {'Parameters':<12} {'Size (MB)':<10}")
    print("-" * 45)
    for name, info in results.items():
        print(f"{name:<20} {info['parameters']:<12,} {info['size_mb']:<10.2f}")

    return results
```

### Inference Speed Benchmarking

```python
def benchmark_inference_speed(models_dict, test_data, num_runs=100):
    """Benchmark inference speed of different models"""

    results = {}

    for name, model in models_dict.items():
        # Warmup
        for _ in range(10):
            _ = model.predict(test_data[:1], verbose=0)

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.predict(test_data, verbose=0)

        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        throughput = len(test_data) / avg_time

        results[name] = {
            'avg_inference_time': avg_time,
            'throughput_fps': throughput
        }

    # Display results
    print(f"{'Model':<20} {'Avg Time (s)':<15} {'Throughput (FPS)':<15}")
    print("-" * 55)
    for name, info in results.items():
        print(f"{name:<20} {info['avg_inference_time']:<15.4f} {info['throughput_fps']:<15.2f}")

    return results
```

### Accuracy vs Size Trade-off Analysis

```python
def accuracy_size_tradeoff_analysis(models_results):
    """Analyze accuracy vs model size trade-offs"""

    import matplotlib.pyplot as plt

    models = list(models_results.keys())
    accuracies = [models_results[m]['accuracy'] for m in models]
    sizes_mb = [models_results[m]['size_mb'] for m in models]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes_mb, accuracies, s=100, alpha=0.7)

    # Add model names as labels
    for i, model in enumerate(models):
        plt.annotate(model, (sizes_mb[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Model Size (MB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Model Size Trade-off')
    plt.grid(True, alpha=0.3)

    # Find Pareto optimal models
    pareto_indices = []
    for i in range(len(models)):
        is_pareto = True
        for j in range(len(models)):
            if i != j:
                if sizes_mb[j] <= sizes_mb[i] and accuracies[j] >= accuracies[i]:
                    if sizes_mb[j] < sizes_mb[i] or accuracies[j] > accuracies[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)

    # Highlight Pareto optimal models
    pareto_sizes = [sizes_mb[i] for i in pareto_indices]
    pareto_accuracies = [accuracies[i] for i in pareto_indices]
    plt.scatter(pareto_sizes, pareto_accuracies,
               s=150, facecolors='none', edgecolors='red', linewidths=2)

    plt.tight_layout()
    plt.show()

    return pareto_indices
```

## 📱 Mobile & Edge Deployment

### TensorFlow Lite Optimization

```python
def create_mobile_optimized_model(model, target_platform='mobile'):
    """Optimize model for mobile deployment"""

    if target_platform == 'mobile':
        # Aggressive optimization for mobile
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Additional mobile optimizations
        converter.target_spec.supported_types = [tf.float16]

    elif target_platform == 'edge_tpu':
        # Optimization for Edge TPU
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        # Representative dataset needed for Edge TPU
        def representative_dataset_gen():
            for _ in range(100):
                yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

        converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    return tflite_model

def test_tflite_model(tflite_model, test_samples):
    """Test TensorFlow Lite model performance"""

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test inference
    predictions = []
    for sample in test_samples:
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output)

    return np.array(predictions)
```

## 🎯 Optimization Workflows

### Complete Optimization Pipeline

```python
def complete_optimization_pipeline(base_model, train_dataset, val_dataset, test_dataset):
    """Complete model optimization workflow"""

    print("🚀 Starting Model Optimization Pipeline")

    # Stage 1: Architecture optimization (if needed)
    print("📐 Stage 1: Architecture Analysis")
    original_size = base_model.count_params()
    print(f"Original model parameters: {original_size:,}")

    # Stage 2: Pruning
    print("✂️ Stage 2: Model Pruning")
    pruned_model = create_pruned_model(base_model, target_sparsity=0.3)
    pruned_model = train_pruned_model(pruned_model, train_dataset, val_dataset)

    # Evaluate pruned model
    pruned_accuracy = pruned_model.evaluate(test_dataset, verbose=0)[1]
    print(f"Pruned model accuracy: {pruned_accuracy:.4f}")

    # Stage 3: Quantization-Aware Training
    print("🎲 Stage 3: Quantization-Aware Training")
    qat_model = create_qat_model(pruned_model)
    qat_model.fit(train_dataset, validation_data=val_dataset, epochs=5)

    qat_accuracy = qat_model.evaluate(test_dataset, verbose=0)[1]
    print(f"QAT model accuracy: {qat_accuracy:.4f}")

    # Stage 4: TensorFlow Lite Conversion
    print("📱 Stage 4: TensorFlow Lite Conversion")
    tflite_model = quantize_model(qat_model, train_dataset)

    # Save optimized model
    with open('optimized_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # Stage 5: Evaluation Summary
    print("📊 Stage 5: Optimization Summary")
    original_accuracy = base_model.evaluate(test_dataset, verbose=0)[1]

    print(f"""
    Optimization Results:
    ┌─────────────────┬──────────────┬──────────────┬──────────────┐
    │ Model           │ Accuracy     │ Parameters   │ Size (MB)    │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Original        │ {original_accuracy:.4f}       │ {original_size:>12,} │ {original_size*4/1024/1024:>12.2f} │
    │ Pruned          │ {pruned_accuracy:.4f}       │ {pruned_model.count_params():>12,} │ {pruned_model.count_params()*4/1024/1024:>12.2f} │
    │ QAT             │ {qat_accuracy:.4f}       │ {qat_model.count_params():>12,} │ {qat_model.count_params()*4/1024/1024:>12.2f} │
    │ TFLite (INT8)   │ ~{qat_accuracy:.4f}       │ {qat_model.count_params():>12,} │ {len(tflite_model)/1024/1024:>12.2f} │
    └─────────────────┴──────────────┴──────────────┴──────────────┘
    """)

    return {
        'original': base_model,
        'pruned': pruned_model,
        'qat': qat_model,
        'tflite': tflite_model
    }
```

## 🔧 Tools & Utilities

### Model Analysis Tools

```python
def analyze_model_complexity(model):
    """Comprehensive model complexity analysis"""

    # Basic metrics
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    # Layer analysis
    layer_info = []
    for i, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        output_shape = layer.output_shape

        layer_info.append({
            'layer_idx': i,
            'layer_name': layer.name,
            'layer_type': type(layer).__name__,
            'parameters': layer_params,
            'output_shape': output_shape
        })

    # Find computational bottlenecks
    bottlenecks = sorted(layer_info, key=lambda x: x['parameters'], reverse=True)[:5]

    print(f"Model Complexity Analysis:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"\nTop 5 Parameter-Heavy Layers:")
    for layer in bottlenecks:
        print(f"  {layer['layer_name']}: {layer['parameters']:,} params")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_info': layer_info,
        'bottlenecks': bottlenecks
    }

# Model visualization
def visualize_optimization_results(results):
    """Visualize optimization results"""

    models = list(results.keys())
    accuracies = []
    sizes = []

    for model_name in models:
        model = results[model_name]
        if hasattr(model, 'evaluate'):
            # For Keras models
            size = model.count_params() * 4 / 1024 / 1024  # MB
        else:
            # For TFLite models
            size = len(model) / 1024 / 1024  # MB

        sizes.append(size)

    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Model sizes
    ax1.bar(models, sizes)
    ax1.set_ylabel('Size (MB)')
    ax1.set_title('Model Size Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Size reduction
    original_size = sizes[0]
    reductions = [(original_size - size) / original_size * 100 for size in sizes]

    ax2.bar(models, reductions)
    ax2.set_ylabel('Size Reduction (%)')
    ax2.set_title('Size Reduction from Original')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
```

## 📚 Additional Resources

- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [TensorFlow Lite Converter](https://www.tensorflow.org/lite/convert)
- [Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Pruning Guide](https://www.tensorflow.org/model_optimization/guide/pruning)
- [Knowledge Distillation Papers](https://arxiv.org/abs/1503.02531)
- [EfficientNet Architecture](https://arxiv.org/abs/1905.11946)
- [MobileNet Architecture](https://arxiv.org/abs/1704.04861)

---

**🎯 Remember**: Start with the easiest optimizations (quantization) and progressively apply more complex techniques. Always measure the accuracy-performance trade-offs!
