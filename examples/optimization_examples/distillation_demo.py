# Location: /examples/optimization_examples/distillation_demo.py

"""
TensorFlow knowledge distillation demonstration.
Shows teacher-student training, performance comparison, and analysis.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import os
from typing import Tuple, Dict, List, Optional, Any, Callable
import json

# Import TensorVerseHub utilities
try:
    from src.model_utils import ModelBuilders
    from src.visualization import setup_plotting_style
    from src.optimization_utils import KnowledgeDistillation
except ImportError:
    print("Warning: TensorVerseHub modules not found. Using standalone implementation.")


class DistillationDemo:
    """Comprehensive knowledge distillation demonstration."""
    
    def __init__(self, alpha: float = 0.7, temperature: float = 3.0):
        """
        Initialize distillation demo.
        
        Args:
            alpha: Weight for distillation loss
            temperature: Temperature for softmax distillation
        """
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_model = None
        self.student_models = {}
        self.training_histories = {}
        self.performance_metrics = {}
        
        # Setup plotting
        try:
            setup_plotting_style()
        except:
            plt.style.use('default')
            sns.set_palette("husl")
    
    def create_teacher_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                           num_classes: int = 10) -> tf.keras.Model:
        """Create a large teacher model."""
        print("👨‍🏫 Creating teacher model (large ResNet-like architecture)...")
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Initial conv block
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        def residual_block(x, filters, stride=1, expand=False):
            shortcut = x
            
            # First conv
            x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            
            # Second conv
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if stride != 1 or expand:
                shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.ReLU()(x)
            return x
        
        # ResNet-like blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2, expand=True)
        x = residual_block(x, 128)
        x = residual_block(x, 256, stride=2, expand=True)
        x = residual_block(x, 256)
        x = residual_block(x, 512, stride=2, expand=True)
        x = residual_block(x, 512)
        
        # Classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        teacher_model = tf.keras.Model(inputs, outputs, name='teacher_model')
        
        teacher_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Teacher model created with {teacher_model.count_params():,} parameters")
        return teacher_model
    
    def create_student_models(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 10) -> Dict[str, tf.keras.Model]:
        """Create various student models of different sizes."""
        print("👨‍🎓 Creating student models...")
        
        students = {}
        
        # Small CNN student
        print("  📱 Creating small CNN student...")
        small_student = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes)  # No softmax for distillation
        ], name='small_student')
        
        students['small'] = small_student
        
        # Medium MobileNet-like student
        print("  📲 Creating medium MobileNet-like student...")
        
        def depthwise_conv_block(x, filters, stride=1):
            """Depthwise separable convolution block."""
            x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            
            x = tf.keras.layers.Conv2D(filters, 1)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return x
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = depthwise_conv_block(x, 64)
        x = depthwise_conv_block(x, 128, stride=2)
        x = depthwise_conv_block(x, 128)
        x = depthwise_conv_block(x, 256, stride=2)
        x = depthwise_conv_block(x, 256)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)  # No softmax
        
        medium_student = tf.keras.Model(inputs, outputs, name='medium_student')
        students['medium'] = medium_student
        
        # Tiny student (ultra-lightweight)
        print("  🐛 Creating tiny student...")
        tiny_student = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5, strides=2, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes)  # No softmax
        ], name='tiny_student')
        
        students['tiny'] = tiny_student
        
        # Print model sizes
        for name, model in students.items():
            print(f"  ✅ {name.capitalize()} student: {model.count_params():,} parameters")
        
        return students
    
    def create_synthetic_data(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 10,
                            num_samples: int = 1500) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create synthetic dataset for demonstration."""
        print("📊 Creating synthetic dataset...")
        
        # Generate random images and labels
        images = tf.random.normal([num_samples] + list(input_shape))
        labels = tf.random.uniform([num_samples], 0, num_classes, dtype=tf.int32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(1000)
        
        # Split train/val/test
        train_size = int(0.7 * num_samples)
        val_size = int(0.2 * num_samples)
        
        train_ds = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = dataset.skip(train_size).take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = dataset.skip(train_size + val_size).batch(32).prefetch(tf.data.AUTOTUNE)
        
        print(f"✅ Dataset created: {train_size} train, {val_size} val, {num_samples - train_size - val_size} test")
        return train_ds, val_ds, test_ds
    
    def train_teacher(self, teacher_model: tf.keras.Model,
                     train_ds: tf.data.Dataset,
                     val_ds: tf.data.Dataset,
                     epochs: int = 15) -> tf.keras.callbacks.History:
        """Train the teacher model."""
        print(f"👨‍🏫 Training teacher model for {epochs} epochs...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2),
            tf.keras.callbacks.ModelCheckpoint('teacher_best.h5', save_best_only=True)
        ]
        
        history = teacher_model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate teacher
        teacher_loss, teacher_acc = teacher_model.evaluate(val_ds, verbose=0)
        print(f"✅ Teacher training completed - Val accuracy: {teacher_acc:.4f}")
        
        return history
    
    def create_distillation_loss(self) -> Callable:
        """Create knowledge distillation loss function."""
        def distillation_loss(y_true, y_pred_student, y_pred_teacher):
            # Convert labels to one-hot for soft loss calculation
            y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred_student.shape[-1])
            
            # Hard loss (standard cross-entropy)
            hard_loss = tf.keras.losses.categorical_crossentropy(y_true_onehot, 
                                                               tf.nn.softmax(y_pred_student))
            
            # Soft loss (knowledge distillation)
            teacher_soft = tf.nn.softmax(y_pred_teacher / self.temperature)
            student_soft = tf.nn.softmax(y_pred_student / self.temperature)
            
            soft_loss = tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft)
            soft_loss *= (self.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
            return total_loss
        
        return distillation_loss
    
    def train_student_with_distillation(self, student_model: tf.keras.Model,
                                      teacher_model: tf.keras.Model,
                                      train_ds: tf.data.Dataset,
                                      val_ds: tf.data.Dataset,
                                      student_name: str,
                                      epochs: int = 10) -> tf.keras.callbacks.History:
        """Train student model using knowledge distillation."""
        print(f"🎓 Training {student_name} student with distillation...")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Training metrics
        train_loss_tracker = tf.keras.metrics.Mean()
        train_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
        val_loss_tracker = tf.keras.metrics.Mean()
        val_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
        
        distillation_loss_fn = self.create_distillation_loss()
        
        @tf.function
        def train_step(x_batch, y_batch):
            # Get teacher predictions (no gradients needed)
            teacher_pred = teacher_model(x_batch, training=False)
            
            with tf.GradientTape() as tape:
                # Get student predictions
                student_pred = student_model(x_batch, training=True)
                
                # Calculate distillation loss
                loss = distillation_loss_fn(y_batch, student_pred, teacher_pred)
                loss = tf.reduce_mean(loss)
            
            # Apply gradients
            gradients = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
            
            # Update metrics
            train_loss_tracker.update_state(loss)
            train_accuracy_tracker.update_state(y_batch, tf.nn.softmax(student_pred))
            
            return loss
        
        @tf.function
        def val_step(x_batch, y_batch):
            teacher_pred = teacher_model(x_batch, training=False)
            student_pred = student_model(x_batch, training=False)
            
            loss = distillation_loss_fn(y_batch, student_pred, teacher_pred)
            loss = tf.reduce_mean(loss)
            
            val_loss_tracker.update_state(loss)
            val_accuracy_tracker.update_state(y_batch, tf.nn.softmax(student_pred))
            
            return loss
        
        # Training loop
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Reset metrics
            train_loss_tracker.reset_states()
            train_accuracy_tracker.reset_states()
            val_loss_tracker.reset_states()
            val_accuracy_tracker.reset_states()
            
            # Training
            for x_batch, y_batch in train_ds:
                train_step(x_batch, y_batch)
            
            # Validation
            for x_batch, y_batch in val_ds:
                val_step(x_batch, y_batch)
            
            # Record metrics
            train_loss = train_loss_tracker.result()
            train_acc = train_accuracy_tracker.result()
            val_loss = val_loss_tracker.result()
            val_acc = val_accuracy_tracker.result()
            
            history['loss'].append(float(train_loss))
            history['accuracy'].append(float(train_acc))
            history['val_loss'].append(float(val_loss))
            history['val_accuracy'].append(float(val_acc))
            
            print(f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                student_model.save_weights(f'{student_name}_student_best.h5')
        
        print(f"✅ {student_name.capitalize()} student training completed - Best val acc: {best_val_acc:.4f}")
        
        # Create history object compatible with Keras
        class HistoryObject:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return HistoryObject(history)
    
    def train_student_baseline(self, student_model: tf.keras.Model,
                             train_ds: tf.data.Dataset,
                             val_ds: tf.data.Dataset,
                             student_name: str,
                             epochs: int = 10) -> tf.keras.callbacks.History:
        """Train student model without distillation (baseline)."""
        print(f"📚 Training {student_name} student baseline (without distillation)...")
        
        # Add softmax layer for baseline training
        student_with_softmax = tf.keras.Sequential([
            student_model,
            tf.keras.layers.Softmax()
        ])
        
        student_with_softmax.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
        ]
        
        history = student_with_softmax.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Copy weights back to original student model
        student_model.set_weights(student_with_softmax.layers[0].get_weights())
        
        val_loss, val_acc = student_with_softmax.evaluate(val_ds, verbose=0)
        print(f"✅ {student_name.capitalize()} baseline completed - Val accuracy: {val_acc:.4f}")
        
        return history
    
    def evaluate_models(self, models: Dict[str, tf.keras.Model], 
                       test_ds: tf.data.Dataset) -> Dict[str, Dict]:
        """Evaluate all models on test set."""
        print("\n📊 Evaluating all models...")
        
        results = {}
        
        for name, model in models.items():
            print(f"  🔍 Evaluating {name}...")
            
            # For models without softmax, add it temporarily
            if name != 'teacher':
                test_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
            else:
                test_model = model
            
            test_loss, test_acc = test_model.evaluate(test_ds, verbose=0)
            
            results[name] = {
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss),
                'parameters': model.count_params(),
                'model_size_mb': model.count_params() * 4 / (1024 * 1024)
            }
            
            print(f"    ✅ Test accuracy: {test_acc:.4f}")
        
        return results
    
    def benchmark_inference(self, models: Dict[str, tf.keras.Model], 
                          test_input: np.ndarray) -> Dict[str, Dict]:
        """Benchmark inference speed of all models."""
        print("\n⚡ Benchmarking inference performance...")
        
        results = {}
        
        for name, model in models.items():
            print(f"  ⏱️ Benchmarking {name}...")
            
            # Warmup
            for _ in range(10):
                _ = model(test_input, training=False)
            
            # Benchmark
            times = []
            for _ in range(50):
                start_time = time.time()
                _ = model(test_input, training=False)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            
            results[name] = {
                'avg_time_ms': float(avg_time),
                'std_time_ms': float(std_time),
                'throughput_fps': float(1000 / avg_time)
            }
            
            print(f"    ⚡ Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        
        return results
    
    def visualize_results(self, evaluation_results: Dict, benchmark_results: Dict, 
                         training_histories: Dict):
        """Create comprehensive visualization of distillation results."""
        print("\n📊 Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(evaluation_results.keys())
        accuracies = [evaluation_results[model]['test_accuracy'] * 100 for model in models]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = ax1.bar(models, accuracies, color=colors)
        ax1.set_title('Test Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 2. Model size comparison
        ax2 = fig.add_subplot(gs[0, 1])
        sizes = [evaluation_results[model]['model_size_mb'] for model in models]
        
        bars = ax2.bar(models, sizes, color=colors)
        ax2.set_title('Model Size Comparison', fontweight='bold')
        ax2.set_ylabel('Size (MB)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{size:.1f}MB', ha='center', va='bottom')
        
        # 3. Inference time comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if benchmark_results:
            times = [benchmark_results[model]['avg_time_ms'] for model in models if model in benchmark_results]
            model_names = [model for model in models if model in benchmark_results]
            
            bars = ax3.bar(model_names, times, color=colors[:len(times)])
            ax3.set_title('Inference Time Comparison', fontweight='bold')
            ax3.set_ylabel('Time (ms)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, time_val in zip(bars, times):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{time_val:.1f}ms', ha='center', va='bottom')
        
        # 4. Training history
        ax4 = fig.add_subplot(gs[1, :])
        for name, history in training_histories.items():
            if hasattr(history, 'history'):
                if 'val_accuracy' in history.history:
                    epochs = range(1, len(history.history['val_accuracy']) + 1)
                    ax4.plot(epochs, history.history['val_accuracy'], 
                            label=f'{name} - Val Accuracy', marker='o')
        
        ax4.set_title('Training History Comparison', fontweight='bold')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Validation Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Efficiency analysis (Accuracy vs Size)
        ax5 = fig.add_subplot(gs[2, 0])
        for i, model in enumerate(models):
            ax5.scatter(sizes[i], accuracies[i], 
                       s=100, color=colors[i], label=model, alpha=0.7)
            ax5.annotate(model, (sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax5.set_title('Accuracy vs Model Size', fontweight='bold')
        ax5.set_xlabel('Model Size (MB)')
        ax5.set_ylabel('Accuracy (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Speed vs Accuracy trade-off
        ax6 = fig.add_subplot(gs[2, 1])
        if benchmark_results:
            for model in model_names:
                if model in evaluation_results:
                    idx = models.index(model)
                    ax6.scatter(benchmark_results[model]['avg_time_ms'], 
                              evaluation_results[model]['test_accuracy'] * 100,
                              s=100, color=colors[idx], label=model, alpha=0.7)
                    ax6.annotate(model, 
                               (benchmark_results[model]['avg_time_ms'], 
                                evaluation_results[model]['test_accuracy'] * 100),
                               xytext=(5, 5), textcoords='offset points')
        
        ax6.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
        ax6.set_xlabel('Inference Time (ms)')
        ax6.set_ylabel('Accuracy (%)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Compression ratio analysis
        ax7 = fig.add_subplot(gs[2, 2])
        if 'teacher' in evaluation_results:
            teacher_size = evaluation_results['teacher']['model_size_mb']
            compression_ratios = [teacher_size / sizes[i] for i in range(len(models))]
            
            bars = ax7.bar(models, compression_ratios, color=colors)
            ax7.set_title('Compression Ratio vs Teacher', fontweight='bold')
            ax7.set_ylabel('Compression Ratio')
            ax7.tick_params(axis='x', rotation=45)
            
            for bar, ratio in zip(bars, compression_ratios):
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{ratio:.1f}x', ha='center', va='bottom')
        
        plt.suptitle('Knowledge Distillation Results Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = 'distillation_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved: {output_path}")
        plt.show()
    
    def run_complete_demo(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                         num_classes: int = 10,
                         teacher_epochs: int = 15,
                         student_epochs: int = 10):
        """Run complete knowledge distillation demonstration."""
        print("🎯 TensorFlow Knowledge Distillation Demo")
        print("=" * 60)
        
        # Create models
        self.teacher_model = self.create_teacher_model(input_shape, num_classes)
        student_models = self.create_student_models(input_shape, num_classes)
        
        # Create data
        train_ds, val_ds, test_ds = self.create_synthetic_data(input_shape, num_classes)
        
        # Train teacher
        teacher_history = self.train_teacher(self.teacher_model, train_ds, val_ds, teacher_epochs)
        self.training_histories['teacher'] = teacher_history
        
        # Train students with and without distillation
        all_models = {'teacher': self.teacher_model}
        
        for name, student_model in student_models.items():
            print(f"\n--- Training {name} student ---")
            
            # Clone student for baseline training
            baseline_student = tf.keras.models.clone_model(student_model)
            baseline_student.set_weights(student_model.get_weights())
            
            # Train with distillation
            distill_history = self.train_student_with_distillation(
                student_model, self.teacher_model, train_ds, val_ds, name, student_epochs
            )
            self.training_histories[f'{name}_distilled'] = distill_history
            
            # Train baseline (without distillation)
            baseline_history = self.train_student_baseline(
                baseline_student, train_ds, val_ds, f'{name}_baseline', student_epochs
            )
            self.training_histories[f'{name}_baseline'] = baseline_history
            
            # Store models
            all_models[f'{name}_distilled'] = student_model
            all_models[f'{name}_baseline'] = baseline_student
        
        # Evaluate all models
        evaluation_results = self.evaluate_models(all_models, test_ds)
        
        # Benchmark inference
        test_input = np.random.random((1,) + input_shape).astype(np.float32)
        benchmark_results = self.benchmark_inference(all_models, test_input)
        
        # Visualize results
        self.visualize_results(evaluation_results, benchmark_results, self.training_histories)
        
        # Print summary
        self.print_summary(evaluation_results)
        
        print("🎉 Knowledge distillation demo completed successfully!")
        return all_models, evaluation_results, benchmark_results
    
    def print_summary(self, evaluation_results: Dict):
        """Print formatted summary of results."""
        print("\n📋 Knowledge Distillation Results Summary")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Parameters':<12} {'Size (MB)':<10} {'Compression':<12}")
        print("-" * 80)
        
        teacher_params = evaluation_results['teacher']['parameters']
        
        for model, results in evaluation_results.items():
            compression = f"{teacher_params / results['parameters']:.1f}x" if model != 'teacher' else "1.0x"
            print(f"{model:<20} {results['test_accuracy']:<10.3f} "
                  f"{results['parameters']:<12,} {results['model_size_mb']:<10.1f} {compression:<12}")


def main():
    """Main function for distillation demo."""
    parser = argparse.ArgumentParser(description='TensorFlow Knowledge Distillation Demo')
    parser.add_argument('--teacher-epochs', type=int, default=15, help='Teacher training epochs')
    parser.add_argument('--student-epochs', type=int, default=10, help='Student training epochs')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--input-height', type=int, default=224, help='Input image height')
    parser.add_argument('--input-width', type=int, default=224, help='Input image width')
    parser.add_argument('--input-channels', type=int, default=3, help='Input image channels')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    
    args = parser.parse_args()
    
    # Run demo
    demo = DistillationDemo(alpha=args.alpha, temperature=args.temperature)
    input_shape = (args.input_height, args.input_width, args.input_channels)
    
    try:
        demo.run_complete_demo(
            input_shape, 
            args.num_classes, 
            args.teacher_epochs, 
            args.student_epochs
        )
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()