# Location: /examples/optimization_examples/quantization_demo.py

"""
TensorFlow model quantization demonstration.
Shows post-training quantization, quantization-aware training, and performance comparison.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
from typing import Tuple, Dict, List, Optional, Any
import json

# Import TensorVerseHub utilities
try:
    from src.data_utils import create_image_classification_pipeline
    from src.model_utils import ModelBuilders
    from src.optimization_utils import ModelQuantization
    from src.visualization import setup_plotting_style
except ImportError:
    print("Warning: TensorVerseHub modules not found. Using standalone implementation.")


class QuantizationDemo:
    """Comprehensive quantization demonstration."""
    
    def __init__(self):
        """Initialize quantization demo."""
        self.original_model = None
        self.quantized_models = {}
        self.performance_metrics = {}
        
        # Setup plotting
        try:
            setup_plotting_style()
        except:
            plt.style.use('default')
    
    def create_demo_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                         num_classes: int = 10) -> tf.keras.Model:
        """Create a demo CNN model for quantization."""
        print("🏗️ Creating demo CNN model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def create_synthetic_data(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 10, 
                            num_samples: int = 1000) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create synthetic dataset for demonstration."""
        print("📊 Creating synthetic dataset...")
        
        # Generate random images and labels
        images = tf.random.normal([num_samples] + list(input_shape))
        labels = tf.random.uniform([num_samples], 0, num_classes, dtype=tf.int32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        # Split train/test
        train_size = int(0.8 * num_samples)
        train_ds = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
        
        print(f"✅ Dataset created: {train_size} train, {num_samples - train_size} test samples")
        return train_ds, test_ds
    
    def train_model(self, model: tf.keras.Model, 
                   train_ds: tf.data.Dataset, 
                   test_ds: tf.data.Dataset,
                   epochs: int = 5) -> tf.keras.callbacks.History:
        """Train the model quickly for demonstration."""
        print(f"🚀 Training model for {epochs} epochs...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=1)
        ]
        
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final performance
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"✅ Training completed - Test accuracy: {test_acc:.4f}")
        
        return history
    
    def demonstrate_post_training_quantization(self, model: tf.keras.Model,
                                             representative_ds: tf.data.Dataset) -> Dict[str, Any]:
        """Demonstrate different post-training quantization techniques."""
        print("\n🔧 Demonstrating Post-Training Quantization")
        print("=" * 50)
        
        quantization_results = {}
        
        # 1. Dynamic Range Quantization (FP32 -> FP16)
        print("1️⃣ Dynamic Range Quantization (FP16)...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_fp16 = converter.convert()
            
            quantization_results['fp16'] = {
                'model': tflite_fp16,
                'size_bytes': len(tflite_fp16),
                'size_mb': len(tflite_fp16) / (1024 * 1024),
                'type': 'FP16'
            }
            print(f"   ✅ FP16 model size: {quantization_results['fp16']['size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ FP16 quantization failed: {e}")
            quantization_results['fp16'] = {'error': str(e)}
        
        # 2. Integer Quantization (FP32 -> INT8)
        print("2️⃣ Integer Quantization (INT8)...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            # Representative dataset for calibration
            def representative_data_gen():
                for input_value in representative_ds.take(100):
                    if isinstance(input_value, tuple):
                        yield [tf.cast(input_value[0], tf.float32)]
                    else:
                        yield [tf.cast(input_value, tf.float32)]
            
            converter.representative_dataset = representative_data_gen
            tflite_int8 = converter.convert()
            
            quantization_results['int8'] = {
                'model': tflite_int8,
                'size_bytes': len(tflite_int8),
                'size_mb': len(tflite_int8) / (1024 * 1024),
                'type': 'INT8'
            }
            print(f"   ✅ INT8 model size: {quantization_results['int8']['size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ INT8 quantization failed: {e}")
            quantization_results['int8'] = {'error': str(e)}
        
        # 3. Full Integer Quantization
        print("3️⃣ Full Integer Quantization...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            def representative_data_gen():
                for input_value in representative_ds.take(100):
                    if isinstance(input_value, tuple):
                        yield [tf.cast(input_value[0], tf.float32)]
                    else:
                        yield [tf.cast(input_value, tf.float32)]
            
            converter.representative_dataset = representative_data_gen
            tflite_full_int8 = converter.convert()
            
            quantization_results['full_int8'] = {
                'model': tflite_full_int8,
                'size_bytes': len(tflite_full_int8),
                'size_mb': len(tflite_full_int8) / (1024 * 1024),
                'type': 'Full INT8'
            }
            print(f"   ✅ Full INT8 model size: {quantization_results['full_int8']['size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ Full INT8 quantization failed: {e}")
            quantization_results['full_int8'] = {'error': str(e)}
        
        return quantization_results
    
    def demonstrate_qat(self, model: tf.keras.Model,
                       train_ds: tf.data.Dataset,
                       test_ds: tf.data.Dataset) -> tf.keras.Model:
        """Demonstrate Quantization-Aware Training."""
        print("\n🎓 Demonstrating Quantization-Aware Training")
        print("=" * 50)
        
        try:
            import tensorflow_model_optimization as tfmot
            
            # Apply QAT
            quantize_model = tfmot.quantization.keras.quantize_model
            qat_model = quantize_model(model)
            
            print("📊 QAT Model Summary:")
            qat_model.summary()
            
            # Compile QAT model
            qat_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("🚀 Fine-tuning with QAT (2 epochs)...")
            qat_model.fit(
                train_ds,
                epochs=2,
                validation_data=test_ds,
                verbose=1
            )
            
            # Evaluate QAT model
            qat_loss, qat_acc = qat_model.evaluate(test_ds, verbose=0)
            print(f"✅ QAT model accuracy: {qat_acc:.4f}")
            
            return qat_model
            
        except ImportError:
            print("❌ TensorFlow Model Optimization not available")
            print("   Install with: pip install tensorflow-model-optimization")
            return None
        except Exception as e:
            print(f"❌ QAT failed: {e}")
            return None
    
    def benchmark_models(self, models: Dict[str, Any], test_input: np.ndarray) -> Dict[str, Dict]:
        """Benchmark different quantized models."""
        print("\n⚡ Benchmarking Model Performance")
        print("=" * 50)
        
        results = {}
        
        # Original model benchmark
        if self.original_model:
            print("🔵 Benchmarking original model...")
            start_time = time.time()
            for _ in range(10):
                _ = self.original_model.predict(test_input, verbose=0)
            orig_time = (time.time() - start_time) / 10
            
            results['original'] = {
                'avg_time_ms': orig_time * 1000,
                'size_mb': self.original_model.count_params() * 4 / (1024 * 1024),
                'accuracy': 'baseline'
            }
            print(f"   ⏱️ Average time: {orig_time * 1000:.2f} ms")
        
        # TFLite models benchmark
        for name, model_data in models.items():
            if 'error' in model_data:
                continue
                
            print(f"🔶 Benchmarking {name} model...")
            try:
                # Load TFLite interpreter
                interpreter = tf.lite.Interpreter(model_content=model_data['model'])
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Benchmark inference
                start_time = time.time()
                for _ in range(10):
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details[0]['index'])
                
                avg_time = (time.time() - start_time) / 10
                
                results[name] = {
                    'avg_time_ms': avg_time * 1000,
                    'size_mb': model_data['size_mb'],
                    'compression_ratio': results['original']['size_mb'] / model_data['size_mb'] if 'original' in results else 1,
                    'speedup': results['original']['avg_time_ms'] / (avg_time * 1000) if 'original' in results else 1
                }
                
                print(f"   ⏱️ Average time: {avg_time * 1000:.2f} ms")
                print(f"   📦 Size: {model_data['size_mb']:.2f} MB")
                print(f"   🚀 Speedup: {results[name]['speedup']:.2f}x")
                
            except Exception as e:
                print(f"   ❌ Benchmark failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def visualize_results(self, quantization_results: Dict, benchmark_results: Dict):
        """Create visualization of quantization results."""
        print("\n📊 Creating Visualizations")
        print("=" * 50)
        
        # Prepare data for plotting
        model_names = []
        sizes = []
        times = []
        
        for name, result in benchmark_results.items():
            if 'error' not in result:
                model_names.append(name.replace('_', ' ').title())
                sizes.append(result['size_mb'])
                times.append(result['avg_time_ms'])
        
        if len(model_names) < 2:
            print("⚠️ Not enough data for visualization")
            return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model sizes comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars1 = ax1.bar(model_names, sizes, color=colors)
        ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Size (MB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars1, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{size:.1f}MB', ha='center', va='bottom')
        
        # Inference time comparison
        bars2 = ax2.bar(model_names, times, color=colors)
        ax2.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Time (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_ms in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{time_ms:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'quantization_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved: {output_path}")
        plt.show()
        
        # Print summary table
        self.print_summary_table(benchmark_results)
    
    def print_summary_table(self, results: Dict):
        """Print formatted summary table."""
        print("\n📋 Quantization Results Summary")
        print("=" * 80)
        print(f"{'Model':<15} {'Size (MB)':<12} {'Time (ms)':<12} {'Compression':<12} {'Speedup':<10}")
        print("-" * 80)
        
        for name, result in results.items():
            if 'error' not in result:
                compression = f"{result.get('compression_ratio', 1):.1f}x" if result.get('compression_ratio') else "N/A"
                speedup = f"{result.get('speedup', 1):.1f}x" if result.get('speedup') else "N/A"
                
                print(f"{name.replace('_', ' ').title():<15} "
                      f"{result['size_mb']:<12.1f} "
                      f"{result['avg_time_ms']:<12.1f} "
                      f"{compression:<12} "
                      f"{speedup:<10}")
    
    def save_results(self, quantization_results: Dict, benchmark_results: Dict, output_dir: str = "quantization_output"):
        """Save all results to files."""
        print(f"\n💾 Saving results to {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save TFLite models
        for name, result in quantization_results.items():
            if 'model' in result:
                model_path = os.path.join(output_dir, f"{name}_model.tflite")
                with open(model_path, 'wb') as f:
                    f.write(result['model'])
                print(f"   💾 {name} model saved: {model_path}")
        
        # Save benchmark results
        results_path = os.path.join(output_dir, "benchmark_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in benchmark_results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in value.items()}
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        print(f"   📊 Benchmark results saved: {results_path}")
    
    def run_complete_demo(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                         num_classes: int = 10, 
                         training_epochs: int = 5):
        """Run the complete quantization demonstration."""
        print("🎯 TensorFlow Model Quantization Demo")
        print("=" * 60)
        
        # Create and train model
        self.original_model = self.create_demo_model(input_shape, num_classes)
        train_ds, test_ds = self.create_synthetic_data(input_shape, num_classes)
        self.train_model(self.original_model, train_ds, test_ds, training_epochs)
        
        # Demonstrate quantization techniques
        quantization_results = self.demonstrate_post_training_quantization(self.original_model, train_ds)
        
        # Demonstrate QAT (optional)
        qat_model = self.demonstrate_qat(self.original_model, train_ds, test_ds)
        if qat_model:
            # Convert QAT model to TFLite
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                qat_tflite = converter.convert()
                
                quantization_results['qat'] = {
                    'model': qat_tflite,
                    'size_bytes': len(qat_tflite),
                    'size_mb': len(qat_tflite) / (1024 * 1024),
                    'type': 'QAT'
                }
                print(f"✅ QAT TFLite size: {quantization_results['qat']['size_mb']:.2f} MB")
            except Exception as e:
                print(f"❌ QAT TFLite conversion failed: {e}")
        
        # Benchmark all models
        test_input = np.random.random((1,) + input_shape).astype(np.float32)
        benchmark_results = self.benchmark_models(quantization_results, test_input)
        
        # Visualize results
        self.visualize_results(quantization_results, benchmark_results)
        
        # Save results
        self.save_results(quantization_results, benchmark_results)
        
        print("\n🎉 Quantization demo completed successfully!")
        return quantization_results, benchmark_results


def main():
    """Main function for quantization demo."""
    parser = argparse.ArgumentParser(description='TensorFlow Model Quantization Demo')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--input-height', type=int, default=224, help='Input image height')
    parser.add_argument('--input-width', type=int, default=224, help='Input image width')
    parser.add_argument('--input-channels', type=int, default=3, help='Input image channels')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--output-dir', type=str, default='quantization_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Run demo
    demo = QuantizationDemo()
    input_shape = (args.input_height, args.input_width, args.input_channels)
    
    try:
        demo.run_complete_demo(input_shape, args.num_classes, args.epochs)
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()