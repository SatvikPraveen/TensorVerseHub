# Location: /examples/optimization_examples/pruning_demo.py

"""
TensorFlow model pruning demonstration.
Shows magnitude-based pruning, structured pruning, and performance analysis.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import os
from typing import Tuple, Dict, List, Optional, Any
import json

# Import TensorFlow Model Optimization
try:
    import tensorflow_model_optimization as tfmot
    TF_MOT_AVAILABLE = True
except ImportError:
    TF_MOT_AVAILABLE = False
    print("⚠️ TensorFlow Model Optimization not available")
    print("   Install with: pip install tensorflow-model-optimization")

# Import TensorVerseHub utilities
try:
    from src.model_utils import ModelBuilders
    from src.visualization import setup_plotting_style
    from src.optimization_utils import ModelPruning
except ImportError:
    print("Warning: TensorVerseHub modules not found. Using standalone implementation.")


class PruningDemo:
    """Comprehensive model pruning demonstration."""
    
    def __init__(self):
        """Initialize pruning demo."""
        self.original_model = None
        self.pruned_models = {}
        self.pruning_history = {}
        self.sparsity_analysis = {}
        
        # Setup plotting
        try:
            setup_plotting_style()
        except:
            plt.style.use('default')
            sns.set_palette("husl")
    
    def create_demo_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                         num_classes: int = 10) -> tf.keras.Model:
        """Create a demo CNN model for pruning."""
        print("🏗️ Creating demo CNN model for pruning...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_shape, name='conv1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(name='pool1'),
            
            tf.keras.layers.Conv2D(128, 3, activation='relu', name='conv2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(name='pool2'),
            
            tf.keras.layers.Conv2D(256, 3, activation='relu', name='conv3'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            tf.keras.layers.Dense(512, activation='relu', name='dense1'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', name='dense2'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        return model
    
    def create_synthetic_data(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 10,
                            num_samples: int = 1200) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
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
        
        print(f"✅ Dataset created: {train_size} train, {val_size} val, {num_samples - train_size - val_size} test samples")
        return train_ds, val_ds, test_ds
    
    def train_baseline_model(self, model: tf.keras.Model,
                           train_ds: tf.data.Dataset,
                           val_ds: tf.data.Dataset,
                           epochs: int = 10) -> tf.keras.callbacks.History:
        """Train baseline model."""
        print(f"🚀 Training baseline model for {epochs} epochs...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=3, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, 
                patience=2,
                monitor='val_loss'
            )
        ]
        
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate baseline
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"✅ Baseline training completed - Val accuracy: {val_acc:.4f}")
        
        return history
    
    def analyze_weight_distribution(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Analyze weight distributions before pruning."""
        print("📊 Analyzing weight distributions...")
        
        layer_stats = {}
        overall_stats = []
        
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weights = layer.kernel.numpy().flatten()
                overall_stats.extend(weights.tolist())
                
                layer_stats[layer.name] = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'zero_fraction': float(np.mean(weights == 0)),
                    'near_zero_fraction': float(np.mean(np.abs(weights) < 0.01)),
                    'total_params': int(len(weights))
                }
        
        overall_stats = np.array(overall_stats)
        analysis = {
            'layer_stats': layer_stats,
            'overall': {
                'mean': float(np.mean(overall_stats)),
                'std': float(np.std(overall_stats)),
                'zero_fraction': float(np.mean(overall_stats == 0)),
                'near_zero_fraction': float(np.mean(np.abs(overall_stats) < 0.01)),
                'total_params': int(len(overall_stats))
            }
        }
        
        print(f"   📈 Overall stats: mean={analysis['overall']['mean']:.4f}, std={analysis['overall']['std']:.4f}")
        print(f"   🕳️ Zero weights: {analysis['overall']['zero_fraction']*100:.2f}%")
        print(f"   📉 Near-zero weights: {analysis['overall']['near_zero_fraction']*100:.2f}%")
        
        return analysis
    
    def demonstrate_magnitude_pruning(self, model: tf.keras.Model,
                                    train_ds: tf.data.Dataset,
                                    val_ds: tf.data.Dataset,
                                    target_sparsity: float = 0.5) -> tf.keras.Model:
        """Demonstrate magnitude-based pruning."""
        print(f"\n✂️ Demonstrating Magnitude-Based Pruning (target: {target_sparsity*100:.0f}% sparsity)")
        print("=" * 70)
        
        if not TF_MOT_AVAILABLE:
            print("❌ TensorFlow Model Optimization not available")
            return None
        
        try:
            # Create pruning schedule
            end_step = 5 * len(train_ds)  # Prune over 5 epochs
            
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=end_step,
                frequency=100
            )
            
            print(f"📅 Pruning schedule: 0% -> {target_sparsity*100:.0f}% over {end_step} steps")
            
            # Apply pruning to model
            def apply_pruning_to_layer(layer):
                # Only prune dense and conv layers
                if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                    return tfmot.sparsity.keras.prune_low_magnitude(
                        layer, pruning_schedule=pruning_schedule
                    )
                return layer
            
            pruned_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_layer
            )
            
            # Copy weights
            pruned_model.set_weights(model.get_weights())
            
            # Compile pruned model
            pruned_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("🏗️ Pruned model architecture:")
            pruned_model.summary()
            
            # Create pruning callbacks
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir='pruning_logs'),
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
            
            # Fine-tune with pruning
            print("🔧 Fine-tuning with pruning...")
            pruning_history = pruned_model.fit(
                train_ds,
                epochs=5,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate pruned model
            pruned_loss, pruned_acc = pruned_model.evaluate(val_ds, verbose=0)
            print(f"✅ Pruned model accuracy: {pruned_acc:.4f}")
            
            # Store results
            self.pruning_history['magnitude'] = pruning_history
            
            return pruned_model
            
        except Exception as e:
            print(f"❌ Magnitude pruning failed: {e}")
            return None
    
    def demonstrate_structured_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """Demonstrate structured pruning (channel/filter removal)."""
        print("\n🏗️ Demonstrating Structured Pruning")
        print("=" * 50)
        
        try:
            # Simple structured pruning: remove channels based on L1 norm
            print("📊 Analyzing filter importance...")
            
            structured_model = tf.keras.models.clone_model(model)
            structured_model.set_weights(model.get_weights())
            
            # For demonstration, we'll simulate structured pruning by
            # reducing the number of filters in convolutional layers
            print("⚠️ Note: This is a simplified structured pruning demonstration")
            print("   In practice, you would use specialized tools or implement")
            print("   more sophisticated filter importance metrics.")
            
            # Create a new model with reduced filters
            reduced_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=model.input_shape[1:]),  # 64->32
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                
                tf.keras.layers.Conv2D(64, 3, activation='relu'),  # 128->64
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                
                tf.keras.layers.Conv2D(128, 3, activation='relu'),  # 256->128
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GlobalAveragePooling2D(),
                
                tf.keras.layers.Dense(256, activation='relu'),  # 512->256
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),  # 256->128
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(model.output_shape[-1], activation='softmax')
            ])
            
            reduced_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"📉 Reduced model parameters: {reduced_model.count_params():,}")
            print(f"📊 Parameter reduction: {(1 - reduced_model.count_params()/model.count_params())*100:.1f}%")
            
            return reduced_model
            
        except Exception as e:
            print(f"❌ Structured pruning failed: {e}")
            return None
    
    def analyze_sparsity(self, model: tf.keras.Model, model_name: str) -> Dict[str, Any]:
        """Analyze sparsity patterns in a model."""
        print(f"🔍 Analyzing sparsity in {model_name} model...")
        
        layer_sparsity = {}
        total_params = 0
        total_zeros = 0
        
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weights = layer.kernel.numpy()
                layer_params = weights.size
                layer_zeros = np.sum(weights == 0)
                layer_sparsity_ratio = layer_zeros / layer_params
                
                layer_sparsity[layer.name] = {
                    'total_params': int(layer_params),
                    'zero_params': int(layer_zeros),
                    'sparsity_ratio': float(layer_sparsity_ratio),
                    'density_ratio': float(1 - layer_sparsity_ratio)
                }
                
                total_params += layer_params
                total_zeros += layer_zeros
        
        overall_sparsity = total_zeros / total_params if total_params > 0 else 0
        
        analysis = {
            'layer_sparsity': layer_sparsity,
            'overall_sparsity': float(overall_sparsity),
            'overall_density': float(1 - overall_sparsity),
            'total_parameters': int(total_params),
            'zero_parameters': int(total_zeros)
        }
        
        print(f"   📊 Overall sparsity: {overall_sparsity*100:.2f}%")
        print(f"   🎯 Effective parameters: {total_params - total_zeros:,} / {total_params:,}")
        
        return analysis
    
    def benchmark_models(self, models: Dict[str, tf.keras.Model], test_input: np.ndarray) -> Dict[str, Dict]:
        """Benchmark inference performance of different models."""
        print("\n⚡ Benchmarking Model Performance")
        print("=" * 50)
        
        results = {}
        
        for name, model in models.items():
            if model is None:
                continue
                
            print(f"🔵 Benchmarking {name} model...")
            
            # Warmup
            for _ in range(5):
                _ = model.predict(test_input, verbose=0)
            
            # Benchmark
            start_time = time.time()
            for _ in range(20):
                _ = model.predict(test_input, verbose=0)
            avg_time = (time.time() - start_time) / 20
            
            # Memory usage (approximate)
            model_size_mb = model.count_params() * 4 / (1024 * 1024)
            
            results[name] = {
                'avg_time_ms': avg_time * 1000,
                'parameters': model.count_params(),
                'model_size_mb': model_size_mb,
                'throughput_fps': 1000 / (avg_time * 1000)
            }
            
            print(f"   ⏱️ Average time: {avg_time * 1000:.2f} ms")
            print(f"   📦 Parameters: {model.count_params():,}")
            print(f"   📊 Throughput: {results[name]['throughput_fps']:.1f} FPS")
        
        return results
    
    def visualize_pruning_results(self, sparsity_analyses: Dict, benchmark_results: Dict):
        """Create comprehensive visualizations of pruning results."""
        print("\n📊 Creating Pruning Visualizations")
        print("=" * 50)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sparsity comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(sparsity_analyses.keys())
        sparsities = [sparsity_analyses[model]['overall_sparsity'] * 100 for model in models]
        
        bars = ax1.bar(models, sparsities, color=['blue', 'orange', 'green'][:len(models)])
        ax1.set_title('Model Sparsity Comparison', fontweight='bold')
        ax1.set_ylabel('Sparsity (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, sparsity in zip(bars, sparsities):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{sparsity:.1f}%', ha='center', va='bottom')
        
        # 2. Parameter count comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if benchmark_results:
            models_bench = list(benchmark_results.keys())
            params = [benchmark_results[model]['parameters'] / 1000 for model in models_bench]  # in thousands
            
            bars = ax2.bar(models_bench, params, color=['blue', 'orange', 'green'][:len(models_bench)])
            ax2.set_title('Parameter Count Comparison', fontweight='bold')
            ax2.set_ylabel('Parameters (K)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, param in zip(bars, params):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                        f'{param:.0f}K', ha='center', va='bottom')
        
        # 3. Inference time comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if benchmark_results:
            times = [benchmark_results[model]['avg_time_ms'] for model in models_bench]
            
            bars = ax3.bar(models_bench, times, color=['blue', 'orange', 'green'][:len(models_bench)])
            ax3.set_title('Inference Time Comparison', fontweight='bold')
            ax3.set_ylabel('Time (ms)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, time_ms in zip(bars, times):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{time_ms:.1f}ms', ha='center', va='bottom')
        
        # 4. Layer-wise sparsity heatmap
        if len(sparsity_analyses) > 1:
            ax4 = fig.add_subplot(gs[1, :2])
            
            # Create sparsity matrix
            all_layers = set()
            for analysis in sparsity_analyses.values():
                all_layers.update(analysis['layer_sparsity'].keys())
            
            sparsity_matrix = []
            layer_names = sorted(list(all_layers))
            
            for model in models:
                model_sparsity = []
                for layer in layer_names:
                    if layer in sparsity_analyses[model]['layer_sparsity']:
                        model_sparsity.append(sparsity_analyses[model]['layer_sparsity'][layer]['sparsity_ratio'])
                    else:
                        model_sparsity.append(0)
                sparsity_matrix.append(model_sparsity)
            
            if sparsity_matrix:
                im = ax4.imshow(sparsity_matrix, cmap='YlOrRd', aspect='auto')
                ax4.set_title('Layer-wise Sparsity Heatmap', fontweight='bold')
                ax4.set_xlabel('Layers')
                ax4.set_ylabel('Models')
                ax4.set_yticks(range(len(models)))
                ax4.set_yticklabels(models)
                ax4.set_xticks(range(len(layer_names)))
                ax4.set_xticklabels(layer_names, rotation=45, ha='right')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Sparsity Ratio')
        
        # 5. Training history (if available)
        if hasattr(self, 'pruning_history') and self.pruning_history:
            ax5 = fig.add_subplot(gs[2, :])
            
            for name, history in self.pruning_history.items():
                if 'val_accuracy' in history.history:
                    epochs = range(1, len(history.history['val_accuracy']) + 1)
                    ax5.plot(epochs, history.history['val_accuracy'], 
                            label=f'{name} - Val Accuracy', marker='o')
                    ax5.plot(epochs, history.history['accuracy'], 
                            label=f'{name} - Train Accuracy', linestyle='--')
            
            ax5.set_title('Pruning Training History', fontweight='bold')
            ax5.set_xlabel('Epochs')
            ax5.set_ylabel('Accuracy')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('TensorFlow Model Pruning Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = 'pruning_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved: {output_path}")
        plt.show()
    
    def save_results(self, models: Dict[str, tf.keras.Model],
                    sparsity_analyses: Dict, 
                    benchmark_results: Dict,
                    output_dir: str = "pruning_output"):
        """Save all pruning results."""
        print(f"\n💾 Saving results to {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in models.items():
            if model is not None:
                model_path = os.path.join(output_dir, f"{name}_model")
                model.save(model_path, save_format='tf')
                print(f"   💾 {name} model saved: {model_path}")
        
        # Save analyses
        results = {
            'sparsity_analyses': sparsity_analyses,
            'benchmark_results': benchmark_results,
            'summary': {
                'models_analyzed': list(models.keys()),
                'pruning_techniques': ['magnitude_based', 'structured'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        results_path = os.path.join(output_dir, "pruning_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   📊 Analysis results saved: {results_path}")
    
    def run_complete_demo(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                         num_classes: int = 10,
                         training_epochs: int = 10,
                         target_sparsity: float = 0.7):
        """Run complete pruning demonstration."""
        print("🎯 TensorFlow Model Pruning Demo")
        print("=" * 60)
        
        # Create and train baseline model
        self.original_model = self.create_demo_model(input_shape, num_classes)
        train_ds, val_ds, test_ds = self.create_synthetic_data(input_shape, num_classes)
        
        baseline_history = self.train_baseline_model(self.original_model, train_ds, val_ds, training_epochs)
        
        # Analyze baseline model
        baseline_analysis = self.analyze_weight_distribution(self.original_model)
        
        # Demonstrate pruning techniques
        models = {'original': self.original_model}
        
        # Magnitude-based pruning
        pruned_model = self.demonstrate_magnitude_pruning(self.original_model, train_ds, val_ds, target_sparsity)
        if pruned_model is not None:
            # Strip pruning wrappers for final model
            if TF_MOT_AVAILABLE:
                final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
                models['magnitude_pruned'] = final_pruned_model
            else:
                models['magnitude_pruned'] = pruned_model
        
        # Structured pruning
        structured_model = self.demonstrate_structured_pruning(self.original_model)
        if structured_model is not None:
            models['structured_pruned'] = structured_model
        
        # Analyze sparsity for all models
        sparsity_analyses = {}
        for name, model in models.items():
            sparsity_analyses[name] = self.analyze_sparsity(model, name)
        
        # Benchmark models
        test_input = np.random.random((1,) + input_shape).astype(np.float32)
        benchmark_results = self.benchmark_models(models, test_input)
        
        # Visualize results
        self.visualize_pruning_results(sparsity_analyses, benchmark_results)
        
        # Save results
        self.save_results(models, sparsity_analyses, benchmark_results)
        
        # Print summary
        print("\n📋 Pruning Demo Summary")
        print("=" * 50)
        for name, analysis in sparsity_analyses.items():
            print(f"{name.replace('_', ' ').title()}:")
            print(f"  Parameters: {analysis['total_parameters']:,}")
            print(f"  Sparsity: {analysis['overall_sparsity']*100:.1f}%")
            if name in benchmark_results:
                print(f"  Inference: {benchmark_results[name]['avg_time_ms']:.1f} ms")
            print()
        
        print("🎉 Pruning demo completed successfully!")
        return models, sparsity_analyses, benchmark_results


def main():
    """Main function for pruning demo."""
    parser = argparse.ArgumentParser(description='TensorFlow Model Pruning Demo')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--target-sparsity', type=float, default=0.7, help='Target sparsity ratio')
    parser.add_argument('--input-height', type=int, default=224, help='Input image height')
    parser.add_argument('--input-width', type=int, default=224, help='Input image width')
    parser.add_argument('--input-channels', type=int, default=3, help='Input image channels')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--output-dir', type=str, default='pruning_output', help='Output directory')
    
    args = parser.parse_args()
    
    if not TF_MOT_AVAILABLE:
        print("❌ TensorFlow Model Optimization required for full demo")
        print("   Install with: pip install tensorflow-model-optimization")
        return
    
    # Run demo
    demo = PruningDemo()
    input_shape = (args.input_height, args.input_width, args.input_channels)
    
    try:
        demo.run_complete_demo(
            input_shape, 
            args.num_classes, 
            args.epochs, 
            args.target_sparsity
        )
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()