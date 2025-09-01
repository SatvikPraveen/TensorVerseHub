# Location: /scripts/train_models.py

"""
Automated model training pipeline for TensorVerseHub.
Supports training multiple model types with different configurations.
"""

import tensorflow as tf
import numpy as np
import argparse
import json
import os
import time
from typing import Dict, Any, List, Tuple, Optional
import yaml
from pathlib import Path

# Import TensorVerseHub utilities
try:
    from src.data_utils import DataPipeline, create_image_classification_pipeline
    from src.model_utils import ModelBuilders, TrainingUtilities, create_classification_model
    from src.visualization import TrainingVisualization, setup_plotting_style
    from src.optimization_utils import ModelQuantization, ModelPruning
    from src.export_utils import quick_export
except ImportError as e:
    print(f"Warning: Could not import TensorVerseHub modules: {e}")
    print("Make sure you've installed the package with: pip install -e .")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON or YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")

def create_synthetic_data(config: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create synthetic dataset for testing."""
    print("📊 Creating synthetic dataset...")
    
    data_config = config.get('data', {})
    batch_size = data_config.get('batch_size', 32)
    
    if config['model_type'] == 'image_classification':
        # Create synthetic image data
        input_shape = tuple(config['model'].get('input_shape', [224, 224, 3]))
        num_classes = config['model'].get('num_classes', 10)
        num_samples = data_config.get('num_samples', 1000)
        
        # Generate random images and labels
        images = tf.random.normal([num_samples] + list(input_shape))
        labels = tf.random.uniform([num_samples], 0, num_classes, dtype=tf.int32)
        
        # Create datasets
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        # Split into train/validation
        train_size = int(0.8 * num_samples)
        train_ds = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    elif config['model_type'] == 'text_classification':
        # Create synthetic text data
        vocab_size = config['model'].get('vocab_size', 10000)
        max_length = config['model'].get('max_length', 128)
        num_classes = config['model'].get('num_classes', 5)
        num_samples = data_config.get('num_samples', 1000)
        
        # Generate random sequences and labels
        sequences = tf.random.uniform([num_samples, max_length], 0, vocab_size, dtype=tf.int32)
        labels = tf.random.uniform([num_samples], 0, num_classes, dtype=tf.int32)
        
        dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
        
        train_size = int(0.8 * num_samples)
        train_ds = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")
    
    print(f"✅ Created synthetic dataset: {train_size} train, {num_samples - train_size} val samples")
    return train_ds, val_ds

def build_model(config: Dict[str, Any]) -> tf.keras.Model:
    """Build model based on configuration."""
    print(f"🏗️ Building {config['model_type']} model...")
    
    model_config = config['model']
    
    if config['model_type'] == 'image_classification':
        model = ModelBuilders.create_cnn_classifier(
            input_shape=tuple(model_config['input_shape']),
            num_classes=model_config['num_classes'],
            architecture=model_config.get('architecture', 'simple'),
            dropout_rate=model_config.get('dropout_rate', 0.5)
        )
    elif config['model_type'] == 'text_classification':
        model = ModelBuilders.create_text_classifier(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config.get('embedding_dim', 128),
            max_length=model_config['max_length'],
            num_classes=model_config['num_classes'],
            architecture=model_config.get('architecture', 'lstm')
        )
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")
    
    # Compile model
    compile_config = config.get('compile', {})
    model.compile(
        optimizer=compile_config.get('optimizer', 'adam'),
        loss=compile_config.get('loss', 'sparse_categorical_crossentropy'),
        metrics=compile_config.get('metrics', ['accuracy'])
    )
    
    print("✅ Model built and compiled successfully")
    return model

def train_model(model: tf.keras.Model, 
                train_ds: tf.data.Dataset, 
                val_ds: tf.data.Dataset,
                config: Dict[str, Any]) -> tf.keras.callbacks.History:
    """Train the model with specified configuration."""
    print("🚀 Starting model training...")
    
    training_config = config.get('training', {})
    model_name = config.get('name', 'model')
    
    # Create callbacks
    callbacks = TrainingUtilities.create_callbacks(
        model_name=model_name,
        patience=training_config.get('patience', 10),
        reduce_lr=training_config.get('reduce_lr', True),
        tensorboard=training_config.get('tensorboard', True)
    )
    
    # Additional callbacks
    if training_config.get('custom_callback', False):
        custom_callback = TrainingUtilities.CustomCallback(val_ds)
        callbacks.append(custom_callback)
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=training_config.get('epochs', 10),
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=training_config.get('verbose', 1)
    )
    
    print("✅ Training completed successfully")
    return history

def evaluate_model(model: tf.keras.Model, 
                  test_ds: tf.data.Dataset, 
                  config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate model performance."""
    print("📊 Evaluating model...")
    
    # Evaluate on test data
    results = model.evaluate(test_ds, verbose=0)
    
    # Create results dictionary
    eval_results = {}
    for i, metric_name in enumerate(model.metrics_names):
        eval_results[metric_name] = float(results[i])
    
    print("📈 Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    
    return eval_results

def save_results(model: tf.keras.Model,
                history: tf.keras.callbacks.History,
                eval_results: Dict[str, float],
                config: Dict[str, Any],
                output_dir: str) -> None:
    """Save model and training results."""
    print(f"💾 Saving results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'saved_model')
    model.save(model_path, save_format='tf')
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create training plots
    try:
        plot_path = os.path.join(output_dir, 'training_plots.png')
        TrainingVisualization.plot_training_history(history, save_path=plot_path)
    except Exception as e:
        print(f"Warning: Could not create training plots: {e}")
    
    print("✅ Results saved successfully")

def optimize_model(model: tf.keras.Model,
                  train_ds: tf.data.Dataset,
                  config: Dict[str, Any],
                  output_dir: str) -> None:
    """Apply model optimization techniques."""
    optimization_config = config.get('optimization', {})
    
    if not optimization_config.get('enabled', False):
        return
    
    print("⚡ Applying model optimizations...")
    
    # Quantization
    if optimization_config.get('quantization', False):
        print("  📊 Applying quantization...")
        try:
            tflite_model = ModelQuantization.quantize_model_post_training(
                model, train_ds.take(100), 'int8'
            )
            
            # Save quantized model
            tflite_path = os.path.join(output_dir, 'quantized_model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"  ✅ Quantized model saved: {tflite_path}")
        except Exception as e:
            print(f"  ❌ Quantization failed: {e}")
    
    # Pruning
    if optimization_config.get('pruning', False):
        print("  ✂️ Applying pruning...")
        try:
            pruning_schedule = ModelPruning.create_pruning_schedule(
                final_sparsity=optimization_config.get('target_sparsity', 0.5)
            )
            
            pruned_model = ModelPruning.create_pruned_model(model, pruning_schedule)
            
            # Save pruned model
            pruned_path = os.path.join(output_dir, 'pruned_model')
            pruned_model.save(pruned_path)
            
            print(f"  ✅ Pruned model saved: {pruned_path}")
        except Exception as e:
            print(f"  ❌ Pruning failed: {e}")

def export_model_formats(model: tf.keras.Model,
                        config: Dict[str, Any],
                        output_dir: str) -> None:
    """Export model to multiple formats."""
    export_config = config.get('export', {})
    
    if not export_config.get('enabled', False):
        return
    
    print("📦 Exporting model to multiple formats...")
    
    formats = export_config.get('formats', ['savedmodel', 'tflite'])
    export_dir = os.path.join(output_dir, 'exports')
    
    try:
        export_stats = quick_export(model, export_dir, config.get('name', 'model'), formats)
        
        # Save export statistics
        stats_path = os.path.join(export_dir, 'export_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(export_stats, f, indent=2, default=str)
        
        print("✅ Model exported successfully")
    except Exception as e:
        print(f"❌ Model export failed: {e}")

def run_training_pipeline(config_path: str, output_dir: str, quick_test: bool = False) -> None:
    """Run complete training pipeline."""
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    if quick_test:
        print("⚡ Running in quick test mode")
        config['training']['epochs'] = min(config['training'].get('epochs', 10), 2)
        config['data']['num_samples'] = 100
    
    print(f"🎯 Training pipeline: {config.get('name', 'Unnamed')}")
    print(f"📄 Config: {config_path}")
    print(f"📁 Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create or load data
        if config['data'].get('type') == 'synthetic':
            train_ds, val_ds = create_synthetic_data(config)
            test_ds = val_ds  # Use validation as test for simplicity
        else:
            # TODO: Implement real data loading
            raise NotImplementedError("Real data loading not implemented yet")
        
        # Build model
        model = build_model(config)
        
        # Print model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        # Train model
        history = train_model(model, train_ds, val_ds, config)
        
        # Evaluate model
        eval_results = evaluate_model(model, test_ds, config)
        
        # Save results
        save_results(model, history, eval_results, config, output_dir)
        
        # Apply optimizations
        optimize_model(model, train_ds, config, output_dir)
        
        # Export model formats
        export_model_formats(model, config, output_dir)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📁 Results saved to: {output_dir}")
        
        # Best metrics
        best_val_acc = max(history.history.get('val_accuracy', [0]))
        print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        raise

def create_default_configs():
    """Create default configuration files."""
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)
    
    # CNN Image Classification Config
    cnn_config = {
        "name": "cnn_image_classifier",
        "model_type": "image_classification",
        "model": {
            "input_shape": [224, 224, 3],
            "num_classes": 10,
            "architecture": "simple",
            "dropout_rate": 0.5
        },
        "data": {
            "type": "synthetic",
            "batch_size": 32,
            "num_samples": 1000
        },
        "compile": {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"]
        },
        "training": {
            "epochs": 20,
            "patience": 5,
            "reduce_lr": True,
            "tensorboard": True,
            "verbose": 1
        },
        "optimization": {
            "enabled": True,
            "quantization": True,
            "pruning": False,
            "target_sparsity": 0.5
        },
        "export": {
            "enabled": True,
            "formats": ["savedmodel", "tflite", "onnx"]
        }
    }
    
    # Text Classification Config
    text_config = {
        "name": "lstm_text_classifier",
        "model_type": "text_classification",
        "model": {
            "vocab_size": 10000,
            "embedding_dim": 128,
            "max_length": 128,
            "num_classes": 5,
            "architecture": "lstm"
        },
        "data": {
            "type": "synthetic",
            "batch_size": 32,
            "num_samples": 800
        },
        "compile": {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"]
        },
        "training": {
            "epochs": 15,
            "patience": 3,
            "reduce_lr": True,
            "tensorboard": True
        },
        "optimization": {
            "enabled": False
        },
        "export": {
            "enabled": True,
            "formats": ["savedmodel", "tflite"]
        }
    }
    
    # Save configs
    with open(f"{configs_dir}/cnn_image_config.json", 'w') as f:
        json.dump(cnn_config, f, indent=2)
    
    with open(f"{configs_dir}/lstm_text_config.json", 'w') as f:
        json.dump(text_config, f, indent=2)
    
    print(f"📝 Created default configs in {configs_dir}/")

def main():
    """Main function for training script."""
    parser = argparse.ArgumentParser(description='TensorVerseHub Model Training Pipeline')
    
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with minimal epochs')
    parser.add_argument('--create-configs', action='store_true', help='Create default configuration files')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations')
    
    args = parser.parse_args()
    
    # Setup plotting style
    try:
        setup_plotting_style()
    except:
        pass
    
    if args.create_configs:
        create_default_configs()
        return
    
    if args.list_configs:
        configs_dir = "configs"
        if os.path.exists(configs_dir):
            print("📋 Available configurations:")
            for config_file in os.listdir(configs_dir):
                if config_file.endswith(('.json', '.yml', '.yaml')):
                    print(f"  • {config_file}")
        else:
            print("No configurations found. Use --create-configs to create default ones.")
        return
    
    if not args.config:
        print("❌ Configuration file required. Use --config to specify or --create-configs for defaults.")
        parser.print_help()
        return
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config_name = Path(args.config).stem
    output_dir = os.path.join(args.output, f"{config_name}_{timestamp}")
    
    # Run training pipeline
    try:
        run_training_pipeline(args.config, output_dir, args.quick_test)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == '__main__':
    main()