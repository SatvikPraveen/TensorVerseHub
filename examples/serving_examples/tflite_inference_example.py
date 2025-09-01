# Location: /examples/serving_examples/tflite_inference_example.py

"""
TensorFlow Lite inference example with performance benchmarking.
Demonstrates efficient inference on mobile-optimized models.
"""

import tensorflow as tf
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Any
import os
from PIL import Image
import argparse


class TFLiteInference:
    """TensorFlow Lite model inference class."""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        """
        Initialize TFLite interpreter.
        
        Args:
            model_path: Path to TFLite model file
            num_threads: Number of CPU threads for inference
        """
        self.model_path = model_path
        self.num_threads = num_threads
        
        # Load interpreter
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Performance tracking
        self.inference_count = 0
        self.total_time = 0
        
        print(f"✅ TFLite model loaded: {model_path}")
        print(f"📊 Model info:")
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Input dtype: {self.input_details[0]['dtype']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
        print(f"   Output dtype: {self.output_details[0]['dtype']}")
    
    def preprocess_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image file
            target_size: Target size (height, width), if None uses model input size
            
        Returns:
            Preprocessed image array
        """
        # Get target size from model if not specified
        if target_size is None:
            input_shape = self.input_details[0]['shape']
            target_size = (input_shape[1], input_shape[2])  # Height, Width
        
        # Load and preprocess image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] if model expects float32
        if self.input_details[0]['dtype'] == np.float32:
            image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Ensure correct dtype
        image_array = image_array.astype(self.input_details[0]['dtype'])
        
        return image_array
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Model predictions
        """
        start_time = time.time()
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self.total_time += inference_time
        self.inference_count += 1
        
        return output_data
    
    def predict_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict image classification with top-k results.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of top predictions with confidence scores
        """
        # Preprocess image
        input_data = self.preprocess_image(image_path)
        
        # Run inference
        predictions = self.predict(input_data)
        
        # Get probabilities (assuming softmax output)
        if predictions.ndim > 1:
            probabilities = predictions[0]
        else:
            probabilities = predictions
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class_id': int(idx),
                'confidence': float(probabilities[idx]),
                'probability': float(probabilities[idx])
            })
        
        return results
    
    def benchmark(self, input_data: np.ndarray, num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_data: Input data for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance metrics
        """
        print(f"🔥 Running benchmark: {warmup_runs} warmup + {num_runs} test runs")
        
        # Warmup runs
        for _ in range(warmup_runs):
            self.predict(input_data)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(input_data)
            times.append(time.time() - start_time)
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]
        
        metrics = {
            'avg_time_ms': np.mean(times_ms),
            'min_time_ms': np.min(times_ms),
            'max_time_ms': np.max(times_ms),
            'std_time_ms': np.std(times_ms),
            'median_time_ms': np.median(times_ms),
            'p95_time_ms': np.percentile(times_ms, 95),
            'p99_time_ms': np.percentile(times_ms, 99),
            'throughput_fps': 1000 / np.mean(times_ms)
        }
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        # Calculate model size
        model_size = os.path.getsize(self.model_path)
        
        info = {
            'model_path': self.model_path,
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'num_threads': self.num_threads,
            'input_details': {
                'shape': self.input_details[0]['shape'].tolist(),
                'dtype': str(self.input_details[0]['dtype']),
                'name': self.input_details[0]['name'],
                'quantization': self.input_details[0]['quantization']
            },
            'output_details': {
                'shape': self.output_details[0]['shape'].tolist(),
                'dtype': str(self.output_details[0]['dtype']),
                'name': self.output_details[0]['name'],
                'quantization': self.output_details[0]['quantization']
            },
            'inference_stats': {
                'total_inferences': self.inference_count,
                'total_time_s': self.total_time,
                'avg_time_ms': (self.total_time / self.inference_count * 1000) if self.inference_count > 0 else 0
            }
        }
        
        return info
    
    def print_model_info(self):
        """Print detailed model information."""
        info = self.get_model_info()
        
        print("\n📋 TFLite Model Information:")
        print(f"   Model file: {info['model_path']}")
        print(f"   Model size: {info['model_size_mb']:.2f} MB")
        print(f"   Threads: {info['num_threads']}")
        
        print(f"\n📥 Input Details:")
        print(f"   Shape: {info['input_details']['shape']}")
        print(f"   Data type: {info['input_details']['dtype']}")
        print(f"   Quantization: {info['input_details']['quantization']}")
        
        print(f"\n📤 Output Details:")
        print(f"   Shape: {info['output_details']['shape']}")
        print(f"   Data type: {info['output_details']['dtype']}")
        print(f"   Quantization: {info['output_details']['quantization']}")
        
        print(f"\n⚡ Performance Stats:")
        print(f"   Total inferences: {info['inference_stats']['total_inferences']}")
        print(f"   Average time: {info['inference_stats']['avg_time_ms']:.2f} ms")


def compare_models(original_model_path: str, tflite_model_path: str, test_image_path: str):
    """Compare original model with TFLite version."""
    print("🔍 Comparing original model vs TFLite model")
    
    # Load original model
    try:
        if original_model_path.endswith('.h5'):
            original_model = tf.keras.models.load_model(original_model_path)
        else:
            original_model = tf.saved_model.load(original_model_path)
        print(f"✅ Original model loaded: {original_model_path}")
    except Exception as e:
        print(f"❌ Failed to load original model: {e}")
        return
    
    # Load TFLite model
    tflite_inference = TFLiteInference(tflite_model_path)
    
    # Prepare test input
    test_input = tflite_inference.preprocess_image(test_image_path)
    
    # Original model inference
    start_time = time.time()
    if hasattr(original_model, 'predict'):
        original_pred = original_model.predict(test_input)
    else:
        original_pred = original_model(test_input)
        if isinstance(original_pred, dict):
            original_pred = list(original_pred.values())[0]
        original_pred = original_pred.numpy()
    original_time = time.time() - start_time
    
    # TFLite inference
    start_time = time.time()
    tflite_pred = tflite_inference.predict(test_input)
    tflite_time = time.time() - start_time
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"   Original model time: {original_time * 1000:.2f} ms")
    print(f"   TFLite model time: {tflite_time * 1000:.2f} ms")
    print(f"   Speedup: {original_time / tflite_time:.2f}x")
    
    # Calculate accuracy difference
    if original_pred.shape == tflite_pred.shape:
        mae = np.mean(np.abs(original_pred - tflite_pred))
        max_diff = np.max(np.abs(original_pred - tflite_pred))
        print(f"   Mean Absolute Error: {mae:.6f}")
        print(f"   Max Absolute Error: {max_diff:.6f}")
        
        # Top predictions comparison
        orig_top = np.argmax(original_pred)
        tflite_top = np.argmax(tflite_pred)
        print(f"   Original top class: {orig_top}")
        print(f"   TFLite top class: {tflite_top}")
        print(f"   Top prediction match: {'✅' if orig_top == tflite_top else '❌'}")
    else:
        print(f"   ⚠️ Output shapes don't match: {original_pred.shape} vs {tflite_pred.shape}")


def main():
    """Main function for TFLite inference example."""
    parser = argparse.ArgumentParser(description='TFLite Inference Example')
    parser.add_argument('--model', required=True, help='Path to TFLite model')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--compare', help='Path to original model for comparison')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Initialize TFLite inference
    tflite_inference = TFLiteInference(args.model, args.threads)
    
    # Print model information
    tflite_inference.print_model_info()
    
    # Test image prediction
    if args.image:
        if os.path.exists(args.image):
            print(f"\n🖼️  Testing image prediction: {args.image}")
            predictions = tflite_inference.predict_image(args.image)
            
            print("🎯 Top predictions:")
            for i, pred in enumerate(predictions):
                print(f"   {i+1}. Class {pred['class_id']}: {pred['confidence']:.4f}")
        else:
            print(f"❌ Image file not found: {args.image}")
    
    # Run benchmark
    if args.benchmark:
        # Create dummy input for benchmarking
        input_shape = tflite_inference.input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(tflite_inference.input_details[0]['dtype'])
        
        print(f"\n⚡ Running performance benchmark...")
        metrics = tflite_inference.benchmark(dummy_input, args.runs)
        
        print("📈 Benchmark Results:")
        print(f"   Average time: {metrics['avg_time_ms']:.2f} ms")
        print(f"   Median time: {metrics['median_time_ms']:.2f} ms")
        print(f"   Min time: {metrics['min_time_ms']:.2f} ms")
        print(f"   Max time: {metrics['max_time_ms']:.2f} ms")
        print(f"   95th percentile: {metrics['p95_time_ms']:.2f} ms")
        print(f"   99th percentile: {metrics['p99_time_ms']:.2f} ms")
        print(f"   Throughput: {metrics['throughput_fps']:.1f} FPS")
        print(f"   Standard deviation: {metrics['std_time_ms']:.2f} ms")
    
    # Compare with original model
    if args.compare and args.image:
        if os.path.exists(args.compare) and os.path.exists(args.image):
            compare_models(args.compare, args.model, args.image)
        else:
            print("❌ Comparison requires both --compare model and --image paths")


if __name__ == '__main__':
    main()