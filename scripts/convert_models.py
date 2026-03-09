#!/usr/bin/env python3
"""
tensorverse-convert  — CLI for converting TensorFlow models between formats.

Usage
-----
    tensorverse-convert --model ./models/final_model --to tflite
    tensorverse-convert --model ./models/final_model --to onnx --output ./converted/
    tensorverse-convert --model ./models/final_model --to all --quantize int8
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


SUPPORTED_FORMATS = ["saved_model", "h5", "tflite", "onnx", "tfjs", "coreml", "all"]
QUANTIZE_OPTIONS = ["none", "default", "int8", "float16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorVerseHub — Model Conversion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the source model (SavedModel dir or .h5 file)",
    )
    parser.add_argument(
        "--to",
        type=str,
        choices=SUPPORTED_FORMATS,
        required=True,
        help="Target format for conversion",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./converted_models",
        help="Output directory for converted models (default: ./converted_models)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=QUANTIZE_OPTIONS,
        default="none",
        help="Quantization type for TFLite target (default: none)",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=None,
        help="Model input shape, e.g. --input-shape 1 224 224 3",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a quick benchmark on the converted TFLite model",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def load_source_model(model_path: str):
    """Load a Keras model from SavedModel directory or .h5 file."""
    import tensorflow as tf

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def convert_to_tflite(model, output_dir: str, quantize: str, quiet: bool) -> str:
    """Convert model to TFLite with optional quantization."""
    import tensorflow as tf

    os.makedirs(output_dir, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    suffix = "fp32"
    if quantize == "default":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        suffix = "quant_default"
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        suffix = "quant_int8"
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        suffix = "quant_fp16"

    tflite_model = converter.convert()
    out_path = os.path.join(output_dir, f"model_{suffix}.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    if not quiet:
        size_kb = len(tflite_model) / 1024
        print(f"  TFLite model saved → {out_path}  ({size_kb:.1f} KB)")
    return out_path


def benchmark_tflite(tflite_path: str, input_shape, quiet: bool):
    """Run a simple latency benchmark on the TFLite model."""
    import time

    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    if input_shape is None:
        input_shape = input_details[0]["shape"]

    dummy_input = np.random.rand(*input_shape).astype(input_details[0]["dtype"])

    # Warm-up
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    for _ in range(5):
        interpreter.invoke()

    # Benchmark
    runs = 50
    start = time.perf_counter()
    for _ in range(runs):
        interpreter.invoke()
    elapsed = (time.perf_counter() - start) / runs * 1000

    if not quiet:
        print(f"  TFLite benchmark  : {elapsed:.2f} ms/inference (avg over {runs} runs)")


def convert_to_onnx(model, output_dir: str, opset: int, quiet: bool) -> str:
    """Convert model to ONNX format via tf2onnx."""
    try:
        import tensorflow as tf
        import tf2onnx  # type: ignore[import]
    except ImportError:
        print("tf2onnx is not installed. Run: pip install tf2onnx")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.onnx")

    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
    _, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset, output_path=out_path
    )

    if not quiet:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  ONNX model saved  → {out_path}  ({size_mb:.2f} MB)")
    return out_path


def convert_to_tfjs(model, output_dir: str, quiet: bool) -> str:
    """Convert model to TensorFlow.js format."""
    try:
        import tensorflowjs as tfjs  # type: ignore[import]
    except ImportError:
        print("tensorflowjs is not installed. Run: pip install tensorflowjs")
        return ""

    out_path = os.path.join(output_dir, "tfjs_model")
    os.makedirs(out_path, exist_ok=True)
    tfjs.converters.save_keras_model(model, out_path)

    if not quiet:
        print(f"  TF.js model saved → {out_path}")
    return out_path


def convert_to_coreml(model, output_dir: str, quiet: bool) -> str:
    """Convert model to CoreML format."""
    try:
        import coremltools as ct  # type: ignore[import]
    except ImportError:
        print("coremltools is not installed. Run: pip install coremltools")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.mlpackage")
    cml_model = ct.convert(model)
    cml_model.save(out_path)

    if not quiet:
        print(f"  CoreML model saved → {out_path}")
    return out_path


def convert_to_h5(model, output_dir: str, quiet: bool) -> str:
    """Save model in legacy .h5 format."""
    import tensorflow as tf

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.h5")
    model.save(out_path)

    if not quiet:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  H5 model saved    → {out_path}  ({size_mb:.2f} MB)")
    return out_path


def convert_to_saved_model(model, output_dir: str, quiet: bool) -> str:
    """Save model in SavedModel format."""
    out_path = os.path.join(output_dir, "saved_model")
    model.save(out_path)

    if not quiet:
        print(f"  SavedModel saved  → {out_path}")
    return out_path


def main():
    args = parse_args()

    import tensorflow as tf

    if not args.quiet:
        print(f"TensorVerseHub Conversion CLI  |  TensorFlow {tf.__version__}")
        print(f"  Source : {args.model}")
        print(f"  Target : {args.to}")
        print(f"  Output : {args.output}")

    model = load_source_model(args.model)

    targets = SUPPORTED_FORMATS[:-1] if args.to == "all" else [args.to]

    for target in targets:
        if not args.quiet:
            print(f"\n→ Converting to {target.upper()} …")

        out_subdir = os.path.join(args.output, target)

        if target == "tflite":
            tflite_path = convert_to_tflite(model, out_subdir, args.quantize, args.quiet)
            if args.benchmark and tflite_path:
                benchmark_tflite(tflite_path, args.input_shape, args.quiet)
        elif target == "onnx":
            convert_to_onnx(model, out_subdir, args.opset, args.quiet)
        elif target == "tfjs":
            convert_to_tfjs(model, out_subdir, args.quiet)
        elif target == "coreml":
            convert_to_coreml(model, out_subdir, args.quiet)
        elif target == "h5":
            convert_to_h5(model, out_subdir, args.quiet)
        elif target == "saved_model":
            convert_to_saved_model(model, out_subdir, args.quiet)

    if not args.quiet:
        print(f"\nConversion complete. Artifacts saved to: {args.output}")


if __name__ == "__main__":
    main()
