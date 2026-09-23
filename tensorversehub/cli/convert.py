"""Convert a Keras model to SavedModel, TFLite, ONNX, TensorFlow.js or Core ML."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List

logger = logging.getLogger("tensorverse.convert")

FORMATS = ("saved_model", "keras", "tflite", "onnx", "tfjs", "coreml", "all")
QUANTIZE = ("none", "dynamic", "float16", "int8")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="Source .keras/.h5 file or SavedModel directory"
    )
    parser.add_argument("--to", choices=FORMATS, required=True)
    parser.add_argument("--output", default="./converted_models")
    parser.add_argument("--quantize", choices=QUANTIZE, default="none", help="TFLite quantization")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Random calibration samples for int8 (when no dataset is given)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the TFLite model")
    parser.add_argument("--quiet", "-q", action="store_true")


def _calibration_data(model: Any, n: int):
    import numpy as np

    from .. import compat

    shape = tuple(compat.model_input_shape(model)[1:])
    return np.random.default_rng(0).random((n, *shape), dtype=np.float32)


def run(args: argparse.Namespace) -> int:

    from .. import compat, configure_tensorflow
    from .. import export_utils as eu

    configure_tensorflow(log_level=logging.WARNING if args.quiet else logging.INFO)
    model = compat.load_model(args.model)
    if not isinstance(model, compat.keras.Model):
        raise SystemExit(
            "Only .keras/.h5 models can be converted; re-export the SavedModel from its Keras source."
        )
    out = Path(args.output)
    targets: List[str] = [f for f in FORMATS if f != "all"] if args.to == "all" else [args.to]
    quant = "float32" if args.quantize == "none" else args.quantize

    for target in targets:
        logger.info("→ converting to %s", target)
        try:
            if target == "saved_model":
                stats = eu.SavedModelExporter.export_savedmodel(model, out / "saved_model")
            elif target == "keras":
                path = out / "model.keras"
                compat.save_model(model, path)
                stats = {"export_path": str(path)}
            elif target == "tflite":
                rep = (
                    _calibration_data(model, args.calibration_samples) if quant == "int8" else None
                )
                path = out / "tflite" / f"model_{quant}.tflite"
                stats = eu.TFLiteExporter.export_tflite(
                    model, path, quant, representative_dataset=rep
                )
                if args.benchmark:
                    sample = _calibration_data(model, 1)
                    bench = eu.TFLiteExporter.benchmark_tflite_model(path, sample, num_runs=50)
                    logger.info("TFLite latency: %.3f ms/inference", bench["avg_inference_time_ms"])
            elif target == "onnx":
                stats = eu.ONNXExporter.export_onnx(
                    model, out / "onnx" / "model.onnx", opset_version=args.opset
                )
            elif target == "tfjs":
                stats = eu.TensorFlowJSExporter.export_tfjs(model, out / "tfjs")
            else:
                stats = eu.CoreMLExporter.export_coreml(model, out / "coreml" / "model.mlpackage")
            logger.info("   saved → %s", stats.get("export_path"))
        except ImportError as exc:
            logger.warning("   skipped %s: %s", target, exc)
            if args.to != "all":
                return 1
    logger.info("Conversion complete: %s", out)
    return 0


def main() -> None:
    from . import main as cli_main

    sys.exit(cli_main(["convert", *sys.argv[1:]]))


if __name__ == "__main__":  # pragma: no cover
    main()
