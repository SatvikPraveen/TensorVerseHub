"""
Multi-format model export: SavedModel, TFLite, ONNX, TensorFlow.js and Core ML.

Every exporter returns a statistics dictionary and never prints; optional
back-ends (``tf2onnx``, ``tensorflowjs``, ``coremltools``) raise ``ImportError``
with the install hint when missing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from . import compat
from .compat import SavedModelPredictor, keras

logger = logging.getLogger(__name__)
PathLike = Union[str, "os.PathLike[str]"]

EXPORT_FORMATS = ("savedmodel", "keras", "tflite", "onnx", "tfjs", "coreml")


def _model_bytes(model: keras.Model) -> int:
    return sum(compat.count_params([w]) * tf.as_dtype(w.dtype).size for w in model.weights)


def _dir_size(path: PathLike) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_interpreter(
    model_path: Optional[PathLike] = None, model_content: Optional[bytes] = None
) -> Any:
    """Create a TFLite interpreter, preferring the standalone LiteRT runtime when installed."""
    kwargs = {"model_path": str(model_path)} if model_path else {"model_content": model_content}
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]
    except ImportError:
        Interpreter = tf.lite.Interpreter  # type: ignore[assignment]
    interpreter = Interpreter(**kwargs)
    interpreter.allocate_tensors()
    return interpreter


def _model_config(model: keras.Model, include_optimizer: bool = False) -> Dict[str, Any]:
    optimizer = getattr(model, "optimizer", None)
    return {
        "name": model.name,
        "input_shape": list(compat.model_input_shape(model)),
        "output_shape": list(model.output_shape),
        "num_parameters": int(model.count_params()),
        "num_layers": len(model.layers),
        "optimizer": (
            optimizer.get_config() if include_optimizer and optimizer is not None else None
        ),
        "loss": str(getattr(model, "loss", None)),
        "metrics": compat.metric_names(model),
        "tensorflow_version": compat.TF_VERSION,
        "keras_version": compat.KERAS_VERSION,
    }


# ---------------------------------------------------------------------------
# SavedModel
# ---------------------------------------------------------------------------


class SavedModelExporter:
    """Inference SavedModel with ``metadata.json`` / ``model_config.json`` sidecars."""

    @staticmethod
    def export_savedmodel(
        model: keras.Model,
        export_path: PathLike,
        include_optimizer: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        export_path = str(export_path)
        compat.export_saved_model(model, export_path)
        config = _model_config(model, include_optimizer)
        Path(export_path, "model_config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        if metadata is not None:
            Path(export_path, "metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        stats: Dict[str, Any] = {
            "export_path": export_path,
            "size_bytes": _dir_size(export_path),
            "original_size_bytes": _model_bytes(model),
        }
        logger.info("SavedModel exported to %s (%.2f MB)", export_path, stats["size_bytes"] / 2**20)
        return stats

    @staticmethod
    def load_savedmodel_with_metadata(
        model_path: PathLike,
    ) -> Tuple[SavedModelPredictor, Dict[str, Any]]:
        predictor = SavedModelPredictor(model_path)
        metadata: Dict[str, Any] = {}
        meta_file = Path(model_path, "metadata.json")
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        config_file = Path(model_path, "model_config.json")
        metadata["model_config"] = (
            json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
        )
        return predictor, metadata


# ---------------------------------------------------------------------------
# TFLite
# ---------------------------------------------------------------------------


class TFLiteExporter:
    """TFLite conversion, benchmarking and numerical validation."""

    QUANTIZATION_TYPES = ("float32", "dynamic", "float16", "int8")

    @staticmethod
    def export_tflite(
        model: keras.Model,
        export_path: PathLike,
        quantization_type: str = "float32",
        representative_dataset: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
        target_ops: Optional[Sequence[str]] = None,
        num_calibration_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Convert and write a ``.tflite`` file.

        ``quantization_type``: ``float32`` (no quantisation), ``dynamic`` (int8 weights),
        ``float16`` or ``int8`` (full integer, needs ``representative_dataset``).
        """
        if quantization_type not in TFLiteExporter.QUANTIZATION_TYPES:
            raise ValueError(
                f"quantization_type must be one of {TFLiteExporter.QUANTIZATION_TYPES}"
            )
        from .optimization_utils import _representative_generator

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantization_type != "float32":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization_type == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == "int8":
            if representative_dataset is None:
                raise ValueError("int8 quantization requires a representative_dataset")
            converter.representative_dataset = _representative_generator(
                representative_dataset, num_calibration_samples
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        if target_ops:
            op_map = {
                "TFLITE_BUILTINS": tf.lite.OpsSet.TFLITE_BUILTINS,
                "TFLITE_BUILTINS_INT8": tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                "SELECT_TF_OPS": tf.lite.OpsSet.SELECT_TF_OPS,
            }
            unknown = set(target_ops) - set(op_map)
            if unknown:
                raise ValueError(f"Unknown target ops: {sorted(unknown)}")
            converter.target_spec.supported_ops = [op_map[o] for o in target_ops]

        try:
            tflite_model = converter.convert()
        except Exception as exc:
            raise RuntimeError(f"TFLite conversion failed: {exc}") from exc

        export_path = str(export_path)
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        Path(export_path).write_bytes(tflite_model)

        original = _model_bytes(model)
        stats = {
            "export_path": export_path,
            "quantization_type": quantization_type,
            "original_size_bytes": original,
            "tflite_size_bytes": len(tflite_model),
            "original_size_mb": original / 2**20,
            "tflite_size_mb": len(tflite_model) / 2**20,
            "compression_ratio": original / len(tflite_model) if tflite_model else 0.0,
        }
        logger.info(
            "TFLite (%s) exported to %s: %.3f MB (%.2fx smaller)",
            quantization_type,
            export_path,
            stats["tflite_size_mb"],
            stats["compression_ratio"],
        )
        return stats

    @staticmethod
    def run_tflite(interpreter: Any, inputs: np.ndarray) -> np.ndarray:
        """Run a batch through an interpreter one sample at a time (handles int8 I/O)."""
        in_detail = interpreter.get_input_details()[0]
        out_detail = interpreter.get_output_details()[0]
        in_scale, in_zero = in_detail.get("quantization", (0.0, 0))
        out_scale, out_zero = out_detail.get("quantization", (0.0, 0))
        outputs = []
        for sample in np.asarray(inputs):
            x = sample[np.newaxis].astype(np.float32)
            if in_detail["dtype"] in (np.int8, np.uint8) and in_scale:
                x = np.round(x / in_scale + in_zero).astype(in_detail["dtype"])
            else:
                x = x.astype(in_detail["dtype"])
            interpreter.set_tensor(in_detail["index"], x)
            interpreter.invoke()
            y = interpreter.get_tensor(out_detail["index"])[0]
            if out_detail["dtype"] in (np.int8, np.uint8) and out_scale:
                y = (y.astype(np.float32) - out_zero) * out_scale
            outputs.append(y)
        return np.stack(outputs)

    @staticmethod
    def benchmark_tflite_model(
        tflite_path: PathLike, test_input: np.ndarray, num_runs: int = 100, warmup: int = 5
    ) -> Dict[str, float]:
        interpreter = make_interpreter(tflite_path)
        in_detail = interpreter.get_input_details()[0]
        out_detail = interpreter.get_output_details()[0]
        sample = np.asarray(test_input)
        if sample.ndim == len(in_detail["shape"]) - 1:
            sample = sample[np.newaxis]
        sample = sample.astype(in_detail["dtype"])
        for _ in range(warmup):
            interpreter.set_tensor(in_detail["index"], sample)
            interpreter.invoke()
        start = time.perf_counter()
        for _ in range(num_runs):
            interpreter.set_tensor(in_detail["index"], sample)
            interpreter.invoke()
            interpreter.get_tensor(out_detail["index"])
        total = time.perf_counter() - start
        return {
            "avg_inference_time_ms": total / num_runs * 1000,
            "total_time_s": total,
            "throughput_fps": num_runs / total if total > 0 else 0.0,
        }

    @staticmethod
    def validate_tflite(
        model: keras.Model, tflite_path: PathLike, sample_inputs: np.ndarray, atol: float = 1e-2
    ) -> Dict[str, Any]:
        """Compare Keras vs TFLite predictions on ``sample_inputs``."""
        keras_out = np.asarray(model.predict(sample_inputs, verbose=0))
        tflite_out = TFLiteExporter.run_tflite(make_interpreter(tflite_path), sample_inputs)
        max_abs = float(np.max(np.abs(keras_out - tflite_out)))
        agree = float(np.mean(np.argmax(keras_out, -1) == np.argmax(tflite_out, -1)))
        return {
            "max_abs_diff": max_abs,
            "argmax_agreement": agree,
            "within_tolerance": max_abs <= atol,
        }


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


class ONNXExporter:
    """ONNX export through ``tf2onnx`` (``pip install tensorversehub[export]``)."""

    @staticmethod
    def export_onnx(
        model: keras.Model,
        export_path: PathLike,
        input_signature: Optional[Sequence[tf.TensorSpec]] = None,
        opset_version: int = 17,
    ) -> Dict[str, Any]:
        try:
            import tf2onnx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "tf2onnx is required for ONNX export: pip install 'tensorversehub[export]'"
            ) from exc

        export_path = str(export_path)
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        signature = list(input_signature or compat.input_signature(model))
        try:
            tf2onnx.convert.from_keras(
                model, input_signature=signature, opset=opset_version, output_path=export_path
            )
        except Exception as exc:
            raise RuntimeError(f"ONNX conversion failed: {exc}") from exc

        original = _model_bytes(model)
        onnx_size = os.path.getsize(export_path)
        stats = {
            "export_path": export_path,
            "opset_version": opset_version,
            "original_size_bytes": original,
            "onnx_size_bytes": onnx_size,
            "original_size_mb": original / 2**20,
            "onnx_size_mb": onnx_size / 2**20,
            "size_ratio": onnx_size / original if original else 0.0,
            "input_names": [s.name for s in signature],
        }
        logger.info("ONNX exported to %s (%.2f MB)", export_path, stats["onnx_size_mb"])
        return stats

    @staticmethod
    def validate_onnx(
        model: keras.Model, onnx_path: PathLike, sample_inputs: np.ndarray, atol: float = 1e-4
    ) -> Dict[str, Any]:
        """Compare Keras vs ONNX Runtime predictions."""
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("onnxruntime is required: pip install onnxruntime") from exc
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        onnx_out = session.run(None, {name: np.asarray(sample_inputs, np.float32)})[0]
        keras_out = np.asarray(model.predict(sample_inputs, verbose=0))
        max_abs = float(np.max(np.abs(keras_out - onnx_out)))
        return {"max_abs_diff": max_abs, "within_tolerance": max_abs <= atol}


# ---------------------------------------------------------------------------
# TensorFlow.js / Core ML
# ---------------------------------------------------------------------------


class TensorFlowJSExporter:
    """TensorFlow.js export (``pip install tensorflowjs``)."""

    @staticmethod
    def export_tfjs(
        model: keras.Model,
        export_path: PathLike,
        quantization_bytes: Optional[int] = None,
        skip_op_check: bool = False,
        strip_debug_ops: bool = True,
    ) -> Dict[str, Any]:
        try:
            import tensorflowjs as tfjs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("tensorflowjs is required: pip install tensorflowjs") from exc

        export_path = str(export_path)
        Path(export_path).mkdir(parents=True, exist_ok=True)
        options: Dict[str, Any] = {
            "skip_op_check": skip_op_check,
            "strip_debug_ops": strip_debug_ops,
        }
        if quantization_bytes:
            options["quantization_dtype_map"] = {{1: "uint8", 2: "uint16"}[quantization_bytes]: "*"}
        with tempfile.TemporaryDirectory() as tmp:
            saved = os.path.join(tmp, "saved_model")
            compat.export_saved_model(model, saved)
            try:
                tfjs.converters.convert_tf_saved_model(saved, export_path, **options)
            except Exception as exc:
                raise RuntimeError(f"TensorFlow.js conversion failed: {exc}") from exc

        original = _model_bytes(model)
        size = _dir_size(export_path)
        return {
            "export_path": export_path,
            "original_size_bytes": original,
            "tfjs_size_bytes": size,
            "original_size_mb": original / 2**20,
            "tfjs_size_mb": size / 2**20,
            "size_ratio": size / original if original else 0.0,
            "quantization_bytes": quantization_bytes,
        }


class CoreMLExporter:
    """Core ML export (``pip install coremltools``; macOS recommended)."""

    @staticmethod
    def export_coreml(
        model: keras.Model,
        export_path: PathLike,
        class_labels: Optional[Sequence[str]] = None,
        minimum_deployment_target: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            import coremltools as ct  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("coremltools is required: pip install coremltools") from exc

        export_path = str(export_path)
        kwargs: Dict[str, Any] = {"source": "tensorflow"}
        if class_labels:
            kwargs["classifier_config"] = ct.ClassifierConfig(list(class_labels))
        if minimum_deployment_target is not None:
            kwargs["minimum_deployment_target"] = minimum_deployment_target
        try:
            mlmodel = ct.convert(model, **kwargs)
            mlmodel.save(export_path)
        except Exception as exc:
            raise RuntimeError(f"Core ML conversion failed: {exc}") from exc
        size = (
            _dir_size(export_path) if os.path.isdir(export_path) else os.path.getsize(export_path)
        )
        original = _model_bytes(model)
        return {
            "export_path": export_path,
            "original_size_bytes": original,
            "coreml_size_bytes": size,
            "original_size_mb": original / 2**20,
            "coreml_size_mb": size / 2**20,
            "size_ratio": size / original if original else 0.0,
        }


# ---------------------------------------------------------------------------
# Multi-format
# ---------------------------------------------------------------------------


class MultiFormatExporter:
    """Export one model to several formats and write ``export_summary.json``."""

    def __init__(self, model: keras.Model, model_name: str = "model") -> None:
        self.model = model
        self.model_name = model_name
        self.export_stats: Dict[str, Dict[str, Any]] = {}

    def export_all_formats(
        self,
        export_dir: PathLike,
        formats: Optional[Sequence[str]] = None,
        representative_dataset: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export to ``formats`` (default: ``savedmodel``, ``keras``, ``tflite``).

        Extra keyword arguments: ``metadata``, ``tflite_quantization``, ``onnx_opset``,
        ``tfjs_quantization``, ``class_labels``.
        """
        formats = list(formats or ("savedmodel", "keras", "tflite"))
        unknown = set(formats) - set(EXPORT_FORMATS)
        if unknown:
            raise ValueError(f"Unknown export formats: {sorted(unknown)}")
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            try:
                if fmt == "savedmodel":
                    stats = SavedModelExporter.export_savedmodel(
                        self.model, export_dir / "savedmodel", metadata=kwargs.get("metadata")
                    )
                elif fmt == "keras":
                    path = export_dir / f"{self.model_name}.keras"
                    compat.save_model(self.model, path)
                    stats = {"export_path": str(path), "size_bytes": path.stat().st_size}
                elif fmt == "tflite":
                    stats = TFLiteExporter.export_tflite(
                        self.model,
                        export_dir / f"{self.model_name}.tflite",
                        quantization_type=kwargs.get("tflite_quantization", "float32"),
                        representative_dataset=representative_dataset,
                    )
                elif fmt == "onnx":
                    stats = ONNXExporter.export_onnx(
                        self.model,
                        export_dir / f"{self.model_name}.onnx",
                        opset_version=kwargs.get("onnx_opset", 17),
                    )
                elif fmt == "tfjs":
                    stats = TensorFlowJSExporter.export_tfjs(
                        self.model,
                        export_dir / "tfjs",
                        quantization_bytes=kwargs.get("tfjs_quantization"),
                    )
                else:  # coreml
                    stats = CoreMLExporter.export_coreml(
                        self.model,
                        export_dir / f"{self.model_name}.mlpackage",
                        class_labels=kwargs.get("class_labels"),
                    )
                self.export_stats[fmt] = stats
            except Exception as exc:
                if raise_on_error:
                    raise
                logger.warning("Failed to export %s: %s", fmt, exc)
                self.export_stats[fmt] = {"error": str(exc)}

        self._write_summary(export_dir)
        return self.export_stats

    def _write_summary(self, export_dir: Path) -> None:
        summary = {
            "model_name": self.model_name,
            "timestamp": _now(),
            "original_model": _model_config(self.model),
            "export_stats": self.export_stats,
        }
        (export_dir / "export_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )


def quick_export(
    model: keras.Model,
    export_dir: PathLike,
    model_name: str = "model",
    formats: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """One-liner around :class:`MultiFormatExporter`."""
    return MultiFormatExporter(model, model_name).export_all_formats(export_dir, formats, **kwargs)


def create_deployment_package(
    model: keras.Model,
    package_path: PathLike,
    model_name: str = "model",
    formats: Optional[Sequence[str]] = None,
    include_metadata: bool = True,
    model_version: str = "1.0.0",
) -> str:
    """Zip exported formats plus ``package_metadata.json`` into ``package_path``."""
    package_path = str(package_path)
    with tempfile.TemporaryDirectory() as tmp:
        exports = os.path.join(tmp, "exports")
        stats = MultiFormatExporter(model, model_name).export_all_formats(exports, formats)
        if include_metadata:
            metadata = {
                "model_name": model_name,
                "model_version": model_version,
                "tensorflow_version": compat.TF_VERSION,
                "keras_version": compat.KERAS_VERSION,
                "export_timestamp": _now(),
                "model_architecture": _model_config(model),
                "export_formats": [k for k, v in stats.items() if "error" not in v],
                "deployment_instructions": {
                    "savedmodel": "tf.saved_model.load(path).signatures['serving_default']",
                    "keras": "keras.models.load_model('model.keras')",
                    "tflite": "tf.lite.Interpreter / ai_edge_litert Interpreter",
                    "onnx": "onnxruntime.InferenceSession('model.onnx')",
                    "tfjs": "tf.loadGraphModel('tfjs/model.json') in the browser",
                },
            }
            Path(tmp, "package_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        Path(package_path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in Path(tmp).rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(tmp))
    logger.info("Deployment package created: %s", package_path)
    return package_path


__all__ = [
    "CoreMLExporter",
    "EXPORT_FORMATS",
    "MultiFormatExporter",
    "ONNXExporter",
    "SavedModelExporter",
    "SavedModelPredictor",
    "TFLiteExporter",
    "TensorFlowJSExporter",
    "create_deployment_package",
    "make_interpreter",
    "quick_export",
]
