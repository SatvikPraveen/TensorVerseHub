"""
Keras 2 / Keras 3 compatibility layer.

TensorFlow 2.16+ ships Keras 3 as ``tf.keras``. Setting ``TF_USE_LEGACY_KERAS=1``
(with the ``tf-keras`` package installed) switches ``tf.keras`` back to Keras 2,
which is still required by ``tensorflow-model-optimization``.

Everything in TensorVerseHub that touches an API that changed between the two
generations goes through this module, so the rest of the code base can be
written once and run under either.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - static analysis sees the real Keras package
    import keras
else:
    keras = tf.keras

PathLike = Union[str, "os.PathLike[str]"]

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def _parse_version(version: str) -> Tuple[int, ...]:
    """Parse ``"2.21.0-rc1"`` → ``(2, 21, 0)`` without depending on ``packaging``."""
    parts: List[int] = []
    for chunk in version.split(".")[:3]:
        match = re.match(r"\d+", chunk)
        if match is None:
            break
        parts.append(int(match.group(0)))
    return tuple(parts) or (0,)


def _detect_keras_version() -> str:
    version_fn = getattr(keras, "version", None)
    if callable(version_fn):
        try:
            return str(version_fn())
        except Exception:  # pragma: no cover - defensive
            pass
    attr = getattr(keras, "__version__", None)
    if isinstance(attr, str):
        return attr
    try:  # legacy tf.keras exposes no version attribute through the lazy loader
        import tf_keras  # type: ignore[import-not-found]

        return str(tf_keras.__version__)
    except ImportError:  # pragma: no cover - TF < 2.16
        return "2.0.0"


TF_VERSION: str = tf.__version__
TF_VERSION_INFO: Tuple[int, ...] = _parse_version(tf.__version__)
KERAS_VERSION: str = _detect_keras_version()
KERAS_VERSION_INFO: Tuple[int, ...] = _parse_version(KERAS_VERSION)
IS_KERAS_3: bool = KERAS_VERSION_INFO[0] >= 3
IS_LEGACY_KERAS: bool = not IS_KERAS_3
MIN_TF_VERSION: Tuple[int, int] = (2, 16)

NATIVE_MODEL_EXTENSION = ".keras"
_KERAS_FILE_SUFFIXES = (".keras", ".h5", ".hdf5")


def check_tensorflow_version(minimum: Tuple[int, int] = MIN_TF_VERSION) -> None:
    """Raise ``ImportError`` if the installed TensorFlow is older than ``minimum``."""
    if TF_VERSION_INFO[:2] < minimum:
        raise ImportError(
            f"TensorVerseHub requires TensorFlow >= {minimum[0]}.{minimum[1]}, "
            f"found {TF_VERSION}. Upgrade with: pip install -U 'tensorflow>={minimum[0]}.{minimum[1]}'"
        )


def require_legacy_keras(feature: str) -> None:
    """
    Raise a helpful error when ``feature`` needs Keras 2.

    ``tensorflow-model-optimization`` (pruning, quantization-aware training,
    weight clustering) has not been ported to Keras 3.  Users can opt in to the
    legacy stack by installing ``tf-keras`` and exporting ``TF_USE_LEGACY_KERAS=1``
    *before* importing TensorFlow.
    """
    if IS_KERAS_3:
        raise RuntimeError(
            f"{feature} requires the legacy Keras 2 API (tensorflow-model-optimization "
            "does not support Keras 3). Install 'tf-keras' and set the environment "
            "variable TF_USE_LEGACY_KERAS=1 before importing TensorFlow."
        )


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------


def is_keras_file(path: PathLike) -> bool:
    return Path(path).suffix.lower() in _KERAS_FILE_SUFFIXES


def is_saved_model_dir(path: PathLike) -> bool:
    return Path(path).is_dir() and (Path(path) / "saved_model.pb").exists()


def export_saved_model(model: keras.Model, path: PathLike, verbose: bool = False) -> str:
    """
    Write an inference-only TensorFlow SavedModel with a ``serving_default`` signature.

    Works for Keras 2 and Keras 3 models and is the format expected by
    TF Serving, TensorRT, TF.js and ``tf2onnx``.
    """
    path = str(path)
    if hasattr(model, "export"):
        try:
            model.export(path, verbose=verbose)
        except TypeError:  # tf-keras' export() has no verbose argument
            model.export(path)
    else:  # pragma: no cover - TF < 2.16
        model.save(path, save_format="tf")
    return path


def save_model(model: keras.Model, path: PathLike, overwrite: bool = True) -> str:
    """
    Save a model in the format implied by ``path``.

    * ``*.keras`` – native Keras format (recommended; reloadable, keeps optimizer)
    * ``*.h5``    – legacy HDF5
    * directory   – inference-only SavedModel via :func:`export_saved_model`
    """
    path_str = str(path)
    if is_keras_file(path_str):
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        model.save(path_str, overwrite=overwrite)
        return path_str
    return export_saved_model(model, path_str)


class SavedModelPredictor:
    """
    Thin callable wrapper around a loaded SavedModel signature.

    Keras 3 cannot reload an exported SavedModel as a ``keras.Model``; this class
    gives both Keras generations a uniform ``predict``/``__call__`` interface.
    """

    def __init__(self, path: PathLike, signature: str = "serving_default") -> None:
        self.path = str(path)
        self.loaded = tf.saved_model.load(self.path)
        if signature not in self.loaded.signatures:
            available = list(self.loaded.signatures.keys())
            raise ValueError(f"Signature '{signature}' not found. Available: {available}")
        self.signature_name = signature
        self.fn = self.loaded.signatures[signature]
        _, kwargs_spec = self.fn.structured_input_signature
        self.input_names: List[str] = list(kwargs_spec.keys())
        self.input_specs: Dict[str, tf.TensorSpec] = dict(kwargs_spec)
        outputs = self.fn.structured_outputs
        self.output_names: List[str] = list(outputs.keys()) if isinstance(outputs, dict) else []

    @property
    def input_shape(self) -> Tuple[Optional[int], ...]:
        spec = self.input_specs[self.input_names[0]]
        return tuple(spec.shape.as_list())

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        if isinstance(inputs, dict):
            feed = {k: tf.convert_to_tensor(v) for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)) and len(self.input_names) > 1:
            feed = {n: tf.convert_to_tensor(v) for n, v in zip(self.input_names, inputs)}
        else:
            spec = self.input_specs[self.input_names[0]]
            feed = {self.input_names[0]: tf.cast(tf.convert_to_tensor(inputs), spec.dtype)}
        result = self.fn(**feed)
        if isinstance(result, dict) and len(result) == 1:
            return next(iter(result.values()))
        return result

    def predict(self, inputs: Any, batch_size: Optional[int] = None) -> np.ndarray:
        if batch_size is None:
            out = self(inputs)
            return out.numpy() if hasattr(out, "numpy") else out
        arr = np.asarray(inputs)
        chunks = [self(arr[i : i + batch_size]).numpy() for i in range(0, len(arr), batch_size)]
        return np.concatenate(chunks, axis=0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SavedModelPredictor(path={self.path!r}, signature={self.signature_name!r})"


def load_model(
    path: PathLike,
    custom_objects: Optional[Dict[str, Any]] = None,
    compile: bool = True,
) -> Union[keras.Model, SavedModelPredictor]:
    """
    Load a ``.keras``/``.h5`` file as a ``keras.Model`` or a SavedModel directory as
    a :class:`SavedModelPredictor` (or a ``keras.Model`` under Keras 2 when possible).
    """
    path_str = str(path)
    if is_keras_file(path_str):
        return keras.models.load_model(path_str, custom_objects=custom_objects, compile=compile)
    if is_saved_model_dir(path_str):
        if IS_LEGACY_KERAS:
            try:
                loaded = keras.models.load_model(
                    path_str, custom_objects=custom_objects, compile=compile
                )
                if isinstance(loaded, keras.Model):
                    return loaded
            except Exception as exc:  # exported (inference-only) SavedModel
                logger.debug("Falling back to SavedModelPredictor: %s", exc)
        return SavedModelPredictor(path_str)
    raise FileNotFoundError(f"No Keras file or SavedModel directory found at: {path_str}")


def checkpoint_filepath(directory: PathLike, name: str = "best", weights_only: bool = False) -> str:
    """Return a ModelCheckpoint-compatible path for the running Keras generation."""
    suffix = ".weights.h5" if weights_only else NATIVE_MODEL_EXTENSION
    return str(Path(directory) / f"{name}{suffix}")


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def count_params(weights: Iterable[Any]) -> int:
    """Count scalar parameters across an iterable of variables/tensors."""
    total = 0
    for w in weights:
        shape = getattr(w, "shape", None)
        if shape is None:
            continue
        dims = shape.as_list() if hasattr(shape, "as_list") else list(shape)
        total += int(np.prod([d if d is not None else 0 for d in dims])) if dims else 1
    return total


def layer_output_shape(layer: keras.layers.Layer) -> Optional[Tuple[Any, ...]]:
    """Best-effort output shape of a layer, or ``None`` when it is not yet built."""
    shape = getattr(layer, "output_shape", None)
    if shape is None:
        try:
            output = layer.output
        except (AttributeError, ValueError, RuntimeError):
            return None
        if isinstance(output, (list, tuple)):
            return tuple(tuple(o.shape) for o in output)
        shape = output.shape
    if isinstance(shape, list):
        return tuple(tuple(s) for s in shape)
    return tuple(shape)


def model_input_shape(model: Any) -> Tuple[Any, ...]:
    shape = getattr(model, "input_shape", None)
    if shape is None:
        raise ValueError("Model has no static input shape (build or call it first).")
    if isinstance(shape, list):
        return tuple(tuple(s) for s in shape)
    return tuple(shape)


def input_signature(model: keras.Model, dtype: tf.DType = tf.float32) -> List[tf.TensorSpec]:
    """Build a ``tf.TensorSpec`` list matching ``model.inputs`` (batch dim = ``None``)."""
    specs = []
    inputs = model.inputs if getattr(model, "inputs", None) else [None]
    shapes = model_input_shape(model)
    if not isinstance(shapes[0], (tuple, list)):
        shapes = (shapes,)
    for i, (inp, shape) in enumerate(zip(inputs, shapes)):
        name = getattr(inp, "name", None) or f"input_{i}"
        name = name.split(":")[0]
        in_dtype = getattr(inp, "dtype", dtype) or dtype
        specs.append(tf.TensorSpec(shape=(None,) + tuple(shape[1:]), dtype=in_dtype, name=name))
    return specs


def metric_names(model: keras.Model) -> List[str]:
    """Names of compiled metrics (Keras 3 hides them behind ``compile_metrics``)."""
    names: List[str] = []
    for metric in getattr(model, "metrics", []):
        inner = getattr(metric, "metrics", None)
        if inner and getattr(metric, "name", "") == "compile_metrics":
            names.extend(m.name for m in inner)
        else:
            names.append(metric.name)
    return names


def compute_loss(model: keras.Model, x: Any, y: Any, y_pred: Any, sample_weight: Any = None) -> Any:
    """Evaluate the compiled loss of ``model`` (works for Keras 2 and 3)."""
    if y is not None and not isinstance(y, (tf.Tensor, dict, list, tuple)):
        y = tf.convert_to_tensor(y)
    if hasattr(model, "compute_loss"):
        return model.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
    return model.compiled_loss(y, y_pred, sample_weight=sample_weight)  # pragma: no cover


def unpack_batch(batch: Any) -> Tuple[Any, Any, Any]:
    """Split a dataset element into ``(x, y, sample_weight)``."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) == 1:
            return batch[0], None, None
    return batch, None, None


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


def set_mixed_precision_policy(policy: str = "mixed_float16") -> None:
    keras.mixed_precision.set_global_policy(policy)


def global_policy_name() -> str:
    policy = keras.mixed_precision.global_policy()
    return getattr(policy, "name", str(policy))


def wrap_loss_scale_optimizer(optimizer: keras.optimizers.Optimizer) -> keras.optimizers.Optimizer:
    """Wrap ``optimizer`` in a ``LossScaleOptimizer`` unless it already is one."""
    lso_cls = keras.mixed_precision.LossScaleOptimizer
    if isinstance(optimizer, lso_cls):
        return optimizer
    return lso_cls(optimizer)


def is_loss_scale_optimizer(optimizer: Any) -> bool:
    return isinstance(optimizer, keras.mixed_precision.LossScaleOptimizer)


def current_loss_scale(optimizer: Any) -> Optional[tf.Tensor]:
    """Current loss scale of a ``LossScaleOptimizer`` (``None`` for plain optimizers)."""
    if not is_loss_scale_optimizer(optimizer):
        return None
    if hasattr(optimizer, "loss_scale"):  # Keras 2
        return tf.convert_to_tensor(optimizer.loss_scale)
    if getattr(optimizer, "built", False) and hasattr(optimizer, "dynamic_scale"):  # Keras 3
        return tf.convert_to_tensor(optimizer.dynamic_scale)
    initial = getattr(optimizer, "initial_scale", None)
    return tf.convert_to_tensor(initial) if initial is not None else None


def scale_loss(optimizer: Any, loss: tf.Tensor) -> tf.Tensor:
    if hasattr(optimizer, "scale_loss"):  # Keras 3
        return optimizer.scale_loss(loss)
    if hasattr(optimizer, "get_scaled_loss"):  # Keras 2
        return optimizer.get_scaled_loss(loss)
    return loss


def unscale_gradients(optimizer: Any, gradients: Sequence[Any]) -> List[Any]:
    """
    Undo loss scaling on ``gradients``.

    Keras 2 exposes ``get_unscaled_gradients``. Keras 3's ``LossScaleOptimizer``
    unscales inside ``apply``; to allow clipping before ``apply`` we divide by the
    current dynamic scale explicitly, and ``apply`` then sees already-unscaled
    gradients — so we return them together with a flag telling the caller to use
    the *inner* optimizer.  See :func:`apply_gradients`.
    """
    if hasattr(optimizer, "get_unscaled_gradients"):
        return list(optimizer.get_unscaled_gradients(list(gradients)))
    return list(gradients)


def apply_gradients(optimizer: Any, gradients: Sequence[Any], variables: Sequence[Any]) -> None:
    """Apply ``(gradient, variable)`` pairs on either Keras generation."""
    pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
    optimizer.apply_gradients(pairs)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def register_serializable(package: str = "TensorVerseHub", name: Optional[str] = None) -> Callable:
    """Decorator that registers a class with Keras' serialization registry."""
    saving = getattr(keras, "saving", None)
    if saving is not None and hasattr(saving, "register_keras_serializable"):
        return saving.register_keras_serializable(package=package, name=name)
    return keras.utils.register_keras_serializable(package=package, name=name)


def version_info() -> Dict[str, Any]:
    """Summary of the runtime used for diagnostics and bug reports."""
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow": TF_VERSION,
        "keras": KERAS_VERSION,
        "keras_generation": 3 if IS_KERAS_3 else 2,
        "legacy_keras_env": os.environ.get("TF_USE_LEGACY_KERAS", ""),
        "numpy": np.__version__,
        "gpus": [g.name for g in gpus],
        "mixed_precision_policy": global_policy_name(),
    }


__all__ = [
    "IS_KERAS_3",
    "IS_LEGACY_KERAS",
    "KERAS_VERSION",
    "MIN_TF_VERSION",
    "NATIVE_MODEL_EXTENSION",
    "SavedModelPredictor",
    "TF_VERSION",
    "apply_gradients",
    "check_tensorflow_version",
    "checkpoint_filepath",
    "compute_loss",
    "count_params",
    "current_loss_scale",
    "export_saved_model",
    "global_policy_name",
    "input_signature",
    "is_keras_file",
    "is_loss_scale_optimizer",
    "is_saved_model_dir",
    "keras",
    "layer_output_shape",
    "load_model",
    "metric_names",
    "model_input_shape",
    "register_serializable",
    "require_legacy_keras",
    "save_model",
    "scale_loss",
    "set_mixed_precision_policy",
    "unpack_batch",
    "unscale_gradients",
    "version_info",
    "wrap_loss_scale_optimizer",
]
