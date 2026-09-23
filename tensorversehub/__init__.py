"""
TensorVerseHub — production utilities for TensorFlow 2.17+ / Keras.

The package is import-cheap: submodules are loaded lazily on first attribute
access and importing it has **no side effects** (no GPU configuration, no
printing, no global plotting style).  Call :func:`configure_tensorflow` explicitly
when you want memory growth, mixed precision or XLA enabled.

Example::

    import tensorversehub as tvh

    tvh.configure_tensorflow(mixed_precision=True)
    model = tvh.model_utils.ModelBuilders.create_cnn_classifier((32, 32, 3), 10)
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List

__version__ = "2.0.0"
__author__ = "Satvik Praveen"
__license__ = "MIT"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SUBMODULES = (
    "compat",
    "data_utils",
    "export_utils",
    "model_utils",
    "optimization_utils",
    "training_utils",
    "visualization",
)

# Public symbols re-exported from submodules (resolved lazily).
_LAZY_ATTRS: Dict[str, str] = {
    # data_utils
    "DataAugmentation": "data_utils",
    "DataPipeline": "data_utils",
    "TFRecordHandler": "data_utils",
    "create_image_classification_pipeline": "data_utils",
    "create_text_classification_pipeline": "data_utils",
    "create_tfrecord_dataset": "data_utils",
    # model_utils
    "CustomLayers": "model_utils",
    "ModelAnalysis": "model_utils",
    "ModelBuilders": "model_utils",
    "TrainingUtilities": "model_utils",
    "create_classification_model": "model_utils",
    "create_transfer_learning_model": "model_utils",
    "load_model_with_metadata": "model_utils",
    "save_model_with_metadata": "model_utils",
    # training_utils
    "CustomTrainingLoop": "training_utils",
    "EarlyStoppingHandler": "training_utils",
    "GradientClipping": "training_utils",
    "LearningRateFinder": "training_utils",
    "MetricsTracker": "training_utils",
    "WarmupCosineSchedule": "training_utils",
    # optimization_utils
    "KnowledgeDistillation": "optimization_utils",
    "MixedPrecisionOptimization": "optimization_utils",
    "ModelCompression": "optimization_utils",
    "ModelPruning": "optimization_utils",
    "ModelQuantization": "optimization_utils",
    "create_inference_optimized_model": "optimization_utils",
    "optimize_for_mobile": "optimization_utils",
    # export_utils
    "MultiFormatExporter": "export_utils",
    "ONNXExporter": "export_utils",
    "SavedModelExporter": "export_utils",
    "SavedModelPredictor": "compat",
    "TFLiteExporter": "export_utils",
    "TensorFlowJSExporter": "export_utils",
    "create_deployment_package": "export_utils",
    "quick_export": "export_utils",
    # visualization
    "AdvancedVisualization": "visualization",
    "DataVisualization": "visualization",
    "ModelVisualization": "visualization",
    "TrainingVisualization": "visualization",
    "quick_model_analysis": "visualization",
    "setup_plotting_style": "visualization",
}

__all__: List[str] = [
    "__version__",
    "configure_tensorflow",
    "about",
    *_SUBMODULES,
    *sorted(_LAZY_ATTRS),
]

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from . import (  # noqa: F401
        compat,
        data_utils,
        export_utils,
        model_utils,
        optimization_utils,
        training_utils,
        visualization,
    )


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module = importlib.import_module(f"{__name__}.{_LAZY_ATTRS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def configure_tensorflow(
    memory_growth: bool = True,
    mixed_precision: bool = False,
    xla: bool = False,
    deterministic: bool = False,
    seed: int | None = None,
    log_level: int = logging.INFO,
) -> Dict[str, Any]:
    """
    Configure the TensorFlow runtime in one call.

    Args:
        memory_growth:  Allocate GPU memory on demand instead of grabbing it all.
        mixed_precision: Enable the ``mixed_float16`` global dtype policy.
        xla:             Turn on XLA JIT compilation for eligible ops.
        deterministic:   Enable op determinism (slower, reproducible).
        seed:            Seed Python, NumPy and TensorFlow RNGs.
        log_level:       Level for the ``tensorversehub`` logger.

    Returns:
        The runtime summary from :func:`tensorversehub.compat.version_info`.
    """
    import tensorflow as tf

    from . import compat

    logging.getLogger(__name__).setLevel(log_level)
    compat.check_tensorflow_version()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus and memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:  # already initialised
                logger.warning("Could not set memory growth on %s: %s", gpu.name, exc)
        logger.info("Configured %d GPU(s) with memory growth", len(gpus))

    if mixed_precision:
        compat.set_mixed_precision_policy("mixed_float16")
        logger.info("Mixed precision policy enabled: mixed_float16")

    if xla:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA JIT compilation enabled")

    if deterministic:
        tf.config.experimental.enable_op_determinism()
        logger.info("Op determinism enabled")

    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
        logger.info("Global random seed set to %d", seed)

    info = compat.version_info()
    logger.info(
        "TensorVerseHub %s | TensorFlow %s | Keras %s",
        __version__,
        info["tensorflow"],
        info["keras"],
    )
    return info


def about() -> Dict[str, Any]:
    """Return version and runtime information (safe to call without a GPU)."""
    from . import compat

    return {"tensorversehub": __version__, **compat.version_info()}
