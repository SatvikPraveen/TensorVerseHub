"""
Model compression and inference optimisation.

* :class:`ModelQuantization` – post-training TFLite quantisation (dynamic / float16 /
  full int8) and quantisation-aware training.
* :class:`ModelPruning` – magnitude pruning via ``tensorflow-model-optimization``
  plus a framework-independent sparsity report.
* :class:`KnowledgeDistillation` – temperature-scaled KL distillation with a
  ``Distiller`` model *and* a manual loop.
* :class:`MixedPrecisionOptimization`, :class:`ModelCompression`,
  :class:`TensorRTOptimization` and convenience functions.

``tensorflow-model-optimization`` only supports the legacy Keras 2 API; the
functions that need it raise a clear error under Keras 3 (see
:func:`tensorversehub.compat.require_legacy_keras`).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from . import compat
from .compat import keras

logger = logging.getLogger(__name__)

QUANTIZATION_TYPES = ("default", "dynamic", "float16", "int8", "int8_fallback")


def _require_tfmot(feature: str) -> Any:
    compat.require_legacy_keras(feature)
    try:
        import tensorflow_model_optimization as tfmot  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"{feature} requires tensorflow-model-optimization: "
            "pip install 'tensorversehub[optimization]'"
        ) from exc
    return tfmot


def _representative_generator(
    dataset: Union[tf.data.Dataset, Iterable[Any], np.ndarray], num_samples: int = 100
) -> Callable[[], Iterable[List[tf.Tensor]]]:
    """Build a TFLite ``representative_dataset`` generator yielding single samples."""

    def gen() -> Iterable[List[tf.Tensor]]:
        count = 0
        if isinstance(dataset, np.ndarray):
            for sample in dataset:
                if count >= num_samples:
                    return
                yield [tf.cast(tf.convert_to_tensor(sample)[tf.newaxis], tf.float32)]
                count += 1
            return
        for batch in dataset:
            x, _, _ = compat.unpack_batch(batch)
            if isinstance(x, dict):
                x = next(iter(x.values()))
            x = tf.convert_to_tensor(x)
            if x.shape.rank == 0:
                continue
            for i in range(int(x.shape[0])):
                if count >= num_samples:
                    return
                yield [tf.cast(x[i : i + 1], tf.float32)]
                count += 1

    return gen


def _tflite_converter(model: keras.Model) -> tf.lite.TFLiteConverter:
    return tf.lite.TFLiteConverter.from_keras_model(model)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------


class ModelQuantization:
    """Post-training quantisation and quantisation-aware training."""

    @staticmethod
    def quantize_model_post_training(
        model: keras.Model,
        representative_dataset: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
        optimization_type: str = "default",
        num_calibration_samples: int = 100,
    ) -> bytes:
        """
        Convert to TFLite with the requested quantisation.

        * ``default``/``dynamic`` – dynamic-range int8 weights, float activations
        * ``float16``             – half-precision weights
        * ``int8``                – full integer (needs ``representative_dataset``)
        * ``int8_fallback``       – int8 where possible, float ops elsewhere
        """
        if optimization_type not in QUANTIZATION_TYPES:
            raise ValueError(
                f"Unknown optimization_type '{optimization_type}'. Choose from {QUANTIZATION_TYPES}"
            )
        converter = _tflite_converter(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if optimization_type == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif optimization_type in ("int8", "int8_fallback"):
            if representative_dataset is None:
                raise ValueError("int8 quantization requires a representative_dataset")
            converter.representative_dataset = _representative_generator(
                representative_dataset, num_calibration_samples
            )
            if optimization_type == "int8":
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        return converter.convert()

    @staticmethod
    def quantize_model_qat(
        model: keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        loss: Optional[Any] = None,
        metrics: Optional[Sequence[Any]] = None,
        verbose: int = 1,
    ) -> keras.Model:
        """Quantisation-aware fine-tuning (legacy Keras + tfmot only)."""
        tfmot = _require_tfmot("Quantization-aware training")
        q_model = tfmot.quantization.keras.quantize_model(model)
        q_model.compile(
            optimizer=optimizer,
            loss=loss or getattr(model, "loss", None) or "sparse_categorical_crossentropy",
            metrics=list(metrics or ["accuracy"]),
        )
        callbacks = []
        if validation_dataset is not None:
            callbacks.append(keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))
        q_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=callbacks,
            verbose=verbose,
        )
        return q_model

    @staticmethod
    def convert_qat_to_tflite(qat_model: keras.Model) -> bytes:
        converter = _tflite_converter(qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class ModelPruning:
    """Magnitude pruning (tfmot) and sparsity reporting."""

    @staticmethod
    def create_pruning_schedule(
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        begin_step: int = 0,
        end_step: int = 10000,
        frequency: int = 100,
    ) -> Any:
        tfmot = _require_tfmot("Pruning schedules")
        if not 0.0 <= initial_sparsity <= final_sparsity < 1.0:
            raise ValueError("Require 0 <= initial_sparsity <= final_sparsity < 1")
        return tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
            frequency=frequency,
        )

    @staticmethod
    def create_pruned_model(
        model: keras.Model,
        pruning_schedule: Any,
        target_layers: Optional[Sequence[str]] = None,
    ) -> keras.Model:
        """Wrap the whole model or only ``target_layers`` with pruning wrappers."""
        tfmot = _require_tfmot("Model pruning")
        prune = tfmot.sparsity.keras.prune_low_magnitude
        if target_layers is None:
            return prune(model, pruning_schedule=pruning_schedule)
        targets = set(target_layers)

        def _clone(layer: keras.layers.Layer) -> keras.layers.Layer:
            if layer.name in targets:
                return prune(layer, pruning_schedule=pruning_schedule)
            return layer

        return keras.models.clone_model(model, clone_function=_clone)

    @staticmethod
    def pruning_callbacks(log_dir: Optional[str] = None) -> List[keras.callbacks.Callback]:
        tfmot = _require_tfmot("Pruning callbacks")
        callbacks: List[keras.callbacks.Callback] = [tfmot.sparsity.keras.UpdatePruningStep()]
        if log_dir:
            callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir))
        return callbacks

    @staticmethod
    def train_pruned_model(
        pruned_model: keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 20,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        loss: Any = "sparse_categorical_crossentropy",
        metrics: Optional[Sequence[Any]] = None,
        log_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> keras.Model:
        pruned_model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics or ["accuracy"]))
        callbacks = ModelPruning.pruning_callbacks(log_dir)
        if validation_dataset is not None:
            callbacks.append(keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
        pruned_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=callbacks,
            verbose=verbose,
        )
        return pruned_model

    @staticmethod
    def finalize_pruned_model(pruned_model: keras.Model) -> keras.Model:
        tfmot = _require_tfmot("strip_pruning")
        return tfmot.sparsity.keras.strip_pruning(pruned_model)

    @staticmethod
    def compute_sparsity(model: keras.Model, threshold: float = 0.0) -> Dict[str, Any]:
        """Fraction of (near-)zero weights per kernel and overall — works on any Keras model."""
        per_layer: Dict[str, float] = {}
        zeros = total = 0
        for layer in model.layers:
            kernel = getattr(layer, "kernel", None)
            if kernel is None:
                continue
            values = np.asarray(kernel.numpy() if hasattr(kernel, "numpy") else kernel)
            layer_zeros = int(np.sum(np.abs(values) <= threshold))
            per_layer[layer.name] = layer_zeros / max(values.size, 1)
            zeros += layer_zeros
            total += values.size
        return {"overall_sparsity": zeros / max(total, 1), "per_layer": per_layer}


# ---------------------------------------------------------------------------
# Knowledge distillation
# ---------------------------------------------------------------------------


def distillation_loss(
    y_true: tf.Tensor,
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    alpha: float = 0.7,
    temperature: float = 3.0,
    teacher_is_probabilities: bool = False,
) -> tf.Tensor:
    """
    Hinton et al. (2015) distillation loss: ``(1-α)·CE(y, s) + α·T²·KL(t_T ‖ s_T)``.

    ``student_logits`` must be raw logits.  ``teacher_logits`` may be logits or, with
    ``teacher_is_probabilities=True``, softmax outputs (converted via ``log``).
    """
    student_logits = tf.cast(student_logits, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)
    if teacher_is_probabilities:
        teacher_logits = tf.math.log(tf.clip_by_value(teacher_logits, 1e-7, 1.0))
    hard = keras.losses.sparse_categorical_crossentropy(y_true, student_logits, from_logits=True)
    teacher_soft = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_log_soft = tf.nn.log_softmax(student_logits / temperature, axis=-1)
    soft = tf.reduce_sum(
        teacher_soft * (tf.math.log(teacher_soft + 1e-7) - student_log_soft), axis=-1
    )
    return (1.0 - alpha) * hard + alpha * (temperature**2) * soft


class KnowledgeDistillation:
    """
    Train a small student model to mimic a larger teacher.

    Args:
        teacher_model: Trained teacher (outputs logits or probabilities).
        alpha:         Weight of the soft (distillation) term.
        temperature:   Softmax temperature.
        teacher_outputs_probabilities: Set when the teacher ends with softmax.
    """

    STUDENT_ARCHITECTURES = ("simple_cnn", "mobilenet")

    def __init__(
        self,
        teacher_model: keras.Model,
        alpha: float = 0.7,
        temperature: float = 3.0,
        teacher_outputs_probabilities: bool = False,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.teacher_model = teacher_model
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.teacher_outputs_probabilities = teacher_outputs_probabilities

    def create_distillation_loss(self) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        def loss_fn(
            y_true: tf.Tensor, student_logits: tf.Tensor, teacher_out: tf.Tensor
        ) -> tf.Tensor:
            return distillation_loss(
                y_true,
                student_logits,
                teacher_out,
                alpha=self.alpha,
                temperature=self.temperature,
                teacher_is_probabilities=self.teacher_outputs_probabilities,
            )

        return loss_fn

    def create_student_model(
        self,
        student_architecture: str,
        input_shape: Tuple[int, ...],
        num_classes: int,
        weights: Optional[str] = None,
    ) -> keras.Model:
        """Student network that outputs **logits** (no final activation)."""
        if student_architecture not in self.STUDENT_ARCHITECTURES:
            raise ValueError(f"student_architecture must be one of {self.STUDENT_ARCHITECTURES}")
        inputs = keras.Input(shape=input_shape, name="image")
        if student_architecture == "simple_cnn":
            x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(0.3)(x)
        else:
            base = keras.applications.MobileNetV2(
                input_shape=input_shape, include_top=False, weights=weights
            )
            base.trainable = True
            x = base(inputs)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(num_classes, name="logits")(x)
        return keras.Model(inputs, outputs, name=f"student_{student_architecture}")

    def train_student_model(
        self,
        student_model: keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 20,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
    ) -> Dict[str, List[float]]:
        """Manual distillation loop; returns a history dict."""
        optimizer = optimizer or keras.optimizers.Adam()
        loss_fn = self.create_distillation_loss()
        acc = keras.metrics.SparseCategoricalAccuracy()

        @tf.function
        def train_step(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            teacher_out = self.teacher_model(x, training=False)
            with tf.GradientTape() as tape:
                student_out = student_model(x, training=True)
                loss = tf.reduce_mean(loss_fn(y, student_out, teacher_out))
            grads = tape.gradient(loss, student_model.trainable_variables)
            compat.apply_gradients(optimizer, grads, student_model.trainable_variables)
            return loss

        history: Dict[str, List[float]] = {"loss": [], "val_loss": [], "val_accuracy": []}
        for epoch in range(epochs):
            losses = [float(train_step(x, y)) for x, y in train_dataset]
            history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
            if validation_dataset is not None:
                acc.reset_state()
                val_losses = []
                for x, y in validation_dataset:
                    t_out = self.teacher_model(x, training=False)
                    s_out = student_model(x, training=False)
                    val_losses.append(float(tf.reduce_mean(loss_fn(y, s_out, t_out))))
                    acc.update_state(y, s_out)
                history["val_loss"].append(float(np.mean(val_losses)))
                history["val_accuracy"].append(float(acc.result()))
            logger.info(
                "Distillation epoch %d/%d: %s",
                epoch + 1,
                epochs,
                {k: round(v[-1], 4) for k, v in history.items() if v},
            )
        return history


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


class MixedPrecisionOptimization:
    """Helpers for ``mixed_float16`` training."""

    @staticmethod
    def enable_mixed_precision(policy: str = "mixed_float16") -> None:
        compat.set_mixed_precision_policy(policy)
        logger.info("Mixed precision policy set to %s", policy)

    @staticmethod
    def disable_mixed_precision() -> None:
        compat.set_mixed_precision_policy("float32")

    @staticmethod
    def create_mixed_precision_model(model: keras.Model) -> keras.Model:
        """Clone ``model`` forcing the final ``Dense`` layer to float32 for numerical stability."""
        last_dense = next(
            (l.name for l in reversed(model.layers) if isinstance(l, keras.layers.Dense)), None
        )

        def _clone(layer: keras.layers.Layer) -> keras.layers.Layer:
            if layer.name == last_dense:
                config = layer.get_config()
                config["dtype"] = "float32"
                return keras.layers.Dense.from_config(config)
            return layer.__class__.from_config(layer.get_config())

        clone = keras.models.clone_model(model, clone_function=_clone)
        clone.set_weights(model.get_weights())
        return clone

    @staticmethod
    def compile_mixed_precision_model(
        model: keras.Model,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        loss: Any = "sparse_categorical_crossentropy",
        metrics: Optional[Sequence[Any]] = None,
    ) -> keras.Model:
        """Compile with a loss-scaled optimizer (required for stable float16 training)."""
        opt = keras.optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer
        model.compile(
            optimizer=compat.wrap_loss_scale_optimizer(opt),
            loss=loss,
            metrics=list(metrics or ["accuracy"]),
        )
        return model


# ---------------------------------------------------------------------------
# Compression pipeline
# ---------------------------------------------------------------------------


def model_size_bytes(model: keras.Model) -> int:
    total = 0
    for w in model.weights:
        total += compat.count_params([w]) * tf.as_dtype(w.dtype).size
    return total


class ModelCompression:
    """Prune → strip → quantise pipeline and size analysis."""

    @staticmethod
    def apply_magnitude_pruning_and_quantization(
        model: keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        target_sparsity: float = 0.5,
        epochs: int = 10,
        quantization: str = "dynamic",
        verbose: int = 1,
    ) -> Tuple[keras.Model, bytes]:
        """Returns ``(stripped_pruned_model, tflite_bytes)``."""
        steps_per_epoch = int(train_dataset.cardinality().numpy())
        if steps_per_epoch <= 0:
            steps_per_epoch = 100
        total_steps = epochs * steps_per_epoch
        schedule = ModelPruning.create_pruning_schedule(
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=total_steps,
            frequency=max(1, min(100, total_steps // 10)),
        )
        pruned = ModelPruning.create_pruned_model(model, schedule)
        pruned = ModelPruning.train_pruned_model(
            pruned, train_dataset, validation_dataset, epochs, verbose=verbose
        )
        stripped = ModelPruning.finalize_pruned_model(pruned)
        tflite = ModelQuantization.quantize_model_post_training(
            stripped, train_dataset if quantization.startswith("int8") else None, quantization
        )
        return stripped, tflite

    @staticmethod
    def analyze_compression_ratio(
        original_model: keras.Model,
        compressed_model: keras.Model,
        tflite_model: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        original_params = original_model.count_params()
        compressed_params = compressed_model.count_params()
        original_mb = model_size_bytes(original_model) / (1024 * 1024)
        compressed_mb = model_size_bytes(compressed_model) / (1024 * 1024)
        analysis: Dict[str, Any] = {
            "original_parameters": original_params,
            "compressed_parameters": compressed_params,
            "parameter_reduction_ratio": original_params / compressed_params
            if compressed_params
            else 0.0,
            "parameter_reduction_percentage": (1 - compressed_params / original_params) * 100
            if original_params
            else 0.0,
            "original_size_mb": original_mb,
            "compressed_size_mb": compressed_mb,
            "size_reduction_ratio": original_mb / compressed_mb if compressed_mb else 0.0,
            "compressed_sparsity": ModelPruning.compute_sparsity(compressed_model)[
                "overall_sparsity"
            ],
        }
        if tflite_model:
            tflite_mb = len(tflite_model) / (1024 * 1024)
            analysis["tflite_size_mb"] = tflite_mb
            analysis["tflite_compression_ratio"] = original_mb / tflite_mb if tflite_mb else 0.0
        return analysis


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def optimize_for_mobile(
    model: keras.Model,
    representative_dataset: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
    target_size_mb: float = 5.0,
    strategies: Sequence[str] = ("dynamic", "float16", "int8"),
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Try several TFLite quantisation strategies and pick the smallest that meets
    ``target_size_mb`` (or the smallest overall).  Returns ``(tflite_bytes, report)``.
    """
    results: Dict[str, Any] = {}
    for strategy in strategies:
        try:
            if strategy.startswith("int8") and representative_dataset is None:
                raise ValueError("no representative dataset")
            tflite = ModelQuantization.quantize_model_post_training(
                model, representative_dataset, strategy
            )
            size_mb = len(tflite) / (1024 * 1024)
            results[strategy] = {
                "model": tflite,
                "size_mb": size_mb,
                "meets_target": size_mb <= target_size_mb,
            }
        except Exception as exc:  # strategy unsupported for this model
            results[strategy] = {"error": str(exc)}

    candidates = {k: v for k, v in results.items() if "model" in v}
    if not candidates:
        raise RuntimeError(f"All quantization strategies failed: {results}")
    meeting = {k: v for k, v in candidates.items() if v["meets_target"]}
    pool = meeting or candidates
    best = min(pool, key=lambda k: pool[k]["size_mb"])
    report: Dict[str, Any] = {
        k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()
    }
    report["selected"] = best
    report["target_size_mb"] = target_size_mb
    return candidates[best]["model"], report


OPTIMIZATION_LEVELS = ("conservative", "moderate", "aggressive")


def create_inference_optimized_model(
    model: keras.Model,
    input_shape: Optional[Tuple[int, ...]] = None,
    optimization_level: str = "aggressive",
) -> keras.Model:
    """
    Clone ``model`` for inference.

    * ``conservative`` – plain clone (fresh graph, no training state)
    * ``moderate``     – clone + attaches a traced ``predict_fn`` (``tf.function``)
    * ``aggressive``   – clone compiled with ``jit_compile=True`` (XLA)
    """
    if optimization_level not in OPTIMIZATION_LEVELS:
        raise ValueError(f"optimization_level must be one of {OPTIMIZATION_LEVELS}")
    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    if optimization_level == "conservative":
        return clone

    if input_shape is None:
        input_shape = tuple(compat.model_input_shape(model)[1:])
    spec = tf.TensorSpec((None, *input_shape), tf.float32)
    predict_fn = tf.function(
        lambda x: clone(x, training=False),
        input_signature=[spec],
        jit_compile=optimization_level == "aggressive",
    )
    clone.predict_fn = predict_fn  # type: ignore[attr-defined]
    if optimization_level == "aggressive":
        clone.compile(
            optimizer=getattr(model, "optimizer", None) or "adam",
            loss=getattr(model, "loss", None),
            jit_compile=True,
        )
    return clone


def benchmark_model_performance(
    original_model: Any,
    optimized_model: Any,
    test_input: Union[tf.Tensor, np.ndarray],
    num_runs: int = 100,
    warmup: int = 3,
) -> Dict[str, float]:
    """Latency comparison; uses ``predict_fn`` when a model exposes one."""

    def _callable(m: Any) -> Callable[[Any], Any]:
        fn = getattr(m, "predict_fn", None)
        if fn is not None:
            return fn
        return lambda x: m(x, training=False)

    x = tf.convert_to_tensor(test_input)
    timings = {}
    for name, model in (("original", original_model), ("optimized", optimized_model)):
        fn = _callable(model)
        for _ in range(warmup):
            fn(x)
        start = time.perf_counter()
        for _ in range(num_runs):
            fn(x)
        timings[name] = (time.perf_counter() - start) / num_runs * 1000
    speedup = timings["original"] / timings["optimized"] if timings["optimized"] > 0 else 0.0
    return {
        "original_avg_time_ms": timings["original"],
        "optimized_avg_time_ms": timings["optimized"],
        "speedup_ratio": speedup,
        "performance_improvement_percent": (speedup - 1) * 100,
    }


class TensorRTOptimization:
    """TensorRT conversion (requires a TensorRT-enabled TensorFlow build)."""

    @staticmethod
    def convert_to_tensorrt(
        model: keras.Model,
        precision: str = "FP16",
        output_dir: Optional[str] = None,
    ) -> Any:
        """Convert via TF-TRT; returns the loaded TRT SavedModel or ``model`` if unavailable."""
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt  # type: ignore
        except ImportError:
            logger.warning("TensorRT not available; returning the original model")
            return model
        tmp = output_dir or tempfile.mkdtemp(prefix="tvh_trt_")
        try:
            saved_model_path = os.path.join(tmp, "saved_model")
            compat.export_saved_model(model, saved_model_path)
            params = trt.TrtConversionParams(
                precision_mode=precision, use_calibration=precision == "INT8"
            )
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=saved_model_path, conversion_params=params
            )
            converter.convert()
            trt_path = os.path.join(tmp, "trt_model")
            converter.save(trt_path)
            return tf.saved_model.load(trt_path)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("TensorRT conversion failed: %s", exc)
            return model


def create_optimization_report(
    original_model: keras.Model,
    optimized_components: Dict[str, Any],
    performance_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Markdown summary of pruning / quantisation / latency results."""
    original_params = original_model.count_params()
    original_mb = model_size_bytes(original_model) / (1024 * 1024)
    lines = [
        "# Model Optimization Report",
        "",
        "## Original Model",
        f"- **Parameters**: {original_params:,}",
        f"- **Layers**: {len(original_model.layers)}",
        f"- **Model Size**: {original_mb:.2f} MB",
        "",
        "## Optimization Results",
    ]
    pruned = optimized_components.get("pruned_model")
    if pruned is not None:
        sparsity = ModelPruning.compute_sparsity(pruned)["overall_sparsity"]
        lines += ["", "### Pruning", f"- **Sparsity**: {sparsity * 100:.1f}%"]
    q_size = optimized_components.get("quantized_model_size")
    if q_size is not None:
        reduction = (original_mb - q_size) / original_mb * 100 if original_mb else 0.0
        lines += [
            "",
            "### Quantization",
            f"- **Quantized Model Size**: {q_size:.2f} MB",
            f"- **Size Reduction**: {reduction:.1f}%",
        ]
    if performance_metrics:
        speedup = performance_metrics.get("speedup_ratio", 1.0)
        lines += [
            "",
            "## Performance Metrics",
            f"- **Original Inference Time**: {performance_metrics.get('original_avg_time_ms', 0):.2f} ms",
            f"- **Optimized Inference Time**: {performance_metrics.get('optimized_avg_time_ms', 0):.2f} ms",
            f"- **Speedup**: {speedup:.2f}x",
            "",
            "## Recommendations",
            "- Excellent optimization results"
            if speedup > 1.5
            else "- Good optimization results"
            if speedup > 1.2
            else "- Consider additional optimization techniques (pruning, int8, XLA)",
        ]
    return "\n".join(lines)


__all__ = [
    "KnowledgeDistillation",
    "MixedPrecisionOptimization",
    "ModelCompression",
    "ModelPruning",
    "ModelQuantization",
    "OPTIMIZATION_LEVELS",
    "QUANTIZATION_TYPES",
    "TensorRTOptimization",
    "benchmark_model_performance",
    "create_inference_optimized_model",
    "create_optimization_report",
    "distillation_loss",
    "model_size_bytes",
    "optimize_for_mobile",
]
