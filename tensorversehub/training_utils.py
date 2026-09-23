"""
Custom training engine and schedule / gradient utilities.

* :class:`WarmupCosineSchedule` – serialisable linear warm-up + cosine annealing.
* :class:`GradientClipping` – global-norm, value and graph-mode percentile clipping.
* :class:`MetricsTracker` / :class:`EarlyStoppingHandler` – bookkeeping for manual loops.
* :class:`LearningRateFinder` – LR range test (Smith 2017).
* :class:`CustomTrainingLoop` – ``GradientTape`` engine with mixed precision, gradient
  accumulation, clipping, XLA and Keras callbacks, working on Keras 2 and 3.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from . import compat
from .compat import keras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------


@compat.register_serializable(name="WarmupCosineSchedule")
class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up followed by cosine annealing to ``min_lr``.

    Args:
        base_lr:      Peak learning rate reached at the end of warm-up.
        total_steps:  Total optimisation steps.
        warmup_steps: Warm-up steps; a float in ``(0, 1)`` is a fraction of ``total_steps``.
        min_lr:       Floor of the cosine decay.
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: Union[int, float] = 0.05,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__()
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        if isinstance(warmup_steps, float) and 0 < warmup_steps < 1:
            self.warmup_steps = int(warmup_steps * total_steps)
        else:
            self.warmup_steps = int(warmup_steps)
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps cannot exceed total_steps")

    def __call__(self, step: Any) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup = tf.constant(float(self.warmup_steps), tf.float32)
        total = tf.constant(float(self.total_steps), tf.float32)
        warmup_lr = self.base_lr * step / tf.maximum(warmup, 1.0)
        progress = tf.clip_by_value((step - warmup) / tf.maximum(total - warmup, 1.0), 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


class GradientClipping:
    """Gradient clipping helpers that work in eager and graph mode."""

    @staticmethod
    def clip_by_global_norm(
        gradients: Sequence[Optional[tf.Tensor]], max_norm: float
    ) -> Tuple[List[Optional[tf.Tensor]], tf.Tensor]:
        """Return ``(clipped_gradients, global_norm)``; ``None`` entries are preserved."""
        clipped, global_norm = tf.clip_by_global_norm(list(gradients), max_norm)
        return list(clipped), global_norm

    @staticmethod
    def clip_by_value(
        gradients: Sequence[Optional[tf.Tensor]], clip_min: float = -1.0, clip_max: float = 1.0
    ) -> List[Optional[tf.Tensor]]:
        return [
            tf.clip_by_value(g, clip_min, clip_max) if g is not None else None for g in gradients
        ]

    @staticmethod
    def percentile_threshold(
        gradients: Sequence[Optional[tf.Tensor]], percentile: float = 95.0
    ) -> tf.Tensor:
        """``percentile``-th percentile of ``|g|`` over all gradients (pure TF ops)."""
        flat = tf.concat(
            [tf.reshape(tf.abs(tf.cast(g, tf.float32)), [-1]) for g in gradients if g is not None],
            axis=0,
        )
        n = tf.shape(flat)[0]
        rank = tf.cast(tf.round(percentile / 100.0 * tf.cast(n - 1, tf.float32)), tf.int32)
        rank = tf.clip_by_value(rank, 0, tf.maximum(n - 1, 0))
        return tf.sort(flat)[rank]

    @staticmethod
    def adaptive_clip(
        gradients: Sequence[Optional[tf.Tensor]], percentile: float = 95.0
    ) -> List[Optional[tf.Tensor]]:
        """Clip element-wise to ± the ``percentile``-th percentile of ``|g|``."""
        threshold = GradientClipping.percentile_threshold(gradients, percentile)
        return [
            tf.clip_by_value(g, -tf.cast(threshold, g.dtype), tf.cast(threshold, g.dtype))
            if g is not None
            else None
            for g in gradients
        ]


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------


class MetricsTracker:
    """Accumulates per-batch metrics and stores per-epoch averages in ``history``."""

    def __init__(self) -> None:
        self._train_accum: Dict[str, List[float]] = {}
        self._val_accum: Dict[str, List[float]] = {}
        self.history: Dict[str, List[float]] = {}

    def update_train(self, **metrics: Union[float, tf.Tensor]) -> None:
        for name, value in metrics.items():
            self._train_accum.setdefault(name, []).append(float(value))

    def update_val(self, **metrics: Union[float, tf.Tensor]) -> None:
        for name, value in metrics.items():
            self._val_accum.setdefault(name, []).append(float(value))

    def commit_epoch(self) -> Dict[str, float]:
        """Average the accumulators into ``history`` and return the epoch summary."""
        summary: Dict[str, float] = {}
        for prefix, accum in (("train", self._train_accum), ("val", self._val_accum)):
            for name, values in accum.items():
                key = f"{prefix}_{name}"
                epoch_value = float(np.mean(values)) if values else float("nan")
                self.history.setdefault(key, []).append(epoch_value)
                summary[key] = epoch_value
        self._train_accum.clear()
        self._val_accum.clear()
        return summary

    @property
    def epochs(self) -> int:
        return max((len(v) for v in self.history.values()), default=0)

    def best(self, key: str, mode: str = "min") -> Tuple[int, float]:
        """``(epoch_index, value)`` of the best recorded value for ``key``."""
        values = self.history[key]
        idx = int(np.nanargmin(values) if mode == "min" else np.nanargmax(values))
        return idx, values[idx]

    def reset(self) -> None:
        self._train_accum.clear()
        self._val_accum.clear()
        self.history.clear()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStoppingHandler:
    """Early stopping with ``min_delta``, patience, baseline and best-weight restoration."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best: bool = True,
        baseline: Optional[float] = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best = restore_best
        self._baseline = baseline
        self._wait: int = 0
        self._best: float = 0.0
        self._best_weights: Optional[List[np.ndarray]] = None
        self.reset()

    @property
    def best(self) -> float:
        return self._best

    @property
    def wait(self) -> int:
        return self._wait

    def update(self, epoch_summary: Dict[str, float], model: Optional[keras.Model] = None) -> bool:
        """Call once per epoch. Returns ``True`` when training should stop."""
        if self.monitor not in epoch_summary:
            raise KeyError(
                f"Monitored metric '{self.monitor}' not in epoch summary "
                f"(available: {sorted(epoch_summary)})"
            )
        current = float(epoch_summary[self.monitor])
        if self.mode == "min":
            improved = current < self._best - self.min_delta
        else:
            improved = current > self._best + self.min_delta

        if improved:
            self._best = current
            self._wait = 0
            if self.restore_best and model is not None:
                self._best_weights = model.get_weights()
        else:
            self._wait += 1

        if self._wait >= self.patience:
            if self.restore_best and self._best_weights is not None and model is not None:
                model.set_weights(self._best_weights)
            return True
        return False

    def reset(self) -> None:
        self._wait = 0
        self._best_weights = None
        if self._baseline is not None:
            self._best = float(self._baseline)
        else:
            self._best = float("inf") if self.mode == "min" else float("-inf")


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------


def get_learning_rate(optimizer: Any) -> float:
    lr = optimizer.learning_rate
    if callable(lr) and not isinstance(lr, tf.Variable):
        lr = lr(optimizer.iterations)
    return float(tf.convert_to_tensor(lr))


def set_learning_rate(optimizer: Any, value: float) -> None:
    lr = optimizer.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(value)
    else:
        optimizer.learning_rate = value


# ---------------------------------------------------------------------------
# LR range finder
# ---------------------------------------------------------------------------


class LearningRateFinder:
    """
    Learning-rate range test (Smith 2017).

    Increases the LR exponentially from ``min_lr`` to ``max_lr`` over ``num_steps``
    batches while recording an exponentially smoothed loss.  Model weights and the
    optimizer LR are restored afterwards.
    """

    def __init__(
        self,
        model: keras.Model,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        num_steps: int = 100,
        beta: float = 0.98,
        divergence_factor: float = 4.0,
    ) -> None:
        if getattr(model, "optimizer", None) is None:
            raise ValueError("Model must be compiled with an optimizer before running the finder")
        self.model = model
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.num_steps = int(num_steps)
        self.beta = float(beta)
        self.divergence_factor = float(divergence_factor)
        self.lrs: List[float] = []
        self.losses: List[float] = []

    def _lr_at_step(self, step: int) -> float:
        return self.min_lr * (self.max_lr / self.min_lr) ** (step / max(self.num_steps, 1))

    def find(
        self, dataset: tf.data.Dataset, loss_fn: Optional[Callable] = None
    ) -> Tuple[List[float], List[float]]:
        """Run the test and return ``(lrs, smoothed_losses)``."""
        original_weights = self.model.get_weights()
        original_lr = get_learning_rate(self.model.optimizer)
        self.lrs, self.losses = [], []
        avg_loss, best_loss = 0.0, float("inf")

        step = 0
        for batch in dataset:
            if step >= self.num_steps:
                break
            x, y, sw = compat.unpack_batch(batch)
            lr = self._lr_at_step(step)
            set_learning_rate(self.model.optimizer, lr)

            with tf.GradientTape() as tape:
                y_pred = self.model(x, training=True)
                if loss_fn is not None:
                    loss = tf.reduce_mean(loss_fn(y, y_pred))
                else:
                    loss = compat.compute_loss(self.model, x, y, y_pred, sw)
            grads = tape.gradient(loss, self.model.trainable_variables)
            compat.apply_gradients(self.model.optimizer, grads, self.model.trainable_variables)

            loss_val = float(loss)
            avg_loss = self.beta * avg_loss + (1 - self.beta) * loss_val
            smoothed = avg_loss / (1 - self.beta ** (step + 1))
            if step > 0 and smoothed > self.divergence_factor * best_loss:
                break
            best_loss = min(best_loss, smoothed)
            self.lrs.append(lr)
            self.losses.append(smoothed)
            step += 1  # noqa: SIM113

        self.model.set_weights(original_weights)
        set_learning_rate(self.model.optimizer, original_lr)
        return self.lrs, self.losses

    def suggest(self, skip_start: int = 10, skip_end: int = 5) -> Optional[float]:
        """LR at the steepest loss decrease (a common heuristic), or ``None``."""
        lrs = self.lrs[skip_start : len(self.lrs) - skip_end]
        losses = self.losses[skip_start : len(self.losses) - skip_end]
        if len(lrs) < 2:
            return None
        grads = np.gradient(np.asarray(losses), np.log10(np.asarray(lrs)))
        return float(lrs[int(np.argmin(grads))])

    def plot(self, skip_start: int = 10, skip_end: int = 5, show: bool = True) -> Any:
        """Plot loss vs LR on a log axis and return the matplotlib figure."""
        import matplotlib.pyplot as plt

        lrs = self.lrs[skip_start : len(self.lrs) - skip_end]
        losses = self.losses[skip_start : len(self.losses) - skip_end]
        if not lrs:
            logger.warning("Not enough data to plot; run find() first")
            return None
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lrs, losses, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log)")
        ax.set_ylabel("Smoothed loss")
        ax.set_title("Learning rate range test")
        ax.grid(alpha=0.4)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Custom training loop
# ---------------------------------------------------------------------------


class CustomTrainingLoop:
    """
    ``GradientTape`` training engine.

    Args:
        model:               Keras model.
        optimizer:           Keras optimizer (wrapped in ``LossScaleOptimizer`` when
                             ``mixed_precision`` is on).
        loss_fn:             ``loss_fn(y_true, y_pred)`` returning per-sample or scalar loss.
        train_metrics:       Metrics updated on training batches.
        val_metrics:         Metrics updated on validation batches.
        clip_norm:           Global-norm clipping threshold (``0`` disables).
        mixed_precision:     Enable ``mixed_float16`` policy + loss scaling.
        accumulation_steps:  Accumulate gradients over N batches before applying.
        jit_compile:         Compile the train step with XLA.
    """

    def __init__(
        self,
        model: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        loss_fn: Callable,
        train_metrics: Optional[Sequence[keras.metrics.Metric]] = None,
        val_metrics: Optional[Sequence[keras.metrics.Metric]] = None,
        clip_norm: float = 0.0,
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
        jit_compile: bool = False,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self.model = model
        self.loss_fn = loss_fn
        self.train_metrics = list(train_metrics or [])
        self.val_metrics = list(val_metrics or [])
        self.clip_norm = float(clip_norm)
        self.mixed_precision = bool(mixed_precision)
        self.accumulation_steps = int(accumulation_steps)
        self.tracker = MetricsTracker()

        if self.mixed_precision:
            if not compat.global_policy_name().startswith("mixed"):
                compat.set_mixed_precision_policy("mixed_float16")
            optimizer = compat.wrap_loss_scale_optimizer(optimizer)
            if compat.IS_KERAS_3 and self.clip_norm > 0:
                # Keras 3 unscales inside LossScaleOptimizer.apply; let the inner
                # optimizer clip *after* unscaling instead of clipping scaled grads.
                optimizer.inner_optimizer.global_clipnorm = self.clip_norm
        self.optimizer = optimizer

        self._accumulators: Optional[List[tf.Variable]] = None
        self._accum_counter = 0
        self._train_step = tf.function(self._train_step_impl, jit_compile=jit_compile)
        self._val_step = tf.function(self._val_step_impl)

    # -- steps ---------------------------------------------------------------
    def _compute_gradients(self, x: Any, y: Any) -> Tuple[tf.Tensor, List[Any], tf.Tensor]:
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = tf.reduce_mean(self.loss_fn(y, y_pred))
            scaled = compat.scale_loss(self.optimizer, loss) if self.mixed_precision else loss
        grads = tape.gradient(scaled, self.model.trainable_variables)
        if self.mixed_precision and not compat.IS_KERAS_3:
            grads = compat.unscale_gradients(self.optimizer, grads)
        clip_here = self.clip_norm > 0 and not (self.mixed_precision and compat.IS_KERAS_3)
        if clip_here:
            grads, global_norm = GradientClipping.clip_by_global_norm(grads, self.clip_norm)
        else:
            global_norm = tf.linalg.global_norm([g for g in grads if g is not None])
            if self.mixed_precision and compat.IS_KERAS_3:
                scale = compat.current_loss_scale(self.optimizer)
                if scale is not None:  # report the *unscaled* norm
                    global_norm = global_norm / tf.cast(scale, global_norm.dtype)
        for m in self.train_metrics:
            m.update_state(y, y_pred)
        return loss, grads, global_norm

    def _train_step_impl(self, x: Any, y: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        loss, grads, global_norm = self._compute_gradients(x, y)
        compat.apply_gradients(self.optimizer, grads, self.model.trainable_variables)
        return loss, global_norm

    def _accumulate_step(self, x: Any, y: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        loss, grads, global_norm = self._compute_gradients(x, y)
        if self._accumulators is None:
            self._accumulators = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.model.trainable_variables
            ]
        for acc, g in zip(self._accumulators, grads):
            if g is not None:
                acc.assign_add(tf.cast(g, acc.dtype) / self.accumulation_steps)
        return loss, global_norm

    def _apply_accumulated(self) -> None:
        assert self._accumulators is not None
        compat.apply_gradients(
            self.optimizer, [a.value() for a in self._accumulators], self.model.trainable_variables
        )
        for acc in self._accumulators:
            acc.assign(tf.zeros_like(acc))

    def _val_step_impl(self, x: Any, y: Any) -> tf.Tensor:
        y_pred = self.model(x, training=False)
        loss = tf.reduce_mean(self.loss_fn(y, y_pred))
        for m in self.val_metrics:
            m.update_state(y, y_pred)
        return loss

    def train_on_batch(self, x: Any, y: Any) -> Dict[str, float]:
        """Run one optimisation step (honouring gradient accumulation)."""
        if self.accumulation_steps == 1:
            loss, gnorm = self._train_step(x, y)
        else:
            loss, gnorm = self._accumulate_step(x, y)
            self._accum_counter += 1
            if self._accum_counter % self.accumulation_steps == 0:
                self._apply_accumulated()
        return {"loss": float(loss), "grad_norm": float(gnorm)}

    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate ``dataset`` with ``loss_fn`` and ``val_metrics``."""
        for m in self.val_metrics:
            m.reset_state()
        losses = []
        for batch in dataset:
            x, y, _ = compat.unpack_batch(batch)
            losses.append(float(self._val_step(x, y)))
        results = {"loss": float(np.mean(losses)) if losses else float("nan")}
        results.update({m.name: float(m.result()) for m in self.val_metrics})
        return results

    # -- fit -----------------------------------------------------------------
    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        early_stopping: Optional[EarlyStoppingHandler] = None,
        callbacks: Optional[Sequence[keras.callbacks.Callback]] = None,
        steps_per_epoch: Optional[int] = None,
        verbose: int = 1,
    ) -> MetricsTracker:
        """Train for ``epochs`` and return the :class:`MetricsTracker` history."""
        cb_list = keras.callbacks.CallbackList(list(callbacks or []), model=self.model)
        cb_list.on_train_begin()

        for epoch in range(epochs):
            cb_list.on_epoch_begin(epoch)
            for m in self.train_metrics + self.val_metrics:
                m.reset_state()

            for step, batch in enumerate(train_ds):
                if steps_per_epoch is not None and step >= steps_per_epoch:
                    break
                x, y, _ = compat.unpack_batch(batch)
                cb_list.on_train_batch_begin(step)
                batch_logs = self.train_on_batch(x, y)
                batch_logs.update({m.name: float(m.result()) for m in self.train_metrics})
                self.tracker.update_train(**batch_logs)
                if verbose >= 2:
                    logger.info(
                        "epoch %d step %d: %s",
                        epoch + 1,
                        step,
                        "  ".join(f"{k}={v:.4f}" for k, v in batch_logs.items()),
                    )
                cb_list.on_train_batch_end(step, logs=batch_logs)

            if self.accumulation_steps > 1 and self._accum_counter % self.accumulation_steps:
                self._apply_accumulated()  # flush a partial accumulation window
                self._accum_counter = 0

            if val_ds is not None:
                val_results = self.evaluate(val_ds)
                self.tracker.update_val(**val_results)

            epoch_summary = self.tracker.commit_epoch()
            epoch_summary["lr"] = get_learning_rate(self.optimizer)
            cb_list.on_epoch_end(epoch, logs=epoch_summary)
            if verbose >= 1:
                logger.info(
                    "Epoch %d/%d — %s",
                    epoch + 1,
                    epochs,
                    "  ".join(f"{k}: {v:.4f}" for k, v in epoch_summary.items()),
                )
            if early_stopping is not None and early_stopping.update(epoch_summary, self.model):
                if verbose >= 1:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        cb_list.on_train_end()
        return self.tracker


__all__ = [
    "CustomTrainingLoop",
    "EarlyStoppingHandler",
    "GradientClipping",
    "LearningRateFinder",
    "MetricsTracker",
    "WarmupCosineSchedule",
    "get_learning_rate",
    "set_learning_rate",
]
