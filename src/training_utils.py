"""
TensorVerseHub training utilities.

Advanced custom training loop components for TensorFlow 2.x / tf.keras:
  - LearningRateFinder      : LR range test (Smith 2017)
  - WarmupCosineSchedule    : Warm-up + cosine annealing LR schedule
  - GradientClipping        : Adaptive and fixed gradient clipping helpers
  - MetricsTracker          : Per-epoch metrics accumulator with history
  - EarlyStoppingHandler    : Reusable early stopping with delta and patience
  - CustomTrainingLoop      : GradientTape-based training engine
"""

import tensorflow as tf
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import math


# ---------------------------------------------------------------------------
# LR Schedule utilities
# ---------------------------------------------------------------------------

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up followed by cosine annealing.

    Args:
        base_lr:       Peak learning rate (after warm-up).
        total_steps:   Total training steps.
        warmup_steps:  Number of warm-up steps. If a float in (0, 1), treated
                       as a fraction of total_steps.
        min_lr:        Minimum LR at the end of the cosine decay (default 0).
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: Union[int, float] = 0.05,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.min_lr = min_lr

        if isinstance(warmup_steps, float) and 0 < warmup_steps < 1:
            self.warmup_steps = int(warmup_steps * total_steps)
        else:
            self.warmup_steps = int(warmup_steps)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Linear warm-up phase
        warmup_lr = self.base_lr * step / tf.maximum(warmup, 1.0)

        # Cosine annealing phase
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self) -> Dict:
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


# ---------------------------------------------------------------------------
# Gradient clipping helpers
# ---------------------------------------------------------------------------

class GradientClipping:
    """
    Gradient clipping utilities.

    Provides static methods for clipping by norm, value, and adaptive norm.
    """

    @staticmethod
    def clip_by_global_norm(
        gradients: List[tf.Tensor], max_norm: float
    ) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """
        Clips a list of gradients by the global norm.

        Returns:
            (clipped_gradients, global_norm)
        """
        clipped, global_norm = tf.clip_by_global_norm(gradients, max_norm)
        return clipped, global_norm

    @staticmethod
    def clip_by_value(
        gradients: List[Optional[tf.Tensor]],
        clip_min: float = -1.0,
        clip_max: float = 1.0,
    ) -> List[Optional[tf.Tensor]]:
        """Clips every gradient tensor element-wise between clip_min and clip_max."""
        return [
            tf.clip_by_value(g, clip_min, clip_max) if g is not None else None
            for g in gradients
        ]

    @staticmethod
    def adaptive_clip(
        gradients: List[Optional[tf.Tensor]],
        percentile: float = 95.0,
    ) -> List[Optional[tf.Tensor]]:
        """
        Clips gradients to the *percentile*-th percentile of their combined
        absolute values.  Useful when gradient magnitudes vary across runs.
        """
        flat = tf.concat(
            [tf.reshape(g, [-1]) for g in gradients if g is not None], axis=0
        )
        threshold = float(np.percentile(np.abs(flat.numpy()), percentile))
        return GradientClipping.clip_by_value(gradients, -threshold, threshold)


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Accumulates and stores per-epoch training / validation metrics.

    Usage::

        tracker = MetricsTracker()
        for epoch in range(epochs):
            for x_batch, y_batch in train_ds:
                loss, acc = ...
                tracker.update_train(loss=loss, accuracy=acc)
            for x_batch, y_batch in val_ds:
                loss, acc = ...
                tracker.update_val(loss=loss, accuracy=acc)
            tracker.commit_epoch()

        print(tracker.history)     # dict of lists
    """

    def __init__(self) -> None:
        self._train_accum: Dict[str, List[float]] = {}
        self._val_accum: Dict[str, List[float]] = {}
        self.history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    def update_train(self, **metrics: Union[float, tf.Tensor]) -> None:
        for name, value in metrics.items():
            self._train_accum.setdefault(name, []).append(float(value))

    def update_val(self, **metrics: Union[float, tf.Tensor]) -> None:
        for name, value in metrics.items():
            self._val_accum.setdefault(name, []).append(float(value))

    def commit_epoch(self) -> Dict[str, float]:
        """Average accumulated metrics and append to history. Returns epoch summary."""
        summary: Dict[str, float] = {}
        for name, values in self._train_accum.items():
            key = f"train_{name}"
            epoch_val = float(np.mean(values))
            self.history.setdefault(key, []).append(epoch_val)
            summary[key] = epoch_val

        for name, values in self._val_accum.items():
            key = f"val_{name}"
            epoch_val = float(np.mean(values))
            self.history.setdefault(key, []).append(epoch_val)
            summary[key] = epoch_val

        self._train_accum.clear()
        self._val_accum.clear()
        return summary

    def reset(self) -> None:
        """Clear all history."""
        self._train_accum.clear()
        self._val_accum.clear()
        self.history.clear()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStoppingHandler:
    """
    Reusable early stopping with delta and optional best-weights restoration.

    Args:
        monitor:        Metric key in the epoch summary dict to watch.
        patience:       How many epochs without improvement to tolerate.
        min_delta:      Minimum change to count as an improvement.
        mode:           'min' or 'max'.
        restore_best:   If True, restore best weights when training stops.
        baseline:       Optional initial baseline — first improvement must beat it.
    """

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
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self._best = baseline if baseline is not None else (
            float("inf") if mode == "min" else float("-inf")
        )
        self._wait = 0
        self._best_weights: Optional[List[np.ndarray]] = None

    @property
    def best(self) -> float:
        return self._best

    def update(
        self, epoch_summary: Dict[str, float], model: Optional[tf.keras.Model] = None
    ) -> bool:
        """
        Call at the end of each epoch.

        Returns True when training should stop.
        """
        if self.monitor not in epoch_summary:
            raise KeyError(
                f"Monitored metric '{self.monitor}' not found in epoch summary. "
                f"Available keys: {list(epoch_summary.keys())}"
            )

        current = epoch_summary[self.monitor]
        improved = (
            current < self._best - self.min_delta
            if self.mode == "min"
            else current > self._best + self.min_delta
        )

        if improved:
            self._best = current
            self._wait = 0
            if self.restore_best and model is not None:
                self._best_weights = [w.numpy() for w in model.weights]
        else:
            self._wait += 1

        if self._wait >= self.patience:
            if self.restore_best and self._best_weights is not None and model is not None:
                for w, val in zip(model.weights, self._best_weights):
                    w.assign(val)
            return True  # Stop

        return False

    def reset(self) -> None:
        self._wait = 0
        self._best_weights = None


# ---------------------------------------------------------------------------
# LR range finder
# ---------------------------------------------------------------------------

class LearningRateFinder:
    """
    Learning rate range test (Smith 2017).

    Gradually increases the LR from *min_lr* to *max_lr* over *num_steps*
    mini-batches, recording the smoothed loss at each step.  The ideal
    starting LR is typically just before the loss begins to diverge.

    Args:
        model:       A compiled tf.keras.Model.
        min_lr:      Starting LR (default 1e-7).
        max_lr:      Ending LR (default 1).
        num_steps:   Mini-batches to iterate (default 100).
        beta:        Smoothing factor for the loss curve (default 0.98).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        num_steps: int = 100,
        beta: float = 0.98,
    ) -> None:
        self.model = model
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.beta = beta

        self.lrs: List[float] = []
        self.losses: List[float] = []

    def _lr_at_step(self, step: int) -> float:
        """Exponentially increasing LR."""
        return self.min_lr * (self.max_lr / self.min_lr) ** (step / self.num_steps)

    def find(
        self,
        dataset: tf.data.Dataset,
        loss_fn: Optional[Callable] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Run the LR range test.

        Args:
            dataset:  A tf.data.Dataset that produces (x, y) batches.
            loss_fn:  Optional custom loss function. If None, uses the model's
                      compiled loss.

        Returns:
            (lrs, smoothed_losses)
        """
        # Save original weights so we can restore them afterwards
        original_weights = [w.numpy() for w in self.model.weights]
        original_lr = float(self.model.optimizer.learning_rate)

        self.lrs = []
        self.losses = []
        avg_loss = 0.0
        best_loss = float("inf")

        step = 0
        for x_batch, y_batch in dataset:
            if step >= self.num_steps:
                break

            lr = self._lr_at_step(step)
            self.model.optimizer.learning_rate.assign(lr)

            with tf.GradientTape() as tape:
                y_pred = self.model(x_batch, training=True)
                if loss_fn is not None:
                    loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                else:
                    loss = self.model.compiled_loss(y_batch, y_pred)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            loss_val = float(loss)
            avg_loss = self.beta * avg_loss + (1 - self.beta) * loss_val
            smoothed = avg_loss / (1 - self.beta ** (step + 1))

            # Stop early if loss has exploded
            if step > 0 and smoothed > 4 * best_loss:
                break

            if smoothed < best_loss:
                best_loss = smoothed

            self.lrs.append(lr)
            self.losses.append(smoothed)
            step += 1

        # Restore original model state
        for w, val in zip(self.model.weights, original_weights):
            w.assign(val)
        self.model.optimizer.learning_rate.assign(original_lr)

        return self.lrs, self.losses

    def plot(self, skip_start: int = 10, skip_end: int = 5) -> None:
        """Plot the LR vs loss curve (requires matplotlib)."""
        import matplotlib.pyplot as plt

        lrs = self.lrs[skip_start: len(self.lrs) - skip_end]
        losses = self.losses[skip_start: len(self.losses) - skip_end]

        if not lrs:
            print("Not enough data to plot. Try running find() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lrs, losses, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Smoothed Loss")
        ax.set_title("Learning Rate Range Test")
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Custom training loop
# ---------------------------------------------------------------------------

class CustomTrainingLoop:
    """
    GradientTape-based training engine.

    Advantages over model.fit():
      - Full control over gradient manipulation (custom clipping, accumulation)
      - Mixed-precision support without AutoCast layers
      - Easy integration of custom metrics and callbacks at the batch level

    Args:
        model:           A tf.keras.Model.
        optimizer:       A tf.keras.optimizers.Optimizer.
        loss_fn:         Loss callable (y_true, y_pred) → scalar tensor.
        train_metrics:   List of tf.keras.metrics.Metric for training.
        val_metrics:     List of tf.keras.metrics.Metric for validation.
        clip_norm:       If > 0, clips gradients by global norm.
        mixed_precision: If True, casts inputs to float16 and scales loss.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Callable,
        train_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        val_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        clip_norm: float = 0.0,
        mixed_precision: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_metrics = train_metrics or []
        self.val_metrics = val_metrics or []
        self.clip_norm = clip_norm
        self.mixed_precision = mixed_precision

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.tracker = MetricsTracker()

    # ------------------------------------------------------------------
    @tf.function
    def _train_step(
        self, x: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        if self.clip_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        for m in self.train_metrics:
            m.update_state(y, y_pred)

        return loss, gradients

    @tf.function
    def _val_step(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        y_pred = self.model(x, training=False)
        loss = self.loss_fn(y, y_pred)
        for m in self.val_metrics:
            m.update_state(y, y_pred)
        return loss

    # ------------------------------------------------------------------
    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        early_stopping: Optional[EarlyStoppingHandler] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1,
    ) -> MetricsTracker:
        """
        Run the custom training loop.

        Args:
            train_ds:       Training tf.data.Dataset yielding (x, y) batches.
            val_ds:         Optional validation dataset.
            epochs:         Number of epochs.
            early_stopping: Optional EarlyStoppingHandler instance.
            callbacks:      Optional list of tf.keras.callbacks.Callback objects.
            verbose:        0 = silent, 1 = epoch summary, 2 = per-batch.

        Returns:
            MetricsTracker with full history.
        """
        cb_list = tf.keras.callbacks.CallbackList(callbacks or [], model=self.model)
        cb_list.on_train_begin()

        for epoch in range(epochs):
            cb_list.on_epoch_begin(epoch)

            # Reset metrics
            for m in self.train_metrics + self.val_metrics:
                m.reset_state()

            # Training batches
            for step, (x_batch, y_batch) in enumerate(train_ds):
                cb_list.on_train_batch_begin(step)
                loss, _ = self._train_step(x_batch, y_batch)
                train_kv = {m.name: m.result() for m in self.train_metrics}
                self.tracker.update_train(loss=float(loss), **{k: float(v) for k, v in train_kv.items()})

                if verbose >= 2:
                    print(
                        f"  Epoch {epoch + 1}/{epochs} — step {step}: "
                        + "  ".join(f"{k}: {v:.4f}" for k, v in train_kv.items())
                    )
                cb_list.on_train_batch_end(step)

            # Validation batches
            if val_ds is not None:
                for x_batch, y_batch in val_ds:
                    val_loss = self._val_step(x_batch, y_batch)
                    val_kv = {m.name: m.result() for m in self.val_metrics}
                    self.tracker.update_val(loss=float(val_loss), **{k: float(v) for k, v in val_kv.items()})

            epoch_summary = self.tracker.commit_epoch()
            cb_list.on_epoch_end(epoch, logs=epoch_summary)

            if verbose >= 1:
                summary_str = "  ".join(f"{k}: {v:.4f}" for k, v in epoch_summary.items())
                print(f"Epoch {epoch + 1}/{epochs} — {summary_str}")

            if early_stopping is not None and early_stopping.update(epoch_summary, self.model):
                if verbose >= 1:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        cb_list.on_train_end()
        return self.tracker
