"""
Plotting helpers for models, training runs and datasets.

All functions are headless-safe: they return the ``matplotlib.figure.Figure`` and
only call ``plt.show()`` when ``show=True``.  Pass ``save_path`` to write a PNG.
"""

from __future__ import annotations

import logging
import os
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import compat
from .compat import keras

logger = logging.getLogger(__name__)
PathLike = Union[str, "os.PathLike[str]"]
HistoryLike = Union[keras.callbacks.History, Dict[str, Sequence[float]]]


def _history_dict(history: HistoryLike) -> Dict[str, List[float]]:
    data = history.history if hasattr(history, "history") else history
    return {k: [float(v) for v in vals] for k, vals in dict(data).items()}


def _finish(fig: Any, save_path: Optional[PathLike], show: bool) -> Any:
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _to_image(img: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0], "gray"
    if img.ndim == 2:
        return img, "gray"
    if img.dtype != np.uint8 and img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1), None


# ---------------------------------------------------------------------------
# Model visualisation
# ---------------------------------------------------------------------------


class ModelVisualization:
    """Architecture diagrams, feature maps and filters."""

    @staticmethod
    def plot_model_architecture(
        model: keras.Model,
        save_path: PathLike = "model_architecture.png",
        show_shapes: bool = True,
        show_layer_names: bool = True,
        rankdir: str = "TB",
        dpi: int = 96,
    ) -> Optional[str]:
        """Render with ``keras.utils.plot_model``; returns the path or ``None`` if graphviz is missing."""
        try:
            keras.utils.plot_model(
                model,
                to_file=str(save_path),
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                rankdir=rankdir,
                expand_nested=True,
                dpi=dpi,
            )
        except (ImportError, OSError, ValueError) as exc:
            logger.warning("plot_model unavailable (install pydot + graphviz): %s", exc)
            return None
        return str(save_path)

    @staticmethod
    def visualize_layer_outputs(
        model: keras.Model,
        input_image: np.ndarray,
        layer_names: Optional[Sequence[str]] = None,
        max_images_per_row: int = 8,
        max_features: int = 32,
        save_dir: Optional[PathLike] = None,
        show: bool = True,
    ) -> List[Any]:
        """Plot feature maps of Conv2D layers for one input image; returns the figures."""
        if layer_names is None:
            layer_names = [l.name for l in model.layers if isinstance(l, keras.layers.Conv2D)]
        outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = keras.Model(inputs=model.inputs, outputs=outputs)
        activations = activation_model.predict(np.expand_dims(input_image, 0), verbose=0)
        if not isinstance(activations, list):
            activations = [activations]

        figures = []
        for name, act in zip(layer_names, activations):
            n_features = min(int(act.shape[-1]), max_features)
            n_cols = min(max_images_per_row, n_features)
            n_rows = int(np.ceil(n_features / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
            fig.suptitle(f"Layer: {name}", fontsize=12)
            for i, ax in enumerate(np.atleast_1d(axes).ravel()):
                ax.axis("off")
                if i < n_features:
                    ax.imshow(act[0, :, :, i], cmap="viridis")
            path = Path(save_dir, f"{name}_activations.png") if save_dir else None
            figures.append(_finish(fig, path, show))
        return figures

    @staticmethod
    def plot_filter_weights(
        model: keras.Model,
        layer_name: str,
        max_filters: int = 64,
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        """Visualise a Conv2D kernel bank (first 3 input channels as RGB)."""
        layer = model.get_layer(layer_name)
        if not isinstance(layer, keras.layers.Conv2D):
            raise ValueError(f"Layer '{layer_name}' is not a Conv2D layer")
        weights = layer.get_weights()[0]
        w_min, w_max = weights.min(), weights.max()
        weights = (weights - w_min) / (w_max - w_min + 1e-8)
        n_filters = min(weights.shape[-1], max_filters)
        n_cols = 8
        n_rows = int(np.ceil(n_filters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        fig.suptitle(f"Filters: {layer_name}", fontsize=12)
        for i, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.axis("off")
            if i >= n_filters:
                continue
            kernel = weights[:, :, : min(3, weights.shape[2]), i]
            if kernel.shape[2] == 1 or kernel.shape[2] == 2:
                ax.imshow(kernel[:, :, 0], cmap="viridis")
            else:
                ax.imshow(kernel)
        return _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Training visualisation
# ---------------------------------------------------------------------------


class TrainingVisualization:
    """Curves, confusion matrices and ROC plots."""

    @staticmethod
    def plot_training_history(
        history: HistoryLike,
        metrics: Optional[Sequence[str]] = None,
        save_path: Optional[PathLike] = None,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = True,
    ) -> Any:
        """Plot ``metric`` and ``val_metric`` curves for each available metric."""
        data = _history_dict(history)
        if metrics is None:
            metrics = [
                k for k in data if not k.startswith("val_") and k not in ("lr", "learning_rate")
            ]
        available = [m for m in metrics if m in data]
        if not available:
            raise ValueError(f"None of {list(metrics)} found in history keys {sorted(data)}")

        fig, axes = plt.subplots(1, len(available), figsize=figsize, squeeze=False)
        for ax, metric in zip(axes[0], available):
            epochs = range(1, len(data[metric]) + 1)
            ax.plot(epochs, data[metric], "-", label=f"train {metric}")
            if f"val_{metric}" in data:
                ax.plot(epochs, data[f"val_{metric}"], "--", label=f"val {metric}")
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)
        return _finish(fig, save_path, show)

    @staticmethod
    def plot_learning_curves(
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        metric_name: str = "Accuracy",
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        """Score vs training-set size with ±1 std bands (``scores`` shaped ``[sizes, folds]``)."""
        train_scores = np.atleast_2d(train_scores)
        val_scores = np.atleast_2d(val_scores)
        fig, ax = plt.subplots(figsize=(8, 5))
        for scores, label in ((train_scores, "Training"), (val_scores, "Validation")):
            mean, std = scores.mean(axis=1), scores.std(axis=1)
            ax.plot(train_sizes, mean, "o-", label=label)
            ax.fill_between(train_sizes, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Training set size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Learning curves — {metric_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        return _finish(fig, save_path, show)

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        save_path: Optional[PathLike] = None,
        figsize: Tuple[int, int] = (8, 6),
        show: bool = True,
    ) -> Any:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        n = cm.shape[0]
        labels = list(class_names) if class_names else [str(i) for i in range(n)]
        ax.set_xticks(range(n), labels, rotation=45, ha="right")
        ax.set_yticks(range(n), labels)
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
        return _finish(fig, save_path, show)

    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        """One-vs-rest ROC curves; ``y_true`` may be sparse labels or one-hot."""
        from sklearn.metrics import auc, roc_curve

        y_true = np.asarray(y_true)
        n_classes = y_pred_proba.shape[1]
        if y_true.ndim == 1:
            y_true = np.eye(n_classes)[y_true.astype(int)]
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            name = class_names[i] if class_names else f"class {i}"
            ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves (one-vs-rest)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        return _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Data visualisation
# ---------------------------------------------------------------------------


class DataVisualization:
    """Dataset exploration plots."""

    @staticmethod
    def plot_image_samples(
        images: np.ndarray,
        labels: Optional[np.ndarray] = None,
        class_names: Optional[Sequence[str]] = None,
        num_samples: int = 16,
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        num_samples = min(num_samples, len(images))
        n_cols = int(np.ceil(np.sqrt(num_samples)))
        n_rows = int(np.ceil(num_samples / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        for i, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if i >= num_samples:
                continue
            img, cmap = _to_image(images[i])
            ax.imshow(img, cmap=cmap)
            if labels is not None:
                label = (
                    int(np.asarray(labels[i]).ravel()[0]) if np.ndim(labels[i]) else int(labels[i])
                )
                title = (
                    class_names[label] if class_names and label < len(class_names) else str(label)
                )
                ax.set_title(title, fontsize=8)
        return _finish(fig, save_path, show)

    @staticmethod
    def plot_class_distribution(
        labels: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        unique, counts = np.unique(np.asarray(labels).ravel(), return_counts=True)
        fig, ax = plt.subplots(figsize=(max(6, len(unique) * 0.6), 4))
        bars = ax.bar(range(len(unique)), counts)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        names = [class_names[int(u)] if class_names else str(u) for u in unique]
        ax.set_xticks(range(len(unique)), names, rotation=45, ha="right")
        ax.set_ylabel("Samples")
        ax.set_title("Class distribution")
        return _finish(fig, save_path, show)

    @staticmethod
    def plot_data_augmentation_samples(
        original_image: np.ndarray,
        augmentation_layer: keras.layers.Layer,
        num_samples: int = 8,
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        n_cols = 3
        n_rows = int(np.ceil((num_samples + 1) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 3 * n_rows), squeeze=False)
        flat = axes.ravel()
        img, cmap = _to_image(original_image)
        flat[0].imshow(img, cmap=cmap)
        flat[0].set_title("Original")
        batch = tf.convert_to_tensor(np.expand_dims(original_image, 0), tf.float32)
        for i in range(1, len(flat)):
            flat[i].axis("off")
            if i > num_samples:
                continue
            augmented = augmentation_layer(batch, training=True)
            augmented = augmented.numpy() if hasattr(augmented, "numpy") else np.asarray(augmented)
            img, cmap = _to_image(augmented[0])
            flat[i].imshow(img, cmap=cmap)
            flat[i].set_title(f"Augmented {i}")
        flat[0].axis("off")
        return _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Advanced visualisation
# ---------------------------------------------------------------------------


class AdvancedVisualization:
    """Gradient diagnostics and dashboards."""

    @staticmethod
    def plot_gradient_flow(
        gradients: Sequence[Optional[tf.Tensor]],
        variables: Sequence[Any],
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        """Mean |gradient| and max |gradient| per trainable variable (from a ``GradientTape``)."""
        names, means, maxes = [], [], []
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            g = np.abs(np.asarray(grad))
            names.append(getattr(var, "name", str(var)).split(":")[0])
            means.append(float(g.mean()))
            maxes.append(float(g.max()))
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))
        x = np.arange(len(names))
        ax.bar(x, maxes, alpha=0.3, color="tab:blue", label="max |grad|")
        ax.bar(x, means, alpha=0.9, color="tab:orange", label="mean |grad|")
        ax.set_xticks(x, names, rotation=90, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("|gradient|")
        ax.set_title("Gradient flow")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        return _finish(fig, save_path, show)

    @staticmethod
    def create_training_dashboard(
        history: HistoryLike,
        model: keras.Model,
        validation_data: Optional[tf.data.Dataset] = None,
        save_path: Optional[PathLike] = None,
        show: bool = True,
    ) -> Any:
        """Loss/accuracy curves, model summary, sample predictions and final metrics."""
        from .model_utils import ModelAnalysis

        data = _history_dict(history)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0:2])
        if "loss" in data:
            epochs = range(1, len(data["loss"]) + 1)
            ax1.plot(epochs, data["loss"], label="train")
            if "val_loss" in data:
                ax1.plot(epochs, data["val_loss"], "--", label="val")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 2:4])
        acc_key = next(
            (k for k in ("accuracy", "acc", "sparse_categorical_accuracy") if k in data), None
        )
        if acc_key:
            epochs = range(1, len(data[acc_key]) + 1)
            ax2.plot(epochs, data[acc_key], label="train")
            if f"val_{acc_key}" in data:
                ax2.plot(epochs, data[f"val_{acc_key}"], "--", label="val")
            ax2.legend()
        ax2.set_title("Accuracy")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.axis("off")
        summary = "\n".join(ModelAnalysis.model_summary_string(model).splitlines()[:18])
        ax3.text(
            0.0, 1.0, summary, transform=ax3.transAxes, fontsize=6, va="top", family="monospace"
        )
        ax3.set_title("Model summary")

        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.axis("off")
        ax4.set_title("Sample predictions")
        if validation_data is not None:
            try:
                for batch in validation_data.take(1):
                    x_val, y_val, _ = compat.unpack_batch(batch)
                    preds = model.predict(x_val[:8], verbose=0)
                    lines = [
                        f"sample {i}: pred={int(np.argmax(p))} true={int(np.asarray(y_val[i]).ravel()[0])}"
                        for i, p in enumerate(preds)
                    ]
                    ax4.text(
                        0.0,
                        1.0,
                        "\n".join(lines),
                        transform=ax4.transAxes,
                        va="top",
                        fontsize=8,
                        family="monospace",
                    )
            except Exception as exc:  # pragma: no cover - best effort
                ax4.text(0.5, 0.5, f"prediction failed: {exc}", ha="center", va="center")

        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")
        final = {k: v[-1] for k, v in data.items() if v}
        text = "   ".join(f"{k}: {v:.4f}" for k, v in final.items())
        ax5.text(
            0.5,
            0.5,
            text or "no metrics",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4),
        )
        fig.suptitle("Training dashboard", fontsize=14, fontweight="bold")
        return _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def quick_model_analysis(
    model: keras.Model,
    history: HistoryLike,
    test_data: Optional[tf.data.Dataset] = None,
    save_dir: Optional[PathLike] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """Training curves, architecture diagram and (optionally) a confusion matrix."""
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Any] = {}
    outputs["history"] = TrainingVisualization.plot_training_history(
        history,
        save_path=save_dir_path / "training_history.png" if save_dir_path else None,
        show=show,
    )
    if save_dir_path:
        outputs["architecture"] = ModelVisualization.plot_model_architecture(
            model, save_dir_path / "model_architecture.png"
        )
    if test_data is not None:
        y_true: List[int] = []
        y_pred: List[int] = []
        for batch in test_data:
            x, y, _ = compat.unpack_batch(batch)
            y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
            y_true.extend(np.asarray(y).ravel())
        outputs["confusion_matrix"] = TrainingVisualization.plot_confusion_matrix(
            np.array(y_true),
            np.array(y_pred),
            save_path=save_dir_path / "confusion_matrix.png" if save_dir_path else None,
            show=show,
        )
    return outputs


def setup_plotting_style(style: str = "default", palette: Optional[str] = None) -> None:
    """Apply a consistent matplotlib style (optionally a seaborn palette)."""
    plt.style.use(style)
    if palette:
        try:
            import seaborn as sns

            sns.set_palette(palette)
        except ImportError:  # pragma: no cover
            logger.debug("seaborn not installed; palette ignored")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "figure.dpi": 100,
        }
    )


def use_headless_backend() -> None:
    """Switch matplotlib to the non-interactive ``Agg`` backend (CI / servers)."""
    matplotlib.use("Agg")


__all__ = [
    "AdvancedVisualization",
    "DataVisualization",
    "ModelVisualization",
    "TrainingVisualization",
    "quick_model_analysis",
    "setup_plotting_style",
    "use_headless_backend",
]
