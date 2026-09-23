"""Evaluate a .keras / SavedModel / TFLite model and write metrics and plots."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ._common import add_data_arguments, has_class_folders, synthetic_arrays, to_dataset

logger = logging.getLogger("tensorverse.evaluate")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="Path to .keras/.h5 file, SavedModel dir or .tflite"
    )
    parser.add_argument(
        "--task",
        choices=["classification", "text_classification", "autoencoder"],
        default="classification",
    )
    add_data_arguments(parser, "./data/test")
    reports = parser.add_argument_group("reports")
    reports.add_argument("--report", action="store_true", help="Print a classification report")
    reports.add_argument("--confusion-matrix", action="store_true")
    reports.add_argument("--roc-curves", action="store_true")
    reports.add_argument("--class-names", nargs="+", default=None)
    reports.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--quiet", "-q", action="store_true")


def load_any_model(path: str) -> Tuple[Any, str]:
    from .. import compat

    if path.endswith(".tflite"):
        from ..export_utils import make_interpreter

        return make_interpreter(path), "tflite"
    model = compat.load_model(path)
    return model, ("keras" if isinstance(model, compat.keras.Model) else "savedmodel")


def load_data(args: argparse.Namespace):
    if args.task == "classification" and has_class_folders(args.data):
        from ..data_utils import create_image_classification_pipeline

        _, val_ds = create_image_classification_pipeline(
            args.data,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size),
            validation_split=0.999,
            seed=args.seed,
        )
        return val_ds
    logger.info("No dataset at '%s' — using synthetic data", args.data)
    x, y = synthetic_arrays(
        args.task, tuple(args.image_size), args.num_classes, args.num_samples, args.seed
    )
    return to_dataset(x, y, args.batch_size)


def predict_all(model: Any, kind: str, dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
    from ..export_utils import TFLiteExporter

    preds, trues = [], []
    for x, y in dataset:
        x_np = x.numpy()
        if kind == "tflite":
            preds.append(TFLiteExporter.run_tflite(model, x_np))
        elif kind == "keras":
            preds.append(model.predict(x_np, verbose=0))
        else:
            preds.append(model.predict(x_np))
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, task: str) -> Dict[str, float]:
    if task == "autoencoder":
        return {
            "mse": float(np.mean((y_pred - y_true) ** 2)),
            "mae": float(np.mean(np.abs(y_pred - y_true))),
        }
    labels = y_pred.argmax(axis=1)
    eps = 1e-7
    nll = -np.log(np.clip(y_pred[np.arange(len(labels)), y_true.astype(int)], eps, 1.0))
    return {"accuracy": float(np.mean(labels == y_true)), "loss": float(np.mean(nll))}


def run(args: argparse.Namespace) -> int:
    from .. import configure_tensorflow

    configure_tensorflow(log_level=logging.WARNING if args.quiet else logging.INFO)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, kind = load_any_model(args.model)
    logger.info("Loaded %s model from %s", kind, args.model)
    dataset = load_data(args)
    y_pred, y_true = predict_all(model, kind, dataset)
    metrics = compute_metrics(y_pred, y_true, args.task)
    if kind == "keras" and args.task != "autoencoder" and getattr(model, "compiled", True):
        try:
            compiled = model.evaluate(dataset, verbose=0, return_dict=True)
            metrics.update({f"compiled_{k}": float(v) for k, v in compiled.items()})
        except Exception as exc:  # uncompiled model
            logger.debug("model.evaluate skipped: %s", exc)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for k, v in metrics.items():
        logger.info("%-20s %.4f", k, v)

    if args.task != "autoencoder":
        labels = y_pred.argmax(axis=1)
        if args.report:
            from sklearn.metrics import classification_report

            report = classification_report(
                y_true.astype(int), labels, target_names=args.class_names, zero_division=0
            )
            print(report)
            (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        if args.confusion_matrix or args.roc_curves:
            from ..visualization import TrainingVisualization, use_headless_backend

            use_headless_backend()
            if args.confusion_matrix:
                TrainingVisualization.plot_confusion_matrix(
                    y_true.astype(int),
                    labels,
                    class_names=args.class_names,
                    save_path=out_dir / "confusion_matrix.png",
                    show=False,
                )
            if args.roc_curves:
                TrainingVisualization.plot_roc_curves(
                    y_true.astype(int),
                    y_pred,
                    class_names=args.class_names,
                    save_path=out_dir / "roc_curves.png",
                    show=False,
                )
    logger.info("Evaluation artifacts saved to %s", out_dir)
    return 0


def main() -> None:
    from . import main as cli_main

    sys.exit(cli_main(["evaluate", *sys.argv[1:]]))


if __name__ == "__main__":  # pragma: no cover
    main()
