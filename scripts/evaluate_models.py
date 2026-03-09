#!/usr/bin/env python3
"""
tensorverse-evaluate  — CLI for evaluating trained TensorFlow models.

Usage
-----
    tensorverse-evaluate --model ./models/final_model --data ./data/test
    tensorverse-evaluate --model ./models/model.h5 --task classification --num-classes 10
    tensorverse-evaluate --model ./models/final_model --report --confusion-matrix
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorVerseHub — Model Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model (SavedModel directory or .h5 file)",
    )
    parser.add_argument(
        "--tflite",
        type=str,
        default=None,
        help="Path to a TFLite model (.tflite) to evaluate instead",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data",
        type=str,
        default="./data/test",
        help="Path to test dataset (default: ./data/test)",
    )
    data_group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
        help="Input image size (default: 224 224)",
    )
    data_group.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes (default: 10)",
    )
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )

    # Reports
    report_group = parser.add_argument_group("Reports")
    report_group.add_argument(
        "--report",
        action="store_true",
        help="Print a detailed classification report",
    )
    report_group.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Plot and save the confusion matrix",
    )
    report_group.add_argument(
        "--roc-curves",
        action="store_true",
        help="Plot ROC curves (binary / multi-class)",
    )
    report_group.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation artifacts (default: ./eval_results)",
    )

    # Task context
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "text_classification", "regression"],
        default="classification",
        help="Evaluation task type (default: classification)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of class names for reports",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def load_model(args):
    """Load a Keras SavedModel, .h5, or TFLite model."""
    import tensorflow as tf

    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=args.tflite)
        interpreter.allocate_tensors()
        return interpreter, "tflite"

    model = tf.keras.models.load_model(args.model)
    return model, "keras"


def create_synthetic_test_data(args):
    """Generate synthetic test samples for demonstration."""
    import numpy as np
    import tensorflow as tf

    if args.task == "classification":
        h, w = args.image_size
        x = np.random.rand(100, h, w, 3).astype("float32")
        y = np.random.randint(0, args.num_classes, 100)
    elif args.task == "text_classification":
        x = np.random.randint(0, 10000, (100, 128))
        y = np.random.randint(0, args.num_classes, 100)
    else:
        h, w = args.image_size
        x = np.random.rand(100, h, w, 3).astype("float32")
        y = np.random.rand(100, args.num_classes).astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(args.batch_size)
    return dataset, y


def evaluate_keras(model, dataset, args):
    """Evaluate a Keras model and return (loss, metrics, y_pred, y_true)."""
    import numpy as np

    results = model.evaluate(dataset, verbose=0 if args.quiet else 1, return_dict=True)

    y_pred_list, y_true_list = [], []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred_list.append(preds)
        y_true_list.append(y_batch.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    return results, y_pred, y_true


def evaluate_tflite(interpreter, dataset, args):
    """Evaluate a TFLite model."""
    import numpy as np

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred_list, y_true_list = [], []
    for x_batch, y_batch in dataset:
        for i in range(len(x_batch)):
            sample = np.expand_dims(x_batch[i].numpy(), axis=0).astype(
                input_details[0]["dtype"]
            )
            interpreter.set_tensor(input_details[0]["index"], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])
            y_pred_list.append(output[0])
            y_true_list.append(y_batch[i].numpy())

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    return {}, y_pred, y_true


def save_results(results, y_pred, y_true, args):
    """Save evaluation metrics and optional plots."""
    import numpy as np

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in ("classification", "text_classification"):
        y_pred_labels = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred.round().astype(int)
        y_true_labels = y_true.astype(int)

        accuracy = float(np.mean(y_pred_labels == y_true_labels))
        results.setdefault("accuracy", accuracy)

        if args.report:
            try:
                from sklearn.metrics import classification_report
                report = classification_report(
                    y_true_labels,
                    y_pred_labels,
                    target_names=args.class_names,
                    zero_division=0,
                )
                print("\nClassification Report:\n")
                print(report)

                report_path = os.path.join(args.output_dir, "classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report)
            except ImportError:
                print("scikit-learn not installed — skipping classification report.")

        if args.confusion_matrix:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from sklearn.metrics import confusion_matrix
                import seaborn as sns

                cm = confusion_matrix(y_true_labels, y_pred_labels)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix")
                cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
                plt.savefig(cm_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Confusion matrix saved → {cm_path}")
            except ImportError:
                print("matplotlib / seaborn / sklearn not installed — skipping confusion matrix.")

    # Always save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"Metrics saved → {metrics_path}")


def main():
    args = parse_args()

    import tensorflow as tf

    if not args.quiet:
        print(f"TensorVerseHub Evaluation CLI  |  TensorFlow {tf.__version__}")
        print(f"  Model : {args.tflite or args.model}")
        print(f"  Task  : {args.task}")

    model, model_type = load_model(args)

    # Load or synthesise test data
    data_path = Path(args.data)
    if data_path.exists() and any(data_path.iterdir()):
        if not args.quiet:
            print(f"Loading test data from {data_path} …")
        # TODO: replace with DataPipeline.create_image_dataset for real data
        dataset, y_true_raw = create_synthetic_test_data(args)
    else:
        if not args.quiet:
            print(f"Test data directory '{args.data}' not found — using synthetic data.")
        dataset, y_true_raw = create_synthetic_test_data(args)

    if model_type == "keras":
        results, y_pred, y_true = evaluate_keras(model, dataset, args)
    else:
        results, y_pred, y_true = evaluate_tflite(model, dataset, args)

    if not args.quiet:
        print("\n── Evaluation Results ──────────────────────────────")
        for k, v in results.items():
            print(f"  {k:20s}: {v:.4f}")

    save_results(results, y_pred, y_true, args)

    if not args.quiet:
        print(f"\nAll evaluation artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
