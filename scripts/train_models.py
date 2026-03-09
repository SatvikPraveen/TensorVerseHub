#!/usr/bin/env python3
"""
tensorverse-train  — CLI for training TensorFlow models.

Usage
-----
    tensorverse-train --task classification --data ./data/images --epochs 20
    tensorverse-train --task text_classification --data ./data/texts --epochs 10
    tensorverse-train --config config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorVerseHub — Model Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "text_classification", "autoencoder", "gan"],
        default="classification",
        help="Training task type (default: classification)",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Path to dataset directory (default: ./data)",
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
        help="Training batch size (default: 32)",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--architecture",
        type=str,
        choices=["simple", "vgg", "resnet", "lstm", "gru", "transformer"],
        default="resnet",
        help="Model architecture (default: resnet)",
    )
    model_group.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet-pretrained weights (for vision models)",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd", "adamw", "rmsprop"],
        default="adam",
        help="Optimizer (default: adam)",
    )
    train_group.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping (default: True)",
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    train_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed-precision (float16) training",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained model (default: ./models)",
    )
    output_group.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="TensorBoard log directory (default: ./logs)",
    )
    output_group.add_argument(
        "--save-format",
        type=str,
        choices=["saved_model", "h5", "both"],
        default="saved_model",
        help="Model save format (default: saved_model)",
    )

    # Config file (overrides CLI args if provided)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other CLI arguments)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and validate a JSON training config file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def build_model(args):
    """Build a tf.keras model according to the parsed arguments."""
    import tensorflow as tf
    from model_utils import ModelBuilders

    image_size = tuple(args.image_size)

    if args.task == "classification":
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(*image_size, 3),
            num_classes=args.num_classes,
            architecture=args.architecture if args.architecture in ("simple", "vgg", "resnet") else "resnet",
        )
    elif args.task == "text_classification":
        model = ModelBuilders.create_text_classifier(
            vocab_size=10000,
            max_length=128,
            num_classes=args.num_classes,
            architecture=args.architecture if args.architecture in ("lstm", "gru", "transformer") else "lstm",
        )
    elif args.task == "autoencoder":
        model, _, _ = ModelBuilders.create_autoencoder(
            input_shape=(*image_size, 3),
            latent_dim=128,
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return model


def get_optimizer(args):
    """Construct the optimizer from parsed arguments."""
    import tensorflow as tf

    lr = args.learning_rate
    optimizers = {
        "adam": tf.keras.optimizers.Adam(lr),
        "sgd": tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
        "adamw": tf.keras.optimizers.AdamW(lr),
        "rmsprop": tf.keras.optimizers.RMSprop(lr),
    }
    return optimizers[args.optimizer]


def build_callbacks(args):
    """Build training callbacks."""
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "checkpoint_epoch{epoch:02d}.weights.h5"),
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        ),
    ]
    if args.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            verbose=1,
        )
    )
    return callbacks


def create_synthetic_dataset(args):
    """
    Create a small synthetic dataset for demonstration when no data directory
    is found. In production, replace this with a real data loader.
    """
    import tensorflow as tf
    import numpy as np

    if args.task == "classification":
        h, w = args.image_size
        x = np.random.rand(200, h, w, 3).astype("float32")
        y = np.random.randint(0, args.num_classes, 200)
    elif args.task == "text_classification":
        x = np.random.randint(0, 10000, (200, 128))
        y = np.random.randint(0, args.num_classes, 200)
    else:  # autoencoder
        h, w = args.image_size
        x = np.random.rand(200, h, w, 3).astype("float32")
        y = x  # reconstruction target

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    n_val = int(len(x) * args.val_split)
    val_ds = dataset.take(n_val).batch(args.batch_size)
    train_ds = dataset.skip(n_val).batch(args.batch_size)
    return train_ds, val_ds


def main():
    args = parse_args()

    # Allow JSON config to override CLI args
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key.replace("-", "_"), value)

    # Lazy TensorFlow import (avoids slow startup for --help)
    import tensorflow as tf

    if not args.quiet:
        print(f"TensorVerseHub Training CLI  |  TensorFlow {tf.__version__}")
        print(f"  Task        : {args.task}")
        print(f"  Architecture: {args.architecture}")
        print(f"  Epochs      : {args.epochs}")
        print(f"  Batch size  : {args.batch_size}")
        print(f"  Output dir  : {args.output_dir}")

    # Mixed precision
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        if not args.quiet:
            print("  Mixed precision: ON (float16)")

    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Build model
    model = build_model(args)

    loss = "sparse_categorical_crossentropy"
    if args.task == "autoencoder":
        loss = "mse"

    model.compile(
        optimizer=get_optimizer(args),
        loss=loss,
        metrics=["accuracy"] if args.task != "autoencoder" else ["mae"],
    )

    if not args.quiet:
        model.summary()

    # Load or synthesise dataset
    data_path = Path(args.data)
    if data_path.exists() and any(data_path.iterdir()):
        if not args.quiet:
            print(f"\nLoading data from {data_path} …")
        # TODO: plug in DataPipeline.create_image_dataset for real data
        train_ds, val_ds = create_synthetic_dataset(args)
    else:
        if not args.quiet:
            print(f"\nData directory '{args.data}' not found — using synthetic data for demonstration.")
        train_ds, val_ds = create_synthetic_dataset(args)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=build_callbacks(args),
        verbose=0 if args.quiet else 1,
    )

    # Save
    final_path = os.path.join(args.output_dir, "final_model")
    if args.save_format in ("saved_model", "both"):
        model.save(final_path)
        if not args.quiet:
            print(f"\nSaved model → {final_path}")
    if args.save_format in ("h5", "both"):
        h5_path = final_path + ".h5"
        model.save(h5_path)
        if not args.quiet:
            print(f"Saved model → {h5_path}")

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    if not args.quiet:
        val_key = "val_accuracy" if "val_accuracy" in history.history else "val_mae"
        best = max(history.history.get(val_key, [0]))
        print(f"\nTraining complete. Best {val_key}: {best:.4f}")
        print(f"History saved → {history_path}")


if __name__ == "__main__":
    main()
