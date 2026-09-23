"""Train a model on an image folder (or synthetic data) and save it as .keras."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ._common import add_data_arguments, has_class_folders, synthetic_arrays, to_dataset

logger = logging.getLogger("tensorverse.train")

TASKS = ("classification", "text_classification", "autoencoder")
ARCHITECTURES = ("simple", "vgg", "resnet", "lstm", "gru", "transformer", "mlp")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=TASKS, default="classification")
    add_data_arguments(parser, "./data")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")

    model = parser.add_argument_group("model")
    model.add_argument("--architecture", choices=ARCHITECTURES, default="simple")
    model.add_argument("--vocab-size", type=int, default=10000)
    model.add_argument("--latent-dim", type=int, default=64, help="Autoencoder latent size")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--optimizer", choices=["adam", "adamw", "sgd", "rmsprop"], default="adam")
    train.add_argument("--patience", type=int, default=5, help="Early-stopping patience")
    train.add_argument("--no-early-stopping", action="store_true")
    train.add_argument("--mixed-precision", action="store_true")
    train.add_argument("--seed", type=int, default=42)

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default="./models")
    out.add_argument("--log-dir", default="./logs")
    out.add_argument("--model-name", default="final_model")
    out.add_argument("--export-savedmodel", action="store_true", help="Also export a SavedModel")
    out.add_argument("--config", type=str, default=None, help="JSON file overriding arguments")
    out.add_argument("--quiet", "-q", action="store_true")


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            for key, value in json.load(fh).items():
                setattr(args, key.replace("-", "_"), value)
    return args


def build_model(args: argparse.Namespace) -> Any:
    from ..model_utils import ModelBuilders

    h, w = args.image_size
    if args.task == "classification":
        arch = args.architecture if args.architecture in ("simple", "vgg", "resnet") else "simple"
        return ModelBuilders.create_cnn_classifier((h, w, 3), args.num_classes, arch)
    if args.task == "text_classification":
        arch = args.architecture if args.architecture in ("lstm", "gru", "transformer") else "lstm"
        return ModelBuilders.create_text_classifier(
            vocab_size=args.vocab_size,
            max_length=128,
            num_classes=args.num_classes,
            architecture=arch,
        )
    autoencoder, _, _ = ModelBuilders.create_autoencoder(
        (h, w, 3),
        encoding_dim=args.latent_dim,
        architecture="conv" if h % 4 == 0 and w % 4 == 0 else "dense",
    )
    return autoencoder


def build_optimizer(args: argparse.Namespace) -> Any:
    from ..compat import keras

    lr = args.learning_rate
    return {
        "adam": lambda: keras.optimizers.Adam(lr),
        "adamw": lambda: keras.optimizers.AdamW(lr),
        "sgd": lambda: keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
        "rmsprop": lambda: keras.optimizers.RMSprop(lr),
    }[args.optimizer]()


def build_callbacks(args: argparse.Namespace) -> list:
    from .. import compat
    from ..compat import keras

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=compat.checkpoint_filepath(args.output_dir, "best_model"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), verbose=0
        ),
    ]
    if importlib.util.find_spec("tensorboard") is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=args.log_dir))
    else:
        logger.warning("tensorboard not installed; skipping TensorBoard logging")
    if not args.no_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            )
        )
    return callbacks


def load_data(args: argparse.Namespace):
    if args.task == "classification" and has_class_folders(args.data):
        from ..data_utils import create_image_classification_pipeline

        logger.info("Loading images from %s", args.data)
        return create_image_classification_pipeline(
            args.data,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size),
            validation_split=args.val_split,
            seed=args.seed,
        )
    logger.info("No dataset at '%s' — using synthetic data", args.data)
    x, y = synthetic_arrays(
        args.task, tuple(args.image_size), args.num_classes, args.num_samples, args.seed
    )
    n_val = max(1, int(len(x) * args.val_split))
    return (
        to_dataset(x[n_val:], y[n_val:], args.batch_size, shuffle=True),
        to_dataset(x[:n_val], y[:n_val], args.batch_size),
    )


def run(args: argparse.Namespace) -> int:
    args = apply_config(args)
    import tensorflow as tf

    from .. import compat, configure_tensorflow
    from ..model_utils import save_model_with_metadata

    configure_tensorflow(
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        log_level=logging.WARNING if args.quiet else logging.INFO,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    is_ae = args.task == "autoencoder"
    model.compile(
        optimizer=build_optimizer(args),
        loss="mse" if is_ae else "sparse_categorical_crossentropy",
        metrics=["mae"] if is_ae else ["accuracy"],
    )
    if not args.quiet:
        model.summary()

    train_ds, val_ds = load_data(args)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=build_callbacks(args),
        verbose=0 if args.quiet else 1,
    )

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    metadata = {
        "task": args.task,
        "architecture": args.architecture,
        "epochs_run": len(hist.get("loss", [])),
        "image_size": list(args.image_size),
        "num_classes": args.num_classes,
        "final_metrics": {k: v[-1] for k, v in hist.items() if v},
    }
    keras_path = save_model_with_metadata(
        model, Path(args.output_dir) / f"{args.model_name}{compat.NATIVE_MODEL_EXTENSION}", metadata
    )
    logger.info("Saved model → %s", keras_path)
    if args.export_savedmodel:
        sm_path = compat.export_saved_model(
            model, Path(args.output_dir) / f"{args.model_name}_savedmodel"
        )
        logger.info("Exported SavedModel → %s", sm_path)

    history_path = Path(args.output_dir) / "training_history.json"
    history_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    logger.info("History saved → %s", history_path)

    key = next((k for k in ("val_accuracy", "val_mae", "val_loss") if k in hist), None)
    if key and not args.quiet:
        best = min(hist[key]) if key != "val_accuracy" else max(hist[key])
        logger.info("Best %s: %.4f  (TensorFlow %s)", key, best, tf.__version__)
    return 0


def main() -> None:
    from . import main as cli_main

    sys.exit(cli_main(["train", *sys.argv[1:]]))


if __name__ == "__main__":  # pragma: no cover
    main()
