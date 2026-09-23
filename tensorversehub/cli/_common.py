"""Shared helpers for CLI sub-commands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger("tensorverse")


def add_data_arguments(parser: argparse.ArgumentParser, default_dir: str) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data",
        type=str,
        default=default_dir,
        help=f"Dataset directory with one sub-folder per class (default: {default_dir}); "
        "synthetic data is used when it does not exist",
    )
    group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("H", "W"),
        help="Input image size (default: 32 32)",
    )
    group.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    group.add_argument("--batch-size", type=int, default=32, help="Batch size")
    group.add_argument(
        "--num-samples", type=int, default=256, help="Synthetic samples to generate (fallback)"
    )


def has_class_folders(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any(child.is_dir() for child in p.iterdir())


def synthetic_arrays(
    task: str, image_size: Tuple[int, int], num_classes: int, num_samples: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic data so the CLIs run end-to-end without a dataset."""
    rng = np.random.default_rng(seed)
    h, w = image_size
    if task == "text_classification":
        x = rng.integers(0, 10000, (num_samples, 128)).astype("int32")
        y = rng.integers(0, num_classes, num_samples)
    elif task == "autoencoder":
        x = rng.random((num_samples, h, w, 3), dtype=np.float32)
        y = x
    else:
        x = rng.random((num_samples, h, w, 3), dtype=np.float32)
        y = rng.integers(0, num_classes, num_samples)
    return x, y


def to_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> Any:
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=0)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
