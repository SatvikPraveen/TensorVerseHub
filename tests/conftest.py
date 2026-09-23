"""Shared fixtures. TensorFlow is imported lazily so `--help` and collection stay fast."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import tensorflow as tf

from tensorversehub import compat
from tensorversehub.compat import keras

IMAGE_SHAPE = (16, 16, 3)
NUM_CLASSES = 3
NUM_SAMPLES = 24
BATCH_SIZE = 8

requires_legacy_keras = pytest.mark.skipif(
    compat.IS_KERAS_3, reason="needs TF_USE_LEGACY_KERAS=1 (tensorflow-model-optimization)"
)
requires_keras3 = pytest.mark.skipif(not compat.IS_KERAS_3, reason="needs Keras 3")


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # missing, or installed but broken against this TensorFlow
        return False


requires_tfmot = pytest.mark.skipif(
    compat.IS_KERAS_3 or not _has("tensorflow_model_optimization"),
    reason="needs legacy Keras + tensorflow-model-optimization",
)
requires_tf2onnx = pytest.mark.skipif(not _has("tf2onnx"), reason="tf2onnx not installed")
requires_onnxruntime = pytest.mark.skipif(
    not _has("onnxruntime"), reason="onnxruntime not installed"
)
requires_fastapi = pytest.mark.skipif(
    not (_has("fastapi") and _has("httpx")), reason="fastapi/httpx not installed"
)


@pytest.fixture(autouse=True)
def _seed_and_policy():
    keras.utils.set_random_seed(0)
    yield
    keras.mixed_precision.set_global_policy("float32")


@pytest.fixture
def images() -> np.ndarray:
    return np.random.default_rng(0).random((NUM_SAMPLES, *IMAGE_SHAPE), dtype=np.float32)


@pytest.fixture
def labels() -> np.ndarray:
    return np.arange(NUM_SAMPLES) % NUM_CLASSES


@pytest.fixture
def image_dataset(images, labels) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)


@pytest.fixture
def cnn_model() -> keras.Model:
    from tensorversehub.model_utils import ModelBuilders

    model = ModelBuilders.create_cnn_classifier(IMAGE_SHAPE, NUM_CLASSES, "simple")
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@pytest.fixture
def trained_model(cnn_model, image_dataset):
    cnn_model.fit(image_dataset, epochs=1, verbose=0)
    return cnn_model
