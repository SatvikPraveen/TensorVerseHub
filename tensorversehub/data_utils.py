"""
tf.data pipelines, TFRecord I/O and batch-level augmentation.

Highlights
----------
* :class:`TFRecordHandler` – type-dispatched ``tf.train.Feature`` encoding, sharded
  writers and matching parse functions for images, text and raw arrays.
* :class:`DataPipeline` – image / text / TFRecord pipelines with caching, shuffling
  and dependency-free random rotation (``tf.raw_ops.ImageProjectiveTransformV3``).
* :class:`DataAugmentation` – Keras preprocessing stack plus correct, vectorised
  MixUp, CutMix and Random Erasing for batched datasets.
"""

from __future__ import annotations

import glob
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from .compat import keras

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE
PathLike = Union[str, "os.PathLike[str]"]

# ---------------------------------------------------------------------------
# TFRecord helpers
# ---------------------------------------------------------------------------


class TFRecordHandler:
    """Create, write and read TFRecord files with type-dispatched features."""

    # -- feature constructors ------------------------------------------------
    @staticmethod
    def _bytes_feature(value: Union[str, bytes, Sequence[Union[str, bytes]]]) -> tf.train.Feature:
        if isinstance(value, (str, bytes)):
            value = [value]
        encoded = [v.encode("utf-8") if isinstance(v, str) else bytes(v) for v in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded))

    @staticmethod
    def _float_feature(value: Union[float, Sequence[float], np.ndarray]) -> tf.train.Feature:
        values = np.asarray(value, dtype=np.float32).ravel().tolist()
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    @staticmethod
    def _int64_feature(value: Union[int, Sequence[int], np.ndarray]) -> tf.train.Feature:
        values = np.asarray(value, dtype=np.int64).ravel().tolist()
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    @classmethod
    def to_feature(cls, value: Any) -> tf.train.Feature:
        """Convert a Python / NumPy / TensorFlow value to the matching ``tf.train.Feature``."""
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        if isinstance(value, (bool, np.bool_)):
            return cls._int64_feature(int(value))
        if isinstance(value, (int, np.integer)):
            return cls._int64_feature(int(value))
        if isinstance(value, (float, np.floating)):
            return cls._float_feature(float(value))
        if isinstance(value, (str, bytes)):
            return cls._bytes_feature(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size == 0:
                return cls._float_feature([])
            if arr.dtype.kind in "biu":
                return cls._int64_feature(arr)
            if arr.dtype.kind == "f":
                return cls._float_feature(arr)
            if arr.dtype.kind in "SUO":
                return cls._bytes_feature([str(v) if not isinstance(v, bytes) else v for v in arr])
        raise TypeError(f"Unsupported feature type: {type(value).__name__}")

    @classmethod
    def _extra_features(cls, extra: Optional[Dict[str, Any]]) -> Dict[str, tf.train.Feature]:
        return {k: cls.to_feature(v) for k, v in (extra or {}).items()}

    # -- serializers ---------------------------------------------------------
    def serialize_image_example(
        self,
        image_path: PathLike,
        label: int,
        additional_features: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Serialize an encoded image file (JPEG/PNG/...) plus its label and shape."""
        image_bytes = tf.io.read_file(str(image_path))
        shape = tf.io.decode_image(image_bytes, expand_animations=False).shape
        features = {
            "image": self._bytes_feature(image_bytes.numpy()),
            "label": self._int64_feature(label),
            "height": self._int64_feature(int(shape[0])),
            "width": self._int64_feature(int(shape[1])),
            "channels": self._int64_feature(int(shape[2])),
            "filename": self._bytes_feature(os.path.basename(str(image_path))),
        }
        features.update(self._extra_features(additional_features))
        return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

    def serialize_array_example(
        self,
        array: np.ndarray,
        label: int,
        additional_features: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Serialize a dense NumPy array as raw bytes with its shape and dtype."""
        arr = np.ascontiguousarray(array)
        features = {
            "array": self._bytes_feature(arr.tobytes()),
            "shape": self._int64_feature(list(arr.shape)),
            "dtype": self._bytes_feature(arr.dtype.name),
            "label": self._int64_feature(label),
        }
        features.update(self._extra_features(additional_features))
        return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

    def serialize_text_example(
        self, text: str, label: int, additional_features: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Serialize a text string with its label and whitespace token count."""
        features = {
            "text": self._bytes_feature(text),
            "label": self._int64_feature(label),
            "text_length": self._int64_feature(len(text.split())),
        }
        features.update(self._extra_features(additional_features))
        return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

    # -- writers -------------------------------------------------------------
    @staticmethod
    def write_tfrecord(
        examples: Iterable[bytes], output_path: PathLike, compression: Optional[str] = None
    ) -> int:
        """Write serialized examples to one file. Returns the number written."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        options = tf.io.TFRecordOptions(compression_type=compression) if compression else None
        count = 0
        with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
            for example in examples:
                writer.write(example)
                count += 1
        logger.info("Wrote %d examples to %s", count, output_path)
        return count

    @classmethod
    def write_sharded(
        cls,
        examples: Sequence[bytes],
        output_dir: PathLike,
        prefix: str = "data",
        num_shards: int = 4,
        compression: Optional[str] = None,
    ) -> List[str]:
        """Write examples round-robin into ``num_shards`` files and return their paths."""
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        paths = [
            str(Path(output_dir) / f"{prefix}-{i:05d}-of-{num_shards:05d}.tfrecord")
            for i in range(num_shards)
        ]
        for i, path in enumerate(paths):
            cls.write_tfrecord(examples[i::num_shards], path, compression=compression)
        return paths

    @staticmethod
    def count_records(path: PathLike, compression: Optional[str] = None) -> int:
        dataset = tf.data.TFRecordDataset(str(path), compression_type=compression)
        return int(dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy())


# ---------------------------------------------------------------------------
# Feature descriptions & parse functions
# ---------------------------------------------------------------------------


def create_feature_description_image() -> Dict[str, tf.io.FixedLenFeature]:
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }


def create_feature_description_text() -> Dict[str, tf.io.FixedLenFeature]:
    return {
        "text": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "text_length": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }


def create_feature_description_array() -> Dict[str, Any]:
    return {
        "array": tf.io.FixedLenFeature([], tf.string),
        "shape": tf.io.VarLenFeature(tf.int64),
        "dtype": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }


def parse_image_tfrecord(
    example_proto: tf.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    channels: int = 3,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode an image example to ``(float32 image in [0, 1], int32 label)``."""
    features = tf.io.parse_single_example(example_proto, create_feature_description_image())
    image = tf.io.decode_image(features["image"], channels=channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image_size is not None:
        image = tf.image.resize(image, image_size)
    return image, tf.cast(features["label"], tf.int32)


def parse_text_tfrecord(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    features = tf.io.parse_single_example(example_proto, create_feature_description_text())
    return features["text"], tf.cast(features["label"], tf.int32)


def parse_array_tfrecord(
    example_proto: tf.Tensor, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode a raw-array example. ``dtype`` must match the dtype used when writing."""
    features = tf.io.parse_single_example(example_proto, create_feature_description_array())
    shape = tf.sparse.to_dense(features["shape"])
    array = tf.io.decode_raw(features["array"], dtype)
    array = tf.reshape(array, shape)
    return array, tf.cast(features["label"], tf.int32)


# ---------------------------------------------------------------------------
# Image ops
# ---------------------------------------------------------------------------


def rotate_image(
    image: tf.Tensor,
    angle: Union[float, tf.Tensor],
    interpolation: str = "BILINEAR",
    fill_mode: str = "REFLECT",
) -> tf.Tensor:
    """
    Rotate an ``HWC`` or ``BHWC`` image tensor by ``angle`` radians about its centre.

    Implemented with the core ``ImageProjectiveTransformV3`` op, so it needs neither
    ``tf.contrib`` nor TensorFlow Addons and runs inside ``tf.data`` graphs.
    """
    image = tf.convert_to_tensor(image)
    squeeze = image.shape.rank == 3
    images = image[tf.newaxis] if squeeze else image
    orig_dtype = images.dtype
    if orig_dtype not in (tf.float32, tf.float64, tf.uint8, tf.int32, tf.float16, tf.bfloat16):
        images = tf.cast(images, tf.float32)

    angle = tf.cast(angle, tf.float32)
    height = tf.cast(tf.shape(images)[1], tf.float32)
    width = tf.cast(tf.shape(images)[2], tf.float32)
    cos, sin = tf.cos(angle), tf.sin(angle)
    x_offset = ((width - 1) - (cos * (width - 1) - sin * (height - 1))) / 2.0
    y_offset = ((height - 1) - (sin * (width - 1) + cos * (height - 1))) / 2.0
    transform = tf.stack([cos, -sin, x_offset, sin, cos, y_offset, 0.0, 0.0])[tf.newaxis]
    transforms = tf.tile(transform, [tf.shape(images)[0], 1])

    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        fill_value=0.0,
        interpolation=interpolation,
        fill_mode=fill_mode,
    )
    rotated = tf.cast(rotated, orig_dtype)
    return rotated[0] if squeeze else rotated


def augment_image(
    image: tf.Tensor,
    flip_horizontal: bool = True,
    brightness: float = 0.2,
    contrast: Tuple[float, float] = (0.8, 1.2),
    saturation: Tuple[float, float] = (0.8, 1.2),
    hue: float = 0.05,
    max_rotation: float = 0.1,
    seed: Optional[int] = None,
) -> tf.Tensor:
    """Stochastic photometric + small-angle geometric augmentation for a float image."""
    if flip_horizontal:
        image = tf.image.random_flip_left_right(image, seed=seed)
    if brightness > 0:
        image = tf.image.random_brightness(image, brightness, seed=seed)
    if contrast is not None:
        image = tf.image.random_contrast(image, *contrast, seed=seed)
    if image.shape[-1] == 3:
        if saturation is not None:
            image = tf.image.random_saturation(image, *saturation, seed=seed)
        if hue > 0:
            image = tf.image.random_hue(image, hue, seed=seed)
    if max_rotation > 0:
        angle = tf.random.uniform([], -max_rotation, max_rotation, seed=seed)
        image = rotate_image(image, angle)
    return tf.clip_by_value(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


class DataPipeline:
    """Build optimised ``tf.data`` pipelines for images, text and TFRecords."""

    def __init__(
        self,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        seed: Optional[int] = None,
        drop_remainder: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.drop_remainder = drop_remainder
        self.tfrecord_handler = TFRecordHandler()

    # -- shared tail ---------------------------------------------------------
    def finalize(
        self, dataset: tf.data.Dataset, shuffle: bool = True, repeat: bool = False
    ) -> tf.data.Dataset:
        """Apply shuffle → repeat → batch → prefetch."""
        if shuffle and self.shuffle_buffer > 0:
            dataset = dataset.shuffle(
                self.shuffle_buffer, seed=self.seed, reshuffle_each_iteration=True
            )
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        return dataset.prefetch(AUTOTUNE)

    # -- images --------------------------------------------------------------
    def create_image_dataset(
        self,
        image_paths: Sequence[str],
        labels: Sequence[int],
        image_size: Tuple[int, int] = (224, 224),
        num_channels: int = 3,
        augment: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        repeat: bool = False,
        augment_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> tf.data.Dataset:
        """Decode, resize, normalise (and optionally augment) images from disk."""
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        dataset = tf.data.Dataset.from_tensor_slices((list(map(str, image_paths)), list(labels)))

        def _load(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image, channels=num_channels, expand_animations=False)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, image_size)
            image.set_shape((*image_size, num_channels))
            return image, label

        dataset = dataset.map(_load, num_parallel_calls=AUTOTUNE)
        if cache:
            dataset = dataset.cache()  # cache the decoded (un-augmented) images
        if augment:
            fn = augment_fn or augment_image
            dataset = dataset.map(lambda x, y: (fn(x), y), num_parallel_calls=AUTOTUNE)
        return self.finalize(dataset, shuffle=shuffle, repeat=repeat)

    def from_arrays(
        self,
        x: Union[np.ndarray, tf.Tensor],
        y: Optional[Union[np.ndarray, tf.Tensor]] = None,
        shuffle: bool = True,
        repeat: bool = False,
        map_fn: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        """Pipeline over in-memory arrays (optionally applying ``map_fn`` per sample)."""
        dataset = tf.data.Dataset.from_tensor_slices((x, y) if y is not None else x)
        if map_fn is not None:
            dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
        return self.finalize(dataset, shuffle=shuffle, repeat=repeat)

    # -- text ----------------------------------------------------------------
    def create_text_dataset(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        max_length: int = 128,
        vocab_size: int = 10000,
        shuffle: bool = True,
        vectorizer: Optional[keras.layers.TextVectorization] = None,
    ) -> Tuple[tf.data.Dataset, keras.layers.TextVectorization]:
        """Tokenise raw strings with ``TextVectorization`` (adapted on ``texts`` if new)."""
        if vectorizer is None:
            vectorizer = keras.layers.TextVectorization(
                max_tokens=vocab_size,
                output_sequence_length=max_length,
                standardize="lower_and_strip_punctuation",
                split="whitespace",
            )
            vectorizer.adapt(list(texts))
        dataset = tf.data.Dataset.from_tensor_slices((list(texts), list(labels)))
        dataset = dataset.map(lambda t, y: (vectorizer(t), y), num_parallel_calls=AUTOTUNE)
        return self.finalize(dataset, shuffle=shuffle), vectorizer

    # -- tfrecords -----------------------------------------------------------
    def create_tfrecord_dataset(
        self,
        tfrecord_files: Union[str, Sequence[str]],
        feature_description: Optional[Dict[str, Any]] = None,
        parse_fn: Optional[Callable[[tf.Tensor], Any]] = None,
        compression_type: Optional[str] = None,
        shuffle: bool = True,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """Read (optionally globbed / sharded) TFRecord files and parse each example."""
        files = expand_paths(tfrecord_files)
        if not files:
            raise FileNotFoundError(f"No TFRecord files matched: {tfrecord_files}")
        if parse_fn is None:
            if feature_description is None:
                raise ValueError("Provide either parse_fn or feature_description")

            def parse_fn(proto: tf.Tensor) -> Any:  # type: ignore[misc]
                return tf.io.parse_single_example(proto, feature_description)

        dataset = tf.data.TFRecordDataset(
            files, compression_type=compression_type, num_parallel_reads=AUTOTUNE
        )
        dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)
        return self.finalize(dataset, shuffle=shuffle, repeat=repeat)

    # -- misc ----------------------------------------------------------------
    @staticmethod
    def create_mixed_precision_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Cast float32 features to float16.

        Only needed for hand-written ``GradientTape`` loops that do not use a Keras
        dtype policy; ``keras.mixed_precision`` autocasts layer inputs by itself.
        """

        def _cast(features: Any, labels: Any) -> Tuple[Any, Any]:
            if isinstance(features, dict):
                features = {
                    k: tf.cast(v, tf.float16) if v.dtype == tf.float32 else v
                    for k, v in features.items()
                }
            elif isinstance(features, tf.Tensor) and features.dtype == tf.float32:
                features = tf.cast(features, tf.float16)
            return features, labels

        return dataset.map(_cast, num_parallel_calls=AUTOTUNE)


def expand_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    """Expand glob patterns / directories into a sorted list of TFRecord file paths."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files: List[str] = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.tfrecord*"))))
        elif any(ch in p for ch in "*?["):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _sample_beta(shape: Sequence[Any], alpha: float, beta: float) -> tf.Tensor:
    """Beta(alpha, beta) samples via two Gamma draws (``tf.random`` has no Beta)."""
    g1 = tf.random.gamma(shape, alpha, 1.0)
    g2 = tf.random.gamma(shape, beta, 1.0)
    return g1 / (g1 + g2)


def _to_one_hot(y: tf.Tensor, num_classes: Optional[int]) -> tf.Tensor:
    """Return float labels; sparse integer labels are one-hot encoded."""
    if y.dtype.is_integer or y.shape.rank == 1:
        if num_classes is None:
            raise ValueError("num_classes is required for sparse (integer) labels")
        return tf.one_hot(tf.cast(y, tf.int32), num_classes, dtype=tf.float32)
    return tf.cast(y, tf.float32)


def _broadcast_lambda(lam: tf.Tensor, rank: int) -> tf.Tensor:
    return tf.reshape(lam, [-1] + [1] * (rank - 1))


class DataAugmentation:
    """Batch-level augmentation utilities."""

    @staticmethod
    def create_augmentation_layer(
        flip: str = "horizontal",
        rotation: float = 0.1,
        zoom: float = 0.1,
        contrast: float = 0.1,
        translation: float = 0.1,
        seed: Optional[int] = None,
    ) -> keras.Sequential:
        """A Keras ``Sequential`` of random preprocessing layers (active only in training)."""
        layers = [keras.layers.RandomFlip(flip, seed=seed)]
        if rotation:
            layers.append(keras.layers.RandomRotation(rotation, seed=seed))
        if zoom:
            layers.append(keras.layers.RandomZoom(zoom, seed=seed))
        if contrast:
            layers.append(keras.layers.RandomContrast(contrast, seed=seed))
        if translation:
            layers.append(keras.layers.RandomTranslation(translation, translation, seed=seed))
        return keras.Sequential(layers, name="augmentation")

    @staticmethod
    def mixup_batch(
        x: tf.Tensor, y: tf.Tensor, alpha: float = 0.2, num_classes: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """MixUp (Zhang et al. 2018) on one batch; returns soft labels."""
        x = tf.convert_to_tensor(x)
        y = _to_one_hot(tf.convert_to_tensor(y), num_classes)
        batch = tf.shape(x)[0]
        lam = _sample_beta([batch], alpha, alpha)
        idx = tf.random.shuffle(tf.range(batch))
        lam_x = _broadcast_lambda(lam, x.shape.rank)
        lam_y = _broadcast_lambda(lam, y.shape.rank)
        mixed_x = lam_x * x + (1.0 - lam_x) * tf.gather(x, idx)
        mixed_y = lam_y * y + (1.0 - lam_y) * tf.gather(y, idx)
        return mixed_x, mixed_y

    @staticmethod
    def mixup(
        dataset: tf.data.Dataset, alpha: float = 0.2, num_classes: Optional[int] = None
    ) -> tf.data.Dataset:
        """Apply :meth:`mixup_batch` to every batch of a *batched* dataset."""
        return dataset.map(
            lambda x, y: DataAugmentation.mixup_batch(x, y, alpha, num_classes),
            num_parallel_calls=AUTOTUNE,
        )

    @staticmethod
    def cutmix_batch(
        x: tf.Tensor, y: tf.Tensor, alpha: float = 1.0, num_classes: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """CutMix (Yun et al. 2019) with per-sample boxes and area-corrected labels."""
        x = tf.convert_to_tensor(x)
        y = _to_one_hot(tf.convert_to_tensor(y), num_classes)
        batch = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        h_f = tf.cast(height, tf.float32)
        w_f = tf.cast(width, tf.float32)

        lam = _sample_beta([batch], alpha, alpha)
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(cut_ratio * h_f, tf.int32)
        cut_w = tf.cast(cut_ratio * w_f, tf.int32)
        cy = tf.random.uniform([batch], 0, height, dtype=tf.int32)
        cx = tf.random.uniform([batch], 0, width, dtype=tf.int32)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)

        rows = tf.range(height)[tf.newaxis, :, tf.newaxis]  # [1, H, 1]
        cols = tf.range(width)[tf.newaxis, tf.newaxis, :]  # [1, 1, W]
        in_rows = (rows >= y1[:, tf.newaxis, tf.newaxis]) & (rows < y2[:, tf.newaxis, tf.newaxis])
        in_cols = (cols >= x1[:, tf.newaxis, tf.newaxis]) & (cols < x2[:, tf.newaxis, tf.newaxis])
        mask = tf.cast(in_rows & in_cols, x.dtype)[..., tf.newaxis]  # [B, H, W, 1]

        idx = tf.random.shuffle(tf.range(batch))
        mixed_x = x * (1.0 - mask) + tf.gather(x, idx) * mask
        box_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        lam_adj = 1.0 - box_area / (h_f * w_f)
        lam_y = _broadcast_lambda(lam_adj, y.shape.rank)
        mixed_y = lam_y * y + (1.0 - lam_y) * tf.gather(y, idx)
        return mixed_x, mixed_y

    @staticmethod
    def cutmix(
        dataset: tf.data.Dataset, alpha: float = 1.0, num_classes: Optional[int] = None
    ) -> tf.data.Dataset:
        """Apply :meth:`cutmix_batch` to every batch of a *batched* image dataset."""
        return dataset.map(
            lambda x, y: DataAugmentation.cutmix_batch(x, y, alpha, num_classes),
            num_parallel_calls=AUTOTUNE,
        )

    @staticmethod
    def random_erasing(
        image: tf.Tensor,
        probability: float = 0.5,
        area_range: Tuple[float, float] = (0.02, 0.33),
        aspect_range: Tuple[float, float] = (0.3, 3.3),
        fill_value: float = 0.0,
    ) -> tf.Tensor:
        """Random Erasing (Zhong et al. 2020) on a single ``HWC`` image."""
        image = tf.convert_to_tensor(image)
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        h_f, w_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

        area = tf.random.uniform([], *area_range) * h_f * w_f
        log_ratio = tf.math.log(tf.constant(aspect_range, tf.float32))
        aspect = tf.exp(tf.random.uniform([], log_ratio[0], log_ratio[1]))
        erase_h = tf.minimum(tf.cast(tf.round(tf.sqrt(area * aspect)), tf.int32), height)
        erase_w = tf.minimum(tf.cast(tf.round(tf.sqrt(area / aspect)), tf.int32), width)
        y0 = tf.random.uniform([], 0, height - erase_h + 1, dtype=tf.int32)
        x0 = tf.random.uniform([], 0, width - erase_w + 1, dtype=tf.int32)

        rows = tf.range(height)[:, tf.newaxis]
        cols = tf.range(width)[tf.newaxis, :]
        mask = (rows >= y0) & (rows < y0 + erase_h) & (cols >= x0) & (cols < x0 + erase_w)
        mask = tf.cast(mask, image.dtype)[..., tf.newaxis]
        erased = image * (1.0 - mask) + tf.cast(fill_value, image.dtype) * mask
        apply = tf.random.uniform([]) < probability
        return tf.where(apply, erased, image)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def create_image_classification_pipeline(
    image_dir: PathLike,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    validation_split: float = 0.2,
    seed: int = 42,
    cache: bool = True,
    label_mode: str = "int",
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Train/validation datasets from a class-per-subdirectory image folder."""
    common = dict(
        validation_split=validation_split,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
    )
    train_ds = keras.utils.image_dataset_from_directory(str(image_dir), subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(str(image_dir), subset="validation", **common)
    rescale = keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    if cache:
        train_ds, val_ds = train_ds.cache(), val_ds.cache()
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def create_text_classification_pipeline(
    texts: Sequence[str],
    labels: Sequence[int],
    batch_size: int = 32,
    max_tokens: int = 10000,
    sequence_length: int = 128,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, keras.layers.TextVectorization]:
    """Shuffle, split and vectorise text; the vectoriser is adapted on the *training* split only."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    texts_arr = np.asarray(texts, dtype=object)[order]
    labels_arr = np.asarray(labels)[order]
    split = int(len(texts) * (1 - validation_split))
    train_texts, val_texts = list(texts_arr[:split]), list(texts_arr[split:])
    train_labels, val_labels = labels_arr[:split], labels_arr[split:]

    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens, output_sequence_length=sequence_length
    )
    vectorizer.adapt(train_texts)

    def _make(t: List[str], y: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((t, y))
        if shuffle:
            ds = ds.shuffle(len(t), seed=seed)
        ds = ds.map(lambda a, b: (vectorizer(a), b), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    return _make(train_texts, train_labels, True), _make(val_texts, val_labels, False), vectorizer


def create_tfrecord_dataset(
    tfrecord_paths: Union[str, Sequence[str]],
    batch_size: int = 32,
    shuffle_buffer: int = 10000,
    parse_fn: Optional[Callable[[tf.Tensor], Any]] = None,
    feature_description: Optional[Dict[str, Any]] = None,
    compression_type: Optional[str] = None,
    repeat: bool = False,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """One-call TFRecord pipeline (defaults to :func:`parse_image_tfrecord`)."""
    pipeline = DataPipeline(batch_size=batch_size, shuffle_buffer=shuffle_buffer, seed=seed)
    if parse_fn is None and feature_description is None:
        parse_fn = parse_image_tfrecord
    return pipeline.create_tfrecord_dataset(
        tfrecord_paths,
        feature_description=feature_description,
        parse_fn=parse_fn,
        compression_type=compression_type,
        shuffle=shuffle_buffer > 0,
        repeat=repeat,
    )


def split_dataset(
    dataset: tf.data.Dataset, fractions: Sequence[float] = (0.8, 0.1, 0.1)
) -> List[tf.data.Dataset]:
    """Split an *unbatched* dataset of known cardinality by consecutive fractions."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    total = int(dataset.cardinality().numpy())
    if total < 0:
        raise ValueError("Dataset cardinality must be known (avoid repeat()/filter() first)")
    splits: List[tf.data.Dataset] = []
    start = 0
    for i, frac in enumerate(fractions):
        size = total - start if i == len(fractions) - 1 else math.floor(total * frac)
        splits.append(dataset.skip(start).take(size))
        start += size
    return splits


def compute_class_weights(labels: Union[Sequence[int], np.ndarray]) -> Dict[int, float]:
    """Inverse-frequency class weights (``n / (k * count_c)``) for ``model.fit(class_weight=...)``."""
    arr = np.asarray(labels).astype(np.int64).ravel()
    classes, counts = np.unique(arr, return_counts=True)
    weights = len(arr) / (len(classes) * counts)
    return {int(c): float(w) for c, w in zip(classes, weights)}


__all__ = [
    "AUTOTUNE",
    "DataAugmentation",
    "DataPipeline",
    "TFRecordHandler",
    "augment_image",
    "compute_class_weights",
    "create_feature_description_array",
    "create_feature_description_image",
    "create_feature_description_text",
    "create_image_classification_pipeline",
    "create_text_classification_pipeline",
    "create_tfrecord_dataset",
    "expand_paths",
    "parse_array_tfrecord",
    "parse_image_tfrecord",
    "parse_text_tfrecord",
    "rotate_image",
    "split_dataset",
]
