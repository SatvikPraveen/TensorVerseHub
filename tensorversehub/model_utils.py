"""
Model building blocks, reference architectures and analysis tools.

* :class:`CustomLayers` – serialisable multi-head attention, sinusoidal positional
  encoding and a pre-LayerNorm Transformer encoder block.
* :class:`ModelBuilders` – CNN (simple / VGG / ResNet), text (LSTM / GRU / Transformer),
  MLP, autoencoder and GAN reference models.
* :class:`TrainingUtilities` – callback bundles and a ``tf.function`` train step.
* :class:`ModelAnalysis` – parameter / memory statistics, profiler-backed FLOPs and
  Markdown reports.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from . import compat
from .compat import keras

logger = logging.getLogger(__name__)
PathLike = Union[str, "os.PathLike[str]"]


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------


@compat.register_serializable(name="MultiHeadAttention")
class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-head scaled dot-product attention (Vaswani et al. 2017).

    Args:
        d_model:   Model width; must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        dropout:   Dropout applied to the attention weights.

    Call signature ``(query, key=None, value=None, attention_mask=None)`` where a
    missing ``key``/``value`` defaults to ``query`` (self-attention).  ``attention_mask``
    is boolean/0-1 with ``True`` = *attend* (Keras convention) and broadcastable to
    ``[batch, num_heads, q_len, k_len]``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.depth = self.d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model, name="query")
        self.wk = keras.layers.Dense(d_model, name="key")
        self.wv = keras.layers.Dense(d_model, name="value")
        self.dense = keras.layers.Dense(d_model, name="output")
        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def build(self, query_shape: Any, key_shape: Any = None, value_shape: Any = None) -> None:
        query_shape = tuple(query_shape)
        key_shape = tuple(key_shape) if key_shape is not None else query_shape
        value_shape = tuple(value_shape) if value_shape is not None else key_shape
        self.wq.build(query_shape)
        self.wk.build(key_shape)
        self.wv.build(value_shape)
        self.dense.build(query_shape[:-1] + (self.d_model,))
        super().build(query_shape)

    def compute_output_shape(self, query_shape: Any, *args: Any, **kwargs: Any) -> Any:
        return tuple(query_shape[:-1]) + (self.d_model,)

    def split_heads(self, x: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = tf.matmul(q, k, transpose_b=True)
        logits = logits / tf.sqrt(tf.cast(tf.shape(k)[-1], logits.dtype))
        if attention_mask is not None:
            keep = tf.cast(attention_mask, tf.bool)
            neg = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(keep, logits, neg)
        weights = tf.nn.softmax(logits, axis=-1)
        weights = self.dropout(weights, training=training)
        return tf.matmul(weights, v), weights

    def call(
        self,
        query: tf.Tensor,
        key: Optional[tf.Tensor] = None,
        value: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
        return_attention_scores: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        key = query if key is None else key
        value = key if value is None else value
        batch_size = tf.shape(query)[0]

        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(value), batch_size)

        attended, weights = self.scaled_dot_product_attention(q, k, v, attention_mask, training)
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        concat = tf.reshape(attended, (batch_size, -1, self.d_model))
        output = self.dense(concat)
        if return_attention_scores:
            return output, weights
        return output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"d_model": self.d_model, "num_heads": self.num_heads, "dropout": self.dropout_rate}
        )
        return config


@compat.register_serializable(name="PositionalEncoding")
class PositionalEncoding(keras.layers.Layer):
    """Sinusoidal positional encoding added to token embeddings."""

    def __init__(self, position: int, d_model: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.position = int(position)
        self.d_model = int(d_model)
        self.pos_encoding = tf.constant(self.positional_encoding(position, d_model))

    @staticmethod
    def get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @classmethod
    def positional_encoding(cls, position: int, d_model: int) -> np.ndarray:
        angle_rads = cls.get_angles(
            np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return angle_rads[np.newaxis, ...].astype(np.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            seq_len, self.position, message="Sequence longer than the encoded max position"
        )
        return x + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)

    def compute_output_shape(self, input_shape: Any) -> Any:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config


@compat.register_serializable(name="TransformerEncoderBlock")
class TransformerEncoderBlock(keras.layers.Layer):
    """Pre-LayerNorm Transformer encoder block (attention + GELU feed-forward)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim or 4 * d_model)
        self.dropout_rate = float(dropout)

        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout, name="mha")
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_2")
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(self.ff_dim, activation="gelu"),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(d_model),
            ],
            name="ffn",
        )
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)

    def build(self, input_shape: Any) -> None:
        input_shape = tuple(input_shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.attention.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        h = self.norm1(x)
        h = self.attention(h, attention_mask=attention_mask, training=training)
        x = x + self.dropout1(h, training=training)
        h = self.ffn(self.norm2(x), training=training)
        return x + self.dropout2(h, training=training)

    def compute_output_shape(self, input_shape: Any) -> Any:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


class CustomLayers:
    """Namespace for TensorVerseHub's custom Keras layers."""

    MultiHeadAttention = MultiHeadAttention
    PositionalEncoding = PositionalEncoding
    TransformerEncoderBlock = TransformerEncoderBlock

    @staticmethod
    def create_padding_mask(token_ids: tf.Tensor, pad_id: int = 0) -> tf.Tensor:
        """``[batch, 1, 1, seq]`` boolean mask, ``True`` where tokens are *not* padding."""
        keep = tf.not_equal(token_ids, pad_id)
        return keep[:, tf.newaxis, tf.newaxis, :]

    @staticmethod
    def create_look_ahead_mask(size: int) -> tf.Tensor:
        """``[size, size]`` causal mask, ``True`` where position j <= i."""
        return tf.linalg.band_part(tf.ones((size, size), tf.bool), -1, 0)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _residual_block(x: tf.Tensor, filters: int, stride: int = 1, name: str = "") -> tf.Tensor:
    shortcut = x
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(
            shortcut
        )
        shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.Add()([x, shortcut])
    return keras.layers.ReLU()(x)


class ModelBuilders:
    """Reference architectures built with the Keras functional API."""

    CNN_ARCHITECTURES = ("simple", "vgg", "resnet")
    TEXT_ARCHITECTURES = ("lstm", "gru", "transformer")

    @staticmethod
    def create_cnn_classifier(
        input_shape: Tuple[int, int, int],
        num_classes: int,
        architecture: str = "simple",
        dropout_rate: float = 0.5,
        name: Optional[str] = None,
    ) -> keras.Model:
        """CNN classifier with a softmax head (``simple`` | ``vgg`` | ``resnet``)."""
        if architecture not in ModelBuilders.CNN_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}'. Choose from {ModelBuilders.CNN_ARCHITECTURES}"
            )
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        inputs = keras.Input(shape=input_shape, name="image")
        x = inputs
        if architecture == "simple":
            for filters in (32, 64, 64):
                x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
                x = keras.layers.MaxPooling2D()(x)
        elif architecture == "vgg":
            for filters in (64, 128, 256):
                x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
                x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
                x = keras.layers.MaxPooling2D()(x)
        else:  # resnet
            x = keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
            for filters, stride in ((64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1)):
                x = _residual_block(x, filters, stride)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        return keras.Model(inputs, outputs, name=name or f"{architecture}_cnn")

    @staticmethod
    def create_text_classifier(
        vocab_size: int,
        max_length: int,
        num_classes: int,
        embedding_dim: int = 128,
        architecture: str = "lstm",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        name: Optional[str] = None,
    ) -> keras.Model:
        """Text classifier over integer token ids (``lstm`` | ``gru`` | ``transformer``)."""
        if architecture not in ModelBuilders.TEXT_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}'. Choose from {ModelBuilders.TEXT_ARCHITECTURES}"
            )
        inputs = keras.Input(shape=(max_length,), dtype="int32", name="tokens")
        x = keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(inputs)

        if architecture == "lstm":
            x = keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=dropout_rate))(x)
        elif architecture == "gru":
            x = keras.layers.Bidirectional(keras.layers.GRU(128, dropout=dropout_rate))(x)
        else:
            x = PositionalEncoding(max_length, embedding_dim)(x)
            for i in range(num_layers):
                x = TransformerEncoderBlock(
                    embedding_dim, num_heads, dropout=dropout_rate, name=f"encoder_{i}"
                )(x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
            x = keras.layers.GlobalAveragePooling1D()(x)
            x = keras.layers.Dense(128, activation="gelu")(x)

        x = keras.layers.Dropout(dropout_rate)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        return keras.Model(inputs, outputs, name=name or f"{architecture}_text_classifier")

    @staticmethod
    def create_mlp(
        input_dim: int,
        num_classes: int,
        hidden_units: Sequence[int] = (256, 128),
        dropout_rate: float = 0.2,
        activation: str = "relu",
        name: str = "mlp",
    ) -> keras.Model:
        """Fully-connected classifier for tabular / flattened inputs."""
        inputs = keras.Input(shape=(input_dim,), name="features")
        x = inputs
        for units in hidden_units:
            x = keras.layers.Dense(units, activation=activation)(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        return keras.Model(inputs, outputs, name=name)

    @staticmethod
    def create_autoencoder(
        input_shape: Tuple[int, ...],
        encoding_dim: int = 64,
        architecture: str = "dense",
    ) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Return ``(autoencoder, encoder, decoder)``; ``conv`` needs H and W divisible by 4."""
        if architecture not in ("dense", "conv"):
            raise ValueError("architecture must be 'dense' or 'conv'")
        if architecture == "conv":
            if len(input_shape) != 3 or input_shape[0] % 4 or input_shape[1] % 4:
                raise ValueError("conv autoencoder needs an (H, W, C) input with H, W % 4 == 0")

        encoder_input = keras.Input(shape=input_shape, name="encoder_input")
        if architecture == "dense":
            x = keras.layers.Flatten()(encoder_input)
            x = keras.layers.Dense(512, activation="relu")(x)
            x = keras.layers.Dense(256, activation="relu")(x)
            encoded = keras.layers.Dense(encoding_dim, activation="relu", name="latent")(x)
        else:
            x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
            x = keras.layers.MaxPooling2D(2, padding="same")(x)
            x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling2D(2, padding="same")(x)
            x = keras.layers.Flatten()(x)
            encoded = keras.layers.Dense(encoding_dim, activation="relu", name="latent")(x)
        encoder = keras.Model(encoder_input, encoded, name="encoder")

        decoder_input = keras.Input(shape=(encoding_dim,), name="decoder_input")
        if architecture == "dense":
            x = keras.layers.Dense(256, activation="relu")(decoder_input)
            x = keras.layers.Dense(512, activation="relu")(x)
            x = keras.layers.Dense(int(np.prod(input_shape)), activation="sigmoid")(x)
            decoded = keras.layers.Reshape(input_shape)(x)
        else:
            small = (input_shape[0] // 4, input_shape[1] // 4, 32)
            x = keras.layers.Dense(int(np.prod(small)), activation="relu")(decoder_input)
            x = keras.layers.Reshape(small)(x)
            x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
            x = keras.layers.UpSampling2D(2)(x)
            x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
            x = keras.layers.UpSampling2D(2)(x)
            decoded = keras.layers.Conv2D(input_shape[-1], 3, activation="sigmoid", padding="same")(
                x
            )
        decoder = keras.Model(decoder_input, decoded, name="decoder")

        ae_input = keras.Input(shape=input_shape, name="autoencoder_input")
        autoencoder = keras.Model(ae_input, decoder(encoder(ae_input)), name="autoencoder")
        return autoencoder, encoder, decoder

    @staticmethod
    def create_gan(
        latent_dim: int,
        output_shape: Tuple[int, int, int],
        generator_architecture: str = "dense",
        discriminator_architecture: str = "dense",
    ) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Return ``(gan, generator, discriminator)``; ``conv`` generator needs H, W % 4 == 0."""
        for arch in (generator_architecture, discriminator_architecture):
            if arch not in ("dense", "conv"):
                raise ValueError("architectures must be 'dense' or 'conv'")
        height, width, channels = output_shape

        g_in = keras.Input(shape=(latent_dim,), name="latent")
        if generator_architecture == "dense":
            x = keras.layers.Dense(256, activation="relu")(g_in)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(512, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(int(np.prod(output_shape)), activation="tanh")(x)
            g_out = keras.layers.Reshape(output_shape)(x)
        else:
            if height % 4 or width % 4:
                raise ValueError("conv generator needs H and W divisible by 4")
            h0, w0 = height // 4, width // 4
            x = keras.layers.Dense(h0 * w0 * 256, use_bias=False)(g_in)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU(0.2)(x)
            x = keras.layers.Reshape((h0, w0, 256))(x)
            x = keras.layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU(0.2)(x)
            x = keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU(0.2)(x)
            g_out = keras.layers.Conv2DTranspose(
                channels, 5, strides=2, padding="same", activation="tanh"
            )(x)
        generator = keras.Model(g_in, g_out, name="generator")

        d_in = keras.Input(shape=output_shape, name="image")
        if discriminator_architecture == "dense":
            x = keras.layers.Flatten()(d_in)
            x = keras.layers.Dense(512)(x)
            x = keras.layers.LeakyReLU(0.2)(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(256)(x)
            x = keras.layers.LeakyReLU(0.2)(x)
            x = keras.layers.Dropout(0.3)(x)
        else:
            x = keras.layers.Conv2D(64, 5, strides=2, padding="same")(d_in)
            x = keras.layers.LeakyReLU(0.2)(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Conv2D(128, 5, strides=2, padding="same")(x)
            x = keras.layers.LeakyReLU(0.2)(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
        d_out = keras.layers.Dense(1, activation="sigmoid", name="validity")(x)
        discriminator = keras.Model(d_in, d_out, name="discriminator")

        discriminator.trainable = False
        gan_in = keras.Input(shape=(latent_dim,), name="gan_latent")
        gan = keras.Model(gan_in, discriminator(generator(gan_in)), name="gan")
        discriminator.trainable = True
        return gan, generator, discriminator


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


class ValidationMonitor(keras.callbacks.Callback):
    """
    Runs a manual validation pass every ``log_freq`` epochs and logs the result.

    Useful when you want validation metrics that differ from the compiled ones.
    """

    def __init__(
        self,
        validation_data: tf.data.Dataset,
        log_freq: int = 1,
        loss_fn: Optional[Callable] = None,
        metric: Optional[keras.metrics.Metric] = None,
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.log_freq = max(1, int(log_freq))
        self.loss_fn = loss_fn or keras.losses.SparseCategoricalCrossentropy()
        self.metric = metric or keras.metrics.SparseCategoricalAccuracy()
        self.history: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if (epoch + 1) % self.log_freq:
            return
        self.metric.reset_state()
        losses = []
        for x_val, y_val in self.validation_data:
            preds = self.model(x_val, training=False)
            losses.append(float(tf.reduce_mean(self.loss_fn(y_val, preds))))
            self.metric.update_state(y_val, preds)
        record = {
            "epoch": epoch + 1,
            "val_loss": float(np.mean(losses)) if losses else float("nan"),
            self.metric.name: float(self.metric.result()),
        }
        self.history.append(record)
        logger.info("ValidationMonitor %s", record)


class TrainingUtilities:
    """Callback bundles and a reusable ``tf.function`` training step."""

    ValidationMonitor = ValidationMonitor

    @staticmethod
    def create_callbacks(
        model_name: str,
        patience: int = 10,
        reduce_lr: bool = True,
        tensorboard: bool = True,
        checkpoint_dir: PathLike = "models/checkpoints",
        log_dir: PathLike = "logs/tensorboard",
        monitor: str = "val_loss",
    ) -> List[keras.callbacks.Callback]:
        """ModelCheckpoint + EarlyStopping (+ ReduceLROnPlateau, TensorBoard)."""
        ckpt_dir = Path(checkpoint_dir) / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks: List[keras.callbacks.Callback] = [
            keras.callbacks.ModelCheckpoint(
                filepath=compat.checkpoint_filepath(ckpt_dir, "best_model"),
                monitor=monitor,
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
            ),
        ]
        if reduce_lr:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor, factor=0.2, patience=max(1, patience // 2), min_lr=1e-7
                )
            )
        if tensorboard:
            if importlib.util.find_spec("tensorboard") is None:
                logger.warning("tensorboard is not installed; skipping the TensorBoard callback")
            else:
                callbacks.append(
                    keras.callbacks.TensorBoard(
                        log_dir=str(Path(log_dir) / model_name), histogram_freq=1
                    )
                )
        return callbacks

    @staticmethod
    def create_custom_training_step(
        model: keras.Model,
        loss_fn: Callable,
        optimizer: keras.optimizers.Optimizer,
        metrics: Optional[Sequence[keras.metrics.Metric]] = None,
        jit_compile: bool = False,
    ) -> Callable[[tf.Tensor, tf.Tensor], Dict[str, tf.Tensor]]:
        """Return a compiled ``train_step(x, y) -> {"loss": ..., <metric>: ...}`` function."""
        metrics = list(metrics or [])

        @tf.function(jit_compile=jit_compile)
        def train_step(x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = tf.reduce_mean(loss_fn(y, predictions))
            gradients = tape.gradient(loss, model.trainable_variables)
            compat.apply_gradients(optimizer, gradients, model.trainable_variables)
            results = {"loss": loss}
            for metric in metrics:
                metric.update_state(y, predictions)
                results[metric.name] = metric.result()
            return results

        return train_step


# ---------------------------------------------------------------------------
# Model analysis
# ---------------------------------------------------------------------------


def _weights_bytes(weights: Sequence[Any]) -> int:
    total = 0
    for w in weights:
        dtype = tf.as_dtype(getattr(w, "dtype", tf.float32))
        total += compat.count_params([w]) * dtype.size
    return total


class ModelAnalysis:
    """Parameter statistics, FLOP estimates and Markdown reports."""

    @staticmethod
    def analyze_model_architecture(model: keras.Model) -> Dict[str, Any]:
        """Parameter counts, dtype-aware size and per-layer details."""
        trainable = compat.count_params(model.trainable_weights)
        non_trainable = compat.count_params(model.non_trainable_weights)
        layer_info = []
        for i, layer in enumerate(model.layers):
            layer_info.append(
                {
                    "index": i,
                    "name": layer.name,
                    "type": type(layer).__name__,
                    "output_shape": str(compat.layer_output_shape(layer)),
                    "params": compat.count_params(layer.weights),
                    "trainable": bool(getattr(layer, "trainable", True)),
                }
            )
        size_bytes = _weights_bytes(model.weights)
        return {
            "name": model.name,
            "total_parameters": trainable + non_trainable,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "total_layers": len(model.layers),
            "model_size_bytes": size_bytes,
            "model_size_mb": size_bytes / (1024 * 1024),
            "input_shape": str(getattr(model, "input_shape", None)),
            "output_shape": str(getattr(model, "output_shape", None)),
            "layer_details": layer_info,
        }

    @staticmethod
    def _analytic_flops(model: keras.Model) -> int:
        flops = 0
        for layer in model.layers:
            out_shape = compat.layer_output_shape(layer)
            if out_shape is None or isinstance(out_shape[0], tuple):
                continue
            out_elems = int(np.prod([d for d in out_shape[1:] if d is not None]))
            if isinstance(layer, keras.layers.Dense):
                in_dim = int(layer.kernel.shape[0])
                flops += 2 * in_dim * int(layer.units)
            elif isinstance(layer, keras.layers.Conv2D):
                kh, kw = layer.kernel_size
                in_ch = int(layer.kernel.shape[2])
                flops += 2 * kh * kw * in_ch * out_elems
        return int(flops)

    @staticmethod
    def compute_model_flops(
        model: keras.Model,
        input_shape: Optional[Tuple[int, ...]] = None,
        batch_size: int = 1,
    ) -> int:
        """
        Forward-pass floating point operations for one batch of ``batch_size``.

        Uses the TensorFlow graph profiler on a frozen concrete function and falls
        back to an analytic Dense/Conv2D estimate if profiling is unavailable.
        """
        if input_shape is None:
            input_shape = tuple(compat.model_input_shape(model)[1:])
        spec = tf.TensorSpec((batch_size, *input_shape), tf.float32)
        try:
            forward = tf.function(lambda x: model(x, training=False))
            concrete = forward.get_concrete_function(spec)
            try:
                from tensorflow.python.framework.convert_to_constants import (
                    convert_variables_to_constants_v2_as_graph,
                )

                frozen, _ = convert_variables_to_constants_v2_as_graph(concrete)
                graph_def = frozen.graph.as_graph_def()
            except Exception:  # pragma: no cover - private API changed
                graph_def = concrete.graph.as_graph_def()
            with tf.Graph().as_default() as graph:
                tf.compat.v1.import_graph_def(graph_def, name="")
                options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                options["output"] = "none"
                profile = tf.compat.v1.profiler.profile(
                    graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=options
                )
            if profile is not None and profile.total_float_ops > 0:
                return int(profile.total_float_ops)
        except Exception as exc:  # pragma: no cover - profiler unavailable
            logger.debug("Profiler FLOP count failed (%s); using analytic estimate", exc)
        return ModelAnalysis._analytic_flops(model) * batch_size

    @staticmethod
    def model_summary_string(model: keras.Model) -> str:
        buffer = io.StringIO()
        model.summary(print_fn=lambda line, *args, **kwargs: buffer.write(line + "\n"))
        return buffer.getvalue()

    @staticmethod
    def create_model_summary_report(
        model: keras.Model,
        input_shape: Optional[Tuple[int, ...]] = None,
        save_path: Optional[PathLike] = None,
    ) -> str:
        """Markdown report with parameter, size and FLOP statistics."""
        analysis = ModelAnalysis.analyze_model_architecture(model)
        try:
            flops = ModelAnalysis.compute_model_flops(model, input_shape)
        except Exception as exc:
            logger.warning("FLOP estimation failed: %s", exc)
            flops = 0

        lines = [
            f"# Model Analysis Report: {model.name}",
            "",
            "## Architecture Overview",
            f"- **Total Parameters**: {analysis['total_parameters']:,}",
            f"- **Trainable Parameters**: {analysis['trainable_parameters']:,}",
            f"- **Non-trainable Parameters**: {analysis['non_trainable_parameters']:,}",
            f"- **Total Layers**: {analysis['total_layers']}",
            f"- **Model Size**: {analysis['model_size_mb']:.2f} MB",
            f"- **Estimated FLOPs (batch=1)**: {flops:,}",
            "",
            "## Layer Details",
        ]
        for layer in analysis["layer_details"]:
            lines.append(
                f"- **{layer['name']}** ({layer['type']}): {layer['output_shape']} "
                f"- {layer['params']:,} params"
            )
        lines += ["", "## Keras Summary", "```", ModelAnalysis.model_summary_string(model), "```"]
        report = "\n".join(lines)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text(report, encoding="utf-8")
            logger.info("Model analysis report saved to %s", save_path)
        return report


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def create_classification_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: Optional[str] = None,
    compile_model: bool = True,
    learning_rate: float = 1e-3,
    vocab_size: int = 10000,
) -> keras.Model:
    """Pick a CNN (3-D input), text model (1-D int input) or MLP and optionally compile it."""
    if len(input_shape) == 3:
        model = ModelBuilders.create_cnn_classifier(
            input_shape, num_classes, architecture=architecture or "simple"
        )
    elif len(input_shape) == 1:
        if architecture in (None, "mlp"):
            model = ModelBuilders.create_mlp(input_shape[0], num_classes)
        else:
            model = ModelBuilders.create_text_classifier(
                vocab_size=vocab_size,
                max_length=input_shape[0],
                num_classes=num_classes,
                architecture=architecture,
            )
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


_APPLICATIONS: Dict[str, Tuple[str, str]] = {
    "ResNet50": ("resnet", "ResNet50"),
    "ResNet50V2": ("resnet_v2", "ResNet50V2"),
    "VGG16": ("vgg16", "VGG16"),
    "MobileNetV2": ("mobilenet_v2", "MobileNetV2"),
    "MobileNetV3Small": ("mobilenet_v3", "MobileNetV3Small"),
    "EfficientNetB0": ("efficientnet", "EfficientNetB0"),
    "EfficientNetV2B0": ("efficientnet_v2", "EfficientNetV2B0"),
    "InceptionV3": ("inception_v3", "InceptionV3"),
    "DenseNet121": ("densenet", "DenseNet121"),
    "ConvNeXtTiny": ("convnext", "ConvNeXtTiny"),
}


def create_transfer_learning_model(
    base_model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    weights: Optional[str] = "imagenet",
    fine_tune: bool = False,
    fine_tune_at: int = 100,
    dropout_rate: float = 0.2,
    include_preprocessing: bool = True,
) -> keras.Model:
    """
    ImageNet backbone + GAP + dropout + softmax head.

    ``include_preprocessing`` inserts the backbone's own ``preprocess_input`` so the
    model accepts raw ``[0, 255]`` pixels.  Pass ``weights=None`` for random init
    (no download).
    """
    if base_model_name not in _APPLICATIONS:
        raise ValueError(
            f"Unsupported base model '{base_model_name}'. Choose from {sorted(_APPLICATIONS)}"
        )
    module_name, class_name = _APPLICATIONS[base_model_name]
    app_module = getattr(keras.applications, module_name)
    base_model = getattr(keras.applications, class_name)(
        weights=weights, include_top=False, input_shape=input_shape
    )
    base_model.trainable = bool(fine_tune)
    if fine_tune:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    if include_preprocessing and hasattr(app_module, "preprocess_input"):
        x = keras.layers.Lambda(app_module.preprocess_input, name="preprocess")(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"{base_model_name}_transfer")


def _metadata_path(save_path: PathLike) -> Path:
    path = Path(save_path)
    if compat.is_keras_file(path):
        return path.with_suffix(path.suffix + ".metadata.json")
    return path / "metadata.json"


def save_model_with_metadata(
    model: keras.Model, save_path: PathLike, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save a ``.keras`` file (sidecar ``*.metadata.json``) or SavedModel dir (``metadata.json``)."""
    path = compat.save_model(model, save_path)
    payload = {
        "tensorflow_version": compat.TF_VERSION,
        "keras_version": compat.KERAS_VERSION,
        "tensorversehub_format": "keras" if compat.is_keras_file(path) else "saved_model",
        **(metadata or {}),
    }
    meta_path = _metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Model saved to %s with metadata %s", path, meta_path)
    return path


def load_model_with_metadata(
    model_path: PathLike, custom_objects: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Inverse of :func:`save_model_with_metadata`."""
    model = compat.load_model(model_path, custom_objects=custom_objects)
    meta_path = _metadata_path(model_path)
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, metadata


__all__ = [
    "CustomLayers",
    "ModelAnalysis",
    "ModelBuilders",
    "MultiHeadAttention",
    "PositionalEncoding",
    "TrainingUtilities",
    "TransformerEncoderBlock",
    "ValidationMonitor",
    "create_classification_model",
    "create_transfer_learning_model",
    "load_model_with_metadata",
    "save_model_with_metadata",
]
