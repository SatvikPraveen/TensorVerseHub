import importlib.util

import numpy as np
import pytest
import tensorflow as tf

from tensorversehub import compat
from tensorversehub import model_utils as mu
from tensorversehub.compat import keras


class TestCustomLayers:
    def test_multi_head_attention_shapes_and_mask(self):
        mha = mu.CustomLayers.MultiHeadAttention(16, 4)
        x = tf.random.normal((2, 5, 16))
        mask = mu.CustomLayers.create_look_ahead_mask(5)
        out, weights = mha(x, attention_mask=mask, return_attention_scores=True)
        assert out.shape == (2, 5, 16) and weights.shape == (2, 4, 5, 5)
        upper = np.triu(np.ones((5, 5)), k=1).astype(bool)
        assert np.all(weights.numpy()[:, :, upper] < 1e-6)
        np.testing.assert_allclose(tf.reduce_sum(weights, -1).numpy(), 1.0, atol=1e-5)
        cross = mha(x, tf.random.normal((2, 7, 16)))
        assert cross.shape == (2, 5, 16)

    def test_multi_head_attention_validation_and_config(self):
        with pytest.raises(ValueError):
            mu.CustomLayers.MultiHeadAttention(10, 4)
        layer = mu.CustomLayers.MultiHeadAttention(8, 2, dropout=0.1)
        clone = mu.CustomLayers.MultiHeadAttention.from_config(layer.get_config())
        assert clone.num_heads == 2 and clone.dropout_rate == 0.1

    def test_padding_mask(self):
        mask = mu.CustomLayers.create_padding_mask(tf.constant([[1, 2, 0], [3, 0, 0]]))
        assert mask.shape == (2, 1, 1, 3)
        assert mask.numpy()[0, 0, 0].tolist() == [True, True, False]

    def test_positional_encoding(self):
        pe = mu.CustomLayers.PositionalEncoding(10, 8)
        enc = pe.pos_encoding.numpy()
        assert enc.shape == (1, 10, 8) and np.abs(enc).max() <= 1.0
        x = tf.zeros((2, 6, 8))
        out = pe(x)
        assert out.shape == (2, 6, 8)
        np.testing.assert_allclose(out.numpy()[0], enc[0, :6], atol=1e-6)
        with pytest.raises(tf.errors.InvalidArgumentError):
            pe(tf.zeros((1, 11, 8)))
        assert mu.PositionalEncoding.from_config(pe.get_config()).position == 10

    def test_transformer_block_and_serialization(self, tmp_path):
        inputs = keras.Input((6,), dtype="int32")
        x = keras.layers.Embedding(30, 8)(inputs)
        x = mu.CustomLayers.PositionalEncoding(6, 8)(x)
        x = mu.CustomLayers.TransformerEncoderBlock(8, 2, ff_dim=16, dropout=0.0)(x)
        out = keras.layers.Dense(2, activation="softmax")(keras.layers.GlobalAveragePooling1D()(x))
        model = keras.Model(inputs, out)
        tokens = np.random.randint(0, 30, (3, 6))
        preds = model.predict(tokens, verbose=0)
        path = tmp_path / "transformer.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        np.testing.assert_allclose(reloaded.predict(tokens, verbose=0), preds, atol=1e-5)


class TestModelBuilders:
    @pytest.mark.parametrize("arch", mu.ModelBuilders.CNN_ARCHITECTURES)
    def test_cnn_classifier(self, arch, images, labels):
        model = mu.ModelBuilders.create_cnn_classifier((16, 16, 3), 3, arch)
        assert model.output_shape == (None, 3)
        model.compile("adam", "sparse_categorical_crossentropy")
        model.fit(images, labels, epochs=1, batch_size=8, verbose=0)
        assert np.allclose(model.predict(images[:2], verbose=0).sum(axis=1), 1.0, atol=1e-5)

    def test_cnn_validation(self):
        with pytest.raises(ValueError):
            mu.ModelBuilders.create_cnn_classifier((8, 8, 3), 3, "unknown")
        with pytest.raises(ValueError):
            mu.ModelBuilders.create_cnn_classifier((8, 8, 3), 0)

    @pytest.mark.parametrize("arch", mu.ModelBuilders.TEXT_ARCHITECTURES)
    def test_text_classifier(self, arch):
        model = mu.ModelBuilders.create_text_classifier(
            50, 8, 2, embedding_dim=16, architecture=arch, num_heads=2, num_layers=1
        )
        tokens = np.random.randint(0, 50, (4, 8))
        assert model.predict(tokens, verbose=0).shape == (4, 2)
        with pytest.raises(ValueError):
            mu.ModelBuilders.create_text_classifier(50, 8, 2, architecture="cnn")

    def test_mlp(self):
        model = mu.ModelBuilders.create_mlp(10, 4, hidden_units=(8,))
        assert model.predict(np.zeros((2, 10), np.float32), verbose=0).shape == (2, 4)

    @pytest.mark.parametrize("arch", ["dense", "conv"])
    def test_autoencoder(self, arch, images):
        ae, enc, dec = mu.ModelBuilders.create_autoencoder((16, 16, 3), 8, arch)
        assert enc.output_shape == (None, 8) and ae.output_shape == (None, 16, 16, 3)
        np.testing.assert_allclose(
            ae.predict(images[:2], verbose=0),
            dec.predict(enc.predict(images[:2], verbose=0), verbose=0),
            atol=1e-5,
        )

    def test_autoencoder_validation(self):
        with pytest.raises(ValueError):
            mu.ModelBuilders.create_autoencoder((15, 15, 3), 8, "conv")
        with pytest.raises(ValueError):
            mu.ModelBuilders.create_autoencoder((16, 16, 3), 8, "vae")

    @pytest.mark.parametrize("g,d", [("dense", "dense"), ("conv", "conv")])
    def test_gan(self, g, d):
        gan, gen, disc = mu.ModelBuilders.create_gan(8, (16, 16, 1), g, d)
        z = np.random.randn(2, 8).astype("float32")
        fake = gen.predict(z, verbose=0)
        assert fake.shape == (2, 16, 16, 1) and disc.predict(fake, verbose=0).shape == (2, 1)
        assert gan.predict(z, verbose=0).shape == (2, 1)
        assert disc.trainable  # restored after GAN assembly
        with pytest.raises(ValueError):
            mu.ModelBuilders.create_gan(8, (15, 15, 1), "conv")


class TestTrainingUtilities:
    def test_create_callbacks(self, tmp_path):
        callbacks = mu.TrainingUtilities.create_callbacks(
            "unit", patience=2, checkpoint_dir=tmp_path / "ck", log_dir=tmp_path / "logs"
        )
        types = {type(c).__name__ for c in callbacks}
        assert {"ModelCheckpoint", "EarlyStopping", "ReduceLROnPlateau"} <= types
        if importlib.util.find_spec("tensorboard") is not None:
            assert "TensorBoard" in types  # skipped automatically when tensorboard is absent
        ckpt = next(c for c in callbacks if type(c).__name__ == "ModelCheckpoint")
        assert str(ckpt.filepath).endswith("best_model.keras")
        minimal = mu.TrainingUtilities.create_callbacks(
            "u2", reduce_lr=False, tensorboard=False, checkpoint_dir=tmp_path
        )
        assert len(minimal) == 2

    def test_custom_training_step_reduces_loss(self, images, labels):
        model = mu.ModelBuilders.create_cnn_classifier((16, 16, 3), 3, "simple", dropout_rate=0.0)
        step = mu.TrainingUtilities.create_custom_training_step(
            model,
            keras.losses.SparseCategoricalCrossentropy(),
            keras.optimizers.Adam(1e-2),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        first = float(step(images, labels)["loss"])
        for _ in range(30):
            result = step(images, labels)
        assert float(result["loss"]) < first and "acc" in result

    def test_validation_monitor(self, cnn_model, image_dataset):
        monitor = mu.TrainingUtilities.ValidationMonitor(image_dataset, log_freq=1)
        cnn_model.fit(image_dataset, epochs=2, verbose=0, callbacks=[monitor])
        assert len(monitor.history) == 2 and "val_loss" in monitor.history[0]


class TestModelAnalysis:
    def test_analyze(self, cnn_model):
        analysis = mu.ModelAnalysis.analyze_model_architecture(cnn_model)
        assert analysis["total_parameters"] == cnn_model.count_params()
        assert analysis["model_size_bytes"] == cnn_model.count_params() * 4
        assert analysis["layer_details"][0]["type"] == "InputLayer"

    def test_flops(self, cnn_model):
        flops = mu.ModelAnalysis.compute_model_flops(cnn_model)
        analytic = mu.ModelAnalysis._analytic_flops(cnn_model)
        assert flops > 0 and analytic > 0
        assert 0.5 < flops / analytic < 2.0
        assert mu.ModelAnalysis.compute_model_flops(cnn_model, (16, 16, 3), batch_size=2) >= flops

    def test_report(self, cnn_model, tmp_path):
        report = mu.ModelAnalysis.create_model_summary_report(
            cnn_model, save_path=tmp_path / "r.md"
        )
        assert "Total Parameters" in report and (tmp_path / "r.md").exists()
        assert "Total params" in mu.ModelAnalysis.model_summary_string(cnn_model)


class TestConvenience:
    def test_create_classification_model(self):
        cnn = mu.create_classification_model((16, 16, 3), 3)
        assert cnn.optimizer is not None
        mlp = mu.create_classification_model((10,), 2, compile_model=False)
        assert mlp.output_shape == (None, 2) and getattr(mlp, "optimizer", None) is None
        text = mu.create_classification_model((8,), 2, architecture="gru", vocab_size=20)
        assert text.predict(np.zeros((1, 8), np.int32), verbose=0).shape == (1, 2)
        with pytest.raises(ValueError):
            mu.create_classification_model((2, 2), 2)

    def test_transfer_learning(self):
        model = mu.create_transfer_learning_model("MobileNetV2", (32, 32, 3), 3, weights=None)
        assert model.output_shape == (None, 3)
        assert model.predict(np.zeros((1, 32, 32, 3), np.float32), verbose=0).shape == (1, 3)
        tuned = mu.create_transfer_learning_model(
            "MobileNetV2",
            (32, 32, 3),
            3,
            weights=None,
            fine_tune=True,
            fine_tune_at=10,
            include_preprocessing=False,
        )
        assert tuned.count_params() == model.count_params()
        with pytest.raises(ValueError):
            mu.create_transfer_learning_model("NotAModel", (32, 32, 3), 3)

    @pytest.mark.parametrize("name", ["model.keras", "saved"])
    def test_save_load_with_metadata(self, cnn_model, images, tmp_path, name):
        path = mu.save_model_with_metadata(cnn_model, tmp_path / name, {"owner": "tests"})
        model, metadata = mu.load_model_with_metadata(path)
        assert metadata["owner"] == "tests" and metadata["keras_version"] == compat.KERAS_VERSION
        preds = (
            model.predict(images[:2], verbose=0)
            if isinstance(model, keras.Model)
            else model.predict(images[:2])
        )
        np.testing.assert_allclose(preds, cnn_model.predict(images[:2], verbose=0), atol=1e-4)
