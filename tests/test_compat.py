import numpy as np
import pytest
import tensorflow as tf

from tensorversehub import compat
from tensorversehub.compat import keras


def test_version_parsing():
    assert compat._parse_version("2.21.0-rc1") == (2, 21, 0)
    assert compat._parse_version("3.15") == (3, 15)
    assert compat.TF_VERSION_INFO >= compat.MIN_TF_VERSION
    assert compat.IS_KERAS_3 != compat.IS_LEGACY_KERAS
    compat.check_tensorflow_version()
    with pytest.raises(ImportError):
        compat.check_tensorflow_version((99, 0))


def test_path_helpers(tmp_path):
    assert compat.is_keras_file("m.keras") and compat.is_keras_file("m.H5")
    assert not compat.is_keras_file(tmp_path)
    assert not compat.is_saved_model_dir(tmp_path)
    assert compat.checkpoint_filepath(tmp_path, "best").endswith("best.keras")
    assert compat.checkpoint_filepath(tmp_path, "w", weights_only=True).endswith("w.weights.h5")


def test_save_and_load_keras_file(cnn_model, images, tmp_path):
    path = compat.save_model(cnn_model, tmp_path / "nested" / "model.keras")
    loaded = compat.load_model(path)
    assert isinstance(loaded, keras.Model)
    np.testing.assert_allclose(
        loaded.predict(images[:2], verbose=0), cnn_model.predict(images[:2], verbose=0), atol=1e-5
    )


def test_export_and_predictor(cnn_model, images, tmp_path):
    path = compat.save_model(cnn_model, tmp_path / "sm")
    assert compat.is_saved_model_dir(path)
    predictor = compat.load_model(path)
    if isinstance(predictor, keras.Model):  # legacy keras may reload a full model
        preds = predictor.predict(images[:4], verbose=0)
    else:
        assert isinstance(predictor, compat.SavedModelPredictor)
        assert predictor.input_shape[1:] == images.shape[1:]
        preds = predictor.predict(images[:4], batch_size=2)
    np.testing.assert_allclose(preds, cnn_model.predict(images[:4], verbose=0), atol=1e-4)
    with pytest.raises(ValueError):
        compat.SavedModelPredictor(path, signature="missing")


def test_load_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        compat.load_model(tmp_path / "nothing")


def test_introspection(cnn_model):
    assert compat.count_params(cnn_model.weights) == cnn_model.count_params()
    assert compat.count_params([]) == 0
    assert compat.layer_output_shape(cnn_model.layers[1])[-1] == 32
    assert compat.model_input_shape(cnn_model) == (None, 16, 16, 3)
    spec = compat.input_signature(cnn_model)
    assert len(spec) == 1 and spec[0].shape.as_list() == [None, 16, 16, 3]
    assert isinstance(compat.metric_names(cnn_model), list)


def test_unpack_batch():
    assert compat.unpack_batch((1, 2)) == (1, 2, None)
    assert compat.unpack_batch((1, 2, 3)) == (1, 2, 3)
    assert compat.unpack_batch(5) == (5, None, None)


def test_compute_loss(trained_model, images, labels):
    y_pred = trained_model(images[:4])
    loss = compat.compute_loss(trained_model, images[:4], labels[:4], y_pred)
    assert float(loss) > 0
    assert "accuracy" in compat.metric_names(trained_model)


def test_loss_scale_optimizer_helpers():
    opt = keras.optimizers.Adam()
    lso = compat.wrap_loss_scale_optimizer(opt)
    assert compat.is_loss_scale_optimizer(lso)
    assert compat.wrap_loss_scale_optimizer(lso) is lso
    scaled = compat.scale_loss(lso, tf.constant(1.0))
    assert float(scaled) >= 1.0
    assert compat.current_loss_scale(opt) is None
    assert compat.current_loss_scale(lso) is not None


def test_policy_roundtrip():
    compat.set_mixed_precision_policy("mixed_float16")
    assert compat.global_policy_name() == "mixed_float16"
    compat.set_mixed_precision_policy("float32")
    assert compat.global_policy_name() == "float32"


def test_require_legacy_keras():
    if compat.IS_KERAS_3:
        with pytest.raises(RuntimeError, match="TF_USE_LEGACY_KERAS"):
            compat.require_legacy_keras("Pruning")
    else:
        compat.require_legacy_keras("Pruning")


def test_version_info_keys():
    info = compat.version_info()
    assert {"tensorflow", "keras", "keras_generation", "gpus", "mixed_precision_policy"} <= set(
        info
    )
