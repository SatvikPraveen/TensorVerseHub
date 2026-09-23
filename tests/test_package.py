import importlib
import subprocess
import sys

import tensorversehub as tvh


def test_version_and_lazy_attributes():
    assert tvh.__version__.count(".") == 2
    assert tvh.ModelBuilders is tvh.model_utils.ModelBuilders
    assert tvh.SavedModelPredictor is tvh.compat.SavedModelPredictor
    for name in tvh.__all__:
        assert getattr(tvh, name) is not None
    assert "data_utils" in dir(tvh)


def test_unknown_attribute_raises():
    import pytest

    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(tvh, "does_not_exist")  # noqa: B009


def test_import_has_no_side_effects():
    """Importing the package must not print, configure GPUs or touch the dtype policy."""
    code = (
        "import io, contextlib, sys\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    import tensorversehub\n"
        "    import tensorversehub.model_utils\n"
        "assert buf.getvalue() == '', repr(buf.getvalue())\n"
        "import tensorflow as tf\n"
        "assert tf.keras.mixed_precision.global_policy().name == 'float32'\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("clean")


def test_about_and_configure():
    info = tvh.about()
    assert info["tensorversehub"] == tvh.__version__
    assert info["keras_generation"] in (2, 3)
    configured = tvh.configure_tensorflow(seed=7, memory_growth=False)
    assert configured["tensorflow"] == info["tensorflow"]


def test_submodules_importable():
    for name in (
        "compat",
        "data_utils",
        "model_utils",
        "training_utils",
        "optimization_utils",
        "export_utils",
        "visualization",
        "cli",
    ):
        assert importlib.import_module(f"tensorversehub.{name}")
