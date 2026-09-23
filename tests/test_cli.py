import json
import subprocess
import sys

import numpy as np
import pytest

from tensorversehub import cli
from tests.conftest import requires_fastapi


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "tensorverse 2." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["train", "--help"])


def test_info_command(capsys):
    assert cli.main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "tensorflow" in payload


def test_console_script_entry():
    result = subprocess.run(
        [sys.executable, "-m", "tensorversehub.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and "train" in result.stdout


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    code = cli.main(
        [
            "--log-level",
            "warning",
            "train",
            "--task",
            "classification",
            "--architecture",
            "simple",
            "--image-size",
            "16",
            "16",
            "--num-classes",
            "3",
            "--num-samples",
            "48",
            "--batch-size",
            "16",
            "--epochs",
            "1",
            "--output-dir",
            str(out),
            "--log-dir",
            str(out / "logs"),
            "--export-savedmodel",
            "--quiet",
        ]
    )
    assert code == 0
    return out


def test_train_artifacts(trained_dir):
    assert (trained_dir / "final_model.keras").exists()
    assert (trained_dir / "final_model.keras.metadata.json").exists()
    assert (trained_dir / "final_model_savedmodel" / "saved_model.pb").exists()
    history = json.loads((trained_dir / "training_history.json").read_text())
    assert len(history["loss"]) == 1


def test_train_config_override(tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps({"epochs": 1, "task": "autoencoder", "image-size": [16, 16], "num-samples": 32})
    )
    code = cli.main(
        [
            "--log-level",
            "warning",
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "ae"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--quiet",
        ]
    )
    assert code == 0 and (tmp_path / "ae" / "final_model.keras").exists()


@pytest.mark.parametrize("model_name", ["final_model.keras", "final_model_savedmodel"])
def test_evaluate(trained_dir, tmp_path, model_name):
    out = tmp_path / "eval"
    code = cli.main(
        [
            "--log-level",
            "warning",
            "evaluate",
            "--model",
            str(trained_dir / model_name),
            "--image-size",
            "16",
            "16",
            "--num-classes",
            "3",
            "--num-samples",
            "24",
            "--output-dir",
            str(out),
            "--report",
            "--confusion-matrix",
            "--roc-curves",
            "--class-names",
            "a",
            "b",
            "c",
            "--quiet",
        ]
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert (out / "confusion_matrix.png").exists() and (out / "classification_report.txt").exists()


def test_convert(trained_dir, tmp_path):
    out = tmp_path / "conv"
    code = cli.main(
        [
            "--log-level",
            "warning",
            "convert",
            "--model",
            str(trained_dir / "final_model.keras"),
            "--to",
            "tflite",
            "--quantize",
            "int8",
            "--output",
            str(out),
            "--benchmark",
            "--quiet",
        ]
    )
    assert code == 0 and (out / "tflite" / "model_int8.tflite").exists()
    code = cli.main(
        [
            "--log-level",
            "warning",
            "convert",
            "--model",
            str(trained_dir / "final_model.keras"),
            "--to",
            "all",
            "--output",
            str(out / "all"),
            "--quiet",
        ]
    )
    assert code == 0 and (out / "all" / "saved_model" / "saved_model.pb").exists()
    code = cli.main(
        [
            "--log-level",
            "warning",
            "evaluate",
            "--model",
            str(out / "tflite" / "model_int8.tflite"),
            "--image-size",
            "16",
            "16",
            "--num-classes",
            "3",
            "--num-samples",
            "8",
            "--output-dir",
            str(tmp_path / "eval_tflite"),
            "--quiet",
        ]
    )
    assert code == 0


def test_convert_rejects_savedmodel(trained_dir, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "convert",
                "--model",
                str(trained_dir / "final_model_savedmodel"),
                "--to",
                "tflite",
                "--output",
                str(tmp_path),
            ]
        )


@requires_fastapi
def test_serve_app(trained_dir):
    from fastapi.testclient import TestClient

    from tensorversehub.cli.serve import build_app

    app = build_app(
        str(trained_dir / "final_model.keras"), class_names=["a", "b", "c"], max_batch_size=2
    )
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/metadata").json()["input_shape"] == [None, 16, 16, 3]
    sample = np.zeros((1, 16, 16, 3)).tolist()
    body = client.post("/predict", json={"instances": sample}).json()
    assert body["predictions"][0]["class"] in {"a", "b", "c"}
    assert client.post("/predict", json={}).status_code == 400
    assert (
        client.post("/predict", json={"instances": np.zeros((3, 16, 16, 3)).tolist()}).status_code
        == 400
    )
