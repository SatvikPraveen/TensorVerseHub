import json
import zipfile

import numpy as np
import pytest

from tensorversehub import compat
from tensorversehub import export_utils as eu
from tests.conftest import requires_onnxruntime, requires_tf2onnx


class TestSavedModel:
    def test_export_and_load(self, trained_model, images, tmp_path):
        stats = eu.SavedModelExporter.export_savedmodel(
            trained_model, tmp_path / "sm", metadata={"tag": "t"}
        )
        assert stats["size_bytes"] > 0 and (tmp_path / "sm" / "model_config.json").exists()
        predictor, metadata = eu.SavedModelExporter.load_savedmodel_with_metadata(tmp_path / "sm")
        assert metadata["tag"] == "t" and metadata["model_config"]["num_layers"] == len(
            trained_model.layers
        )
        np.testing.assert_allclose(
            predictor.predict(images[:3]), trained_model.predict(images[:3], verbose=0), atol=1e-4
        )
        assert "SavedModelPredictor" in repr(predictor)


class TestTFLite:
    @pytest.mark.parametrize("kind", ["float32", "dynamic", "float16"])
    def test_export_and_validate(self, trained_model, images, image_dataset, tmp_path, kind):
        path = tmp_path / f"m_{kind}.tflite"
        stats = eu.TFLiteExporter.export_tflite(trained_model, path, kind)
        assert path.exists() and stats["tflite_size_bytes"] == path.stat().st_size
        check = eu.TFLiteExporter.validate_tflite(trained_model, path, images[:4], atol=5e-2)
        assert check["within_tolerance"] and check["argmax_agreement"] == 1.0

    def test_int8_and_errors(self, trained_model, images, image_dataset, tmp_path):
        path = tmp_path / "int8.tflite"
        eu.TFLiteExporter.export_tflite(
            trained_model, path, "int8", representative_dataset=image_dataset
        )
        check = eu.TFLiteExporter.validate_tflite(trained_model, path, images[:4], atol=0.25)
        assert check["within_tolerance"]
        bench = eu.TFLiteExporter.benchmark_tflite_model(path, images[0], num_runs=3, warmup=1)
        assert bench["avg_inference_time_ms"] > 0 and bench["throughput_fps"] > 0
        with pytest.raises(ValueError):
            eu.TFLiteExporter.export_tflite(trained_model, path, "int8")
        with pytest.raises(ValueError):
            eu.TFLiteExporter.export_tflite(trained_model, path, "int4")
        with pytest.raises(ValueError):
            eu.TFLiteExporter.export_tflite(trained_model, path, target_ops=["BOGUS"])

    def test_select_tf_ops(self, trained_model, tmp_path):
        stats = eu.TFLiteExporter.export_tflite(
            trained_model, tmp_path / "sel.tflite", target_ops=["TFLITE_BUILTINS", "SELECT_TF_OPS"]
        )
        assert stats["compression_ratio"] > 0


@requires_tf2onnx
class TestONNX:
    def test_export(self, trained_model, images, tmp_path):
        stats = eu.ONNXExporter.export_onnx(trained_model, tmp_path / "m.onnx", opset_version=15)
        assert stats["onnx_size_bytes"] > 0 and stats["input_names"] == ["image"]

    @requires_onnxruntime
    def test_validate(self, trained_model, images, tmp_path):
        eu.ONNXExporter.export_onnx(trained_model, tmp_path / "m.onnx")
        assert eu.ONNXExporter.validate_onnx(trained_model, tmp_path / "m.onnx", images[:3])[
            "within_tolerance"
        ]


class TestMultiFormat:
    def test_export_all(self, trained_model, tmp_path):
        exporter = eu.MultiFormatExporter(trained_model, "unit")
        stats = exporter.export_all_formats(
            tmp_path,
            formats=["savedmodel", "keras", "tflite", "coreml"],
            metadata={"a": 1},
            tflite_quantization="dynamic",
        )
        assert "error" in stats["coreml"] or "export_path" in stats["coreml"]
        assert (tmp_path / "unit.keras").exists() and (tmp_path / "unit.tflite").exists()
        summary = json.loads((tmp_path / "export_summary.json").read_text())
        assert summary["model_name"] == "unit" and set(summary["export_stats"]) == {
            "savedmodel",
            "keras",
            "tflite",
            "coreml",
        }
        with pytest.raises(ValueError):
            exporter.export_all_formats(tmp_path, formats=["pickle"])
        with pytest.raises((ImportError, RuntimeError)):
            eu.MultiFormatExporter(trained_model).export_all_formats(
                tmp_path, formats=["coreml"], raise_on_error=True
            )

    def test_quick_export_and_package(self, trained_model, tmp_path):
        stats = eu.quick_export(trained_model, tmp_path / "q", formats=["keras", "tflite"])
        assert set(stats) == {"keras", "tflite"}
        package = eu.create_deployment_package(
            trained_model, tmp_path / "pkg.zip", formats=["keras", "tflite"]
        )
        names = zipfile.ZipFile(package).namelist()
        assert "package_metadata.json" in names and any(n.endswith(".tflite") for n in names)
        meta = json.loads(zipfile.ZipFile(package).read("package_metadata.json"))
        assert (
            meta["export_formats"] == ["keras", "tflite"]
            and meta["keras_version"] == compat.KERAS_VERSION
        )


def test_make_interpreter_from_content(trained_model, tmp_path):
    path = tmp_path / "c.tflite"
    eu.TFLiteExporter.export_tflite(trained_model, path)
    interpreter = eu.make_interpreter(model_content=path.read_bytes())
    assert interpreter.get_input_details()[0]["shape"].tolist() == [1, 16, 16, 3]
