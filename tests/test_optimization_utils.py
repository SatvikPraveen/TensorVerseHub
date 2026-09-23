import numpy as np
import pytest
import tensorflow as tf

from tensorversehub import compat
from tensorversehub import optimization_utils as ou
from tensorversehub.compat import keras
from tests.conftest import requires_tfmot


class TestQuantization:
    @pytest.mark.parametrize("kind", ["default", "dynamic", "float16", "int8", "int8_fallback"])
    def test_post_training(self, trained_model, image_dataset, kind):
        tflite = ou.ModelQuantization.quantize_model_post_training(
            trained_model, image_dataset, kind
        )
        assert isinstance(tflite, bytes) and len(tflite) > 1000

    def test_numpy_representative_and_errors(self, trained_model, images):
        assert ou.ModelQuantization.quantize_model_post_training(trained_model, images, "int8")
        with pytest.raises(ValueError):
            ou.ModelQuantization.quantize_model_post_training(trained_model, None, "int8")
        with pytest.raises(ValueError):
            ou.ModelQuantization.quantize_model_post_training(trained_model, None, "int4")

    def test_optimize_for_mobile(self, trained_model, image_dataset):
        model_bytes, report = ou.optimize_for_mobile(
            trained_model, image_dataset, target_size_mb=100
        )
        assert report["selected"] in report and report[report["selected"]]["meets_target"]
        assert len(model_bytes) == int(report[report["selected"]]["size_mb"] * 2**20)
        tiny, report = ou.optimize_for_mobile(
            trained_model, None, target_size_mb=1e-6, strategies=("dynamic", "int8")
        )
        assert "error" in report["int8"] and report["selected"] == "dynamic"
        with pytest.raises(RuntimeError):
            ou.optimize_for_mobile(trained_model, None, strategies=("int8",))

    def test_qat_requires_legacy(self, trained_model, image_dataset):
        if compat.IS_KERAS_3:
            with pytest.raises(RuntimeError):
                ou.ModelQuantization.quantize_model_qat(trained_model, image_dataset, epochs=1)


class TestPruning:
    def test_compute_sparsity(self, cnn_model):
        report = ou.ModelPruning.compute_sparsity(cnn_model)
        assert report["overall_sparsity"] == pytest.approx(0.0) and report["per_layer"]
        kernel = cnn_model.layers[1].kernel
        kernel.assign(tf.zeros_like(kernel))
        assert (
            ou.ModelPruning.compute_sparsity(cnn_model)["per_layer"][cnn_model.layers[1].name]
            == 1.0
        )

    def test_pruning_requires_legacy(self, cnn_model):
        if compat.IS_KERAS_3:
            with pytest.raises(RuntimeError, match="TF_USE_LEGACY_KERAS"):
                ou.ModelPruning.create_pruning_schedule()

    @requires_tfmot
    @pytest.mark.legacy_keras
    def test_prune_train_strip(self, trained_model, image_dataset):
        schedule = ou.ModelPruning.create_pruning_schedule(
            final_sparsity=0.5, end_step=6, frequency=1
        )
        pruned = ou.ModelPruning.create_pruned_model(trained_model, schedule)
        pruned = ou.ModelPruning.train_pruned_model(
            pruned, image_dataset, image_dataset, epochs=2, verbose=0
        )
        stripped = ou.ModelPruning.finalize_pruned_model(pruned)
        assert ou.ModelPruning.compute_sparsity(stripped)["overall_sparsity"] >= 0.4
        assert stripped.count_params() == trained_model.count_params()
        with pytest.raises(ValueError):
            ou.ModelPruning.create_pruning_schedule(initial_sparsity=0.9, final_sparsity=0.5)

    @requires_tfmot
    @pytest.mark.legacy_keras
    def test_compression_pipeline_and_qat(self, trained_model, image_dataset):
        model, tflite = ou.ModelCompression.apply_magnitude_pruning_and_quantization(
            trained_model, image_dataset, image_dataset, target_sparsity=0.5, epochs=2, verbose=0
        )
        analysis = ou.ModelCompression.analyze_compression_ratio(trained_model, model, tflite)
        assert analysis["compressed_sparsity"] >= 0.4 and analysis["tflite_size_mb"] > 0
        qat = ou.ModelQuantization.quantize_model_qat(
            trained_model, image_dataset, epochs=1, verbose=0
        )
        assert len(ou.ModelQuantization.convert_qat_to_tflite(qat)) > 1000


class TestDistillation:
    def test_loss_properties(self):
        y = tf.constant([0, 1])
        logits = tf.constant([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        ce = keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        np.testing.assert_allclose(
            ou.distillation_loss(y, logits, logits, alpha=0.0).numpy(), ce.numpy(), atol=1e-6
        )
        assert float(tf.reduce_max(ou.distillation_loss(y, logits, logits, alpha=1.0))) < 1e-5
        probs = tf.nn.softmax(logits)
        assert (
            float(
                tf.reduce_max(
                    ou.distillation_loss(y, logits, probs, alpha=1.0, teacher_is_probabilities=True)
                )
            )
            < 1e-4
        )

    def test_student_training(self, trained_model, image_dataset):
        kd = ou.KnowledgeDistillation(
            trained_model, alpha=0.5, temperature=2.0, teacher_outputs_probabilities=True
        )
        student = kd.create_student_model("simple_cnn", (16, 16, 3), 3)
        assert student.layers[-1].activation.__name__ == "linear"
        history = kd.train_student_model(student, image_dataset, image_dataset, epochs=2)
        assert len(history["loss"]) == 2 and 0.0 <= history["val_accuracy"][-1] <= 1.0
        with pytest.raises(ValueError):
            kd.create_student_model("resnet", (16, 16, 3), 3)
        with pytest.raises(ValueError):
            ou.KnowledgeDistillation(trained_model, alpha=2.0)


class TestMixedPrecision:
    def test_clone_and_compile(self, cnn_model, images):
        clone = ou.MixedPrecisionOptimization.create_mixed_precision_model(cnn_model)
        last = clone.layers[-1]
        policy = getattr(last, "dtype_policy", None)
        assert (policy.name if policy is not None else last.dtype) == "float32"
        np.testing.assert_allclose(
            clone.predict(images[:2], verbose=0),
            cnn_model.predict(images[:2], verbose=0),
            atol=1e-5,
        )
        ou.MixedPrecisionOptimization.compile_mixed_precision_model(clone)
        assert compat.is_loss_scale_optimizer(clone.optimizer)
        ou.MixedPrecisionOptimization.enable_mixed_precision()
        assert compat.global_policy_name() == "mixed_float16"
        ou.MixedPrecisionOptimization.disable_mixed_precision()


class TestInferenceOptimization:
    @pytest.mark.parametrize("level", ou.OPTIMIZATION_LEVELS)
    def test_levels(self, trained_model, images, level):
        optimized = ou.create_inference_optimized_model(trained_model, optimization_level=level)
        expected = trained_model.predict(images[:2], verbose=0)
        np.testing.assert_allclose(optimized.predict(images[:2], verbose=0), expected, atol=1e-5)
        if level != "conservative":
            np.testing.assert_allclose(
                optimized.predict_fn(images[:2]).numpy(), expected, atol=1e-4
            )

    def test_invalid_level_and_benchmark(self, trained_model, images):
        with pytest.raises(ValueError):
            ou.create_inference_optimized_model(trained_model, optimization_level="max")
        optimized = ou.create_inference_optimized_model(
            trained_model, optimization_level="moderate"
        )
        result = ou.benchmark_model_performance(
            trained_model, optimized, images[:2], num_runs=2, warmup=1
        )
        assert result["original_avg_time_ms"] > 0 and "speedup_ratio" in result

    def test_report_and_size(self, trained_model):
        assert ou.model_size_bytes(trained_model) == trained_model.count_params() * 4
        report = ou.create_optimization_report(
            trained_model,
            {"pruned_model": trained_model, "quantized_model_size": 0.01},
            {"speedup_ratio": 2.0, "original_avg_time_ms": 2.0, "optimized_avg_time_ms": 1.0},
        )
        assert "Pruning" in report and "Quantization" in report and "Excellent" in report
