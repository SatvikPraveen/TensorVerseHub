import numpy as np
import pytest
import tensorflow as tf

from tensorversehub import compat
from tensorversehub import model_utils as mu
from tensorversehub import training_utils as tu
from tensorversehub.compat import keras


class TestSchedule:
    def test_warmup_cosine_values(self):
        sched = tu.WarmupCosineSchedule(1e-3, total_steps=100, warmup_steps=0.1, min_lr=1e-5)
        assert sched.warmup_steps == 10
        assert float(sched(0)) == 0.0
        assert float(sched(10)) == pytest.approx(1e-3)
        assert float(sched(100)) == pytest.approx(1e-5, rel=1e-3)
        assert float(sched(55)) < float(sched(10))
        assert float(sched(tf.Variable(5))) == pytest.approx(5e-4)

    def test_validation_and_config(self):
        with pytest.raises(ValueError):
            tu.WarmupCosineSchedule(1e-3, 0)
        with pytest.raises(ValueError):
            tu.WarmupCosineSchedule(1e-3, 10, warmup_steps=20)
        sched = tu.WarmupCosineSchedule(1e-3, 50, warmup_steps=5)
        restored = tu.WarmupCosineSchedule.from_config(sched.get_config())
        assert float(restored(25)) == float(sched(25))
        opt = keras.optimizers.SGD(sched)
        assert "learning_rate" in opt.get_config()


class TestGradientClipping:
    def test_global_norm_and_value(self):
        grads = [tf.constant([3.0, 4.0]), None]
        clipped, norm = tu.GradientClipping.clip_by_global_norm(grads, 1.0)
        assert float(norm) == pytest.approx(5.0) and clipped[1] is None
        assert float(tf.norm(clipped[0])) == pytest.approx(1.0)
        vals = tu.GradientClipping.clip_by_value([tf.constant([-2.0, 2.0]), None], -1, 1)
        assert vals[0].numpy().tolist() == [-1.0, 1.0] and vals[1] is None

    def test_percentile_and_adaptive(self):
        grads = [tf.constant([1.0, -5.0, 3.0]), tf.constant([[0.5]])]
        assert float(tu.GradientClipping.percentile_threshold(grads, 100.0)) == 5.0
        assert float(tu.GradientClipping.percentile_threshold(grads, 0.0)) == 0.5
        clipped = tu.GradientClipping.adaptive_clip(grads, 50.0)
        assert np.abs(clipped[0].numpy()).max() <= 3.0

        @tf.function
        def graph_mode(g):
            return tu.GradientClipping.adaptive_clip(g, 50.0)

        assert graph_mode(grads)[0].shape == (3,)


class TestTrackerAndEarlyStopping:
    def test_metrics_tracker(self):
        tracker = tu.MetricsTracker()
        tracker.update_train(loss=2.0)
        tracker.update_train(loss=1.0, acc=tf.constant(0.5))
        tracker.update_val(loss=0.5)
        summary = tracker.commit_epoch()
        assert summary == {"train_loss": 1.5, "train_acc": 0.5, "val_loss": 0.5}
        tracker.update_train(loss=0.1)
        tracker.commit_epoch()
        assert tracker.epochs == 2 and tracker.best("train_loss") == (1, pytest.approx(0.1))
        tracker.reset()
        assert tracker.history == {}

    def test_early_stopping(self, cnn_model):
        handler = tu.EarlyStoppingHandler(patience=2, min_delta=0.0)
        original = cnn_model.get_weights()
        assert not handler.update({"val_loss": 1.0}, cnn_model)
        cnn_model.set_weights([w * 0 for w in original])
        assert not handler.update({"val_loss": 1.5}, cnn_model)
        assert handler.update({"val_loss": 1.4}, cnn_model)  # patience exhausted → restore
        np.testing.assert_array_equal(cnn_model.get_weights()[0], original[0])
        assert handler.best == 1.0
        with pytest.raises(KeyError):
            handler.update({"loss": 1.0})
        with pytest.raises(ValueError):
            tu.EarlyStoppingHandler(mode="avg")
        maximise = tu.EarlyStoppingHandler(monitor="val_acc", mode="max", patience=1, baseline=0.9)
        assert maximise.update({"val_acc": 0.5})  # never beats baseline


class TestLearningRateFinder:
    def test_find_restores_state(self, cnn_model, image_dataset):
        before = cnn_model.get_weights()
        finder = tu.LearningRateFinder(cnn_model, min_lr=1e-5, max_lr=1e-1, num_steps=5)
        lrs, losses = finder.find(image_dataset.repeat())
        assert len(lrs) == len(losses) and 1 <= len(lrs) <= 5
        assert lrs == sorted(lrs)
        np.testing.assert_array_equal(cnn_model.get_weights()[0], before[0])
        assert tu.get_learning_rate(cnn_model.optimizer) == pytest.approx(1e-3)
        assert finder.suggest(skip_start=0, skip_end=0) in lrs or finder.suggest() is None
        assert finder.plot(skip_start=0, skip_end=0, show=False) is not None

    def test_requires_compiled_model(self):
        model = mu.ModelBuilders.create_mlp(4, 2)
        with pytest.raises(ValueError):
            tu.LearningRateFinder(model)


class EpochCounter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = 0
        self.logs = None

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1
        self.logs = logs


class TestCustomTrainingLoop:
    def _loop(self, **kwargs):
        model = mu.ModelBuilders.create_cnn_classifier((16, 16, 3), 3, "simple")
        return tu.CustomTrainingLoop(
            model,
            keras.optimizers.Adam(1e-3),
            keras.losses.SparseCategoricalCrossentropy(),
            train_metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
            val_metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
            **kwargs,
        )

    def test_fit_with_callbacks_and_early_stopping(self, image_dataset):
        loop = self._loop(clip_norm=1.0)
        counter = EpochCounter()
        tracker = loop.fit(
            image_dataset,
            image_dataset,
            epochs=3,
            callbacks=[counter],
            early_stopping=tu.EarlyStoppingHandler(patience=1),
            verbose=0,
        )
        assert {"train_loss", "train_grad_norm", "train_acc", "val_loss", "val_acc"} <= set(
            tracker.history
        )
        assert 1 <= counter.epochs <= 3 and "lr" in counter.logs
        assert all(v > 0 for v in tracker.history["train_grad_norm"])  # pre-clip norm
        results = loop.evaluate(image_dataset)
        assert set(results) == {"loss", "acc"}

    def test_gradient_accumulation_matches_shapes(self, image_dataset):
        loop = self._loop(accumulation_steps=2)
        tracker = loop.fit(image_dataset, epochs=1, verbose=0, steps_per_epoch=3)
        assert len(tracker.history["train_loss"]) == 1
        assert loop._accumulators is not None
        assert all(float(tf.reduce_max(tf.abs(a))) == 0.0 for a in loop._accumulators)  # flushed

    def test_mixed_precision_loop(self, image_dataset):
        loop = self._loop(mixed_precision=True, clip_norm=1.0)
        assert compat.is_loss_scale_optimizer(loop.optimizer)
        tracker = loop.fit(image_dataset, image_dataset, epochs=1, verbose=0)
        assert np.isfinite(tracker.history["train_loss"][0])
        assert tracker.history["train_grad_norm"][0] < 1e3  # reported unscaled

    def test_invalid_accumulation(self):
        with pytest.raises(ValueError):
            self._loop(accumulation_steps=0)

    def test_set_learning_rate(self):
        opt = keras.optimizers.Adam(1e-3)
        tu.set_learning_rate(opt, 5e-4)
        assert tu.get_learning_rate(opt) == pytest.approx(5e-4)
