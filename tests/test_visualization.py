import numpy as np
import pytest
import tensorflow as tf
from matplotlib.figure import Figure

from tensorversehub import visualization as vz
from tensorversehub.compat import keras
from tensorversehub.data_utils import DataAugmentation

HISTORY = {
    "loss": [1.0, 0.5, 0.3],
    "val_loss": [1.1, 0.6, 0.5],
    "accuracy": [0.3, 0.6, 0.8],
    "lr": [1e-3] * 3,
}


class TestTraining:
    def test_history_dict_and_object(self, tmp_path):
        fig = vz.TrainingVisualization.plot_training_history(
            HISTORY, save_path=tmp_path / "h.png", show=False
        )
        assert isinstance(fig, Figure) and (tmp_path / "h.png").exists()
        assert len(fig.axes) == 2  # loss + accuracy, lr excluded
        history = keras.callbacks.History()
        history.history = HISTORY
        assert isinstance(
            vz.TrainingVisualization.plot_training_history(history, metrics=["loss"], show=False),
            Figure,
        )
        with pytest.raises(ValueError):
            vz.TrainingVisualization.plot_training_history(HISTORY, metrics=["f1"], show=False)

    def test_curves_matrix_roc(self, tmp_path):
        sizes = np.array([10, 20, 30])
        fig = vz.TrainingVisualization.plot_learning_curves(
            sizes, np.random.rand(3, 2), np.random.rand(3, 2), show=False
        )
        assert isinstance(fig, Figure)
        y_true = np.array([0, 1, 2, 2, 1, 0])
        fig = vz.TrainingVisualization.plot_confusion_matrix(
            y_true,
            y_true[::-1],
            class_names=list("abc"),
            normalize=True,
            save_path=tmp_path / "cm.png",
            show=False,
        )
        assert (tmp_path / "cm.png").exists()
        probs = np.random.dirichlet(np.ones(3), size=6)
        assert isinstance(
            vz.TrainingVisualization.plot_roc_curves(y_true, probs, show=False), Figure
        )
        assert isinstance(
            vz.TrainingVisualization.plot_roc_curves(
                np.eye(3)[y_true], probs, class_names=list("abc"), show=False
            ),
            Figure,
        )


class TestData:
    def test_samples_distribution_augmentation(self, images, labels):
        assert isinstance(
            vz.DataVisualization.plot_image_samples(
                images, labels, class_names=list("abc"), num_samples=5, show=False
            ),
            Figure,
        )
        gray = images[..., :1]
        assert isinstance(
            vz.DataVisualization.plot_image_samples(gray, num_samples=4, show=False), Figure
        )
        assert isinstance(
            vz.DataVisualization.plot_class_distribution(
                labels, class_names=list("abc"), show=False
            ),
            Figure,
        )
        layer = DataAugmentation.create_augmentation_layer()
        assert isinstance(
            vz.DataVisualization.plot_data_augmentation_samples(
                images[0], layer, num_samples=3, show=False
            ),
            Figure,
        )


class TestModel:
    def test_layer_outputs_and_filters(self, trained_model, images, tmp_path):
        figs = vz.ModelVisualization.visualize_layer_outputs(
            trained_model, images[0], save_dir=tmp_path, show=False
        )
        assert len(figs) == 3 and any(p.suffix == ".png" for p in tmp_path.iterdir())
        conv = next(l for l in trained_model.layers if isinstance(l, keras.layers.Conv2D))
        assert isinstance(
            vz.ModelVisualization.plot_filter_weights(trained_model, conv.name, show=False), Figure
        )
        with pytest.raises(ValueError):
            vz.ModelVisualization.plot_filter_weights(
                trained_model, trained_model.layers[-1].name, show=False
            )

    def test_plot_model_architecture(self, trained_model, tmp_path):
        result = vz.ModelVisualization.plot_model_architecture(trained_model, tmp_path / "arch.png")
        assert result is None or (tmp_path / "arch.png").exists()


class TestAdvanced:
    def test_gradient_flow_and_dashboard(
        self, trained_model, images, labels, image_dataset, tmp_path
    ):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(labels, trained_model(images))
            )
        grads = tape.gradient(loss, trained_model.trainable_variables)
        fig = vz.AdvancedVisualization.plot_gradient_flow(
            grads, trained_model.trainable_variables, show=False
        )
        assert isinstance(fig, Figure)
        fig = vz.AdvancedVisualization.create_training_dashboard(
            HISTORY, trained_model, image_dataset, save_path=tmp_path / "dash.png", show=False
        )
        assert (tmp_path / "dash.png").exists()

    def test_quick_analysis(self, trained_model, image_dataset, tmp_path):
        outputs = vz.quick_model_analysis(
            trained_model, HISTORY, image_dataset, save_dir=tmp_path, show=False
        )
        assert {"history", "confusion_matrix"} <= set(outputs)
        assert (tmp_path / "training_history.png").exists() and (
            tmp_path / "confusion_matrix.png"
        ).exists()


def test_style_helpers():
    vz.setup_plotting_style(palette="deep")
    vz.use_headless_backend()
