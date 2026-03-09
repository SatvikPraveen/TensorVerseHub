"""
Hyperparameter tuning demonstration using Keras Tuner.
Covers RandomSearch, Hyperband, and BayesianOptimization strategies.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from typing import Tuple, Dict, Any, Optional

# Optional integration with TensorVerseHub utilities
try:
    from src.visualization import setup_plotting_style
    _HAS_TVH = True
except ImportError:
    _HAS_TVH = False

try:
    import keras_tuner as kt
    _HAS_KERAS_TUNER = True
except ImportError:
    _HAS_KERAS_TUNER = False
    print("Warning: keras-tuner not found. Install with: pip install keras-tuner>=1.3.5,<1.4.0")


class HyperparameterTuningDemo:
    """Demonstrates automated hyperparameter search with Keras Tuner."""

    def __init__(self, output_dir: str = "tuning_results", max_trials: int = 10):
        self.output_dir = output_dir
        self.max_trials = max_trials
        self.results: Dict[str, Any] = {}

        os.makedirs(self.output_dir, exist_ok=True)

        if _HAS_TVH:
            try:
                setup_plotting_style()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess CIFAR-10."""
        print("📦 Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        # Use a smaller subset to keep demo fast
        x_train, y_train = x_train[:5000], y_train[:5000]
        x_test, y_test = x_test[:1000], y_test[:1000]

        print(f"  Train: {x_train.shape}  Test: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    # ------------------------------------------------------------------
    # Model builders (hypermodel functions)
    # ------------------------------------------------------------------

    @staticmethod
    def build_mlp(hp: "kt.HyperParameters") -> tf.keras.Model:
        """Build a tunable MLP for CIFAR-10 classification."""
        n_units = hp.Int("units", min_value=64, max_value=512, step=64)
        n_layers = hp.Int("num_layers", min_value=1, max_value=3)
        dropout_rate = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])

        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Flatten()(inputs)

        for _ in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def build_cnn(hp: "kt.HyperParameters") -> tf.keras.Model:
        """Build a tunable CNN for CIFAR-10 classification."""
        n_filters = hp.Int("filters", min_value=16, max_value=64, step=16)
        n_conv_layers = hp.Int("conv_layers", min_value=1, max_value=3)
        use_bn = hp.Boolean("batch_norm")
        dense_units = hp.Int("dense_units", min_value=64, max_value=256, step=64)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = inputs

        filters = n_filters
        for i in range(n_conv_layers):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            if use_bn:
                x = tf.keras.layers.BatchNormalization()(x)
            if i < n_conv_layers - 1:
                x = tf.keras.layers.MaxPooling2D()(x)
            filters = min(filters * 2, 128)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    def run_random_search(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """RandomSearch — uniform sampling over the search space."""
        if not _HAS_KERAS_TUNER:
            print("⚠️ Skipping RandomSearch (keras-tuner not installed)")
            return {}

        print("\n🎲 RandomSearch Hyperparameter Tuning")
        print("=" * 50)
        print("Strategy: Random uniform sampling — simple, unbiased, parallelisable.")

        tuner = kt.RandomSearch(
            hypermodel=self.build_mlp,
            objective="val_accuracy",
            max_trials=self.max_trials,
            seed=42,
            project_name="random_search",
            directory=self.output_dir,
            overwrite=True,
        )

        tuner.search_space_summary()

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)]
        tuner.search(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_val_acc = tuner.oracle.get_best_trials(1)[0].score

        result = {
            "strategy": "RandomSearch",
            "best_val_accuracy": float(best_val_acc),
            "best_hyperparameters": best_hps.values,
        }
        self.results["random_search"] = result

        print(f"✅ Best val accuracy: {best_val_acc:.4f}")
        print(f"   Best HPs: {best_hps.values}")

        return result

    def run_hyperband(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Hyperband — successive halving for efficient search."""
        if not _HAS_KERAS_TUNER:
            print("⚠️ Skipping Hyperband (keras-tuner not installed)")
            return {}

        print("\n⚡ Hyperband Hyperparameter Tuning")
        print("=" * 50)
        print("Strategy: Successive halving with random configurations — much faster than RandomSearch.")

        tuner = kt.Hyperband(
            hypermodel=self.build_cnn,
            objective="val_accuracy",
            max_epochs=20,
            factor=3,
            seed=42,
            project_name="hyperband",
            directory=self.output_dir,
            overwrite=True,
        )

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)]
        tuner.search(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_val_acc = tuner.oracle.get_best_trials(1)[0].score

        result = {
            "strategy": "Hyperband",
            "best_val_accuracy": float(best_val_acc),
            "best_hyperparameters": best_hps.values,
        }
        self.results["hyperband"] = result

        print(f"✅ Best val accuracy: {best_val_acc:.4f}")
        print(f"   Best HPs: {best_hps.values}")

        return result

    def run_bayesian_optimization(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """BayesianOptimization — GP surrogate guides search toward promising regions."""
        if not _HAS_KERAS_TUNER:
            print("⚠️ Skipping BayesianOptimization (keras-tuner not installed)")
            return {}

        print("\n🧠 BayesianOptimization Hyperparameter Tuning")
        print("=" * 50)
        print("Strategy: Gaussian process surrogate model — learns from previous trials.")

        tuner = kt.BayesianOptimization(
            hypermodel=self.build_mlp,
            objective="val_accuracy",
            max_trials=self.max_trials,
            seed=42,
            project_name="bayesian",
            directory=self.output_dir,
            overwrite=True,
        )

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)]
        tuner.search(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_val_acc = tuner.oracle.get_best_trials(1)[0].score

        result = {
            "strategy": "BayesianOptimization",
            "best_val_accuracy": float(best_val_acc),
            "best_hyperparameters": best_hps.values,
        }
        self.results["bayesian"] = result

        print(f"✅ Best val accuracy: {best_val_acc:.4f}")
        print(f"   Best HPs: {best_hps.values}")

        return result

    # ------------------------------------------------------------------
    # Retrain best model
    # ------------------------------------------------------------------

    def retrain_best_model(
        self,
        strategy: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 30,
    ) -> tf.keras.Model:
        """Retrain the best config from the chosen strategy on full train data."""
        if not _HAS_KERAS_TUNER or strategy not in self.results:
            print(f"⚠️ No results found for strategy '{strategy}'")
            return None

        print(f"\n🏆 Retraining best model from {strategy}")
        print("=" * 50)

        best_hps_values = self.results[strategy]["best_hyperparameters"]
        project_map = {
            "random_search": ("random_search", self.build_mlp),
            "hyperband": ("hyperband", self.build_cnn),
            "bayesian": ("bayesian", self.build_mlp),
        }

        if strategy not in project_map:
            print(f"⚠️ Unknown strategy: {strategy}")
            return None

        project_name, builder = project_map[strategy]

        tuner_cls = {
            "random_search": kt.RandomSearch,
            "hyperband": kt.Hyperband,
            "bayesian": kt.BayesianOptimization,
        }[strategy]

        kwargs: Dict[str, Any] = dict(
            hypermodel=builder,
            objective="val_accuracy",
            seed=42,
            project_name=project_name,
            directory=self.output_dir,
            overwrite=False,
        )
        if strategy == "hyperband":
            kwargs["max_epochs"] = 20
            kwargs["factor"] = 3
        else:
            kwargs["max_trials"] = self.max_trials

        tuner = tuner_cls(**kwargs)  # type: ignore[operator]
        best_hps = tuner.get_best_hyperparameters(1)[0]

        model = tuner.hypermodel.build(best_hps)

        val_split = int(0.8 * len(x_train))
        x_tr, y_tr = x_train[:val_split], y_train[:val_split]
        x_v, y_v = x_train[val_split:], y_train[val_split:]

        history = model.fit(
            x_tr, y_tr,
            validation_data=(x_v, y_v),
            epochs=epochs,
            batch_size=64,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            ],
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n🎯 Final test accuracy: {test_acc:.4f}  |  test loss: {test_loss:.4f}")

        self.results["final"] = {
            "strategy": strategy,
            "best_hyperparameters": best_hps_values,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
        }

        return model

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Bar chart comparing best validation accuracy across strategies."""
        strategies = {k: v for k, v in self.results.items() if k != "final"}
        if not strategies:
            print("No results to plot.")
            return

        names = [v["strategy"] for v in strategies.values()]
        scores = [v["best_val_accuracy"] for v in strategies.values()]
        colors = ["#2196F3", "#FF9800", "#4CAF50"][: len(names)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, scores, color=colors, edgecolor="black", linewidth=0.8)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{score:.4f}",
                ha="center", va="bottom", fontsize=10,
            )

        ax.set_xlabel("Tuning Strategy", fontsize=12)
        ax.set_ylabel("Best Validation Accuracy", fontsize=12)
        ax.set_title("Keras Tuner Strategy Comparison — CIFAR-10", fontsize=13)
        ax.set_ylim(0, min(1.0, max(scores) + 0.05))
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 Comparison chart saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, path: Optional[str] = None) -> None:
        """Persist tuning results to JSON."""
        path = path or os.path.join(self.output_dir, "tuning_results.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keras Tuner hyperparameter search demo on CIFAR-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["random_search", "hyperband", "bayesian", "all"],
        default=["all"],
        help="Which search strategies to run (default: all)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of trials for RandomSearch / BayesianOptimization (default: 10)",
    )
    parser.add_argument(
        "--retrain-epochs",
        type=int,
        default=30,
        metavar="E",
        help="Epochs for final retrain of best model (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        default="tuning_results",
        metavar="DIR",
        help="Directory to store Keras Tuner logs and results (default: tuning_results)",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save comparison chart to output_dir instead of displaying it",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip final best-model retrain step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not _HAS_KERAS_TUNER:
        print("ERROR: keras-tuner is required. Install it with:\n  pip install 'keras-tuner>=1.3.5,<1.4.0'")
        return

    strategies = args.strategies
    if "all" in strategies:
        strategies = ["random_search", "hyperband", "bayesian"]

    demo = HyperparameterTuningDemo(
        output_dir=args.output_dir,
        max_trials=args.max_trials,
    )

    (x_train, y_train), (x_test, y_test) = demo.load_dataset()

    # 80/20 train/validation split
    val_cut = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:val_cut], y_train[:val_cut]
    x_v, y_v = x_train[val_cut:], y_train[val_cut:]

    if "random_search" in strategies:
        demo.run_random_search(x_tr, y_tr, x_v, y_v)

    if "hyperband" in strategies:
        demo.run_hyperband(x_tr, y_tr, x_v, y_v)

    if "bayesian" in strategies:
        demo.run_bayesian_optimization(x_tr, y_tr, x_v, y_v)

    # Compare strategies
    if len(demo.results) > 1:
        save_path = (
            os.path.join(args.output_dir, "strategy_comparison.png")
            if args.save_plot else None
        )
        demo.plot_comparison(save_path=save_path)

    # Retrain best overall model
    if not args.no_retrain and demo.results:
        best_strategy = max(
            demo.results,
            key=lambda k: demo.results[k].get("best_val_accuracy", -1),
        )
        demo.retrain_best_model(
            strategy=best_strategy,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=args.retrain_epochs,
        )

    demo.save_results()

    print("\n✅ Hyperparameter tuning demo complete.")
    print(f"   Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()
