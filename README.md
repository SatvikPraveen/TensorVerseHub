# TensorVerseHub

[![CI](https://img.shields.io/github/actions/workflow/status/SatvikPraveen/TensorVerseHub/ci.yml?branch=main&label=CI&logo=github-actions)](https://github.com/SatvikPraveen/TensorVerseHub/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/SatvikPraveen/TensorVerseHub/docs.yml?branch=main&label=docs&logo=materialformkdocs)](https://satvikpraveen.github.io/TensorVerseHub)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17%20→%202.21-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3%20(%2B%20legacy%202)-red?logo=keras)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
![License](https://img.shields.io/badge/License-MIT-green)

A TensorFlow learning hub that is also a real, tested Python package: **27 curriculum
notebooks** from tensors to diffusion models, plus `tensorversehub`, a typed utility
library and CLI for data pipelines, custom training loops, model compression and
multi-format export. Every utility runs on **Keras 3** *and* legacy **Keras 2**
(`TF_USE_LEGACY_KERAS=1`), and is exercised by the test-suite on both.

## What's inside

| Module | Highlights |
|---|---|
| `tensorversehub.data_utils` | Type-dispatched TFRecord writer/reader (images, text, raw arrays, sharded), `tf.data` pipelines, dependency-free random rotation, vectorised **MixUp / CutMix / Random Erasing** |
| `tensorversehub.model_utils` | Serialisable **multi-head attention**, positional encoding and pre-LN Transformer block; CNN (simple/VGG/ResNet), text (LSTM/GRU/Transformer), MLP, autoencoder and GAN builders; profiler-backed **FLOP counting**; transfer learning over 10 `keras.applications` backbones |
| `tensorversehub.training_utils` | `GradientTape` engine with **mixed precision, gradient accumulation, global-norm clipping, XLA** and Keras callbacks; warm-up + cosine schedule; LR range finder; graph-mode percentile clipping |
| `tensorversehub.optimization_utils` | Post-training quantisation (dynamic / float16 / **full int8**), QAT and magnitude **pruning** (tfmot), temperature-scaled **knowledge distillation**, mixed-precision helpers, XLA inference clones, TensorRT |
| `tensorversehub.export_utils` | SavedModel with metadata, **TFLite** (with LiteRT fallback, int8 I/O handling and numerical validation), **ONNX** via tf2onnx (+ ONNX Runtime validation), TF.js, Core ML, zipped deployment packages |
| `tensorversehub.visualization` | Headless-safe training curves, confusion matrices, ROC, feature maps, filters, gradient-flow and a training dashboard — every function returns a `Figure` |
| `tensorversehub.compat` | The Keras 2 / Keras 3 bridge: save/load/export, loss scaling, output shapes, signatures |
| `tensorverse` CLI | `train`, `evaluate`, `convert`, `serve` (FastAPI) and `info`; runs end-to-end on synthetic data when no dataset is present |

## Quick start

```bash
git clone https://github.com/SatvikPraveen/TensorVerseHub.git
cd TensorVerseHub
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,export]"          # library + tests + ONNX/TFLite tooling
# pip install -e ".[all]"               # + notebooks, serving, docs
```

```python
import tensorversehub as tvh
from tensorversehub.compat import keras

tvh.configure_tensorflow(seed=42)                      # explicit, no import side effects

model = tvh.ModelBuilders.create_cnn_classifier((32, 32, 3), 10, "resnet")
loop = tvh.CustomTrainingLoop(
    model, keras.optimizers.AdamW(1e-3), keras.losses.SparseCategoricalCrossentropy(),
    train_metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    clip_norm=1.0, mixed_precision=True, accumulation_steps=2,
)
history = loop.fit(train_ds, val_ds, epochs=10, early_stopping=tvh.EarlyStoppingHandler(patience=3))

tvh.export_utils.quick_export(model, "exports/", formats=["keras", "savedmodel", "tflite", "onnx"])
```

```bash
tensorverse train --architecture resnet --epochs 10 --data ./data/images --export-savedmodel
tensorverse evaluate --model models/final_model.keras --report --confusion-matrix
tensorverse convert --model models/final_model.keras --to tflite --quantize int8 --benchmark
tensorverse serve   --model models/final_model.keras --port 8000
```

### Keras 3 or legacy Keras 2?

TensorFlow ≥ 2.16 ships Keras 3 as `tf.keras`; everything in this repository works with it.
`tensorflow-model-optimization` (pruning, quantisation-aware training) still requires Keras 2:

```bash
pip install -e ".[optimization]"        # tf-keras + tensorflow-model-optimization
export TF_USE_LEGACY_KERAS=1            # before importing TensorFlow
```

Functions that need it raise a clear `RuntimeError` under Keras 3 instead of failing obscurely.

## Curriculum

| Track | Notebooks | Topics |
|---|---|---|
| [Foundations](docs/notebooks/01_foundations.md) | 01–03 | Tensors, eager/graph execution, `tf.data`, TFRecords, profiling |
| [Keras](docs/notebooks/02_keras.md) | 04–06 | Sequential/functional APIs, custom layers, callbacks |
| [Computer Vision](docs/notebooks/03_vision.md) | 07–09 | CNNs, transfer learning, segmentation |
| [NLP](docs/notebooks/04_nlp.md) | 10–12 | Text layers, attention & Transformers, TF Hub |
| [Generative](docs/notebooks/05_generative.md) | 13–15 | GANs, VAEs, diffusion |
| [Optimization & Export](docs/notebooks/06_optimization.md) | 16–18 | Quantisation, pruning, TFLite / ONNX / TF.js |
| [Advanced](docs/notebooks/07_advanced.md) | 19–20 | Distribution strategies, research re-implementations |
| [Reinforcement Learning](docs/notebooks/08_rl.md) | 23 | DQN, policy gradients, actor–critic |
| [Capstones](docs/notebooks/capstone.md) | 21–22 | Multimodal system, end-to-end MLOps pipeline |
| [Supplementary](docs/notebooks/supplementary.md) | — | Federated learning, meta-learning, time series, Keras Tuner |

Start with `jupyter lab notebooks/01_tensorflow_foundations/`. The full overview lives in
[docs/curriculum.md](docs/curriculum.md).

## Repository layout

```
TensorVerseHub/
├── tensorversehub/          # the package (typed, lazy-import, side-effect free)
│   ├── compat.py            # Keras 2/3 bridge
│   ├── data_utils.py        # TFRecords, tf.data, augmentation
│   ├── model_utils.py       # layers, builders, analysis
│   ├── training_utils.py    # custom training engine, schedules, clipping
│   ├── optimization_utils.py# quantisation, pruning, distillation
│   ├── export_utils.py      # SavedModel / TFLite / ONNX / TF.js / Core ML
│   ├── visualization.py     # plots and dashboards
│   └── cli/                 # `tensorverse` sub-commands
├── notebooks/               # 27 curriculum notebooks
├── examples/                # optimisation demos and serving apps (Flask, Streamlit, TFLite)
├── tests/                   # pytest suite, runs on Keras 3 and legacy Keras
├── docs/                    # MkDocs site (API reference is generated from docstrings)
├── .github/workflows/       # CI matrix, docs deploy, PyPI/GHCR publish
├── Dockerfile · docker-compose.yml · Makefile · pyproject.toml
```

## Development

```bash
make install        # editable install with dev + export extras
make lint           # ruff check + ruff format --check
make typecheck      # mypy
make test           # pytest (Keras 3)
make test-legacy    # pytest with TF_USE_LEGACY_KERAS=1 (enables pruning/QAT tests)
make docs-serve     # live documentation preview
```

CI runs ruff, mypy, the test-suite on Python 3.10/3.11/3.12 against TensorFlow 2.17, 2.18
and 2.21 (Keras 3 and legacy Keras), validates all notebooks, builds the docs in strict
mode, builds the wheel and the Docker image. Tags matching `v*.*.*` publish to PyPI
(trusted publishing) and GHCR.

### Docker

```bash
docker compose up -d jupyter   # JupyterLab on http://localhost:8888
docker compose up -d           # + FastAPI model server (:8000), Streamlit (:8501), TensorBoard (:6006)
```

## Documentation

The site at **https://satvikpraveen.github.io/TensorVerseHub** contains the curriculum
guide, generated API reference, deployment guides, model-optimisation guide, best
practices and troubleshooting. See [docs/CHANGELOG.md](docs/CHANGELOG.md) for release
notes and [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## License

MIT — see [LICENSE](LICENSE).
