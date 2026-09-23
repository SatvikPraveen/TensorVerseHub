# TensorVerseHub

> A comprehensive, production-grade TensorFlow learning hub — 27 notebooks, a typed utility package (`tensorversehub`), a CLI and real-world deployment examples.

---

## What Is TensorVerseHub?

TensorVerseHub is a **focused, zero-fluff reference** for TensorFlow practitioners at every level. It combines:

- **27 Jupyter notebooks** — a progressive curriculum from tensor basics to diffusion models and RL
- **The `tensorversehub` package** — typed, side-effect-free data, model, training, optimization, export and visualization utilities that run on Keras 3 *and* legacy Keras 2
- **Serving examples** — FastAPI model server (`tensorverse serve`), Flask REST API, Streamlit dashboard and TFLite inference
- **CLI** — one `tensorverse` command with `train`, `evaluate`, `convert`, `serve` and `info` sub-commands
- **Docker support** — multi-stage Dockerfile + Docker Compose for Jupyter, the API, Streamlit and TensorBoard

---

## Requirements

| Requirement | Supported |
|-------------|-----------|
| Python | 3.10, 3.11, 3.12 |
| TensorFlow | 2.16 – 2.21 (Keras 3 by default) |
| Legacy Keras 2 | Optional — `pip install -e ".[optimization]"` + `export TF_USE_LEGACY_KERAS=1`; only needed for pruning and quantisation-aware training |

---

## Curriculum At a Glance

| Track | Notebooks | Topics |
|-------|-----------|--------|
| Foundation | 01 – 06 | Tensors, tf.data, Keras APIs, debugging |
| Computer Vision | 07 – 09 | CNNs, transfer learning, segmentation |
| NLP | 10 – 12 | Text layers, Transformers, TF Hub |
| Generative Models | 13 – 15 | GANs, VAEs, Diffusion models |
| Optimization | 16 – 18 | Quantization, pruning, ONNX/TFLite/CoreML |
| Advanced | 19 – 20 | Distributed training, research implementations |
| RL | 23 | DQN, Policy Gradients, Actor-Critic |
| Capstone | 21 – 22 | Multimodal AI, end-to-end MLOps pipeline |
| Supplementary | — | Federated learning, meta-learning, time series |

---

## Quick Start

```bash
git clone https://github.com/SatvikPraveen/TensorVerseHub.git
cd TensorVerseHub
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,export]"      # library + tests + ONNX/TFLite tooling
# pip install -e ".[all]"           # + notebooks, serving, docs
jupyter lab
```

```python
import tensorversehub as tvh
from tensorversehub.compat import keras

tvh.configure_tensorflow(seed=42)   # explicit — importing the package has no side effects

model = tvh.ModelBuilders.create_cnn_classifier((32, 32, 3), 10, "resnet")
tvh.export_utils.quick_export(model, "exports/", formats=["keras", "savedmodel", "tflite"])
```

```bash
tensorverse train --architecture resnet --epochs 10 --data ./data/images
tensorverse serve --model models/final_model.keras --port 8000
```

Or with Docker:

```bash
docker compose up -d jupyter
# → http://localhost:8888
```

---

## Project Structure

```
TensorVerseHub/
├── notebooks/          # 27 learning notebooks
├── tensorversehub/     # The package: compat, data_utils, model_utils, training_utils,
│   └── cli/            #   optimization_utils, export_utils, visualization + `tensorverse` CLI
├── examples/           # Serving & optimization examples
├── tests/              # pytest suite (runs on Keras 3 and legacy Keras 2)
├── docs/               # This documentation
└── pyproject.toml      # Packaging, ruff, mypy and pytest configuration
```

See [Installation](installation.md) for full setup details and the [API Reference](api/compat.md) for the package.
