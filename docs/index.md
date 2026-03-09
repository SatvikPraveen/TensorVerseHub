# TensorVerseHub

> A comprehensive, production-grade TensorFlow learning hub — 27 notebooks, 3,700+ lines of utilities, and real-world deployment examples.

---

## What Is TensorVerseHub?

TensorVerseHub is a **focused, zero-fluff reference** for TensorFlow practitioners at every level. It combines:

- **27 Jupyter notebooks** — a progressive curriculum from tensor basics to diffusion models and RL
- **Production-grade `src/` utilities** — 3,700+ lines of ready-to-use data, model, optimization, export, and visualization helpers
- **Serving examples** — Flask REST API, Streamlit dashboard, FastAPI, and TFLite inference
- **CLI tools** — `tensorverse-train`, `tensorverse-evaluate`, `tensorverse-convert`
- **Docker support** — multi-stage Dockerfile + docker-compose for all services

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
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Or with Docker:

```bash
docker compose up jupyter
# → http://localhost:8888
```

---

## Project Structure

```
TensorVerseHub/
├── notebooks/          # 27 learning notebooks
├── src/                # Production utilities
├── examples/           # Serving & optimization examples
├── scripts/            # CLI tools
├── tests/              # Comprehensive test suite
└── docs/               # This documentation
```

See [Installation](installation.md) for full setup details.
