# Installation Guide

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.12 |
| TensorFlow | 2.17 | 2.21 |
| RAM | 8 GB | 32 GB |
| VRAM (GPU) | — | 8 GB |
| Storage | 10 GB | 50 GB |

TensorFlow ≥ 2.16 ships **Keras 3** as `tf.keras`; everything in TensorVerseHub works with it. The legacy Keras 2 stack is only needed for pruning and quantisation-aware training (see below).

---

## Option 1 — Local Installation (Recommended for Learning)

```bash
# 1. Clone the repository
git clone https://github.com/SatvikPraveen/TensorVerseHub.git
cd TensorVerseHub

# 2. Create a virtual environment inside the project
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install the package (editable) with the extras you need
pip install --upgrade pip
pip install -e ".[dev,export]"   # library + tests + ONNX/TFLite tooling
# pip install -e ".[all]"        # + notebooks, serving, docs

# 4. Verify the install
tensorverse info                 # prints package / TensorFlow / Keras versions as JSON

# 5. Launch Jupyter Lab (needs the `notebooks` extra or `.[all]`)
jupyter lab
```

`requirements.txt` still exists and installs the **full notebook environment** in one go
(`pip install -r requirements.txt`) if you prefer that over extras.

### Extras

| Extra | Installs | Use it for |
|-------|----------|------------|
| `dev` | pytest, ruff, mypy, pre-commit, build, twine | Running tests, linting, contributing |
| `export` | tf2onnx, onnx, onnxruntime | ONNX export and validation |
| `serving` | fastapi, uvicorn, flask, streamlit | `tensorverse serve`, Flask and Streamlit examples |
| `notebooks` | jupyterlab, pandas, seaborn, tensorflow-datasets, tensorflow-hub, keras-tuner, … | The curriculum notebooks |
| `optimization` | tf-keras, tensorflow-model-optimization | Pruning / QAT (legacy Keras 2) |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | Building this site |
| `all` | Everything above | — |

### Legacy Keras 2 (pruning and quantisation-aware training only)

`tensorflow-model-optimization` still requires Keras 2:

```bash
pip install -e ".[optimization]"   # or: make install-legacy
export TF_USE_LEGACY_KERAS=1       # before importing TensorFlow
```

Functions that need it raise a clear `RuntimeError` under Keras 3 instead of failing obscurely.

---

## Option 2 — Docker (Recommended for Reproducibility)

All services are defined in `docker-compose.yml`.

```bash
# Start Jupyter Lab only
docker compose up -d jupyter
# → http://localhost:8888

# Start all services (Jupyter + FastAPI model server + Streamlit + TensorBoard)
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Available Services

| Service | URL | Description |
|---------|-----|-------------|
| Jupyter Lab | http://localhost:8888 | Interactive notebooks |
| API (`tensorverse serve`) | http://localhost:8000 | FastAPI model-serving endpoint |
| Streamlit | http://localhost:8501 | Model demo dashboard |
| TensorBoard | http://localhost:6006 | Training metrics |

See the [Docker deployment guide](deployment/docker.md) for build targets and volumes.

---

## Option 3 — Development Installation

For contributing or running tests:

```bash
make install             # pip install -e ".[dev,export]"
pre-commit install       # install git hooks (ruff, nbstripout, …)
make check               # ruff lint + format check + mypy
make test                # unit tests on Keras 3
make test-legacy         # same suite with TF_USE_LEGACY_KERAS=1 (enables pruning/QAT tests)
```

`pyproject.toml` is the single configuration file for packaging, ruff, mypy, pytest and coverage.

---

## GPU Support

TensorVerseHub automatically uses available GPUs. Call `tvh.configure_tensorflow(memory_growth=True)` to allocate GPU memory on demand.

```bash
# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

On Linux, `pip install "tensorflow[and-cuda]"` pulls in matching CUDA/cuDNN wheels for TensorFlow ≥ 2.16. There is no separate `tensorflow-gpu` package any more.

---

## Dependency Highlights

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.17, < 2.22 | Core framework (Keras 3) |
| `numpy` | ≥ 1.26 | Arrays |
| `tensorflow-model-optimization` | ≥ 0.8 (optional, `optimization` extra) | Pruning, QAT — needs `tf-keras` + `TF_USE_LEGACY_KERAS=1` |
| `tf2onnx` / `onnxruntime` | ≥ 1.16 / ≥ 1.17 (`export` extra) | ONNX export and validation |
| `fastapi` / `uvicorn` | ≥ 0.110 / ≥ 0.29 (`serving` extra) | `tensorverse serve` |
| `flask` | ≥ 3.0 (`serving` extra) | Flask REST API example |
| `streamlit` | ≥ 1.32 (`serving` extra) | Interactive dashboard example |

See [pyproject.toml](https://github.com/SatvikPraveen/TensorVerseHub/blob/main/pyproject.toml) for the authoritative list and [requirements.txt](https://github.com/SatvikPraveen/TensorVerseHub/blob/main/requirements.txt) for the full notebook environment.
