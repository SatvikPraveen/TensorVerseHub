# Installation Guide

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.10 |
| RAM | 8 GB | 32 GB |
| VRAM (GPU) | — | 8 GB |
| Storage | 10 GB | 50 GB |

---

## Option 1 — Local Installation (Recommended for Learning)

```bash
# 1. Clone the repository
git clone https://github.com/SatvikPraveen/TensorVerseHub.git
cd TensorVerseHub

# 2. Create a virtual environment inside the project
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# 5. Launch Jupyter Lab
jupyter lab
```

---

## Option 2 — Docker (Recommended for Reproducibility)

All services are defined in `docker-compose.yml`.

```bash
# Start Jupyter Lab only
docker compose up jupyter
# → http://localhost:8888

# Start all services (Jupyter + Flask + Streamlit + TensorBoard)
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
| Flask API | http://localhost:5000 | REST model-serving API |
| Streamlit | http://localhost:8501 | Model demo dashboard |
| TensorBoard | http://localhost:6006 | Training metrics |

---

## Option 3 — Development Installation

For contributing or running tests:

```bash
pip install -e ".[dev]"
pre-commit install       # install git hooks
make test                # run unit tests
```

---

## GPU Support

TensorVerseHub automatically uses available GPUs.

```bash
# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For CUDA setup, install the matching `tensorflow-gpu` package and ensure CUDA ≥ 11.8 + cuDNN ≥ 8.6 are installed.

---

## Dependency Highlights

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.13.x | Core framework |
| `tensorflow-model-optimization` | 0.7.x | Pruning, quantization |
| `tensorflow-hub` | 0.15.x | Pre-trained model hub |
| `tf2onnx` | 1.15.x | ONNX export |
| `tensorflowjs` | 4.10.x | Browser deployment |
| `flask` | 3.x | REST API serving |
| `streamlit` | 1.28.x | Interactive dashboards |

See [requirements.txt](https://github.com/SatvikPraveen/TensorVerseHub/blob/main/requirements.txt) for the full list with pinned versions.
