# Docker Deployment

TensorVerseHub ships with a multi-stage `Dockerfile` and a `docker-compose.yml` that orchestrates four services.

---

## Quick Start

```bash
# Build and start all services
docker compose up -d

# Start only Jupyter Lab
docker compose up -d jupyter

# View status
docker compose ps

# Tear down
docker compose down
```

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| `jupyter` | 8888 | Jupyter Lab — all notebooks |
| `flask-api` | 5000 | Flask REST API for model serving |
| `streamlit` | 8501 | Interactive model dashboard |
| `tensorboard` | 6006 | Real-time training metrics |

---

## Building the Docker Image

```bash
# Build the image locally (jupyter target)
docker build --target jupyter -t tensorversehub:latest .

# Run interactively
docker run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks tensorversehub:latest
```

---

## Production Deployment

For production, use the `flask-api` or `fastapi` target with environment-specific secrets:

```bash
docker build --target flask-api -t tensorversehub-api:latest .
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -v /path/to/models:/app/models \
  tensorversehub-api:latest
```

---

## Volumes

Three named volumes are created automatically:

| Volume | Purpose |
|--------|---------|
| `models_cache` | Trained models persisted across restarts |
| `data_cache` | Datasets and preprocessed data |
| `tensorboard_logs` | TensorBoard event files |

---

## Troubleshooting

**Container won't start?**  
```bash
docker compose logs jupyter
```

**TensorFlow can't see GPU inside Docker?**  
Use the `tensorflow/tensorflow:2.13.0-gpu` base image and add `--gpus all` to `docker run`.
