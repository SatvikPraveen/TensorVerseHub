# Docker Deployment

TensorVerseHub ships with a multi-stage `Dockerfile` and a `docker-compose.yml` that orchestrates four services.

---

## Quick Start

```bash
# Build and start all services
docker compose up -d            # or: make docker-up

# Start only Jupyter Lab
docker compose up -d jupyter

# View status / logs
docker compose ps
docker compose logs -f          # or: make docker-logs

# Tear down
docker compose down             # or: make docker-down
```

---

## Services

| Service | Port | Image target | Description |
|---------|------|--------------|-------------|
| `jupyter` | 8888 | `jupyter` | Jupyter Lab — all notebooks |
| `api` | 8000 | `api` | FastAPI model server (`tensorverse serve`) |
| `streamlit` | 8501 | `jupyter` | Interactive model dashboard |
| `tensorboard` | 6006 | `runtime` | Real-time training metrics |

The [Flask example](flask.md) is not a Compose service; run it manually with
`python examples/serving_examples/flask_tensorflow_api.py`.

---

## Image Targets

| Target | Contents |
|--------|----------|
| `runtime` | The `tensorversehub` package with the `export` and `serving` extras (base for everything else) |
| `api` | `runtime` + `tensorverse serve` as the entrypoint; model path from `TVH_MODEL` |
| `jupyter` (default) | `runtime` + the `notebooks` extra and JupyterLab |

```bash
# Build the default (jupyter) image locally
docker build --target jupyter -t tensorversehub:latest .   # or: make docker-build

# Run interactively
docker run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks tensorversehub:latest
```

---

## Production Deployment

For production, use the `api` target and mount your trained `.keras` model:

```bash
docker build --target api -t tensorversehub-api:latest .
docker run -p 8000:8000 \
  -e TVH_MODEL=/app/models/final_model.keras \
  -v /path/to/models:/app/models \
  tensorversehub-api:latest

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.1, 0.2, 0.3]]}'
```

The `api` image exposes `/health`, `/metadata` and `/predict` and has a built-in Docker health check against `/health`. Add authentication, rate limiting and TLS at the reverse proxy in front of it.

---

## Volumes

Three named volumes are created automatically:

| Volume | Mounted at | Purpose |
|--------|------------|---------|
| `models` | `/app/models` | Trained models persisted across restarts (shared by `jupyter`, `api`, `streamlit`) |
| `data` | `/app/data` | Datasets and preprocessed data |
| `logs` | `/app/logs` | TensorBoard event files |

The `jupyter` service additionally bind-mounts `./notebooks`, `./examples` and `./docs` so edits are visible on the host.

---

## Troubleshooting

**Container won't start?**  
```bash
docker compose logs jupyter
```

**TensorFlow can't see GPU inside Docker?**  
The images install `tensorflow-cpu`. For GPU inference, build from a `tensorflow/tensorflow:2.21.0-gpu` base (or install `tensorflow[and-cuda]`), install the package with `pip install .[export,serving]`, and add `--gpus all` to `docker run`.
