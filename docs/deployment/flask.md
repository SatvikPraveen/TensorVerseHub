# Flask REST API

The Flask serving example (`examples/serving_examples/flask_tensorflow_api.py`) exposes a production-ready REST API for real-time model inference.

---

## Running Locally

```bash
cd examples/serving_examples
python flask_tensorflow_api.py
# → http://localhost:5000
```

Or via Make:

```bash
make serve-flask
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model metadata |
| `POST` | `/predict` | Single image inference |
| `POST` | `/predict/batch` | Batch inference |
| `GET` | `/metrics` | Prometheus-style metrics |

---

## Example Request

```python
import requests, base64

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:5000/predict",
    json={"image": img_b64, "top_k": 5}
)
print(response.json())
# {"predictions": [{"class": "cat", "confidence": 0.92}, ...]}
```

---

## Docker

```bash
docker compose up -d flask-api
curl http://localhost:5000/health
```
