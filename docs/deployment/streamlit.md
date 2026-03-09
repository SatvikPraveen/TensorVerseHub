# Streamlit Dashboard

The Streamlit demo (`examples/serving_examples/streamlit_tensorflow_demo.py`) provides an interactive web UI for exploring model predictions without writing any code.

---

## Running Locally

```bash
streamlit run examples/serving_examples/streamlit_tensorflow_demo.py
# → http://localhost:8501
```

Or via Make:

```bash
make serve-streamlit
```

---

## Features

- Upload an image and get real-time predictions
- View class probabilities as a bar chart
- Toggle between different loaded models
- Inspect preprocessing steps visually

---

## Docker

```bash
docker compose up -d streamlit
# → http://localhost:8501
```
