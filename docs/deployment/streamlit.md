# Streamlit Dashboard

The Streamlit demo (`examples/serving_examples/streamlit_tensorflow_demo.py`) provides an interactive web UI for exploring model predictions without writing any code.

---

## Running Locally

```bash
pip install -e ".[serving]"      # streamlit and friends
streamlit run examples/serving_examples/streamlit_tensorflow_demo.py
# → http://localhost:8501
```

The demo starts in **Demo Mode** (simulated predictions). Untick it in the sidebar and enter the path to a `.keras` file, SavedModel directory or `.tflite` file to run a real model.

---

## Features

- Upload an image and get real-time predictions
- View class probabilities as a bar chart
- Switch between image and text classification and inspect model metadata
- Point the dashboard at any of your exported models

---

## Docker

```bash
docker compose up -d streamlit
# → http://localhost:8501
```

The service shares the `models` volume with the `jupyter` and `api` services, so a model trained in a notebook is available under `/app/models` here.
