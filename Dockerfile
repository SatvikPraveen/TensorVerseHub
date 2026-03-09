# ============================================================
# TensorVerseHub - Dockerfile
# Multi-stage build for production-grade TensorFlow environment
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first (better layer caching)
COPY requirements.txt .

# Install Python dependencies into a prefix for copying
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.10-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 tensorverse && \
    useradd --uid 1000 --gid tensorverse --shell /bin/bash --create-home tensorverse

# Copy project files
COPY --chown=tensorverse:tensorverse . .

# Remove venv if it was copied in
RUN rm -rf /app/venv

# Switch to non-root user
USER tensorverse

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/app/src:/app

# Expose ports for serving examples
EXPOSE 5000 8501 8000

# Default command — launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# --- Stage 3: Jupyter (default target) ---
FROM runtime AS jupyter

# Health check for the Jupyter server
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# --- Stage 4: Flask API ---
FROM runtime AS flask-api

WORKDIR /app/examples/serving_examples

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "flask_tensorflow_api.py"]

# --- Stage 5: FastAPI ---
FROM runtime AS fastapi

WORKDIR /app/examples/serving_examples

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "fastapi_tensorflow_api:app", "--host", "0.0.0.0", "--port", "8000"]
