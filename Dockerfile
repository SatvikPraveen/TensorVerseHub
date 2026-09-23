# ============================================================
# TensorVerseHub — multi-stage image
#   runtime  : package + serving/export extras (base for all services)
#   jupyter  : JupyterLab with the full notebook stack (default)
#   api      : FastAPI model server (tensorverse serve)
# ============================================================
ARG PYTHON_VERSION=3.12

# --- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY tensorversehub ./tensorversehub
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install "tensorflow-cpu>=2.16,<2.22" \
    && /opt/venv/bin/pip install ".[export,serving]"

# --- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 tensorverse \
    && useradd --uid 1000 --gid tensorverse --create-home --shell /bin/bash tensorverse
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=tensorverse:tensorverse . .
USER tensorverse
EXPOSE 8000
CMD ["tensorverse", "--help"]

# --- api -------------------------------------------------------------------
FROM runtime AS api
ENV TVH_MODEL=/app/models/final_model.keras
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["sh", "-c", "tensorverse serve --model \"$TVH_MODEL\" --host 0.0.0.0 --port 8000"]

# --- jupyter (default) -----------------------------------------------------
FROM runtime AS jupyter
USER root
RUN /opt/venv/bin/pip install --no-cache-dir ".[notebooks]" && chown -R tensorverse:tensorverse /app
USER tensorverse
EXPOSE 8888 6006 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password="]
