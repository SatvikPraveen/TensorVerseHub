# Location: /examples/docker/Dockerfile

# Multi-stage build for TensorFlow development and deployment
FROM tensorflow/tensorflow:2.15.0-gpu AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    vim \
    htop \
    tree \
    graphviz \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    jupyter-server-proxy \
    jupyterlab-git

# Copy source code
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY setup.py .
COPY README.md .

# Install TensorVerseHub package in development mode
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Development stage
FROM base AS development

# Expose ports for Jupyter, TensorBoard, and Flask
EXPOSE 8888 6006 5000

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV JUPYTER_ENABLE_LAB=yes

# Create Jupyter config
RUN jupyter lab --generate-config

# Configure Jupyter Lab
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# Default command for development
CMD ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--port=8888"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY --from=base /app/src ./src
COPY --from=base /app/examples ./examples
COPY --from=base /app/scripts ./scripts

# Remove development dependencies
RUN pip uninstall -y jupyter jupyterlab ipywidgets

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tensorverse
RUN chown -R tensorverse:tensorverse /app
USER tensorverse

# Expose only production ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command for production
CMD ["python", "examples/serving_examples/flask_tensorflow_api.py", "--host", "0.0.0.0", "--port", "5000"]

# Inference stage (minimal)
FROM tensorflow/tensorflow:2.15.0 AS inference

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir \
    flask \
    pillow \
    numpy

# Copy only inference code
COPY examples/serving_examples/flask_tensorflow_api.py .
COPY examples/serving_examples/tflite_inference_example.py .

# Create model directory
RUN mkdir -p /app/models

# Expose inference port
EXPOSE 5000

# Default command for inference
CMD ["python", "flask_tensorflow_api.py", "--host", "0.0.0.0", "--port", "5000"]