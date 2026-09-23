# ============================================================
# TensorVerseHub — developer tasks
# ============================================================
SHELL   := /bin/bash
PYTHON  ?= python3
PIP     := $(PYTHON) -m pip
PYTEST  := $(PYTHON) -m pytest
RUFF    := $(PYTHON) -m ruff
MYPY    := $(PYTHON) -m mypy
PKG     := tensorversehub

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────
.PHONY: install install-all install-legacy
install:  ## Editable install with dev + export extras
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,export]"

install-all:  ## Editable install with every extra (notebooks, serving, docs, ...)
	$(PIP) install -e ".[all]"

install-legacy:  ## Add the legacy Keras 2 stack (tf-keras + tensorflow-model-optimization)
	$(PIP) install -e ".[optimization]"
	@echo "Run with: TF_USE_LEGACY_KERAS=1"

# ── Quality ───────────────────────────────────────────────────
.PHONY: format lint typecheck check
format:  ## Auto-format and auto-fix with ruff
	$(RUFF) check --fix $(PKG) tests examples
	$(RUFF) format $(PKG) tests

lint:  ## Lint with ruff (no changes)
	$(RUFF) check $(PKG) tests examples
	$(RUFF) format --check $(PKG) tests

typecheck:  ## Static type check with mypy
	$(MYPY) $(PKG)

check: lint typecheck  ## Lint + type check

# ── Tests ─────────────────────────────────────────────────────
.PHONY: test test-legacy test-all test-cov
test:  ## Unit tests (Keras 3, excluding slow/gpu)
	$(PYTEST) tests -m "not slow and not gpu"

test-legacy:  ## Unit tests under legacy Keras 2 (enables tfmot pruning/QAT tests)
	TF_USE_LEGACY_KERAS=1 $(PYTEST) tests -m "not slow and not gpu"

test-all:  ## Everything, including slow tests
	$(PYTEST) tests

test-cov:  ## Tests with an HTML coverage report
	$(PYTEST) tests -m "not slow and not gpu" --cov=$(PKG) --cov-report=html:htmlcov --cov-report=term-missing
	@echo "open htmlcov/index.html"

# ── Docs ──────────────────────────────────────────────────────
.PHONY: docs-serve docs-build
docs-serve:  ## Live-preview the documentation site
	mkdocs serve

docs-build:  ## Build the static documentation site (strict)
	mkdocs build --strict

# ── Docker ────────────────────────────────────────────────────
.PHONY: docker-build docker-up docker-down docker-logs
docker-build:  ## Build the Jupyter image
	docker build --target jupyter -t tensorversehub:latest .

docker-up:  ## Start all services (Jupyter, API, Streamlit, TensorBoard)
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## Tail service logs
	docker compose logs -f

# ── CLI shortcuts ─────────────────────────────────────────────
.PHONY: train evaluate convert serve
train:  ## Quick synthetic training run (writes ./models/final_model.keras)
	tensorverse train --epochs 2 --image-size 32 32 --output-dir ./models

evaluate:  ## Evaluate ./models/final_model.keras with reports
	tensorverse evaluate --model ./models/final_model.keras --image-size 32 32 --report --confusion-matrix

convert:  ## Convert ./models/final_model.keras to every format
	tensorverse convert --model ./models/final_model.keras --to all --output ./converted_models

serve:  ## Serve ./models/final_model.keras on :8000
	tensorverse serve --model ./models/final_model.keras --port 8000

# ── Packaging ─────────────────────────────────────────────────
.PHONY: build clean
build:  ## Build sdist + wheel
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:  ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info site
