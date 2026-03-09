# ============================================================
# TensorVerseHub — Makefile
# Development task automation
# ============================================================

SHELL := /bin/bash
PYTHON := python3
PIP    := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK  := $(PYTHON) -m black
ISORT  := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY   := $(PYTHON) -m mypy

SRC_DIRS   := src scripts examples tests
VENV_DIR   := venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help message
	@echo "TensorVerseHub — Development Commands"
	@echo "======================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────
.PHONY: venv
venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created at ./$(VENV_DIR)"
	@echo "Activate with: source $(VENV_ACTIVATE)"

.PHONY: install
install:  ## Install project in editable mode with dev dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: install-all
install-all:  ## Install all optional dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"
	$(PIP) install -r requirements.txt

.PHONY: install-docs
install-docs:  ## Install documentation dependencies
	$(PIP) install -e ".[docs]"

# ── Code Quality ──────────────────────────────────────────────
.PHONY: format
format:  ## Auto-format code with black + isort
	$(BLACK) $(SRC_DIRS)
	$(ISORT) $(SRC_DIRS)

.PHONY: format-check
format-check:  ## Check formatting without modifying files
	$(BLACK) --check --diff $(SRC_DIRS)
	$(ISORT) --check-only --diff $(SRC_DIRS)

.PHONY: lint
lint:  ## Lint code with flake8
	$(FLAKE8) $(SRC_DIRS) \
		--max-line-length=100 \
		--extend-ignore=E203,W503 \
		--count --statistics

.PHONY: typecheck
typecheck:  ## Run static type checking with mypy
	$(MYPY) src/ --ignore-missing-imports

.PHONY: check
check: format-check lint typecheck  ## Run all checks (format, lint, typecheck)

# ── Testing ───────────────────────────────────────────────────
.PHONY: test
test:  ## Run unit tests (excluding slow/GPU tests)
	$(PYTEST) tests/ \
		-v --tb=short \
		--ignore=tests/test_notebooks.py \
		--ignore=tests/test_stress_and_performance.py \
		-m "not slow and not gpu"

.PHONY: test-all
test-all:  ## Run all tests including slow ones
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-integration
test-integration:  ## Run integration tests only
	$(PYTEST) tests/test_integration.py -v --tb=short

.PHONY: test-coverage
test-coverage:  ## Run tests with HTML coverage report
	$(PYTEST) tests/ \
		--cov=src \
		--cov-report=html:htmlcov \
		--cov-report=term-missing \
		--ignore=tests/test_notebooks.py \
		--ignore=tests/test_stress_and_performance.py \
		-m "not slow and not gpu"
	@echo "Coverage report: htmlcov/index.html"

.PHONY: test-notebooks
test-notebooks:  ## Validate notebook JSON structure
	$(PYTHON) - <<'EOF'
	import json, pathlib, sys
	errors = []
	for nb in pathlib.Path("notebooks").rglob("*.ipynb"):
	    try:
	        with open(nb) as f: json.load(f)
	    except json.JSONDecodeError as e:
	        errors.append(f"{nb}: {e}")
	if errors:
	    [print(f"ERROR: {e}") for e in errors]; sys.exit(1)
	print(f"All {len(list(pathlib.Path('notebooks').rglob('*.ipynb')))} notebooks are valid JSON.")
	EOF

# ── Docker ────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:  ## Build Docker image (jupyter target)
	docker build --target jupyter -t tensorversehub:latest .

.PHONY: docker-up
docker-up:  ## Start all services via docker-compose
	docker compose up -d

.PHONY: docker-down
docker-down:  ## Stop all docker-compose services
	docker compose down

.PHONY: docker-logs
docker-logs:  ## Tail logs from all docker-compose services
	docker compose logs -f

.PHONY: docker-jupyter
docker-jupyter:  ## Start only the Jupyter service
	docker compose up -d jupyter
	@echo "Jupyter Lab available at http://localhost:8888"

.PHONY: docker-clean
docker-clean:  ## Remove containers, networks, and volumes
	docker compose down -v --remove-orphans

# ── Jupyter ───────────────────────────────────────────────────
.PHONY: jupyter
jupyter:  ## Launch Jupyter Lab locally
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

.PHONY: tensorboard
tensorboard:  ## Launch TensorBoard on ./logs
	tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# ── Documentation ─────────────────────────────────────────────
.PHONY: docs-serve
docs-serve:  ## Serve documentation site locally (hot-reload)
	mkdocs serve

.PHONY: docs-build
docs-build:  ## Build static documentation site
	mkdocs build --clean

.PHONY: docs-deploy
docs-deploy:  ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy --force

# ── Serving Examples ──────────────────────────────────────────
.PHONY: serve-flask
serve-flask:  ## Run the Flask model-serving API locally
	cd examples/serving_examples && $(PYTHON) flask_tensorflow_api.py

.PHONY: serve-streamlit
serve-streamlit:  ## Run the Streamlit dashboard locally
	streamlit run examples/serving_examples/streamlit_tensorflow_demo.py

# ── Model Optimisation Demo ───────────────────────────────────
.PHONY: demo-quantize
demo-quantize:  ## Run quantization demo
	$(PYTHON) examples/optimization_examples/quantization_demo.py

.PHONY: demo-prune
demo-prune:  ## Run pruning demo
	$(PYTHON) examples/optimization_examples/pruning_demo.py

.PHONY: demo-distill
demo-distill:  ## Run knowledge distillation demo
	$(PYTHON) examples/optimization_examples/distillation_demo.py

# ── Pre-commit ────────────────────────────────────────────────
.PHONY: pre-commit-install
pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

.PHONY: pre-commit-run
pre-commit-run:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

# ── Cleaning ──────────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove Python cache and build artifacts
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -not -path "./venv/*" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info

.PHONY: clean-all
clean-all: clean  ## Remove everything including the venv
	rm -rf $(VENV_DIR) site/
