# Changelog

All notable changes to TensorVerseHub are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `Dockerfile` + `docker-compose.yml` for containerised Jupyter, Flask, Streamlit, and TensorBoard
- `.dockerignore` for optimised Docker build context
- `.github/workflows/ci.yml` — GitHub Actions CI: lint, unit tests (Python 3.9/3.10/3.11), Docker build
- `.github/workflows/publish.yml` — automated PyPI + GitHub Container Registry publishing on version tags
- `.github/ISSUE_TEMPLATE/` — structured bug report and feature request templates
- `.github/PULL_REQUEST_TEMPLATE.md`
- `scripts/train_models.py` — `tensorverse-train` CLI
- `scripts/evaluate_models.py` — `tensorverse-evaluate` CLI
- `scripts/convert_models.py` — `tensorverse-convert` CLI (SavedModel, TFLite, ONNX, TF.js, CoreML)
- `pyproject.toml` — modern PEP 518 build configuration (replaces legacy `setup.py` as primary config)
- `Makefile` — developer task automation (format, lint, test, docker, docs, serve)
- `pytest.ini` — pytest configuration
- `.editorconfig` — cross-editor code style
- `.pre-commit-config.yaml` — automated hooks: black, isort, flake8, nbstripout, yamllint
- `mkdocs.yml` + full documentation site (index, installation, API reference, deployment guides)
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`
- `.vscode/settings.json` and `.vscode/extensions.json` for consistent development experience

### Fixed
- **Notebook 21** (`21_multimodal_ai_system.ipynb`): Replaced `optimize_models()` placeholder with
  real TensorRT API pattern and TFLite float16 fallback
- **Notebook 22** (`22_end_to_end_ml_pipeline.ipynb`): Replaced random feature-importance bar chart
  with actual first Dense layer weight-magnitude extraction

---

## [1.0.0] — 2025-12-01

### Added
- 27 Jupyter notebooks spanning Foundations → Capstone Projects
- Production utilities in `src/`: `data_utils`, `model_utils`, `optimization_utils`,
  `export_utils`, `visualization` (3,720+ lines total)
- Serving examples: Flask REST API, Streamlit dashboard, TFLite inference, FastAPI
- Optimisation examples: quantization, pruning, knowledge distillation
- Comprehensive test suite: 1,800+ lines across 9 test files
- Documentation: Quick reference, best practices, model optimisation guide, troubleshooting,
  architecture diagrams, practical examples
- `requirements.txt` with pinned dependencies
- `setup.py` with package metadata and console scripts
- `.gitignore`

[Unreleased]: https://github.com/SatvikPraveen/TensorVerseHub/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/SatvikPraveen/TensorVerseHub/releases/tag/v1.0.0
