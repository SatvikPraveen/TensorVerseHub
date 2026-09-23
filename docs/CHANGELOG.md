# Changelog

All notable changes to TensorVerseHub are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [2.0.0] — 2026-09-23

A ground-up modernisation. The utilities are now a real, tested, typed package that
targets TensorFlow 2.16–2.21 with Keras 3, while keeping legacy Keras 2 working for
`tensorflow-model-optimization`.

### Added
- `tensorversehub.compat` — Keras 2/3 bridge (`save_model`, `load_model`, `export_saved_model`,
  `SavedModelPredictor`, loss-scaling helpers, output-shape / signature introspection,
  `require_legacy_keras`).
- `tensorversehub.training_utils.CustomTrainingLoop` gains gradient accumulation, XLA
  (`jit_compile`), correct mixed-precision loss scaling on both Keras generations, gradient-norm
  reporting, `evaluate()` and `train_on_batch()`.
- `WarmupCosineSchedule` is registered as a Keras serialisable; `LearningRateFinder.suggest()`.
- Vectorised **MixUp**, **CutMix** (real masks, area-corrected labels) and **Random Erasing**;
  `rotate_image` built on `tf.raw_ops.ImageProjectiveTransformV3`; `serialize_array_example`,
  sharded TFRecord writing, `count_records`, `split_dataset`, `compute_class_weights`.
- `TransformerEncoderBlock`, causal / padding mask helpers, `ModelBuilders.create_mlp`,
  profiler-backed `ModelAnalysis.compute_model_flops`, dtype-aware model size,
  `ValidationMonitor` callback, ten `keras.applications` backbones with built-in preprocessing.
- Full-integer int8 TFLite export with int8 I/O, `TFLiteExporter.validate_tflite`,
  `ONNXExporter.validate_onnx`, `make_interpreter` (LiteRT fallback), `int8_fallback`
  quantisation, `ModelPruning.compute_sparsity`, `distillation_loss`, XLA inference clones.
- Unified `tensorverse` CLI (`train`, `evaluate`, `convert`, `serve`, `info`) with a
  FastAPI server factory and synthetic-data fallbacks.
- New executable test-suite (190+ tests) run in CI on Python 3.10/3.11/3.12 × TensorFlow
  2.16/2.18/2.21, plus a legacy-Keras job; notebook validation, strict docs build, wheel build.
- `docs.yml` workflow deploying MkDocs to GitHub Pages; Dependabot; `py.typed`.
- Curriculum pages generated from the notebooks (`docs/curriculum.md`, `docs/notebooks/`).

### Changed
- **Breaking:** `src/` is now the installable package `tensorversehub/`
  (`from tensorversehub.model_utils import ModelBuilders`). CLI scripts moved to
  `tensorversehub.cli`; console scripts `tensorverse-*` remain.
- **Breaking:** requires Python ≥ 3.10 and TensorFlow ≥ 2.17 (Keras 3). Python 3.9 and
  TensorFlow ≤ 2.16 are no longer supported (2.16's TFLite converter aborts the process on
  int8 conversion of Keras 3 models).
- Importing the package has no side effects (no printing, GPU configuration or global plot
  style). Use `tensorversehub.configure_tensorflow()` explicitly.
- `MultiHeadAttention` call signature is `(query, key=None, value=None, attention_mask=None)`
  with `True` = attend (Keras convention); layers are serialisable via `get_config`.
- Text classifiers use bidirectional RNNs / real Transformer encoder stacks; GAN and
  autoencoder builders validate shapes.
- Models are saved as `.keras`; SavedModel directories are inference-only exports.
- Visualisation functions no longer call `plt.show()` unconditionally; they return the figure
  and accept `show=` / `save_path=`.
- `optimize_for_mobile` picks the smallest strategy that meets the target (the old selection
  logic always returned the first strategy).
- Knowledge distillation uses logits, temperature-scaled KL divergence and `from_logits=True`
  (the old loss mixed probabilities and logits).
- Tooling: ruff + mypy replace black / isort / flake8; `pyproject.toml` is the single
  configuration file (`setup.py`, `pytest.ini` removed); Docker image uses a venv, Python 3.12
  and a FastAPI `api` target.

### Fixed
- `import src` failed on every supported TensorFlow because `__init__` demanded TF ≥ 2.15 with a
  lexicographic version compare.
- `tf.contrib.image.rotate` and `tf.random.beta` (both removed in TF 2) in the augmentation code;
  CutMix previously returned the unmodified batch.
- TFRecord feature dispatch mis-typed lists of floats as int64.
- Percentile gradient clipping called `.numpy()` inside graph code.
- `LearningRateFinder` used `compiled_loss`, removed in Keras 3.
- `compute_model_flops` wrote profiler traces to a `logdir` folder on every call.
- CI never passed: tests imported classes that did not exist (`ModelExporter`,
  `SavedModelHandler`, `MetricsVisualizer`); `black` was pinned to a version newer than CI.
- Dockerfile referenced a non-existent `fastapi_tensorflow_api.py`; `.dockerignore` was
  git-ignored.

### Removed
- `setup.py`, `pytest.ini`, `scripts/` (now `tensorversehub/cli/`), the aspirational test files.

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
