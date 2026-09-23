# Contributing to TensorVerseHub

Thank you for considering a contribution! This document explains how to get started.

---

## Ways to Contribute

- **Bug reports** — open an issue using the Bug Report template
- **Feature requests** — open an issue using the Feature Request template  
- **Notebook improvements** — fix errors, add explanations, or extend examples
- **New utilities** — add functions to the `tensorversehub/` package (with tests and docstrings)
- **Documentation** — improve guides, fix typos
- **Tests** — increase coverage, add edge cases

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/TensorVerseHub.git
cd TensorVerseHub

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Set up the virtual environment
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev,export]"

# 4. Install pre-commit hooks
pre-commit install
```

---

## Code Style

We use **ruff** for linting and formatting (line-length 100) and **mypy** for type checking.

```bash
make format     # ruff --fix + ruff format
make lint       # ruff check + ruff format --check
make typecheck  # mypy tensorversehub
make check      # lint + typecheck
```

All hooks also run automatically on every `git commit` (via pre-commit).

---

## Running Tests

```bash
make test          # unit tests on Keras 3
make test-legacy   # same suite with TF_USE_LEGACY_KERAS=1 (enables pruning/QAT tests)
make test-cov      # unit tests + HTML coverage report (htmlcov/)
```

New code **must** include corresponding tests in `tests/`.

---

## Notebook Conventions

- Use numbered prefixes matching the curriculum (e.g. `07_cnn_architectures_keras.ipynb`)
- Clear all cell outputs before committing (`nbstripout` runs automatically via pre-commit)
- Include a **Markdown overview cell** at the top with learning objectives
- End with a **Summary** section

---

## Pull Request Process

1. Ensure `make check` passes without errors
2. Ensure `make test` passes
3. Fill in the Pull Request template completely
4. Request a review from a maintainer
5. Squash your commits before merging (keep history clean)

---

## Commit Message Format

```
<type>: <short summary>

[optional body]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `style`

Examples:
```
feat: add CutMix augmentation to DataAugmentation class
fix: correct ONNX export signature for variable-length inputs
docs: add TFLite deployment guide
```

---

## Questions?

Open a Discussion on GitHub or file an issue — we're happy to help!

---

## Releasing

1. Bump `version` in `pyproject.toml`, `__version__` in `tensorversehub/__init__.py` and
   `CITATION.cff`; move the `[Unreleased]` notes in `docs/CHANGELOG.md` under the new version.
2. Tag and push: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
3. The **Publish** workflow builds the wheel/sdist (and checks the tag matches the version),
   pushes `ghcr.io/satvikpraveen/tensorversehub:X.Y.Z`, and creates a GitHub release with the
   changelog entry and artifacts.
4. PyPI upload runs only when the repository variable `PYPI_PUBLISH` is `true` and a
   [trusted publisher](https://docs.pypi.org/trusted-publishers/) for this repository /
   the `pypi` environment is configured on PyPI.
