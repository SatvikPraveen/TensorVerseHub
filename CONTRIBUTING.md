# Contributing to TensorVerseHub

**Location: `/CONTRIBUTING.md`**

Thank you for your interest in contributing to TensorVerseHub! This document provides guidelines for contributing to our comprehensive TensorFlow learning platform.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.15+
- Git
- Docker (optional but recommended)

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/TensorVerseHub.git
cd TensorVerseHub

# Create virtual environment
python -m venv tensorverse_env
source tensorverse_env/bin/activate  # On Windows: tensorverse_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy pre-commit

# Setup pre-commit hooks
pre-commit install
```

## 📝 Contribution Types

### 1. Notebook Contributions

- **New tutorials**: Follow the progressive learning structure
- **Improvements**: Bug fixes, better explanations, updated examples
- **Optimizations**: Performance improvements, best practices

### 2. Code Contributions

- **Utilities**: New functions in `src/` modules
- **Examples**: Serving, inference, optimization examples
- **Tests**: Unit tests, integration tests, notebook tests

### 3. Documentation

- **API documentation**: Docstrings, type hints
- **Guides**: Best practices, troubleshooting
- **README updates**: Feature documentation, examples

### 4. Model Contributions

- **Pre-trained models**: Well-documented, optimized models
- **Model architectures**: Custom tf.keras implementations
- **Optimization examples**: Quantization, pruning, distillation

## 🔄 Development Workflow

### 1. Issue Creation

```bash
# Create issue for:
# - Bug reports with reproducible examples
# - Feature requests with detailed specifications
# - Documentation improvements
```

### 2. Branch Strategy

```bash
# Create feature branch
git checkout -b feature/notebook-transformer-improvements
git checkout -b fix/data-pipeline-bug
git checkout -b docs/api-reference-update
```

### 3. Code Standards

#### Python Code Style

```python
# Use Black formatting
black src/ tests/ examples/

# Follow PEP 8 with these settings:
# - Line length: 88 characters
# - Use type hints
# - Comprehensive docstrings

def create_cnn_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    """
    Create a CNN model with configurable architecture.

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled tf.keras.Model ready for training

    Example:
        >>> model = create_cnn_model((224, 224, 3), 10)
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    """
    # Implementation here
```

#### Notebook Standards

```python
# Cell 1: Clear title and learning objectives
"""
# Transformer Architecture Implementation with tf.keras

## Learning Objectives
- Implement multi-head attention mechanism
- Build complete transformer encoder/decoder
- Train on text classification task
- Evaluate model performance and interpretability
"""

# Cell 2: Comprehensive imports with versions
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# Cell 3: Clear explanations before code
"""
## Multi-Head Attention Implementation

The attention mechanism allows the model to focus on different parts of the input
sequence when processing each element. We'll implement this using tf.keras.layers.
"""

# Implementation with detailed comments
```

### 4. Testing Requirements

#### Unit Tests

```python
# tests/test_model_utils.py
import pytest
import tensorflow as tf
from src.model_utils import create_cnn_model

def test_cnn_model_creation():
    """Test CNN model creation with valid parameters."""
    model = create_cnn_model((224, 224, 3), 10)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 10)

def test_cnn_model_invalid_input():
    """Test CNN model creation with invalid parameters."""
    with pytest.raises(ValueError):
        create_cnn_model((224, 224), 10)  # Invalid input shape
```

#### Notebook Testing

```python
# All notebooks must be executable without errors
# Use nbval for automated notebook testing
pytest --nbval notebooks/01_tensorflow_foundations/
```

### 5. Documentation Standards

#### Docstring Format

```python
def train_model(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    epochs: int = 10,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
) -> tf.keras.callbacks.History:
    """
    Train a tf.keras model with comprehensive monitoring.

    This function provides a standardized training pipeline with built-in
    monitoring, checkpointing, and early stopping capabilities.

    Args:
        model: Compiled tf.keras.Model to train
        train_data: Training dataset (tf.data.Dataset)
        validation_data: Validation dataset for monitoring
        epochs: Number of training epochs
        callbacks: Additional callbacks for training

    Returns:
        Training history containing loss and metrics

    Raises:
        ValueError: If model is not compiled
        TypeError: If datasets are not tf.data.Dataset objects

    Example:
        >>> model = create_cnn_model((224, 224, 3), 10)
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        >>> history = train_model(model, train_ds, val_ds, epochs=50)
        >>> plot_training_history(history)
    """
    # Implementation
```

## 📋 Pull Request Process

### 1. Pre-submission Checklist

- [ ] Code follows style guidelines (Black, Flake8)
- [ ] All tests pass locally
- [ ] New code has comprehensive tests
- [ ] Documentation is updated
- [ ] Notebooks execute without errors
- [ ] No sensitive data in commits

### 2. PR Description Template

```markdown
## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix/feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Notebook improvement/addition

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Notebooks tested manually
- [ ] Performance impact assessed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All checks pass
```

### 3. Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: Maintainer reviews for quality, style, functionality
3. **Testing**: Manual testing for notebooks and examples
4. **Documentation**: Verify documentation updates
5. **Approval**: Final approval and merge

## 🐛 Bug Reports

### Bug Report Template

````markdown
**Bug Description**
Clear description of the bug.

**Environment**

- Python version:
- TensorFlow version:
- CUDA version (if applicable):
- Operating System:

**Reproduction Steps**

1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Code Example**

```python
# Minimal reproducible example
```
````

**Error Messages**

```
Full error traceback
```

**Additional Context**
Screenshots, logs, or other relevant information.

````

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
High-level description of how this could be implemented.

**Examples**
```python
# Code examples of how the feature would be used
````

**Additional Context**
References, similar implementations, or other relevant information.

```

## 📚 Documentation Guidelines

### 1. Markdown Standards
- Use clear headings and structure
- Include code examples for all features
- Add links to relevant resources
- Keep language clear and accessible

### 2. Code Documentation
- Comprehensive docstrings for all public functions
- Type hints for all parameters and returns
- Examples in docstrings
- Clear variable names and comments

### 3. Notebook Documentation
- Learning objectives at the beginning
- Step-by-step explanations
- Visualizations and plots
- Summary and next steps

## 🏆 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Contributor list
- Documentation credits

## ❓ Questions and Support

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Create an Issue
- **Feature requests**: Create an Issue with feature request template
- **Pull request help**: Comment on your PR

Thank you for contributing to TensorVerseHub! 🚀
```
