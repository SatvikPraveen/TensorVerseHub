#!/bin/bash

# TensorVerseHub Project Structure Generator
# TensorFlow with tf.keras Integration - Complete Learning Hub
# Run: chmod +x create_tensorversehub.sh && ./create_tensorversehub.sh

set -e  # Exit on any error

echo "🚀 Creating TensorVerseHub: TensorFlow with tf.keras Mastery Hub"
echo "=================================================================="

# Create main project directory
PROJECT_NAME="TensorVerseHub"
if [ -d "$PROJECT_NAME" ]; then
    read -p "⚠️  Directory $PROJECT_NAME already exists. Remove it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_NAME"
        echo "✅ Removed existing directory"
    else
        echo "❌ Aborted. Please remove or rename existing directory."
        exit 1
    fi
fi

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

echo "📁 Creating directory structure..."

# Create main directories
mkdir -p .github/workflows
mkdir -p data/{sample_images/{classification,detection},sample_text,tfrecords_examples,synthetic}
mkdir -p models/{checkpoints/{cnn_model,transformer_model,multimodal_model},saved_models/{image_classifier/1,text_classifier/1,multimodal_system/1},tflite,onnx}
mkdir -p src
mkdir -p examples/{serving_examples,docker,optimization_examples}
mkdir -p tests
mkdir -p docs/assets/{architecture_diagrams,workflow_diagrams,screenshots}
mkdir -p scripts
mkdir -p benchmarks/results
mkdir -p logs/{tensorboard/{cnn_experiments,transformer_experiments,capstone_projects},training_logs,experiment_configs}

# Create notebook directories
mkdir -p notebooks/01_tensorflow_foundations
mkdir -p notebooks/02_neural_networks_with_keras
mkdir -p notebooks/03_computer_vision
mkdir -p notebooks/04_natural_language_processing
mkdir -p notebooks/05_generative_models
mkdir -p notebooks/06_model_optimization
mkdir -p notebooks/07_advanced_topics
mkdir -p notebooks/capstone_projects

echo "📝 Creating configuration files..."

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Jupyter Notebooks
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.conda/
anaconda3/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# TensorFlow specific
*.pb
*.h5
*.hdf5
*.keras
logs/
tensorboard_logs/
*.ckpt*
saved_model/
checkpoint

# Data files (keep structure, ignore large files)
data/**/*.csv
data/**/*.json
data/**/*.pkl
data/**/*.zip
data/**/*.tar.gz
data/**/*.jpg
data/**/*.png
data/**/*.txt
!data/**/readme.txt
models/**/*.h5
models/**/*.pb
models/**/*.tflite
models/**/*.onnx
models/**/variables/

# Temporary files
tmp/
temp/
.cache/

# Documentation builds
docs/_build/
site/
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
# Core TensorFlow with tf.keras
tensorflow>=2.15.0,<2.16.0
tensorflow-hub>=0.16.0
tensorflow-datasets>=4.9.0
tensorflow-model-optimization>=0.7.4

# Data processing and numerical computing
numpy>=1.24.0,<1.27.0
pandas>=2.0.0,<2.3.0
pillow>=10.0.0
opencv-python>=4.8.0

# Visualization and plotting
matplotlib>=3.7.0,<3.9.0
seaborn>=0.12.0,<0.14.0
plotly>=5.15.0,<5.18.0

# Jupyter ecosystem
jupyter>=1.0.0
jupyterlab>=4.0.0,<4.1.0
ipywidgets>=8.0.0,<8.2.0
ipykernel>=6.25.0

# Machine Learning utilities
scikit-learn>=1.3.0,<1.4.0
scipy>=1.11.0,<1.12.0

# Model export and optimization  
onnx>=1.15.0,<1.16.0
tf2onnx>=1.16.0,<1.17.0

# Development and testing
pytest>=7.4.0,<8.0.0
black>=23.7.0,<24.0.0
flake8>=6.0.0,<7.0.0
pre-commit>=3.3.0,<4.0.0

# Progress bars and utilities
tqdm>=4.65.0,<5.0.0
requests>=2.31.0,<3.0.0

# Optional web serving (for examples)
flask>=2.3.0,<3.0.0
streamlit>=1.28.0,<1.29.0

# Documentation
mkdocs>=1.5.0,<2.0.0
mkdocs-material>=9.4.0,<10.0.0
EOF

# setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "TensorFlow with tf.keras mastery hub"

setup(
    name="tensorversehub",
    version="1.0.0",
    author="TensorFlow Practitioner",
    author_email="practitioner@tensorversehub.com",
    description="A comprehensive TensorFlow with tf.keras mastery hub from basics to advanced optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SatvikPraveen/TensorVerseHub",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: TensorFlow",
    ],
    python_requires=">=3.8,<3.12",
    install_requires=[
        "tensorflow>=2.15.0,<2.16.0",
        "tensorflow-hub>=0.16.0",
        "numpy>=1.24.0,<1.27.0",
        "pandas>=2.0.0,<2.3.0",
        "matplotlib>=3.7.0,<3.9.0",
        "jupyter>=1.0.0",
        "scikit-learn>=1.3.0,<1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0,<8.0.0",
            "black>=23.7.0,<24.0.0",
            "flake8>=6.0.0,<7.0.0",
            "pre-commit>=3.3.0,<4.0.0",
        ],
        "viz": [
            "seaborn>=0.12.0,<0.14.0",
            "plotly>=5.15.0,<5.18.0",
        ],
        "optimization": [
            "tensorflow-model-optimization>=0.7.4",
            "onnx>=1.15.0,<1.16.0",
            "tf2onnx>=1.16.0,<1.17.0",
        ],
        "serving": [
            "flask>=2.3.0,<3.0.0",
            "streamlit>=1.28.0,<1.29.0",
        ],
        "docs": [
            "mkdocs>=1.5.0,<2.0.0",
            "mkdocs-material>=9.4.0,<10.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tensorversehub=tensorversehub.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/SatvikPraveen/TensorVerseHub/issues",
        "Source": "https://github.com/SatvikPraveen/TensorVerseHub",
        "Documentation": "https://tensorversehub.readthedocs.io/",
    },
)
EOF

# GitHub Actions CI
cat > .github/workflows/ci.yml << 'EOF'
name: TensorVerseHub CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .[dev]
        
    - name: Lint with flake8
      run: |
        flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
        
    - name: Test with pytest
      run: |
        pytest tests/ -v --tb=short
        
    - name: Check code formatting
      run: |
        black --check src/ tests/
EOF

# Main README
cat > README.md << 'EOF'
# TensorVerseHub 🚀

> **A Comprehensive TensorFlow with tf.keras Mastery Hub — From Foundations to Advanced Optimization**

[![CI](https://github.com/SatvikPraveen/TensorVerseHub/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/TensorVerseHub/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

TensorVerseHub is a comprehensive, portfolio-grade project that demonstrates mastery of **TensorFlow with tf.keras** from fundamentals to advanced optimization. This repository provides:

- **22 high-quality Jupyter notebooks** covering core TensorFlow concepts with tf.keras integration
- **Model optimization examples** using TensorFlow Model Optimization toolkit
- **Cross-platform export utilities** (TFLite, ONNX, TensorFlow.js)
- **Two capstone projects** showcasing end-to-end TensorFlow workflows
- **Clean, reusable utilities** for TensorFlow development with tf.keras

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/SatvikPraveen/TensorVerseHub.git
cd TensorVerseHub

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

jupyter lab
```

### Learning Path

1. **Start with TensorFlow Foundations** → `notebooks/01_tensorflow_foundations/`
2. **Learn tf.keras Integration** → `notebooks/02_neural_networks_with_keras/`
3. **Choose your domain** → Computer Vision, NLP, or Generative Models
4. **Master Model Optimization** → `notebooks/06_model_optimization/`
5. **Complete Capstones** → `notebooks/capstone_projects/`

## 💼 Portfolio Highlights

✅ **TensorFlow + tf.keras Expertise** - 22 notebooks covering core concepts and advanced techniques  
✅ **Model Optimization Skills** - Complete optimization pipeline with TensorFlow tools  
✅ **Cross-Platform Development** - Export models for mobile, web, and edge deployment  
✅ **Research Implementation** - Custom tf.keras components and latest paper implementations  
✅ **Production Readiness** - End-to-end projects with real-world complexity  

## 🛠️ Key Technologies

- **TensorFlow 2.15+** with tf.keras as primary high-level API
- **TensorFlow Hub** for pre-trained model integration
- **TensorFlow Model Optimization** for quantization and pruning
- **TensorFlow Lite** for mobile and edge deployment
- **tf.distribute** for distributed and mixed precision training
- **Custom tf.keras components** for research flexibility

## 📋 Development Setup

```bash
pip install -e .[dev]
pytest tests/
black src/ tests/
flake8 src/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to master TensorFlow with tf.keras?** Start with `notebooks/01_tensorflow_foundations/01_tensors_operations_execution.ipynb`

*TensorVerseHub - Where TensorFlow mastery begins and tf.keras excellence is achieved.*
EOF

# License
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 TensorVerseHub

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo "📦 Creating source files with placeholders..."

# Create placeholder files for source modules
cat > src/__init__.py << 'EOF'
"""
TensorVerseHub/src/__init__.py

TensorFlow with tf.keras utilities package.
TODO: Implement full module imports and functionality.
"""

__version__ = "1.0.0"
__author__ = "TensorFlow Practitioner"

# TODO: Add imports when modules are implemented
# from .data_utils import create_tf_dataset, load_tfrecords
# from .model_utils import create_tensorflow_model, TensorFlowKerasLayer
# from .visualization import plot_training_history, visualize_tensorflow_model
# from .optimization_utils import optimize_tensorflow_model, quantize_model
# from .export_utils import export_to_tflite, export_to_onnx

pass
EOF

# Placeholder source files
touch src/data_utils.py
echo '"""TensorVerseHub/src/data_utils.py - TensorFlow data utilities with tf.keras integration. TODO: Implement data pipeline utilities."""' > src/data_utils.py

touch src/model_utils.py  
echo '"""TensorVerseHub/src/model_utils.py - TensorFlow model utilities with tf.keras integration. TODO: Implement model building utilities."""' > src/model_utils.py

touch src/visualization.py
echo '"""TensorVerseHub/src/visualization.py - Visualization utilities for TensorFlow models. TODO: Implement visualization functions."""' > src/visualization.py

touch src/optimization_utils.py
echo '"""TensorVerseHub/src/optimization_utils.py - TensorFlow model optimization utilities. TODO: Implement optimization functions."""' > src/optimization_utils.py

touch src/export_utils.py
echo '"""TensorVerseHub/src/export_utils.py - TensorFlow model export utilities. TODO: Implement export functions."""' > src/export_utils.py

echo "🧪 Creating test placeholders..."

# Create placeholder test files
cat > tests/__init__.py << 'EOF'
"""TensorVerseHub/tests - Test suite for TensorFlow utilities."""
pass
EOF

touch tests/test_data_utils.py
echo '"""TensorVerseHub/tests/test_data_utils.py - Tests for data utilities. TODO: Implement tests."""' > tests/test_data_utils.py

touch tests/test_model_utils.py
echo '"""TensorVerseHub/tests/test_model_utils.py - Tests for model utilities. TODO: Implement tests."""' > tests/test_model_utils.py

touch tests/test_optimization.py
echo '"""TensorVerseHub/tests/test_optimization.py - Tests for optimization utilities. TODO: Implement tests."""' > tests/test_optimization.py

touch tests/test_visualization.py
echo '"""TensorVerseHub/tests/test_visualization.py - Tests for visualization utilities. TODO: Implement tests."""' > tests/test_visualization.py

touch tests/test_export_utils.py
echo '"""TensorVerseHub/tests/test_export_utils.py - Tests for export utilities. TODO: Implement tests."""' > tests/test_export_utils.py

echo "🎯 Creating example placeholders..."

# Example placeholders
touch examples/serving_examples/flask_tensorflow_api.py
echo '"""TensorVerseHub/examples/serving_examples/flask_tensorflow_api.py - Flask API for serving TensorFlow models. TODO: Implement Flask serving."""' > examples/serving_examples/flask_tensorflow_api.py

touch examples/optimization_examples/quantization_demo.py
echo '"""TensorVerseHub/examples/optimization_examples/quantization_demo.py - TensorFlow model quantization demo. TODO: Implement quantization examples."""' > examples/optimization_examples/quantization_demo.py

touch examples/docker/Dockerfile
echo '# TensorVerseHub/examples/docker/Dockerfile - Docker setup for TensorFlow development. TODO: Implement Docker configuration.' > examples/docker/Dockerfile

touch examples/docker/docker-compose.yml
echo '# TensorVerseHub/examples/docker/docker-compose.yml - Docker Compose for development services. TODO: Implement Docker Compose configuration.' > examples/docker/docker-compose.yml

echo "📚 Creating notebook placeholders..."

# Create all 22 notebook placeholders
notebooks=(
    "01_tensorflow_foundations/01_tensors_operations_execution.ipynb"
    "01_tensorflow_foundations/02_data_pipelines_tfrecords.ipynb"
    "01_tensorflow_foundations/03_debugging_profiling.ipynb"
    "02_neural_networks_with_keras/04_keras_sequential_functional_apis.ipynb"
    "02_neural_networks_with_keras/05_keras_custom_layers_models.ipynb"
    "02_neural_networks_with_keras/06_keras_callbacks_optimization.ipynb"
    "03_computer_vision/07_cnn_architectures_keras.ipynb"
    "03_computer_vision/08_transfer_learning_applications.ipynb"
    "03_computer_vision/09_image_segmentation_keras.ipynb"
    "04_natural_language_processing/10_text_processing_keras_layers.ipynb"
    "04_natural_language_processing/11_transformers_attention_keras.ipynb"
    "04_natural_language_processing/12_nlp_applications_tfhub.ipynb"
    "05_generative_models/13_gans_with_tensorflow_keras.ipynb"
    "05_generative_models/14_vaes_advanced_gans_keras.ipynb"
    "05_generative_models/15_diffusion_models_keras.ipynb"
    "06_model_optimization/16_tensorflow_model_optimization.ipynb"
    "06_model_optimization/17_model_export_tflite_conversion.ipynb"
    "06_model_optimization/18_cross_platform_model_export.ipynb"
    "07_advanced_topics/19_distributed_training_strategies.ipynb"
    "07_advanced_topics/20_research_implementations_keras.ipynb"
    "capstone_projects/21_multimodal_ai_system.ipynb"
    "capstone_projects/22_end_to_end_ml_pipeline.ipynb"
)

for notebook in "${notebooks[@]}"; do
    notebook_path="notebooks/$notebook"
    cat > "$notebook_path" << EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# $(basename "$notebook" .ipynb | tr '_' ' ' | sed 's/\b\w/\u&/g')\n",
    "**Location: TensorVerseHub/$notebook_path**\n",
    "\n",
    "TODO: Implement comprehensive TensorFlow + tf.keras learning content.\n",
    "\n",
    "## Learning Objectives\n",
    "- TODO: Define specific learning objectives\n",
    "- TODO: List key TensorFlow concepts covered\n",
    "- TODO: Outline tf.keras integration points"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "print(f\"TensorFlow version: {tf.__version__}\")\n",
    "# TODO: Add comprehensive implementation"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8+"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
done

echo "📄 Creating documentation placeholders..."

# Documentation placeholders
touch docs/QUICK_REFERENCE.md
echo '# TensorFlow + tf.keras Quick Reference - TensorVerseHub/docs/QUICK_REFERENCE.md. TODO: Implement comprehensive reference guide.' > docs/QUICK_REFERENCE.md

touch docs/TENSORFLOW_KERAS_BEST_PRACTICES.md
echo '# TensorFlow + tf.keras Best Practices - TensorVerseHub/docs/TENSORFLOW_KERAS_BEST_PRACTICES.md. TODO: Implement best practices guide.' > docs/TENSORFLOW_KERAS_BEST_PRACTICES.md

touch docs/MODEL_OPTIMIZATION_GUIDE.md
echo '# Model Optimization Guide - TensorVerseHub/docs/MODEL_OPTIMIZATION_GUIDE.md. TODO: Implement optimization guide.' > docs/MODEL_OPTIMIZATION_GUIDE.md

touch docs/CONCEPT_MAP.md
echo '# TensorFlow Concept Map - TensorVerseHub/docs/CONCEPT_MAP.md. TODO: Implement visual learning progression.' > docs/CONCEPT_MAP.md

touch CONTRIBUTING.md
echo '# Contributing to TensorVerseHub - TODO: Implement contribution guidelines.' > CONTRIBUTING.md

# Data sample files
echo 'TensorVerseHub/data/sample_images/ - Sample image datasets for computer vision experiments. TODO: Add sample images and usage instructions.' > data/sample_images/readme.txt

echo 'TensorVerseHub/data/sample_text/ - Sample text datasets for NLP experiments. TODO: Add sample text data and usage instructions.' > data/sample_text/readme.txt

echo 'TensorVerseHub/data/tfrecords_examples/ - TFRecord examples for efficient data loading. TODO: Add TFRecord samples and utilities.' > data/tfrecords_examples/readme.txt

echo ""
echo "✅ TensorVerseHub Project Structure Generated Successfully!"
echo "=================================================================="
echo ""
echo "📋 What was created:"
echo "✅ Complete directory structure with proper organization"
echo "✅ Core configuration files (requirements.txt, setup.py, .gitignore)"
echo "✅ CI/CD pipeline with GitHub Actions"
echo "✅ Professional README and LICENSE"
echo "✅ Placeholder source modules (6 files) ready for implementation"
echo "✅ Placeholder test files (5 files) ready for implementation"
echo "✅ All 22 notebook placeholders with proper structure"
echo "✅ Example file placeholders for serving and optimization"
echo "✅ Documentation structure with placeholder guides"
echo ""
echo "🚀 Quick Start Commands:"
echo "cd TensorVerseHub"
echo "python -m venv venv"
echo "source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "pip install -r requirements.txt"
echo "pip install -e ."
echo "jupyter lab"
echo ""
echo "🎯 Next Steps:"
echo "1. Implement source modules in src/ directory"
echo "2. Fill in notebook content following the learning progression"
echo "3. Add comprehensive tests in tests/ directory"
echo "4. Create example applications in examples/"
echo "5. Complete documentation guides in docs/"
echo ""
echo "📊 Project Statistics:"
echo "- 22 notebook placeholders created"
echo "- 6 source utility modules ready for implementation"
echo "- 5 test modules ready for implementation"
echo "- Complete CI/CD and documentation structure"
echo "- Professional project organization"
echo ""
echo "Ready to implement TensorFlow mastery content! 🚀"