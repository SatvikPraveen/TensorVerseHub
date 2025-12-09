# TensorVerseHub

![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Keras](https://img.shields.io/badge/Keras-3.0%2B-red?logo=keras)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab%20%26%20Notebook-orange?logo=jupyter)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?logo=windows)
![Issues](https://img.shields.io/github/issues/SatvikPraveen/TensorVerseHub?color=red)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

A comprehensive TensorFlow learning hub featuring 27 high-quality Jupyter notebooks, production-ready utilities, and practical examples. Built with TensorFlow 2.13+ and tf.keras, this repository serves as a complete reference for learning TensorFlow from fundamentals to advanced implementations.

## 🎯 Overview

TensorVerseHub is a **focused, production-grade learning resource** for TensorFlow practitioners. It combines theoretical understanding with hands-on implementations through a progressive curriculum spanning computer vision, NLP, generative modeling, reinforcement learning, and advanced optimization techniques.

### Key Highlights

- **27 Comprehensive Notebooks** - Core curriculum (23) + advanced supplementary materials (4)
- **Multi-Domain Coverage** - Computer vision, NLP, GANs, diffusion models, RL, time series, federated learning
- **Production Utilities** - Ready-to-use helper functions in `src/` folder
- **Practical Deployment Examples** - Flask APIs, Streamlit demos, TFLite inference
- **Model Optimization** - Quantization, pruning, distillation with real examples
- **Quality Testing** - Automated tests ensuring code reliability
- **Essential Documentation** - Best practices, quick reference, troubleshooting guides

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended for advanced models)
- 50GB+ storage space

### Installation

```bash
# Clone the repository
git clone https://github.com/SatvikPraveen/TensorVerseHub.git
cd TensorVerseHub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"

# Launch Jupyter
jupyter lab
```

### Quick Navigation

Start your learning journey:

1. **Fundamentals** → `notebooks/01_tensorflow_foundations/01_tensors_operations_execution.ipynb`
2. **Keras Models** → `notebooks/02_neural_networks_with_keras/04_keras_sequential_functional_apis.ipynb`
3. **Your Domain** → Choose Computer Vision, NLP, or Generative Models
4. **Optimization** → `notebooks/06_model_optimization/16_tensorflow_model_optimization.ipynb`
5. **Capstone** → `notebooks/capstone_projects/22_end_to_end_ml_pipeline.ipynb`

## 📚 Learning Curriculum

### 🔰 Foundation Track (Notebooks 1-6)

**TensorFlow Fundamentals**

- `01_tensors_operations_execution.ipynb` - Core tensor operations and eager execution
- `02_data_pipelines_tfrecords.ipynb` - Efficient data loading with tf.data and TFRecords
- `03_debugging_profiling.ipynb` - Performance optimization and debugging techniques

**Keras Deep Learning**

- `04_keras_sequential_functional_apis.ipynb` - Model building paradigms
- `05_keras_custom_layers_models.ipynb` - Custom component development
- `06_keras_callbacks_optimization.ipynb` - Training optimization strategies

### 🎯 Specialization Track (Notebooks 7-15)

**Computer Vision (Notebooks 7-9)**

- `07_cnn_architectures_keras.ipynb` - CNN design patterns and implementations
- `08_transfer_learning_applications.ipynb` - Pre-trained model utilization
- `09_image_segmentation_keras.ipynb` - Pixel-level classification techniques

**Natural Language Processing (Notebooks 10-12)**

- `10_text_processing_keras_layers.ipynb` - Text preprocessing and embedding layers
- `11_transformers_attention_keras.ipynb` - Attention mechanisms and transformer architectures
- `12_nlp_applications_tfhub.ipynb` - TensorFlow Hub integration for NLP tasks

**Generative Modeling (Notebooks 13-15)**

- `13_gans_with_tensorflow_keras.ipynb` - Generative Adversarial Network implementations
- `14_vaes_advanced_gans_keras.ipynb` - Variational Autoencoders and advanced GAN variants
- `15_diffusion_models_keras.ipynb` - State-of-the-art diffusion model architectures

### 🚢 Production Track (Notebooks 16-22)

**Model Optimization (Notebooks 16-18)**

- `16_tensorflow_model_optimization.ipynb` - Quantization, pruning, and distillation
- `17_model_export_tflite_conversion.ipynb` - Mobile and edge deployment
- `18_cross_platform_model_export.ipynb` - ONNX and multi-framework compatibility

**Advanced Implementation (Notebooks 19-20)**

- `19_distributed_training_strategies.ipynb` - Multi-GPU and TPU training
- `20_research_implementations_keras.ipynb` - Cutting-edge research reproduction

**Capstone Projects (Notebooks 21-22)**

- `21_multimodal_ai_system.ipynb` - Vision-language model integration
- `22_end_to_end_ml_pipeline.ipynb` - Complete MLOps workflow implementation

### 🎓 Advanced Topics (Notebook 23)

**Reinforcement Learning**

- `23_rl_fundamentals_keras.ipynb` - Deep Q-Learning, Policy Gradients, Actor-Critic algorithms with CartPole examples

### 📖 Supplementary Materials

**Advanced Topics Beyond Core Curriculum**

- `meta_learning_fewshot.ipynb` - Few-shot learning, Siamese networks, Prototypical networks, metric learning
- `time_series_forecasting.ipynb` - LSTM and Transformer-based time series prediction with preprocessing pipelines
- `federated_learning.ipynb` - Privacy-preserving federated learning, differential privacy, distributed training
- `reinforcement_learning.ipynb` - Comprehensive RL algorithms (in development)

## 🏗️ Project Structure

```
TensorVerseHub/
├── notebooks/                    # 27 comprehensive learning notebooks
│   ├── 01_tensorflow_foundations/        # Tensors, operations, execution modes
│   ├── 02_neural_networks_with_keras/   # Model building paradigms
│   ├── 03_computer_vision/               # CNNs, transfer learning, segmentation
│   ├── 04_natural_language_processing/  # Text processing, transformers, NLP
│   ├── 05_generative_models/            # GANs, VAEs, diffusion models
│   ├── 06_model_optimization/           # Quantization, pruning, distillation
│   ├── 07_advanced_topics/              # Distributed training, research implementations
│   ├── 08_reinforcement_learning/       # Deep Q-Learning, Policy Gradients, Actor-Critic
│   ├── capstone_projects/               # End-to-end ML pipelines
│   └── supplementary/                   # Advanced topics beyond core curriculum
│       ├── federated_learning.ipynb     # Privacy-preserving distributed learning
│       ├── meta_learning_fewshot.ipynb  # Few-shot learning with metric learning
│       ├── time_series_forecasting.ipynb # LSTM/Transformer time series
│       └── reinforcement_learning.ipynb # Advanced RL implementations
├── src/                          # Production utility modules
│   ├── data_utils.py            # Data preprocessing and tf.data pipelines
│   ├── model_utils.py           # Model building and architecture helpers
│   ├── optimization_utils.py    # Model compression utilities
│   ├── export_utils.py          # Multi-format model export (SavedModel, TFLite, ONNX)
│   └── visualization.py         # Visualization and metrics plotting
├── examples/                     # Practical deployment patterns
│   ├── serving_examples/        # Flask API, Streamlit, TFLite inference
│   └── optimization_examples/   # Quantization, pruning, distillation demos
├── docs/                         # Essential documentation
│   ├── QUICK_REFERENCE.md       # Essential TensorFlow commands and APIs
│   ├── TENSORFLOW_KERAS_BEST_PRACTICES.md  # Production coding standards
│   ├── MODEL_OPTIMIZATION_GUIDE.md         # Compression techniques
│   ├── TROUBLESHOOTING.md       # Common issues and solutions
│   └── assets/                  # Architecture diagrams
├── tests/                        # Automated testing suite
│   ├── test_model_utils.py
│   ├── test_data_utils.py
│   ├── test_optimization.py
│   └── test_notebooks.py        # Notebook execution verification
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation config
└── LICENSE                      # MIT License
```

## 🔧 Core Features

### Complete Learning Path

| Track | Notebooks | Coverage |
|-------|-----------|----------|
| **Fundamentals** | 01-06 | Tensors, Keras APIs, optimization |
| **Computer Vision** | 07-09 | CNNs, transfer learning, segmentation |
| **Natural Language Processing** | 10-12 | Text processing, transformers, embeddings |
| **Generative Models** | 13-15 | GANs, VAEs, diffusion models |
| **Optimization** | 16-18 | Quantization, pruning, export formats |
| **Advanced** | 19-22 | Distributed training, research, capstones |
| **Reinforcement Learning** | 23 | DQN, policy gradients, actor-critic |
| **Supplementary** | 24-27 | Federated learning, meta-learning, time series |

### Production Utilities

**Data Pipelines**
```python
from src.data_utils import create_tfrecord_dataset, augmentation_pipeline

# Efficient data loading
dataset = create_tfrecord_dataset(
    tfrecord_paths=["data/train/*.tfrecord"],
    batch_size=32,
    shuffle_buffer=10000
)
```

**Model Optimization**
```python
from src.optimization_utils import quantize_model_int8, prune_structured

# Post-training quantization
quantized = quantize_model_int8(model, representative_data)

# Structured pruning
pruned = prune_structured(model, sparsity=0.5)
```

**Multi-Format Export**
```python
from src.export_utils import export_multi_format

# Export to SavedModel, TFLite, ONNX
export_multi_format(
    model=trained_model,
    output_dir="models/exports/",
    formats=["savedmodel", "tflite", "onnx"]
)
```

### Deployment Examples

- **Flask API** - RESTful endpoints for model serving
- **Streamlit** - Interactive web interfaces for demos
- **TFLite** - Mobile and edge device inference
- **Optimization** - Quantization and pruning demonstrations

## 🧪 Quality Assurance

Automated tests ensure reliability and code quality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_model_utils.py -v

# Test notebook execution
python -m pytest tests/test_notebooks.py -v
```

## 📚 Documentation

Essential guides for learning and production use:

- **[Quick Reference Guide](docs/QUICK_REFERENCE.md)** - Essential TensorFlow commands and APIs
- **[Best Practices](docs/TENSORFLOW_KERAS_BEST_PRACTICES.md)** - Production-ready coding standards
- **[Model Optimization](docs/MODEL_OPTIMIZATION_GUIDE.md)** - Compression and acceleration techniques
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Why This Project?

TensorVerseHub provides:

✅ **Complete curriculum** - From basics to research-grade implementations  
✅ **Production focus** - Real-world patterns and best practices  
✅ **Multi-domain** - Computer vision, NLP, generative models, RL, time series  
✅ **Hands-on learning** - Executable notebooks with real datasets  
✅ **Modern stack** - TensorFlow 2.13+, tf.keras, and latest ML research  
✅ **Practical utilities** - Copy-paste ready code for common tasks  
✅ **Portfolio value** - Demonstrates mastery across TensorFlow ecosystem  

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/SatvikPraveen/TensorVerseHub/issues)
- **Questions**: [GitHub Discussions](https://github.com/SatvikPraveen/TensorVerseHub/discussions)

---

**Built with ❤️ for the TensorFlow and machine learning community.**
