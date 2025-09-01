#!/bin/bash

# Location: /scripts/setup_environment.sh

# TensorVerseHub Environment Setup Script
# Automated setup for TensorFlow development environment

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    log_info "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Detected macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        log_info "Detected Windows (Git Bash/Cygwin)"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    log_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -ge 8 ]]; then
        log_success "Python $PYTHON_VERSION detected (compatible)"
    else
        log_error "Python 3.8+ required, found Python $PYTHON_VERSION"
        exit 1
    fi
}

# Check pip installation
check_pip() {
    log_info "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    else
        log_error "pip is not installed. Please install pip first."
        exit 1
    fi
    
    # Upgrade pip
    log_info "Upgrading pip..."
    $PIP_CMD install --upgrade pip
    log_success "pip upgraded successfully"
}

# Check CUDA installation (optional)
check_cuda() {
    log_info "Checking CUDA installation..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log_success "CUDA $CUDA_VERSION detected"
        
        # Check if CUDA version is compatible with TensorFlow
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)
        
        if [[ $CUDA_MAJOR -ge 11 ]] && [[ $CUDA_MINOR -ge 8 ]]; then
            log_success "CUDA version is compatible with TensorFlow 2.15+"
            CUDA_AVAILABLE=true
        else
            log_warning "CUDA version may not be compatible. TensorFlow 2.15+ requires CUDA 11.8+"
            CUDA_AVAILABLE=false
        fi
    else
        log_warning "CUDA not detected. Will install CPU-only TensorFlow."
        CUDA_AVAILABLE=false
    fi
}

# Create virtual environment
create_venv() {
    log_info "Setting up Python virtual environment..."
    
    VENV_NAME="tensorverse_env"
    
    if [[ -d "$VENV_NAME" ]]; then
        log_warning "Virtual environment '$VENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            log_info "Removed existing virtual environment"
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_NAME"
    log_success "Virtual environment created: $VENV_NAME"
    
    # Activation instructions
    if [[ "$OS" == "windows" ]]; then
        ACTIVATE_CMD="source $VENV_NAME/Scripts/activate"
    else
        ACTIVATE_CMD="source $VENV_NAME/bin/activate"
    fi
    
    log_info "To activate the environment, run: $ACTIVATE_CMD"
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Check if we're in virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        log_warning "Not in virtual environment. Installing system-wide..."
        read -p "Continue? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Installation cancelled"
            exit 1
        fi
    fi
    
    # Install TensorFlow
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        log_info "Installing TensorFlow with GPU support..."
        $PIP_CMD install "tensorflow[and-cuda]>=2.15.0"
    else
        log_info "Installing CPU-only TensorFlow..."
        $PIP_CMD install "tensorflow>=2.15.0"
    fi
    
    # Install main requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing requirements from requirements.txt..."
        $PIP_CMD install -r requirements.txt
    else
        log_warning "requirements.txt not found. Installing core dependencies..."
        
        # Core dependencies
        $PIP_CMD install \
            numpy>=1.24.0 \
            pandas>=2.0.0 \
            matplotlib>=3.7.0 \
            seaborn>=0.12.0 \
            pillow>=10.0.0 \
            scikit-learn>=1.3.0 \
            jupyter>=1.0.0 \
            jupyterlab>=4.0.0 \
            tensorboard>=2.15.0
    fi
    
    log_success "Dependencies installed successfully"
}

# Install TensorVerseHub package
install_package() {
    log_info "Installing TensorVerseHub package..."
    
    if [[ -f "setup.py" ]]; then
        $PIP_CMD install -e .
        log_success "TensorVerseHub installed in development mode"
    else
        log_warning "setup.py not found. Skipping package installation."
    fi
}

# Setup Jupyter Lab
setup_jupyter() {
    log_info "Configuring Jupyter Lab..."
    
    # Generate Jupyter config
    jupyter lab --generate-config 2>/dev/null || true
    
    # Create Jupyter config directory if it doesn't exist
    JUPYTER_DIR="$HOME/.jupyter"
    mkdir -p "$JUPYTER_DIR"
    
    # Configure Jupyter Lab settings
    cat > "$JUPYTER_DIR/jupyter_lab_config.py" << EOF
# TensorVerseHub Jupyter Lab Configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.allow_root = False
c.ServerApp.token = ''
c.ServerApp.password = ''
c.LabApp.open_browser = True
EOF
    
    # Install Jupyter extensions
    log_info "Installing Jupyter Lab extensions..."
    $PIP_CMD install \
        jupyterlab-git \
        jupyterlab-lsp \
        jupyter-lsp \
        ipywidgets
    
    log_success "Jupyter Lab configured successfully"
}

# Create project directories
create_directories() {
    log_info "Creating project directories..."
    
    DIRS=(
        "data/sample_images/classification"
        "data/sample_images/detection"
        "data/sample_text"
        "data/tfrecords_examples"
        "data/synthetic"
        "models/checkpoints"
        "models/saved_models"
        "models/tflite"
        "models/onnx"
        "logs/tensorboard"
        "logs/training_logs"
        "logs/experiment_configs"
        "benchmarks/results"
    )
    
    for dir in "${DIRS[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "Project directories created"
}

# Setup TensorBoard
setup_tensorboard() {
    log_info "Setting up TensorBoard..."
    
    # Create TensorBoard log directories
    mkdir -p logs/tensorboard/{cnn_experiments,transformer_experiments,capstone_projects}
    
    # Create TensorBoard startup script
    cat > "start_tensorboard.sh" << EOF
#!/bin/bash
# TensorBoard startup script
tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006 --reload_interval=1
EOF
    
    chmod +x start_tensorboard.sh
    log_success "TensorBoard configured"
}

# Download sample data
download_sample_data() {
    log_info "Setting up sample datasets..."
    
    # Create sample text data
    cat > "data/sample_text/movie_reviews.txt" << EOF
This movie is absolutely fantastic! Amazing acting and storyline.
Terrible film, waste of time and money. Poor acting throughout.
Average movie, nothing special but watchable on a weekend.
Brilliant cinematography and outstanding performances by the cast.
Boring and predictable plot, couldn't keep me engaged.
EOF
    
    # Create sample configuration files
    cat > "data/sample_text/readme.txt" << EOF
Sample Text Dataset
==================

This directory contains sample text data for classification tasks:

- movie_reviews.txt: Sample movie reviews for sentiment analysis
- news_articles.txt: Sample news articles for topic classification  
- qa_pairs.txt: Question-answering pairs for QA tasks

Use these files to test text preprocessing pipelines and NLP models.
EOF
    
    # Create synthetic data
    cat > "data/synthetic/generate_data.py" << EOF
import numpy as np
import pandas as pd

# Generate synthetic time series data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=1000, freq='D')
values = np.cumsum(np.random.randn(1000)) + 100
timeseries_df = pd.DataFrame({'date': dates, 'value': values})
timeseries_df.to_csv('data/synthetic/timeseries.csv', index=False)

# Generate synthetic tabular data
features = np.random.randn(1000, 10)
target = (features[:, 0] + features[:, 1] * 0.5 + np.random.randn(1000) * 0.1)
tabular_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(10)])
tabular_df['target'] = target
tabular_df.to_csv('data/synthetic/tabular_data.csv', index=False)

print("Synthetic datasets generated successfully!")
EOF
    
    # Generate the synthetic data
    if command -v python3 &> /dev/null; then
        python3 data/synthetic/generate_data.py
    fi
    
    log_success "Sample data configured"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test TensorFlow import
    $PYTHON_CMD -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
    
    # Test GPU availability
    GPU_AVAILABLE=$($PYTHON_CMD -c "import tensorflow as tf; print(len(tf.config.experimental.list_physical_devices('GPU')) > 0)" 2>/dev/null)
    
    if [[ "$GPU_AVAILABLE" == "True" ]]; then
        log_success "GPU support available"
    else
        log_info "Running in CPU-only mode"
    fi
    
    # Test package imports
    $PYTHON_CMD -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
print('All core packages imported successfully')
"
    
    log_success "Installation verification completed"
}

# Create startup scripts
create_scripts() {
    log_info "Creating convenience scripts..."
    
    # Jupyter Lab startup script
    cat > "start_jupyter.sh" << EOF
#!/bin/bash
# Start Jupyter Lab for TensorVerseHub
echo "Starting Jupyter Lab for TensorVerseHub..."
echo "Access at: http://localhost:8888"
source tensorverse_env/bin/activate 2>/dev/null || source tensorverse_env/Scripts/activate 2>/dev/null
jupyter lab --port=8888 --no-browser --allow-root
EOF
    
    # Model training script
    cat > "train_model.sh" << EOF
#!/bin/bash
# Quick model training script
echo "Starting model training..."
source tensorverse_env/bin/activate 2>/dev/null || source tensorverse_env/Scripts/activate 2>/dev/null
python scripts/train_models.py "\$@"
EOF
    
    # Environment activation script
    if [[ "$OS" == "windows" ]]; then
        ACTIVATE_PATH="tensorverse_env/Scripts/activate"
    else
        ACTIVATE_PATH="tensorverse_env/bin/activate"
    fi
    
    cat > "activate_env.sh" << EOF
#!/bin/bash
# Activate TensorVerseHub environment
echo "Activating TensorVerseHub environment..."
source $ACTIVATE_PATH
echo "Environment activated! TensorFlow \$(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo ""
echo "Available commands:"
echo "  jupyter lab          - Start Jupyter Lab"
echo "  tensorboard --logdir=logs/tensorboard - Start TensorBoard"
echo "  python scripts/train_models.py - Train models"
echo ""
EOF
    
    # Make scripts executable
    chmod +x start_jupyter.sh train_model.sh start_tensorboard.sh activate_env.sh
    
    log_success "Convenience scripts created"
}

# Print final instructions
print_instructions() {
    log_success "🎉 TensorVerseHub setup completed successfully!"
    
    echo ""
    echo "📋 Setup Summary:"
    echo "  ✅ Python $PYTHON_VERSION verified"
    echo "  ✅ Virtual environment created: tensorverse_env"
    echo "  ✅ TensorFlow installed with $([ "$CUDA_AVAILABLE" == true ] && echo "GPU" || echo "CPU") support"
    echo "  ✅ All dependencies installed"
    echo "  ✅ Project directories created"
    echo "  ✅ Jupyter Lab configured"
    echo "  ✅ Sample data prepared"
    echo ""
    echo "🚀 Quick Start:"
    echo "  1. Activate environment:  source activate_env.sh"
    echo "  2. Start Jupyter Lab:     ./start_jupyter.sh"
    echo "  3. Start TensorBoard:     ./start_tensorboard.sh"
    echo "  4. Open browser:          http://localhost:8888 (Jupyter) | http://localhost:6006 (TensorBoard)"
    echo ""
    echo "📚 Next Steps:"
    echo "  • Explore notebooks in notebooks/ directory"
    echo "  • Check out examples in examples/ directory"
    echo "  • Run tests with: pytest tests/"
    echo "  • Read documentation in docs/ directory"
    echo ""
    echo "💡 Useful Commands:"
    echo "  • Train models:          python scripts/train_models.py"
    echo "  • Evaluate models:       python scripts/evaluate_models.py"
    echo "  • Convert models:        python scripts/convert_models.py"
    echo "  • Start Flask API:       python examples/serving_examples/flask_tensorflow_api.py"
    echo ""
    echo "🐳 Docker Alternative:"
    echo "  • Build container:       docker-compose up --build"
    echo "  • Development mode:      docker-compose -f docker-compose.yml -f examples/docker/docker-compose.yml up"
    echo ""
    log_success "Happy learning with TensorVerseHub! 🤖"
}

# Main installation function
main() {
    echo "🚀 TensorVerseHub Environment Setup"
    echo "==================================="
    echo ""
    
    # Parse command line arguments
    SKIP_VENV=false
    SKIP_DEPS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-venv)
                SKIP_VENV=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-venv    Skip virtual environment creation"
                echo "  --skip-deps    Skip dependency installation"
                echo "  --help, -h     Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_os
    check_python
    check_pip
    check_cuda
    
    if [[ "$SKIP_VENV" == false ]]; then
        create_venv
    fi
    
    if [[ "$SKIP_DEPS" == false ]]; then
        install_dependencies
        install_package
    fi
    
    setup_jupyter
    create_directories
    setup_tensorboard
    download_sample_data
    verify_installation
    create_scripts
    print_instructions
}

# Run main function
main "$@"