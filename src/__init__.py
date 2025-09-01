# Location: /src/__init__.py

"""
TensorVerseHub - Comprehensive TensorFlow learning hub with tf.keras integration.

This package provides utilities, models, and tools for TensorFlow 2.15+ development
with a focus on practical machine learning implementation and deployment.
"""

import tensorflow as tf

# Version information
__version__ = "1.0.0"
__author__ = "TensorVerseHub Contributors"
__email__ = "contact@tensorversehub.com"

# Ensure minimum TensorFlow version
MIN_TF_VERSION = "2.15.0"
if tf.__version__ < MIN_TF_VERSION:
    raise ImportError(
        f"TensorVerseHub requires TensorFlow >= {MIN_TF_VERSION}, "
        f"but found version {tf.__version__}. Please upgrade TensorFlow."
    )

# Import main modules
from . import data_utils
from . import model_utils
from . import visualization
from . import optimization_utils
from . import export_utils

# Import key classes and functions for convenient access
from .data_utils import (
    DataPipeline,
    TFRecordHandler,
    DataAugmentation,
    create_image_classification_pipeline,
    create_text_classification_pipeline
)

from .model_utils import (
    ModelBuilders,
    CustomLayers,
    TrainingUtilities,
    ModelAnalysis,
    create_classification_model,
    create_transfer_learning_model
)

from .visualization import (
    ModelVisualization,
    TrainingVisualization,
    DataVisualization,
    AdvancedVisualization,
    quick_model_analysis,
    setup_plotting_style
)

from .optimization_utils import (
    ModelQuantization,
    ModelPruning,
    KnowledgeDistillation,
    MixedPrecisionOptimization,
    optimize_for_mobile,
    create_inference_optimized_model
)

from .export_utils import (
    SavedModelExporter,
    TFLiteExporter,
    ONNXExporter,
    TensorFlowJSExporter,
    MultiFormatExporter,
    quick_export,
    create_deployment_package
)

# Configure TensorFlow settings for optimal performance
def configure_tensorflow(memory_growth: bool = True,
                         mixed_precision: bool = False,
                         xla: bool = False) -> None:
    """
    Configure TensorFlow settings for optimal performance.
    
    Args:
        memory_growth: Enable GPU memory growth
        mixed_precision: Enable mixed precision training
        xla: Enable XLA compilation
    """
    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s) with memory growth: {memory_growth}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Mixed precision
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    
    # XLA compilation
    if xla:
        tf.config.optimizer.set_jit(True)
        print("XLA compilation enabled")


# Auto-configure TensorFlow on import
configure_tensorflow()

# Set up plotting style
try:
    setup_plotting_style()
except Exception:
    pass  # Ignore if matplotlib not available

# Package information
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Configuration
    'configure_tensorflow',
    
    # Data utilities
    'DataPipeline',
    'TFRecordHandler', 
    'DataAugmentation',
    'create_image_classification_pipeline',
    'create_text_classification_pipeline',
    
    # Model utilities
    'ModelBuilders',
    'CustomLayers',
    'TrainingUtilities',
    'ModelAnalysis',
    'create_classification_model',
    'create_transfer_learning_model',
    
    # Visualization
    'ModelVisualization',
    'TrainingVisualization',
    'DataVisualization', 
    'AdvancedVisualization',
    'quick_model_analysis',
    'setup_plotting_style',
    
    # Optimization
    'ModelQuantization',
    'ModelPruning',
    'KnowledgeDistillation',
    'MixedPrecisionOptimization',
    'optimize_for_mobile',
    'create_inference_optimized_model',
    
    # Export utilities
    'SavedModelExporter',
    'TFLiteExporter', 
    'ONNXExporter',
    'TensorFlowJSExporter',
    'MultiFormatExporter',
    'quick_export',
    'create_deployment_package'
]

print(f"TensorVerseHub v{__version__} loaded successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Display GPU information
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("No GPUs detected - using CPU")

# Check for additional TensorFlow components
try:
    import tensorflow_hub
    print(f"TensorFlow Hub available: {tensorflow_hub.__version__}")
except ImportError:
    pass

try:
    import tensorflow_model_optimization
    print(f"TensorFlow Model Optimization available: {tensorflow_model_optimization.__version__}")
except ImportError:
    pass

try:
    import tensorflowjs
    print(f"TensorFlow.js available: {tensorflowjs.__version__}")
except ImportError:
    pass

print("Ready for TensorFlow development! 🚀")