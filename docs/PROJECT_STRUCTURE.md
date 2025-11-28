.
├── .DS_Store
├── .github
│   └── workflows
│       ├── ci.yml
│       └── deploy-docs.yml
├── .gitignore
├── benchmarks
│   ├── inference_benchmarks.py
│   ├── memory_profiling.py
│   ├── results
│   └── training_benchmarks.py
├── CONTRIBUTING.md
├── create_tensorversehub.sh
├── data
│   ├── sample_images
│   │   ├── classification
│   │   ├── detection
│   │   └── readme.txt
│   ├── sample_text
│   │   └── readme.txt
│   ├── synthetic
│   └── tfrecords_examples
│       └── readme.txt
├── docs
│   ├── assets
│   │   ├── architecture_diagrams
│   │   │   ├── distributed_training_strategies.svg
│   │   │   ├── model_optimization_techniques.svg
│   │   │   ├── neural_network_architectures.svg
│   │   │   ├── tensorflow_deployment_architecture.svg
│   │   │   ├── tensorflow_ecosystem_overview.svg
│   │   │   └── tensorflow_keras_architecture.svg
│   │   ├── screenshots
│   │   └── workflow_diagrams
│   │       ├── mlops_pipeline_workflow.svg
│   │       ├── model_evaluation_workflow.svg
│   │       ├── tensorflow_data_pipeline_workflow.svg
│   │       └── tensorflow_training_workflow.svg
│   ├── CONCEPT_MAP.md
│   ├── MODEL_OPTIMIZATION_GUIDE.md
│   ├── QUICK_REFERENCE.md
│   ├── TENSORFLOW_KERAS_BEST_PRACTICES.md
│   └── TROUBLESHOOTING.md
├── examples
│   ├── docker
│   │   ├── docker-compose.yml
│   │   └── Dockerfile
│   ├── optimization_examples
│   │   ├── distillation_demo.py
│   │   ├── pruning_demo.py
│   │   └── quantization_demo.py
│   └── serving_examples
│       ├── flask_tensorflow_api.py
│       ├── streamlit_tensorflow_demo.py
│       └── tflite_inference_example.py
├── LICENSE
├── logs
│   ├── experiment_configs
│   ├── tensorboard
│   │   ├── capstone_projects
│   │   ├── cnn_experiments
│   │   └── transformer_experiments
│   └── training_logs
├── models
│   ├── checkpoints
│   │   ├── cnn_model
│   │   ├── multimodal_model
│   │   └── transformer_model
│   ├── onnx
│   ├── saved_models
│   │   ├── image_classifier
│   │   │   └── 1
│   │   ├── multimodal_system
│   │   │   └── 1
│   │   └── text_classifier
│   │       └── 1
│   └── tflite
├── notebooks
│   ├── 01_tensorflow_foundations
│   │   ├── 01_tensors_operations_execution.ipynb
│   │   ├── 02_data_pipelines_tfrecords.ipynb
│   │   └── 03_debugging_profiling.ipynb
│   ├── 02_neural_networks_with_keras
│   │   ├── 04_keras_sequential_functional_apis.ipynb
│   │   ├── 05_keras_custom_layers_models.ipynb
│   │   └── 06_keras_callbacks_optimization.ipynb
│   ├── 03_computer_vision
│   │   ├── 07_cnn_architectures_keras.ipynb
│   │   ├── 08_transfer_learning_applications.ipynb
│   │   └── 09_image_segmentation_keras.ipynb
│   ├── 04_natural_language_processing
│   │   ├── 10_text_processing_keras_layers.ipynb
│   │   ├── 11_transformers_attention_keras.ipynb
│   │   └── 12_nlp_applications_tfhub.ipynb
│   ├── 05_generative_models
│   │   ├── 13_gans_with_tensorflow_keras.ipynb
│   │   ├── 14_vaes_advanced_gans_keras.ipynb
│   │   └── 15_diffusion_models_keras.ipynb
│   ├── 06_model_optimization
│   │   ├── 16_tensorflow_model_optimization.ipynb
│   │   ├── 17_model_export_tflite_conversion.ipynb
│   │   └── 18_cross_platform_model_export.ipynb
│   ├── 07_advanced_topics
│   │   ├── 19_distributed_training_strategies.ipynb
│   │   └── 20_research_implementations_keras.ipynb
│   └── capstone_projects
│       ├── 21_multimodal_ai_system.ipynb
│       └── 22_end_to_end_ml_pipeline.ipynb
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── scripts
│   ├── convert_models.py
│   ├── evaluate_models.py
│   ├── generate_docs.py
│   ├── setup_environment.sh
│   └── train_models.py
├── setup.py
├── src
│   ├── __init__.py
│   ├── data_utils.py
│   ├── export_utils.py
│   ├── model_utils.py
│   ├── optimization_utils.py
│   └── visualization.py
└── tests
    ├── __init__.py
    ├── test_data_utils.py
    ├── test_integration.py
    ├── test_model_utils.py
    ├── test_notebooks.py
    ├── test_optimization.py
    └── test_tensorflow_keras_layers.py

54 directories, 80 files
