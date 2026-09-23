# Curriculum overview

27 notebooks arranged as a progressive path. Every notebook is self-contained and runs on CPU;
GPU/TPU sections are clearly marked. Launch with `jupyter lab` from the repository root.

| Track | Notebooks | Focus |
|---|---|---|
| [01 — TensorFlow Foundations](notebooks/01_foundations.md) | 3 | Tensors, eager vs graph execution, `tf.data` pipelines, TFRecords, debugging and profiling. |
| [02 — Neural Networks with Keras](notebooks/02_keras.md) | 3 | Sequential and functional APIs, custom layers and models, callbacks and training optimisation. |
| [03 — Computer Vision](notebooks/03_vision.md) | 3 | CNN architectures, transfer learning with `keras.applications`, image segmentation. |
| [04 — Natural Language Processing](notebooks/04_nlp.md) | 3 | Text preprocessing layers, attention and Transformers, TensorFlow Hub encoders. |
| [05 — Generative Models](notebooks/05_generative.md) | 3 | GANs, variational autoencoders and diffusion models built from scratch in Keras. |
| [06 — Model Optimization & Export](notebooks/06_optimization.md) | 3 | Quantization, pruning, distillation and cross-platform export (TFLite, ONNX, TF.js). |
| [07 — Advanced Topics](notebooks/07_advanced.md) | 2 | Distribution strategies and research-paper re-implementations. |
| [08 — Reinforcement Learning](notebooks/08_rl.md) | 1 | DQN, policy gradients and actor–critic on classic control tasks. |
| [Capstone Projects](notebooks/capstone.md) | 2 | End-to-end systems: a multimodal AI pipeline and a full MLOps workflow. |
| [Supplementary](notebooks/supplementary.md) | 4 | Federated learning, meta-learning / few-shot, time-series forecasting and Keras Tuner. |

## Suggested order

1. Foundations → Keras (notebooks 01–06)
2. One specialisation track: Vision, NLP or Generative (07–15)
3. Optimization & export (16–18) — pairs with `tensorversehub.optimization_utils` and `export_utils`
4. Advanced topics, RL and the capstones (19–23)
5. Supplementary notebooks as needed
