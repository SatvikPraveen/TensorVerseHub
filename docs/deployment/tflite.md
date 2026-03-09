# TFLite Inference

TFLite models are optimised for edge devices and mobile platforms. TensorVerseHub provides utilities to export, benchmark, and run TFLite models.

---

## Converting a Model

```python
from src.optimization_utils import ModelQuantization

# Post-training INT8 quantization
tflite_bytes = ModelQuantization.quantize_model_post_training(
    model=my_model,
    representative_dataset=calibration_ds,
    optimization_type="int8",
)

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_bytes)
```

Or with the CLI:

```bash
tensorverse-convert --model ./models/final_model --to tflite --quantize int8
```

---

## Running Inference

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
sample = np.expand_dims(image, axis=0).astype(input_details[0]["dtype"])
interpreter.set_tensor(input_details[0]["index"], sample)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
```

---

## Benchmarking

```bash
tensorverse-convert \
  --model ./models/final_model \
  --to tflite \
  --quantize int8 \
  --benchmark
# → TFLite benchmark: 3.21 ms/inference (avg over 50 runs)
```

---

## Size Comparison

| Format | Typical Size | Latency |
|--------|-------------|---------|
| SavedModel (fp32) | ~100 MB | baseline |
| TFLite default | ~45 MB | ~1.5× faster |
| TFLite float16 | ~50 MB | ~1.3× faster |
| TFLite int8 | ~25 MB | ~2–4× faster |
