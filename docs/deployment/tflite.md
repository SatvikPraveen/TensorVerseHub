# TFLite Inference

TFLite models are optimised for edge devices and mobile platforms. TensorVerseHub provides utilities to export, validate, benchmark, and run TFLite models. Post-training quantisation works on Keras 3 out of the box — no legacy Keras stack needed.

---

## Converting a Model

```python
from tensorversehub.compat import load_model
from tensorversehub.export_utils import TFLiteExporter

model = load_model("models/final_model.keras")

# Post-training INT8 quantization (representative dataset required for int8)
result = TFLiteExporter.export_tflite(
    model,
    "model_int8.tflite",
    quantization_type="int8",          # "float32" | "dynamic" | "float16" | "int8"
    representative_dataset=calibration_ds,
)
print(result["size_mb"])
```

If you want the raw bytes instead of a file, `ModelQuantization.quantize_model_post_training(model, calibration_ds, "int8")` in `tensorversehub.optimization_utils` returns them.

Or with the CLI:

```bash
tensorverse convert --model ./models/final_model.keras --to tflite --quantize int8
```

---

## Running Inference

```python
import numpy as np
from tensorversehub.export_utils import make_interpreter

# Uses the standalone LiteRT runtime (`pip install ai-edge-litert`) when installed,
# otherwise falls back to tf.lite.Interpreter.
interpreter = make_interpreter("model_int8.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
sample = np.expand_dims(image, axis=0).astype(input_details[0]["dtype"])
interpreter.set_tensor(input_details[0]["index"], sample)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
```

`TFLiteExporter.run_tflite(interpreter, batch)` wraps the set/invoke/get sequence and handles int8 input/output scaling for you.

---

## Validating and Benchmarking

```python
# Compare Keras vs TFLite predictions on a few samples
report = TFLiteExporter.validate_tflite(model, "model_int8.tflite", sample_inputs)
print(report["max_abs_diff"], report["argmax_agreement"])

# Latency statistics
stats = TFLiteExporter.benchmark_tflite_model("model_int8.tflite", sample_inputs[:1], num_runs=100)
```

```bash
tensorverse convert \
  --model ./models/final_model.keras \
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
