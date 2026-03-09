#!/usr/bin/env python3
"""
tensorverse-serve  — CLI for serving TensorFlow models via FastAPI.

Usage
-----
    tensorverse-serve --model ./models/my_model
    tensorverse-serve --model ./models/my_model.tflite --host 0.0.0.0 --port 8000
    tensorverse-serve --model ./models/my_model.h5 --class-names cat dog bird
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("tensorverse-serve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorVerseHub — Model Serving CLI (FastAPI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to a SavedModel directory, .h5 file, or .tflite file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1)",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        metavar="CLASS",
        help="Optional list of class label names for classification models",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="Expected input shape (excluding batch dim), e.g. --input-shape 224 224 3",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum allowed batch size per request (default: 32)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change (development only)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )
    return parser.parse_args()


def _detect_model_type(model_path: str) -> str:
    """Infer model format from path."""
    p = Path(model_path)
    if p.suffix == ".tflite":
        return "tflite"
    if p.suffix in (".h5", ".keras"):
        return "keras"
    if p.is_dir() or (p / "saved_model.pb").exists():
        return "saved_model"
    # Fallback: try loading as SavedModel
    return "saved_model"


def build_app(model_path: str, class_names: Optional[List[str]],
              input_shape: Optional[List[int]], max_batch_size: int):
    """
    Build and return the FastAPI application.

    Separated from main() so it can be imported by uvicorn directly
    via --factory mode and by unit tests.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import numpy as np
    except ImportError as exc:
        logger.error("fastapi or numpy not installed: %s", exc)
        logger.error("Install with: pip install fastapi uvicorn numpy")
        sys.exit(1)

    model_type = _detect_model_type(model_path)
    logger.info("Loading model  : %s  (type=%s)", model_path, model_type)

    # ── Load model ────────────────────────────────────────────────────────────
    model = None
    tflite_interpreter = None

    if model_type == "tflite":
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("tensorflow not installed")
            sys.exit(1)
        tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
        tflite_interpreter.allocate_tensors()
        input_details  = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        logger.info("TFLite model loaded. Input shape: %s", input_details[0]["shape"])
    else:
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("tensorflow not installed")
            sys.exit(1)
        if model_type == "keras":
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.saved_model.load(model_path)
        logger.info("Model loaded successfully.")

    # ── Server metadata ───────────────────────────────────────────────────────
    server_start = time.time()
    request_count = {"total": 0, "errors": 0}

    # ── FastAPI app ───────────────────────────────────────────────────────────
    app = FastAPI(
        title="TensorVerseHub Model Server",
        description="REST API for TensorFlow model inference",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Liveness probe."""
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - server_start, 1),
            "model_type": model_type,
            "model_path": model_path,
        }

    @app.get("/metadata")
    def metadata():
        """Model metadata."""
        meta = {
            "model_type": model_type,
            "model_path": model_path,
            "class_names": class_names,
            "max_batch_size": max_batch_size,
            "requests_total": request_count["total"],
            "requests_errors": request_count["errors"],
        }
        if model_type == "tflite":
            meta["input_shape"]  = tflite_interpreter.get_input_details()[0]["shape"].tolist()
            meta["output_shape"] = tflite_interpreter.get_output_details()[0]["shape"].tolist()
        elif model is not None and hasattr(model, "input_shape"):
            meta["input_shape"]  = list(model.input_shape)
            meta["output_shape"] = list(model.output_shape)
        return meta

    @app.post("/predict")
    async def predict(payload: dict):
        """
        Run inference on one or more inputs.

        Request body:
            {
              "instances": [[...], [...]]   // list of input arrays
            }

        Response:
            {
              "predictions": [...],
              "latency_ms": 12.3
            }
        """
        request_count["total"] += 1
        if "instances" not in payload:
            request_count["errors"] += 1
            raise HTTPException(status_code=400, detail="Request body must contain 'instances' key")

        try:
            instances = np.array(payload["instances"], dtype=np.float32)
        except Exception as exc:
            request_count["errors"] += 1
            raise HTTPException(status_code=422, detail=f"Could not parse instances: {exc}")

        if instances.shape[0] > max_batch_size:
            request_count["errors"] += 1
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {instances.shape[0]} exceeds max_batch_size={max_batch_size}",
            )

        t0 = time.perf_counter()
        try:
            if model_type == "tflite":
                inp_detail = tflite_interpreter.get_input_details()[0]
                out_detail = tflite_interpreter.get_output_details()[0]
                # TFLite doesn't support dynamic batching; run per-sample
                outputs = []
                for sample in instances:
                    tflite_interpreter.set_tensor(inp_detail["index"], sample[np.newaxis])
                    tflite_interpreter.invoke()
                    outputs.append(tflite_interpreter.get_tensor(out_detail["index"])[0])
                preds = np.array(outputs)
            elif hasattr(model, "__call__"):
                preds = model(instances, training=False).numpy()
            else:
                preds = model.predict(instances)
        except Exception as exc:
            request_count["errors"] += 1
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000

        # Build response
        if class_names and preds.ndim == 2:
            top_classes = [class_names[int(p.argmax())] for p in preds]
            top_scores  = [float(p.max()) for p in preds]
            result = [
                {"class": cls, "score": score, "probabilities": prob.tolist()}
                for cls, score, prob in zip(top_classes, top_scores, preds)
            ]
        else:
            result = preds.tolist()

        return JSONResponse({
            "predictions": result,
            "latency_ms": round(latency_ms, 3),
        })

    return app


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        logger.error("Model path does not exist: %s", model_path)
        sys.exit(1)

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    # Write a temp config file so uvicorn can find the app factory
    # (needed because build_app requires runtime arguments)
    import tempfile, textwrap

    launcher = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / "src")!r})
        sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
        from serve_model import build_app
        app = build_app(
            model_path={model_path!r},
            class_names={args.class_names!r},
            input_shape={args.input_shape!r},
            max_batch_size={args.max_batch_size},
        )
    """).strip()

    tmp_dir = Path(tempfile.mkdtemp())
    launcher_path = tmp_dir / "_tv_serve_app.py"
    launcher_path.write_text(launcher)
    sys.path.insert(0, str(tmp_dir))

    logger.info("Starting TensorVerseHub model server")
    logger.info("  Model   : %s", model_path)
    logger.info("  Address : http://%s:%d", args.host, args.port)
    logger.info("  Docs    : http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        "_tv_serve_app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
