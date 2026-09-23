"""Serve a model over a FastAPI REST endpoint (/predict, /health, /metadata)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("tensorverse.serve")

CONFIG_ENV = "TVH_SERVE_CONFIG"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", "-m", required=True, help=".keras/.h5 file, SavedModel dir or .tflite"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--reload", action="store_true", help="Auto-reload (development only)")


class ModelRunner:
    """Uniform ``predict(np.ndarray) -> np.ndarray`` over Keras, SavedModel and TFLite."""

    def __init__(self, model_path: str) -> None:
        from .. import compat

        self.model_path = model_path
        if model_path.endswith(".tflite"):
            from ..export_utils import make_interpreter

            self.kind = "tflite"
            self.model = make_interpreter(model_path)
            self.input_shape = self.model.get_input_details()[0]["shape"].tolist()
            self.output_shape = self.model.get_output_details()[0]["shape"].tolist()
        else:
            self.model = compat.load_model(model_path)
            self.kind = "keras" if isinstance(self.model, compat.keras.Model) else "savedmodel"
            self.input_shape = list(self.model.input_shape)
            self.output_shape = list(getattr(self.model, "output_shape", []) or [])

    def predict(self, x: Any) -> Any:
        if self.kind == "tflite":
            from ..export_utils import TFLiteExporter

            return TFLiteExporter.run_tflite(self.model, x)
        if self.kind == "keras":
            return self.model.predict(x, verbose=0)
        return self.model.predict(x)


def build_app(
    model_path: str,
    class_names: Optional[Sequence[str]] = None,
    max_batch_size: int = 32,
) -> Any:
    """Create the FastAPI application (importable by tests and uvicorn)."""
    import numpy as np

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("fastapi is required: pip install 'tensorversehub[serving]'") from exc

    from .. import __version__

    runner = ModelRunner(model_path)
    started = time.time()
    counters = {"total": 0, "errors": 0}
    class_list = list(class_names) if class_names else None

    app = FastAPI(title="TensorVerseHub Model Server", version=__version__)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - started, 1),
            "model_type": runner.kind,
            "model_path": model_path,
        }

    @app.get("/metadata")
    def metadata() -> Dict[str, Any]:
        return {
            "model_type": runner.kind,
            "model_path": model_path,
            "class_names": class_list,
            "input_shape": runner.input_shape,
            "output_shape": runner.output_shape,
            "max_batch_size": max_batch_size,
            **{f"requests_{k}": v for k, v in counters.items()},
        }

    @app.post("/predict")
    def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
        counters["total"] += 1
        if "instances" not in payload:
            counters["errors"] += 1
            raise HTTPException(400, "Request body must contain an 'instances' list")
        try:
            instances = np.asarray(payload["instances"], dtype=np.float32)
        except Exception as exc:
            counters["errors"] += 1
            raise HTTPException(422, f"Could not parse instances: {exc}") from exc
        if instances.ndim == 0 or instances.shape[0] > max_batch_size:
            counters["errors"] += 1
            raise HTTPException(400, f"Batch size must be between 1 and {max_batch_size}")
        t0 = time.perf_counter()
        try:
            preds = np.asarray(runner.predict(instances))
        except Exception as exc:
            counters["errors"] += 1
            raise HTTPException(500, f"Inference failed: {exc}") from exc
        latency = (time.perf_counter() - t0) * 1000
        if class_list and preds.ndim == 2 and preds.shape[1] == len(class_list):
            result: List[Any] = [
                {
                    "class": class_list[int(p.argmax())],
                    "score": float(p.max()),
                    "probabilities": p.tolist(),
                }
                for p in preds
            ]
        else:
            result = preds.tolist()
        return {"predictions": result, "latency_ms": round(latency, 3)}

    return app


def create_app() -> Any:
    """uvicorn factory reading its configuration from ``TVH_SERVE_CONFIG``."""
    config = json.loads(os.environ.get(CONFIG_ENV, "{}"))
    if "model_path" not in config:
        raise RuntimeError(f"{CONFIG_ENV} must contain a model_path")
    return build_app(**config)


def run(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is required: pip install 'tensorversehub[serving]'")
        return 1
    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        logger.error("Model path does not exist: %s", model_path)
        return 1
    os.environ[CONFIG_ENV] = json.dumps(
        {
            "model_path": model_path,
            "class_names": args.class_names,
            "max_batch_size": args.max_batch_size,
        }
    )
    logger.info("Serving %s at http://%s:%d (docs at /docs)", model_path, args.host, args.port)
    uvicorn.run(
        "tensorversehub.cli.serve:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def main() -> None:
    from . import main as cli_main

    sys.exit(cli_main(["serve", *sys.argv[1:]]))


if __name__ == "__main__":  # pragma: no cover
    main()
