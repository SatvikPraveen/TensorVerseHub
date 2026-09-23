"""
``tensorverse`` command-line interface.

Sub-commands (also installed as stand-alone ``tensorverse-*`` scripts)::

    tensorverse train    --task classification --epochs 5
    tensorverse evaluate --model models/final_model.keras --report
    tensorverse convert  --model models/final_model.keras --to tflite --quantize int8
    tensorverse serve    --model models/final_model.keras --port 8000
    tensorverse info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Callable, Dict, List, Optional, Sequence

from .. import __version__

COMMANDS: Dict[str, str] = {
    "train": "tensorversehub.cli.train",
    "evaluate": "tensorversehub.cli.evaluate",
    "convert": "tensorversehub.cli.convert",
    "serve": "tensorversehub.cli.serve",
}


def configure_logging(level: str = "info") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _info(args: argparse.Namespace) -> int:
    from .. import about

    print(json.dumps(about(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    import importlib

    parser = argparse.ArgumentParser(
        prog="tensorverse",
        description="TensorVerseHub — train, evaluate, convert and serve TensorFlow models.",
    )
    parser.add_argument("--version", action="version", version=f"tensorverse {__version__}")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    for name, module_path in COMMANDS.items():
        module = importlib.import_module(module_path)
        sub = subparsers.add_parser(name, help=module.__doc__.strip().splitlines()[0])
        module.add_arguments(sub)
        sub.set_defaults(func=module.run)

    info = subparsers.add_parser("info", help="Print runtime / version information as JSON")
    info.set_defaults(func=_info)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    func: Callable[[argparse.Namespace], int] = args.func
    return int(func(args) or 0)


def _entry(command: str) -> Callable[[], None]:
    """Build a ``tensorverse-<command>`` console-script entry point."""

    def entry() -> None:
        sys.exit(main([command, *sys.argv[1:]]))

    entry.__name__ = f"{command}_entry"
    return entry


__all__ = ["COMMANDS", "build_parser", "configure_logging", "main"]
