"""Shared TFLite interpreter loading (ai-edge-litert, with the old tflite-runtime as fallback)."""

from __future__ import annotations

from pathlib import Path


def load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older Pi installs
        from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=str(model_path))
