"""Sentence embeddings via fastembed (ONNX). No PyTorch."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "fastembed"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized row per text."""
        ...


class FastEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = CACHE_DIR) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return normalize(vectors)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


_lock = threading.Lock()
_default: FastEmbedder | None = None


def default_embedder() -> FastEmbedder:
    """Load the model once per process."""
    global _default
    with _lock:
        if _default is None:
            _default = FastEmbedder()
        return _default
