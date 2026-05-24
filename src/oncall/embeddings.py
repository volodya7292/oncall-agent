"""Embedding client + numeric helpers for operator memory.

Uses an in-process sentence-transformers model (default
`nomic-ai/nomic-embed-text-v1.5`). No daemon, no network calls at runtime
— the model is downloaded once from HuggingFace into `~/.cache/huggingface/`
on first use and loaded from disk thereafter.

Storage format: float32 packed via numpy.tobytes — fixed bytes per row,
fast unpack with np.frombuffer.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Protocol

import numpy as np


log = logging.getLogger(__name__)


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingClient:
    """In-process sentence-transformers embedder.

    Lazy-loads the model on first `embed()` call so daemon startup isn't
    blocked by the (~1–3s on CPU) load. Subsequent calls reuse the loaded
    model; `encode` is wrapped in `asyncio.to_thread` so the event loop
    stays responsive during inference.

    `trust_remote_code=True` is required by some models (e.g.
    `nomic-ai/nomic-embed-text-v1.5`) and is harmless for the rest.
    """

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    def _load_model_sync(self) -> Any:
        from sentence_transformers import SentenceTransformer
        log.info("loading embedding model %r (this may download on first run)",
                 self._model_name)
        return SentenceTransformer(self._model_name, trust_remote_code=True)

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    async def warmup(self) -> None:
        """Eagerly load the model. Call once at startup if you want the
        first user turn to skip the load cost."""
        await self._ensure_loaded()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_loaded()
        arr = await asyncio.to_thread(
            model.encode, texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return arr.tolist()


# ---- storage helpers -------------------------------------------------------


def pack(vec: list[float] | np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def unpack(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


# ---- similarity ------------------------------------------------------------


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of `query` (shape D) against every row of `matrix` (N x D).
    Returns shape (N,). Empty matrix → empty array."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    qn = float(np.linalg.norm(query))
    if qn == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    # avoid divide-by-zero; the dot is also 0 for a zero row, so the result
    # stays 0 regardless of the divisor.
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix @ query) / (norms * qn)


# ---- fuzzy token overlap ---------------------------------------------------

# Matches identifiers, hostnames, paths, emails — anything boundary-bracketed
# by non-name punctuation. Lowercased so the overlap is case-insensitive.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-/@:]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if t}


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of token sets. 0..1.

    Cheap re-rank signal: catches exact-identifier matches (hostnames,
    paths) that pure embeddings can score lower than they should.
    """
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def hybrid_score(
    cos: float, text_a: str, text_b: str, *, alpha: float, beta: float,
) -> float:
    """Hybrid similarity = alpha * cosine + beta * Jaccard token overlap.

    Used in two places: (a) memory retrieval scoring (query text vs each
    stored memory), and (b) dedup-pass clustering (memory vs memory). Same
    weights so the two contexts agree on what "similar" means.
    """
    return alpha * cos + beta * token_overlap(text_a, text_b)
