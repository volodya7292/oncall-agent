"""Embedding client + numeric helpers for operator memory.

Uses a local Ollama daemon for embeddings (default model
`nomic-embed-text:137m-v1.5-fp16`). We pass `keep_alive: "4h"` on every
embed call so Ollama keeps the model resident across our daemon restarts
— first user message after `oncall service start` skips the cold load.

Storage format: float32 packed via numpy.tobytes — fixed bytes per row,
fast unpack with np.frombuffer.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

import numpy as np


log = logging.getLogger(__name__)


# How long Ollama should keep the embedding model loaded after a call.
# 4h covers normal dev rebuild/restart cycles; longer = more idle RAM,
# shorter = first embed after a coffee break pays the load again.
OLLAMA_KEEP_ALIVE = "4h"


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingClient:
    """Local-Ollama embeddings via the /api/embed endpoint.

    Model stays resident in the Ollama process (separate from our daemon)
    so our restarts don't pay cold-load latency, as long as a request
    lands within `keep_alive` of the last one. Falls back gracefully if
    Ollama is down — the operator just sees empty memory for that turn.
    """

    def __init__(self, host: str, *, model: str, timeout: float = 30.0) -> None:
        import httpx
        self._host = host.rstrip("/")
        self._model = model
        self._http = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # /api/embed accepts `input` as either str or list[str]. We always
        # send a list so the response shape is uniform.
        r = await self._http.post(
            f"{self._host}/api/embed",
            json={
                "model": self._model,
                "input": texts,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
        )
        r.raise_for_status()
        body = r.json()
        return [list(v) for v in body.get("embeddings", [])]

    async def ensure_model(self) -> None:
        """Pull the embed model into Ollama if it isn't there yet.

        On a fresh Ollama volume the model is absent and `/api/embed` errors
        ("model not found") rather than auto-pulling — so we provision it
        here before warmup. No-op once the model is resident, so it's safe to
        run on every daemon start. The pull can be hundreds of MB, so we
        stream `/api/pull` to completion with no read timeout instead of
        blocking on a single slow response."""
        import httpx

        r = await self._http.get(f"{self._host}/api/tags")
        r.raise_for_status()
        have = {m.get("name") for m in r.json().get("models", [])}
        if self._model in have:
            return
        log.info("ollama: embed model %s absent, pulling", self._model)
        async with self._http.stream(
            "POST",
            f"{self._host}/api/pull",
            json={"model": self._model},
            timeout=httpx.Timeout(30.0, read=None),
        ) as resp:
            resp.raise_for_status()
            # Drain the NDJSON progress stream; the final line is the
            # {"status": "success"} that means the blob is on disk.
            async for _ in resp.aiter_lines():
                pass
        log.info("ollama: pull complete for %s", self._model)

    async def warmup(self) -> None:
        """Trigger Ollama to load the model now (one throwaway embed). On
        a freshly-started Ollama daemon this is the only way to force the
        load before a real user request hits us."""
        await self.embed(["warmup"])

    async def aclose(self) -> None:
        await self._http.aclose()


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
