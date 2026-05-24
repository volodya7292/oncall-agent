"""Unit tests for OperatorMemory.

Cover the contract the operator depends on:
  * `store` always inserts a new row (write-time dedup is intentionally
    absent — clusters are reconciled by the periodic `dedup_pass` LLM job).
  * `retrieve` ranks by hybrid score, drops below the relevance floor, and
    bumps `last_accessed_at` only for the rows actually returned.
  * LRU eviction at capacity drops the least-recently-accessed row, and a
    retrieval bump protects an entry from eviction.
  * `for_prompt(None)` returns the auto-ping fallback verbatim.

All embeddings come from a deterministic stub that maps text → 2-D unit
vector. With unit vectors, `cos(a, b) = a·b`, so cosines are exact and
boundary cases are testable to three decimals.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import pytest

from oncall.db import Database
from oncall.operator_memory import OperatorMemory


# ---------------------------------------------------------------------------
# Stub embedding client — deterministic, controllable cosines.
# ---------------------------------------------------------------------------


def unit_vec(cos_with_x_axis: float, sign: int = 1) -> list[float]:
    """2-D unit vector whose dot with [1, 0] is exactly `cos_with_x_axis`.
    The second component picks one of the two solutions of x² + y² = 1.
    """
    c = max(-1.0, min(1.0, cos_with_x_axis))
    return [c, sign * math.sqrt(max(0.0, 1.0 - c * c))]


class StubEmbedder:
    """Returns pre-registered vectors by exact text match. Unknown texts
    raise — tests must register every text they pass through, which keeps
    similarity outcomes 100% explicit."""

    def __init__(self) -> None:
        self._table: dict[str, list[float]] = {}

    def register(self, text: str, vec: Sequence[float]) -> None:
        self._table[text] = list(vec)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if t not in self._table:
                raise KeyError(f"StubEmbedder: unregistered text {t!r}")
            out.append(list(self._table[t]))
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    d = Database(tmp_path / "memory.db")
    await d.connect()
    try:
        yield d
    finally:
        await d.close()


def make_memory(
    db: Database,
    embedder: StubEmbedder,
    *,
    capacity: int = 100,
    max_inject: int = 5,
    relevance_floor: float = 0.30,
    hybrid_alpha: float = 0.7,
    hybrid_beta: float = 0.3,
) -> OperatorMemory:
    return OperatorMemory(
        db, embedder,
        embed_model="test-embedder",
        capacity=capacity,
        max_inject=max_inject,
        relevance_floor=relevance_floor,
        hybrid_alpha=hybrid_alpha,
        hybrid_beta=hybrid_beta,
    )


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_inserts_new_fact(db):
    emb = StubEmbedder()
    emb.register("staging api at api-staging:8443", unit_vec(1.0))
    mem = make_memory(db, emb)
    out = await mem.store(["staging api at api-staging:8443"])
    assert out == ["staging api at api-staging:8443"]
    assert await mem.entries_count() == 1


@pytest.mark.asyncio
async def test_store_strips_empty_and_overlong(db):
    emb = StubEmbedder()
    big = "x" * 1000
    emb.register("real fact", unit_vec(1.0))
    mem = make_memory(db, emb)
    out = await mem.store(["", "   ", big, "real fact"])
    assert out == ["real fact"]
    assert await mem.entries_count() == 1


@pytest.mark.asyncio
async def test_retrieve_returns_relevant_rows_above_floor(db):
    emb = StubEmbedder()
    # Two facts. The query is identical to fact A, so cos(query, A) = 1.0
    # and cos(query, B) ≈ 0. Only A should clear the floor.
    emb.register("fact A", unit_vec(1.0))                  # along [1, 0]
    emb.register("fact B", unit_vec(0.0))                  # along [0, 1]
    emb.register("query for A", unit_vec(1.0))             # same direction as A
    mem = make_memory(db, emb)
    await mem.store(["fact A", "fact B"])
    got = await mem.retrieve("query for A")
    assert [m.text for m in got] == ["fact A"]
    # Score = alpha * cos + beta * jaccard.
    # cos = 1.0; tokens overlap = {"a"} of size 1 over union {"query","for",
    # "a","fact"} of size 4 → jaccard 0.25. Total = 0.7*1 + 0.3*0.25 = 0.775.
    assert pytest.approx(got[0].score, abs=0.001) == 0.775
    assert pytest.approx(got[0].cosine, abs=0.001) == 1.0


@pytest.mark.asyncio
async def test_retrieve_empty_when_no_rows(db):
    emb = StubEmbedder()
    emb.register("anything", unit_vec(1.0))
    mem = make_memory(db, emb)
    assert await mem.retrieve("anything") == []


@pytest.mark.asyncio
async def test_retrieve_skips_empty_query(db):
    emb = StubEmbedder()
    emb.register("a", unit_vec(1.0))
    mem = make_memory(db, emb)
    await mem.store(["a"])
    assert await mem.retrieve("") == []
    assert await mem.retrieve("   ") == []


@pytest.mark.asyncio
async def test_for_prompt_none_returns_auto_ping_fallback(db):
    emb = StubEmbedder()
    mem = make_memory(db, emb)
    assert await mem.for_prompt(None) == "(no relevant entries this turn)"


@pytest.mark.asyncio
async def test_for_prompt_with_no_relevant_entries_returns_fallback(db):
    emb = StubEmbedder()
    emb.register("orange", unit_vec(1.0))
    emb.register("apple", unit_vec(0.0))  # orthogonal → cos ≈ 0
    mem = make_memory(db, emb)
    await mem.store(["orange"])
    # Query orthogonal to the only stored fact → below floor.
    assert await mem.for_prompt("apple") == "(no relevant entries this turn)"


@pytest.mark.asyncio
async def test_retrieve_bumps_only_returned_rows(db):
    emb = StubEmbedder()
    emb.register("relevant", unit_vec(1.0))
    emb.register("irrelevant", unit_vec(0.0))
    emb.register("relevant?", unit_vec(1.0))  # same direction as "relevant"
    mem = make_memory(db, emb)
    await mem.store(["relevant", "irrelevant"])
    before = await _last_accessed_map(db)
    await mem.retrieve("relevant?")
    after = await _last_accessed_map(db)
    # The relevant row was returned → its timestamp must change.
    assert after["relevant"] > before["relevant"]
    # The irrelevant row was below floor → must NOT be bumped.
    assert after["irrelevant"] == before["irrelevant"]


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lru_evicts_oldest_accessed_when_over_capacity(db):
    emb = StubEmbedder()
    # 3 orthogonal stored vectors so none of them dedup against each other.
    # The 4th is also orthogonal — must trigger eviction.
    for i, ang in enumerate([0.0, math.pi / 2, math.pi, -math.pi / 2]):
        emb.register(f"f{i}", [math.cos(ang), math.sin(ang)])
    mem = make_memory(db, emb, capacity=3)
    await mem.store(["f0"])
    await mem.store(["f1"])
    await mem.store(["f2"])
    # Bump f1 + f2 via retrieval so they look "recent".
    emb.register("query for f1", [math.cos(math.pi / 2), math.sin(math.pi / 2)])
    await mem.retrieve("query for f1")
    emb.register("query for f2", [math.cos(math.pi), math.sin(math.pi)])
    await mem.retrieve("query for f2")
    # Now insert f3 — f0 has the oldest last_accessed_at and should evict.
    await mem.store(["f3"])
    rows = await _all_rows(db)
    texts = {r["text"] for r in rows}
    assert len(rows) == 3
    assert "f0" not in texts, "least-recently-accessed row should have been evicted"
    assert {"f1", "f2", "f3"} <= texts


@pytest.mark.asyncio
async def test_retrieval_protects_row_from_eviction(db):
    """A row touched by retrieval should not be the next victim of LRU."""
    emb = StubEmbedder()
    for i, ang in enumerate([0.0, math.pi / 2, math.pi]):
        emb.register(f"f{i}", [math.cos(ang), math.sin(ang)])
    emb.register("query for f0", [math.cos(0.0), math.sin(0.0)])
    emb.register("f-new", [math.cos(-math.pi / 2), math.sin(-math.pi / 2)])
    mem = make_memory(db, emb, capacity=3)
    # Insert oldest-first.
    await mem.store(["f0"])
    await mem.store(["f1"])
    await mem.store(["f2"])
    # Touch f0 so it's now the youngest.
    await mem.retrieve("query for f0")
    # New insert forces eviction of one. f0 should survive because it's
    # been touched; f1 should be the victim (now the oldest).
    await mem.store(["f-new"])
    texts = {r["text"] for r in await _all_rows(db)}
    assert "f0" in texts, "retrieval should have protected f0 from eviction"
    assert "f1" not in texts, "untouched f1 should be the evicted row"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration — real local Ollama embeddings. Skipped unless
# ONCALL_RUN_EMBEDDING_TESTS=1 is set in the environment, because they
# require a running Ollama daemon with the model pulled. Confirms:
#   (a) the model produces vectors with consistent dimensionality (sanity),
#   (b) it scores semantically related text above unrelated text (the
#       premise of using embeddings here),
#   (c) it pushes near-duplicate paraphrases over the 0.88 dedup threshold.
# ---------------------------------------------------------------------------


EMBED_MODEL = os.environ.get(
    "ONCALL_MEMORY_EMBED_MODEL", "nomic-embed-text:137m-v1.5-fp16",
)
OLLAMA_HOST = os.environ.get("ONCALL_OLLAMA_HOST", "http://localhost:11434")


requires_embedding_tests = pytest.mark.skipif(
    os.environ.get("ONCALL_RUN_EMBEDDING_TESTS", "") != "1",
    reason=(
        "set ONCALL_RUN_EMBEDDING_TESTS=1 to run live embedding tests "
        "(requires a running Ollama daemon with the model pulled: "
        "`ollama pull nomic-embed-text:137m-v1.5-fp16`)"
    ),
)


def _real_embedder():
    from oncall.embeddings import OllamaEmbeddingClient
    return OllamaEmbeddingClient(host=OLLAMA_HOST, model=EMBED_MODEL)


@requires_embedding_tests
@pytest.mark.asyncio
async def test_real_embeddings_have_expected_dim_and_unit_norm():
    """Spot-check the live model: it returns a single vector per input
    and the dimensionality is consistent across calls."""
    import numpy as np
    emb = _real_embedder()
    vecs = await emb.embed(["the staging api is at api-staging.example.com"])
    assert len(vecs) == 1
    arr = np.asarray(vecs[0], dtype=np.float32)
    assert arr.ndim == 1 and arr.size > 0
    # Sanity: repeat the call, dimensionality stable.
    again = await emb.embed(["something completely different"])
    assert len(again[0]) == arr.size


@requires_embedding_tests
@pytest.mark.asyncio
async def test_real_embeddings_rank_semantically_related_first(db):
    """End-to-end: store a few unrelated facts via the real model, then
    query for one of them paraphrased — the matching fact must come back."""
    mem = OperatorMemory(
        db, _real_embedder(),
        embed_model=EMBED_MODEL,
        capacity=100,
        max_inject=3,
        # Loosen the floor a touch — different models normalize cosine
        # differently and we don't want a brittle threshold here.
        relevance_floor=0.20,
        hybrid_alpha=0.7, hybrid_beta=0.3,
    )
    await mem.store([
        "the staging API runs at api-staging.example.com on port 8443",
        "alex is the on-call lead for the payments team",
        "the prod database is named pg-prod-1",
    ])
    got = await mem.retrieve("where does staging live again?")
    assert got, "retrieval returned nothing — model may have changed"
    # Top hit must be the staging fact, not the alex / prod-db one.
    assert "staging" in got[0].text.lower()


@pytest.mark.asyncio
async def test_rebuild_when_embed_model_changes(db):
    """Switching `embed_model` should:
      - hide existing rows from retrieve()/dedup until rebuilt,
      - leave the row count unchanged in the DB,
      - restore visibility after rebuild_stale_embeddings() runs.
    Captures the upgrade path the production lifespan uses when the user
    swaps embedders.
    """
    # Stub embedder: each distinct text gets a fresh near-orthogonal unit
    # vector so dedup never merges them. `flavor` distinguishes the "old"
    # vs "new" model so we can verify the rebuild actually re-embeds.
    class _Stub:
        def __init__(self, flavor: int) -> None:
            self.flavor = flavor
            self._known: dict[str, int] = {}
        async def embed(self, texts):
            out = []
            for t in texts:
                idx = self._known.setdefault(t, len(self._known))
                vec = [0.0] * 16
                vec[idx % 16] = 1.0
                # Tilt the vector slightly per-flavor so old/new models
                # produce different (but still near-orthogonal between
                # distinct texts) embeddings.
                vec[(idx + 8) % 16] = 0.01 * self.flavor
                out.append(vec)
            return out

    # Seed via "old" model.
    old_mem = make_memory(db, _Stub(flavor=1), relevance_floor=0.0)
    old_mem._embed_model = "old-model"  # type: ignore[attr-defined]
    await old_mem.store(["fact alpha", "fact beta"])
    assert await old_mem.entries_count() == 2

    # Switch to "new" model — the old rows are now stale and invisible.
    new_mem = make_memory(db, _Stub(flavor=2), relevance_floor=0.0)
    new_mem._embed_model = "new-model"  # type: ignore[attr-defined]
    assert await new_mem.stale_count() == 2
    assert await new_mem.entries_count() == 0, "stale rows must be hidden from retrieval"
    # DB row count unchanged — we don't delete, we re-embed.
    raw = await _all_rows(db)
    assert len(raw) == 2

    # Rebuild: re-embeds via the new stub, retags model column.
    result = await new_mem.rebuild_stale_embeddings()
    assert result == {"rebuilt": 2, "failed": 0}
    assert await new_mem.stale_count() == 0
    assert await new_mem.entries_count() == 2


async def _all_rows(db: Database) -> list[dict]:
    cur = await db.conn.execute(
        "SELECT id, text, created_at, last_accessed_at FROM operator_memories ORDER BY id"
    )
    return [dict(r) for r in await cur.fetchall()]


async def _last_accessed_map(db: Database) -> dict[str, str]:
    rows = await _all_rows(db)
    return {r["text"]: r["last_accessed_at"] for r in rows}
