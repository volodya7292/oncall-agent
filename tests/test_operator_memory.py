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

import asyncio
import json
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

    async def embed(
        self, texts: list[str], *, kind: str = "document",
    ) -> list[list[float]]:
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
    # Off by default so the floor/LRU/exclude cases below stay about the
    # thing they test. Production defaults to 0.6; the gate has its own test.
    relative_gate: float = 0.0,
    standing_cap: int = 30,
) -> OperatorMemory:
    return OperatorMemory(
        db, embedder,
        embed_model="test-embedder",
        capacity=capacity,
        max_inject=max_inject,
        relevance_floor=relevance_floor,
        hybrid_alpha=hybrid_alpha,
        hybrid_beta=hybrid_beta,
        relative_gate=relative_gate,
        standing_cap=standing_cap,
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
# Standing (behavioral) memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standing_rows_survive_lru_eviction(db):
    """A standing row is never retrieved, so its last_accessed_at never
    advances — under a plain LRU sweep it would be the FIRST thing deleted,
    silently un-instructing the agent. Capacity pressure must fall entirely
    on the contextual rows."""
    emb = StubEmbedder()
    for i, ang in enumerate([0.0, math.pi / 2, math.pi, -math.pi / 2]):
        emb.register(f"f{i}", [math.cos(ang), math.sin(ang)])
    emb.register("always reply in one line", unit_vec(0.5, sign=-1))
    mem = make_memory(db, emb, capacity=3)
    # Oldest row in the store, and never touched again.
    await mem.store(["always reply in one line"], standing=True)
    for i in range(4):
        await mem.store([f"f{i}"])

    rows = await _all_rows(db)
    texts = {r["text"] for r in rows}
    assert "always reply in one line" in texts
    # Capacity counts only the contextual rows, so the standing one is not
    # occupying a slot either.
    assert len([r for r in rows if not r["standing"]]) == 3


@pytest.mark.asyncio
async def test_standing_rows_are_not_returned_by_retrieve(db):
    """Standing rows are injected wholesale every session. If retrieval also
    scored them, a topic-matching instruction would be injected twice — and
    a standing row at the peak would drag the relative gate up and squeeze
    out the facts the query was actually for."""
    emb = StubEmbedder()
    emb.register("always reply in one line", unit_vec(1.0))
    emb.register("the deploy host is build-01", unit_vec(0.9))
    emb.register("how do you reply", unit_vec(1.0))
    mem = make_memory(db, emb, relative_gate=0.0)
    await mem.store(["always reply in one line"], standing=True)
    await mem.store(["the deploy host is build-01"])

    hits = await mem.retrieve("how do you reply")
    assert [m.text for m in hits] == ["the deploy host is build-01"]
    assert [m.text for m in await mem.standing_rows()] == [
        "always reply in one line",
    ]


@pytest.mark.asyncio
async def test_standing_cap_refuses_rather_than_evicting(db):
    """Standing rows sit in every context window, so the cap is a hard
    refusal — not an LRU drop. Over-cap writes must not silently displace
    an instruction the owner never withdrew."""
    emb = StubEmbedder()
    for i in range(3):
        emb.register(f"rule {i}", unit_vec(1.0 - i * 0.1))
    mem = make_memory(db, emb, standing_cap=2)
    assert await mem.store(["rule 0"], standing=True) == ["rule 0"]
    assert await mem.store(["rule 1"], standing=True) == ["rule 1"]
    assert await mem.store(["rule 2"], standing=True) == []
    assert [m.text for m in await mem.standing_rows()] == ["rule 0", "rule 1"]


@pytest.mark.asyncio
async def test_dedup_merge_inherits_standing(db):
    """Merging a standing row with a paraphrase of itself must keep the
    survivor standing. Otherwise dedup — a background job the owner never
    sees — quietly demotes an instruction to a relevance-gated fact."""
    emb = StubEmbedder()
    emb.register("always reply in one line", unit_vec(1.0))
    emb.register("keep replies to one line", unit_vec(0.99))
    emb.register("reply in one line only", unit_vec(1.0))
    mem = make_memory(db, emb)
    await mem.store(["always reply in one line"], standing=True)
    await mem.store(["keep replies to one line"])

    rows = await _all_rows(db)
    ids = [r["id"] for r in rows]

    class _MergeLLM:
        async def chat(self, *, model, messages, tools, max_tokens=None,
                       reasoning_effort=None):
            return {"content": json.dumps({
                "merge_groups": [{"ids": ids, "text": "reply in one line only"}],
            })}

    await mem.dedup_pass(
        _MergeLLM(), model="test", reasoning_effort="low",
        cluster_threshold=0.5,
    )

    survivors = await _all_rows(db)
    assert [r["text"] for r in survivors] == ["reply in one line only"]
    assert survivors[0]["standing"] == 1


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
#   (c) it retrieves ACROSS languages — the store is bilingual, and an
#       English-only embedder silently reduces retrieval to noise.
# ---------------------------------------------------------------------------


EMBED_MODEL = os.environ.get(
    "ONCALL_MEMORY_EMBED_MODEL", "embeddinggemma:300m",
)
OLLAMA_HOST = os.environ.get("ONCALL_OLLAMA_HOST", "http://localhost:11434")


requires_embedding_tests = pytest.mark.skipif(
    os.environ.get("ONCALL_RUN_EMBEDDING_TESTS", "") != "1",
    reason=(
        "set ONCALL_RUN_EMBEDDING_TESTS=1 to run live embedding tests "
        "(requires a running Ollama daemon with the model pulled: "
        "`ollama pull embeddinggemma:300m`)"
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


@requires_embedding_tests
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expect_token",
    [
        # Ukrainian question -> fact stored in English.
        ("чи можу я позичити комусь гроші?", "lend"),
        # English question -> fact stored in Ukrainian.
        ("what kind of noodles do I like?", "локшині"),
    ],
)
async def test_real_embeddings_retrieve_across_languages(db, query, expect_token):
    """Regression: an English-only embedder (nomic-embed-text, the default
    until db98ece) encoded LANGUAGE rather than meaning, so a Ukrainian
    question never reached a fact stored in English. On the real store that
    put the wanted row at rank 89/95 — below the floor — and the operator
    correctly reported it knew nothing about a fact it had held for weeks.

    Cross-language recall@10 was 11% while same-language recall stayed 100%,
    which is exactly why it went unnoticed: nothing errored, memory just
    quietly stopped working for half the store. The embedder must therefore
    be multilingual, and this pins that property to the live model rather
    than to anyone's belief about the model card.
    """
    mem = OperatorMemory(
        db, _real_embedder(),
        embed_model=EMBED_MODEL,
        capacity=100,
        max_inject=3,
        # Floor 0 on purpose: this pins the cross-language RANKING (right fact
        # first), not the absolute floor. Floor calibration is store-size
        # dependent and belongs in config, not here — on this tiny synthetic
        # store true-positive scores run lower than on the real 95-row store.
        # Under an English-only embedder the ranking itself broke (the wanted
        # row sank to rank 89/95), so a top-hit assertion still catches the
        # regression without coupling to a magic floor number.
        relevance_floor=0.0,
        hybrid_alpha=0.7, hybrid_beta=0.3,
    )
    # Deliberately mixed-language store, mirroring production: facts are
    # written in whichever language the user spoke.
    await mem.store([
        "the user refuses to lend money to anyone, no exceptions",
        "the prod database is named pg-prod-1",
        "користувач віддає перевагу тонкій ручній локшині",
        "користувач не має водійських прав",
    ])
    got = await mem.retrieve(query)
    assert got, f"cross-language retrieval returned nothing for {query!r} — " \
                f"is {EMBED_MODEL} multilingual?"
    assert expect_token in got[0].text.lower(), (
        f"cross-language retrieval ranked {got[0].text!r} above the intended "
        f"fact for {query!r} — embedder may have regressed to English-only"
    )


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
        async def embed(self, texts, *, kind: str = "document"):
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
        "SELECT id, text, standing, created_at, last_accessed_at "
        "FROM operator_memories ORDER BY id"
    )
    return [dict(r) for r in await cur.fetchall()]


async def _last_accessed_map(db: Database) -> dict[str, str]:
    rows = await _all_rows(db)
    return {r["text"]: r["last_accessed_at"] for r in rows}


# ---------------------------------------------------------------------------
# dedup_pass: what the LLM arbiter is actually shown
# ---------------------------------------------------------------------------


class _CapturingLLM:
    """Records the cluster payload; merges nothing."""

    def __init__(self) -> None:
        self.messages: list[list[dict]] = []

    async def chat(self, *, model, messages, tools, max_tokens=None,
                   reasoning_effort=None):
        self.messages.append(messages)
        return {"role": "assistant", "content": '{"merge_groups": []}',
                "tool_calls": []}


@pytest.mark.asyncio
async def test_dedup_cluster_payload_carries_timestamps_oldest_first(db):
    """The arbiter can only tell a paraphrase from a superseded fact if it can
    see WHEN each memory was recorded, and in what order.

    Regression guard for two easy breaks: `memory_all_rows` not selecting
    `created_at` (KeyError on every pass), and the payload arriving in id order
    rather than chronological order — which would invite the LLM to treat the
    stale value as current.
    """
    embedder = StubEmbedder()
    # Same attribute, value changed → must land in one cluster (cos = 1.0).
    old, new = "The user lives in Berlin.", "The user lives in Munich."
    for t in (old, new):
        embedder.register(t, [1.0, 0.0])
    mem = make_memory(db, embedder)

    # Insert NEW first so id order and time order disagree; if the payload were
    # sorted by id the assertion below would catch it.
    await mem.store([new])
    await asyncio.sleep(0.01)
    await mem.store([old])
    await db.conn.execute(
        "UPDATE operator_memories SET created_at = ? WHERE text = ?",
        ("2026-01-01T00:00:00+00:00", old),
    )
    await db.conn.execute(
        "UPDATE operator_memories SET created_at = ? WHERE text = ?",
        ("2026-07-01T00:00:00+00:00", new),
    )
    await db.conn.commit()

    llm = _CapturingLLM()
    await mem.dedup_pass(llm, model="test-model")

    assert llm.messages, "cluster should have reached the arbiter"
    payload = json.loads(llm.messages[0][1]["content"].split("Memories:\n", 1)[1])
    assert [p["text"] for p in payload] == [old, new], "must be oldest-first"
    assert all("recorded_at" in p for p in payload), "arbiter needs timestamps"
    assert payload[0]["recorded_at"] < payload[1]["recorded_at"]


@pytest.mark.asyncio
async def test_retrieve_excludes_before_applying_the_limit(db):
    """Regression: `exclude_ids` must shrink the candidate pool, not the
    result set.

    The session-injection caller excludes memories already shown in this
    session. It used to take the top-`max_inject` and filter afterwards, so
    a session that had seen the best matches got NOTHING injected even
    though plenty of eligible memories scored above the floor — the failure
    that made a long-running session look amnesiac. Excluding first spends
    the budget on rows the caller can actually use.
    """
    emb = StubEmbedder()
    # No token overlap with the query, so jaccard is 0 for every row and
    # score is exactly 0.7 * cos — ranking is purely the registered cosines.
    emb.register("alpha", unit_vec(1.0))
    emb.register("beta", unit_vec(0.9))
    emb.register("gamma", unit_vec(0.8))
    emb.register("zzz", unit_vec(1.0))
    mem = make_memory(db, emb, max_inject=2, relevance_floor=0.30)
    await mem.store(["alpha", "beta", "gamma"])

    ranked = await mem.retrieve("zzz", limit=10)
    assert [m.text for m in ranked] == ["alpha", "beta", "gamma"]
    top_two = {m.id for m in ranked[:2]}

    # Budget is 2 and the top 2 are excluded; the third must still surface.
    got = await mem.retrieve("zzz", exclude_ids=top_two)
    assert [m.text for m in got] == ["gamma"]


@pytest.mark.asyncio
async def test_relative_gate_scales_against_all_rows_not_the_eligible_pool(db):
    """Regression: a session that has been shown the good matches must not
    then be fed the best of the leftovers.

    `relative_gate` keeps candidates scoring within a fraction of the BEST
    score for the query. If that peak were taken after `exclude_ids` were
    removed, the top *remaining* row would define the scale and always pass —
    so a long-lived session would inject steadily worse memories, which is
    exactly what the live store did once 108 of 109 rows had been shown (a
    request to re-read a chat pulled in an unrelated third party's chat id).

    Measured against the full row set, the leftovers are judged against the
    query's real peak and correctly dropped.
    """
    emb = StubEmbedder()
    # Scores are 0.7 * cos (no token overlap with the query): alpha 0.70,
    # beta 0.63, gamma 0.28 — gamma is well under 0.6 * 0.70 = 0.42.
    emb.register("alpha", unit_vec(1.0))
    emb.register("beta", unit_vec(0.9))
    emb.register("gamma", unit_vec(0.4))
    emb.register("zzz", unit_vec(1.0))
    mem = make_memory(db, emb, relevance_floor=0.10, relative_gate=0.6)
    await mem.store(["alpha", "beta", "gamma"])

    ranked = await mem.retrieve("zzz", limit=10)
    assert [m.text for m in ranked] == ["alpha", "beta"], "gamma is off-topic"

    # Exclude both survivors. gamma is now the best *eligible* row and clears
    # the floor — a pool-relative peak would inject it. It must stay out.
    got = await mem.retrieve("zzz", exclude_ids={m.id for m in ranked})
    assert got == []


def test_dedup_candidate_prune_never_drops_an_edge():
    """The dedup pass scores only the pairs `_candidate_pairs` returns, so the
    prune must be exact: alpha*cos + beta is an upper bound on hybrid (jaccard
    ≤ 1), and anything it excludes provably cannot reach the gate.

    Checked against brute force over every pair, including the knife-edge case
    where the bound lands exactly ON the threshold — a `>` there instead of
    `>=` would silently stop merging identical memories.
    """
    import numpy as np
    from oncall.embeddings import hybrid_score
    from oncall.operator_memory import _candidate_pairs

    alpha, beta, threshold = 0.7, 0.3, 0.60
    rng = np.random.default_rng(7)
    texts = [
        " ".join(f"tok{t}" for t in rng.integers(0, 12, size=6))
        for _ in range(40)
    ]
    cos = rng.uniform(-0.2, 1.0, size=(40, 40)).astype(np.float32)
    cos = (cos + cos.T) / 2
    # A pair whose ceiling is EXACTLY the threshold: 0.7*(3/7) + 0.3 = 0.6,
    # reachable only because the identical texts push jaccard to 1.0.
    cos[0, 1] = cos[1, 0] = 3 / 7
    texts[0] = texts[1] = "tok1 tok2 tok3"

    candidates = set(_candidate_pairs(
        cos, alpha=alpha, beta=beta, threshold=threshold,
    ))
    real_edges = {
        (i, j)
        for i in range(40) for j in range(i + 1, 40)
        if hybrid_score(
            float(cos[i, j]), texts[i], texts[j], alpha=alpha, beta=beta,
        ) >= threshold
    }
    assert real_edges - candidates == set()
    assert (0, 1) in candidates
    # And it prunes something, or it isn't buying anything.
    assert len(candidates) < 40 * 39 // 2
