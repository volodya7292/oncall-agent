"""SQLite-backed operator memory with semantic retrieval and LRU eviction.

One row per short declarative fact extracted from a user turn (see
`memory_extractor.py`). Each row holds the fact text plus a packed-float32
embedding; retrieval scores all rows with `alpha * cosine + beta *
token_overlap` and returns the top-K above a relevance floor. Picked rows
have their `last_accessed_at` bumped — frequently-retrieved entries survive
LRU eviction at capacity.

This module never decides WHAT to remember (that's the extractor's job) and
never decides what to forget (LRU does that automatically). The two writes
are `store(facts)` and the implicit `_maybe_evict()` it triggers when over
capacity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .audit import fmt, operator_log
from .db import Database, iso
from .embeddings import (
    EmbeddingClient,
    cosine_matrix,
    token_overlap,
    unpack,
)
from .models import utcnow


log = logging.getLogger(__name__)

# Hard upper bound on a single fact's text length. The extractor's prompt
# tells it to keep facts ≤200 chars; this is a defense-in-depth ceiling so
# a misbehaving extractor can't bloat the prompt budget.
MAX_ENTRY_CHARS = 400


@dataclass
class Memory:
    id: int
    text: str
    score: float
    cosine: float
    last_accessed_at: str


class MemoryStore(Protocol):
    """The slice of OperatorMemory the Operator actually depends on. Tests
    can substitute any object that satisfies these methods."""

    async def store(
        self, facts: list[str], *, source_turn: str | None = None,
    ) -> list[str]: ...
    async def retrieve(
        self, query: str, *, limit: int | None = None,
    ) -> list[Memory]: ...
    async def for_prompt(self, query: str | None) -> str: ...
    async def entries_count(self) -> int: ...


class OperatorMemory:
    def __init__(
        self,
        db: Database,
        embedder: EmbeddingClient,
        *,
        capacity: int,
        max_inject: int,
        relevance_floor: float,
        hybrid_alpha: float,
        hybrid_beta: float,
        dedup_sim: float,
    ) -> None:
        self._db = db
        self._embed = embedder
        self._capacity = capacity
        self._max_inject = max_inject
        self._floor = relevance_floor
        self._alpha = hybrid_alpha
        self._beta = hybrid_beta
        self._dedup_sim = dedup_sim

    # ---- writes ------------------------------------------------------------

    async def store(
        self, facts: list[str], *, source_turn: str | None = None,
    ) -> list[str]:
        """Embed each fact; for each, near-duplicate-merge against existing
        rows (cos ≥ dedup_sim) or INSERT; evict by LRU until count ≤ capacity.

        Returns the texts that were actually written or updated (in input
        order). Empty/whitespace/over-length facts are skipped silently.
        Embedding failures raise — the caller surfaces extraction errors.
        """
        cleaned = []
        for f in facts:
            s = (f or "").strip().replace("\n", " ")
            if s and len(s) <= MAX_ENTRY_CHARS:
                cleaned.append(s)
        if not cleaned:
            return []

        vecs = await self._embed.embed(cleaned)
        out: list[str] = []
        stored = 0
        updated = 0
        for text, vec in zip(cleaned, vecs):
            qvec = np.asarray(vec, dtype=np.float32)
            existing = await self._all_rows()
            if existing:
                matrix = np.vstack([unpack(r["embedding"]) for r in existing])
                cosines = cosine_matrix(qvec, matrix)
                if cosines.size:
                    idx = int(np.argmax(cosines))
                    best = float(cosines[idx])
                    if best >= self._dedup_sim:
                        await self._update_row(
                            int(existing[idx]["id"]),
                            text=text,
                            embedding=qvec,
                            source_turn=source_turn,
                        )
                        out.append(text)
                        updated += 1
                        continue
            await self._insert_row(
                text=text, embedding=qvec, source_turn=source_turn,
            )
            out.append(text)
            stored += 1
            await self._maybe_evict()

        operator_log.info("memory_store " + fmt(
            stored=stored, updated=updated, capacity=self._capacity,
        ))
        return out

    # ---- reads -------------------------------------------------------------

    async def retrieve(
        self, query: str, *, limit: int | None = None,
    ) -> list[Memory]:
        """Hybrid retrieval. Returns up to `limit` (default `max_inject`)
        memories with score ≥ `relevance_floor`, ordered by score desc.
        Bumps `last_accessed_at` for the returned rows (and only those —
        below-floor candidates do NOT extend their LRU lifetime)."""
        q = (query or "").strip()
        if not q:
            return []
        lim = limit if limit is not None else self._max_inject
        rows = await self._all_rows()
        if not rows:
            return []

        try:
            qvec_list = (await self._embed.embed([q]))[0]
        except Exception:
            log.exception("embedding call failed in retrieve()")
            return []
        qvec = np.asarray(qvec_list, dtype=np.float32)
        matrix = np.vstack([unpack(r["embedding"]) for r in rows])
        cosines = cosine_matrix(qvec, matrix)

        scored: list[tuple[float, float, dict[str, Any]]] = []
        for r, c in zip(rows, cosines):
            fuzzy = token_overlap(q, r["text"])
            score = self._alpha * float(c) + self._beta * fuzzy
            if score >= self._floor:
                scored.append((score, float(c), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        picked = scored[:lim]

        if picked:
            await self._bump_access(*(int(p[2]["id"]) for p in picked))

        operator_log.info("memory_retrieve " + fmt(
            candidates=len(rows),
            picked=len(picked),
            max_score=round(picked[0][0], 3) if picked else 0.0,
        ))
        return [
            Memory(
                id=int(r["id"]),
                text=r["text"],
                score=score,
                cosine=cos_val,
                last_accessed_at=r["last_accessed_at"],
            )
            for score, cos_val, r in picked
        ]

    async def for_prompt(self, query: str | None) -> str:
        """Render the memory section for the system prompt. `None` skips
        retrieval entirely (used for auto-ping turns whose synthetic note
        isn't a meaningful retrieval key)."""
        if query is None:
            return "(no relevant entries this turn)"
        memories = await self.retrieve(query)
        if not memories:
            return "(no relevant entries this turn)"
        return "\n".join(f"- {m.text}" for m in memories)

    async def entries_count(self) -> int:
        row = await (await self._db.conn.execute(
            "SELECT COUNT(*) AS n FROM operator_memories"
        )).fetchone()
        return int(row["n"]) if row else 0

    # ---- internals ---------------------------------------------------------

    async def _all_rows(self) -> list[dict[str, Any]]:
        rows = await (await self._db.conn.execute(
            "SELECT id, text, embedding, last_accessed_at "
            "FROM operator_memories ORDER BY id"
        )).fetchall()
        return [dict(r) for r in rows]

    async def _insert_row(
        self,
        *,
        text: str,
        embedding: np.ndarray,
        source_turn: str | None,
    ) -> int:
        now = iso(utcnow())
        cur = await self._db.conn.execute(
            "INSERT INTO operator_memories "
            "(text, embedding, source_turn, created_at, last_accessed_at, access_count) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (text, embedding.tobytes(), source_turn, now, now),
        )
        await self._db.conn.commit()
        return cur.lastrowid or 0

    async def _update_row(
        self,
        row_id: int,
        *,
        text: str,
        embedding: np.ndarray,
        source_turn: str | None,
    ) -> None:
        now = iso(utcnow())
        await self._db.conn.execute(
            "UPDATE operator_memories "
            "SET text = ?, embedding = ?, "
            "    source_turn = COALESCE(?, source_turn), "
            "    last_accessed_at = ?, "
            "    access_count = access_count + 1 "
            "WHERE id = ?",
            (text, embedding.tobytes(), source_turn, now, row_id),
        )
        await self._db.conn.commit()

    async def _bump_access(self, *row_ids: int) -> None:
        if not row_ids:
            return
        now = iso(utcnow())
        placeholders = ",".join("?" * len(row_ids))
        await self._db.conn.execute(
            f"UPDATE operator_memories "
            f"SET last_accessed_at = ?, access_count = access_count + 1 "
            f"WHERE id IN ({placeholders})",
            (now, *row_ids),
        )
        await self._db.conn.commit()

    async def _maybe_evict(self) -> None:
        n = await self.entries_count()
        if n <= self._capacity:
            return
        overflow = n - self._capacity
        await self._db.conn.execute(
            "DELETE FROM operator_memories WHERE id IN ("
            "  SELECT id FROM operator_memories "
            "  ORDER BY last_accessed_at ASC, id ASC LIMIT ?"
            ")",
            (overflow,),
        )
        await self._db.conn.commit()
