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

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .audit import fmt, operator_log
from .db import Database
from .embeddings import (
    EmbeddingClient,
    cosine_matrix,
    hybrid_score,
    unpack,
)
from .metrics import timed


log = logging.getLogger(__name__)

# Hard upper bound on a single fact's text length. The extractor's prompt
# tells it to keep facts short; this is a defense-in-depth ceiling so a
# misbehaving extractor — or the dedup LLM combining several long facts —
# can't bloat the prompt budget.
MAX_ENTRY_CHARS = 512


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
    async def get_by_id(self, memory_id: int) -> Memory | None: ...
    async def delete_by_id(self, memory_id: int) -> bool: ...
    async def for_prompt(self, query: str | None) -> str: ...
    async def entries_count(self) -> int: ...


class OperatorMemory:
    def __init__(
        self,
        db: Database,
        embedder: EmbeddingClient,
        *,
        embed_model: str,
        capacity: int,
        max_inject: int,
        relevance_floor: float,
        hybrid_alpha: float,
        hybrid_beta: float,
    ) -> None:
        self._db = db
        self._embed = embedder
        # Name tag written to each row's `model` column. Retrieval filters
        # to rows whose model matches this; rows from older models are
        # invisible until `rebuild_stale_embeddings()` re-embeds them.
        # Tests pass a sentinel string here — production threads the real
        # model id from settings.oncall_memory_embed_model.
        self._embed_model = embed_model
        self._capacity = capacity
        self._max_inject = max_inject
        self._floor = relevance_floor
        self._alpha = hybrid_alpha
        self._beta = hybrid_beta

    # ---- writes ------------------------------------------------------------

    async def store(
        self, facts: list[str], *, source_turn: str | None = None,
    ) -> list[str]:
        """Embed each fact and INSERT it as a new row; evict by LRU until
        count ≤ capacity.

        No write-time dedup — clusters of near-duplicates are reconciled by
        the periodic `dedup_pass()` background job, which uses the LLM to
        tell paraphrase merges from same-template-different-entity cases
        that embeddings alone confuse.

        Returns the texts that were actually written (in input order).
        Empty/whitespace/over-length facts are skipped silently. Embedding
        failures raise — the caller surfaces extraction errors.
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
        for text, vec in zip(cleaned, vecs):
            qvec = np.asarray(vec, dtype=np.float32)
            await self._insert_row(
                text=text, embedding=qvec, source_turn=source_turn,
            )
            out.append(text)
            await self._maybe_evict()

        operator_log.info("memory_store " + fmt(
            stored=len(out), capacity=self._capacity,
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

        # Rolling "memory" window (surfaced in /status). Scoped to the real
        # retrieval work — embed + score + LRU bump. The empty-query/no-rows
        # exits above are deliberately outside it: they return in ~0ms and
        # would drag the percentiles toward zero.
        with timed("memory") as t:
            embed_start = time.monotonic()
            try:
                qvec_list = (await self._embed.embed([q]))[0]
            except Exception:
                log.exception("embedding call failed in retrieve()")
                t.ok = False
                return []
            embed_s = time.monotonic() - embed_start
            score_start = time.monotonic()
            qvec = np.asarray(qvec_list, dtype=np.float32)
            matrix = np.vstack([unpack(r["embedding"]) for r in rows])
            cosines = cosine_matrix(qvec, matrix)

            scored: list[tuple[float, float, dict[str, Any]]] = []
            for r, c in zip(rows, cosines):
                score = hybrid_score(
                    float(c), q, r["text"],
                    alpha=self._alpha, beta=self._beta,
                )
                if score >= self._floor:
                    scored.append((score, float(c), r))
            scored.sort(key=lambda t: t[0], reverse=True)
            picked = scored[:lim]
            score_s = time.monotonic() - score_start

            if picked:
                await self._bump_access(*(int(p[2]["id"]) for p in picked))

        operator_log.info("memory_retrieve " + fmt(
            candidates=len(rows),
            picked=len(picked),
            max_score=round(picked[0][0], 3) if picked else 0.0,
            embed_ms=int(embed_s * 1000),
            score_ms=int(score_s * 1000),
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

    async def get_by_id(self, memory_id: int) -> Memory | None:
        """Look up a memory by its primary key. Used by the operator's
        `reply_to_dm` tool to verify that an `authority_memory_id` actually
        resolves to an existing row before sending an autonomous reply.
        Returns None if the id doesn't exist (or has been LRU-evicted)."""
        row = await self._db.memory_get(memory_id, self._embed_model)
        if row is None:
            return None
        return Memory(
            id=int(row["id"]),
            text=str(row["text"]),
            score=0.0,
            cosine=0.0,
            last_accessed_at=str(row["last_accessed_at"]),
        )

    async def delete_by_id(self, memory_id: int) -> bool:
        """Hard-delete one memory by id. Returns True if a row was deleted,
        False if no row matched (already evicted, wrong id, or wrong model).
        Used by the operator's `forget_memory` tool when the user explicitly
        asks to drop a memory. We filter on `model` so a stale-but-pending-
        rebuild row can't be deleted via a different model's id namespace."""
        return await self._db.memory_delete(memory_id, self._embed_model)

    async def dedup_pass(
        self,
        llm: Any,
        *,
        model: str,
        reasoning_effort: str = "medium",
        cluster_threshold: float = 0.80,
        max_cluster_size: int = 8,
    ) -> dict[str, int]:
        """Background dedup pass. Finds connected components in the cosine
        graph (edges ≥ `cluster_threshold`), then for each multi-row cluster
        asks the LLM to either consolidate into one entry or keep separate.

        The LLM is the arbiter, not heuristics — write-time dedup catches only
        near-identical phrasing; this pass picks up the harder cases (templates
        that swap one detail) and treats them correctly by reading the texts.

        Returns counts for logging. Failures only log; the pass is idempotent
        so the next run retries."""
        rows = await self._all_rows()
        if len(rows) < 2:
            return {"clusters_found": 0, "merged": 0, "kept": 0, "failed": 0}
        matrix = np.vstack([unpack(r["embedding"]) for r in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms
        cos = normed @ normed.T

        # Hybrid similarity (same formula retrieval uses): alpha*cos + beta*
        # Jaccard token overlap. Identifier-rich texts that embeddings rank
        # low get a fuzzy boost; pure-template paraphrases still cluster on
        # cosine.
        hybrid = np.zeros_like(cos)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                h = hybrid_score(
                    float(cos[i, j]), rows[i]["text"], rows[j]["text"],
                    alpha=self._alpha, beta=self._beta,
                )
                hybrid[i, j] = h
                hybrid[j, i] = h

        # Pairs the LLM has previously reviewed and decided NOT to merge.
        # Skipping these as edges avoids burning LLM calls re-asking every
        # 5 min about the same John-vs-Jane situations. id_a < id_b.
        skip = await self._load_skip_pairs()

        n = len(rows)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        any_edge = False
        for i in range(n):
            for j in range(i + 1, n):
                if hybrid[i, j] < cluster_threshold:
                    continue
                a = int(rows[i]["id"])
                b = int(rows[j]["id"])
                if (min(a, b), max(a, b)) in skip:
                    continue
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
                any_edge = True
        if not any_edge:
            return {"clusters_found": 0, "merged": 0, "kept": 0, "failed": 0}
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        clusters = [g for g in groups.values() if len(g) >= 2]

        merged_n = kept_n = failed_n = 0
        for indices in clusters:
            if len(indices) > max_cluster_size:
                sub = hybrid[np.ix_(indices, indices)]
                top = np.argsort(-sub.sum(axis=1))[:max_cluster_size]
                indices = [indices[k] for k in top]
            cluster_rows = [rows[i] for i in indices]
            cluster_ids = {int(r["id"]) for r in cluster_rows}
            try:
                decision = await self._dedup_decide(
                    llm, model=model, reasoning_effort=reasoning_effort,
                    cluster_rows=cluster_rows,
                )
            except Exception:
                log.exception("memory_dedup: LLM call failed for cluster %s",
                              sorted(cluster_ids))
                failed_n += 1
                continue
            raw_groups = (decision or {}).get("merge_groups") or []
            if not isinstance(raw_groups, list):
                raw_groups = []
            # Validate. Each group: ids ⊆ cluster_ids, |ids| ≥ 2, text non-empty
            # and within MAX_ENTRY_CHARS. Skip silently anything malformed —
            # the rest of the cluster is untouched.
            valid_groups: list[tuple[list[int], str]] = []
            for g in raw_groups:
                if not isinstance(g, dict):
                    continue
                raw_ids = g.get("ids") or []
                if not isinstance(raw_ids, list):
                    continue
                try:
                    gids = [int(i) for i in raw_ids if int(i) in cluster_ids]
                except (TypeError, ValueError):
                    continue
                if len(gids) < 2:
                    continue
                text = (g.get("text") or "").strip()
                if not text or len(text) > MAX_ENTRY_CHARS:
                    operator_log.info("memory_dedup_bad_merge " + fmt(
                        ids=sorted(gids), reason="empty-or-overlong",
                    ))
                    failed_n += 1
                    continue
                valid_groups.append((gids, text))
            merged_ids = {i for gids, _ in valid_groups for i in gids}
            if not valid_groups:
                kept_n += 1
                operator_log.info("memory_dedup_keep " + fmt(
                    ids=sorted(cluster_ids),
                ))
            else:
                for gids, text in valid_groups:
                    try:
                        vec = (await self._embed.embed([text]))[0]
                    except Exception:
                        log.exception("memory_dedup: embed failed for merged text")
                        failed_n += 1
                        continue
                    await self._insert_row(
                        text=text,
                        embedding=np.asarray(vec, dtype=np.float32),
                        source_turn="dedup",
                    )
                    for rid in gids:
                        await self.delete_by_id(rid)
                    merged_n += 1
                    operator_log.info("memory_dedup_merge " + fmt(
                        ids=sorted(gids), text=text,
                    ))
            # Record skip pairs for every surviving id-pair the LLM saw and
            # did NOT merge. Survivors = cluster ids minus anything that
            # ended up in a merge group (those rows are deleted; recording
            # pairs that reference them is moot and would just clutter the
            # table).
            survivors = sorted(cluster_ids - merged_ids)
            new_skip_pairs: list[tuple[int, int]] = []
            for ai in range(len(survivors)):
                for bi in range(ai + 1, len(survivors)):
                    new_skip_pairs.append((survivors[ai], survivors[bi]))
            if new_skip_pairs:
                await self._record_skip_pairs(new_skip_pairs)
        operator_log.info("memory_dedup " + fmt(
            clusters_found=len(clusters),
            merged=merged_n, kept=kept_n, failed=failed_n,
        ))
        return {
            "clusters_found": len(clusters),
            "merged": merged_n,
            "kept": kept_n,
            "failed": failed_n,
        }

    async def _load_skip_pairs(self) -> set[tuple[int, int]]:
        return await self._db.memory_load_skip_pairs()

    async def _record_skip_pairs(self, pairs: list[tuple[int, int]]) -> None:
        await self._db.memory_record_skip_pairs(pairs)

    async def _dedup_decide(
        self,
        llm: Any,
        *,
        model: str,
        reasoning_effort: str,
        cluster_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """One LLM call: decide merge-or-keep for one cluster. Returns the
        parsed JSON dict or None on parse/empty response (caller logs)."""
        # `recorded_at` is what lets the arbiter tell a paraphrase from a fact
        # that CHANGED. Without it, "lives in Berlin" and "lives in Munich"
        # look like a contradiction it must keep both of — and the operator is
        # left holding two mutually exclusive memories with no way to know
        # which one is current.
        items = json.dumps(
            [
                {
                    "id": int(r["id"]),
                    "text": str(r["text"]),
                    "recorded_at": str(r["created_at"]),
                }
                for r in sorted(cluster_rows, key=lambda r: str(r["created_at"]))
            ],
            ensure_ascii=False,
        )
        system = (
            "You deduplicate stored memories. Return STRICT JSON, no prose.\n"
            "Schema:\n"
            '  {"merge_groups": [\n'
            '    {"ids": [<int>, <int>, ...], "text": "<consolidated text '
            "preserving every detail from each member, ≤512 chars>\"},\n"
            "    ...\n"
            "  ]}\n"
            "Each entry is a SUBSET of the input memories you want to "
            "consolidate into one new entry. Memories are listed oldest-first "
            "and each carries `recorded_at`.\n"
            "Merge a group when its members either:\n"
            "  (a) state the SAME fact (paraphrases of one another), or\n"
            "  (b) state the same attribute of the SAME entity, but the value "
            "CHANGED over time — the user moved, renamed something, switched "
            "tools. Here the NEWEST `recorded_at` wins: write the consolidated "
            "text as the current fact. Keep the superseded value only if it "
            "still carries usable signal (e.g. 'uses Postgres, migrated from "
            "MySQL in 2024'), and never phrase it so the stale value could be "
            "mistaken for current.\n"
            "Memories that refer to DIFFERENT entities (person, host, "
            "version, identifier, scope, etc.) MUST NOT share a group — a "
            "newer memory about a different entity supersedes nothing.\n"
            "A preference or opinion that merely differs is NOT automatically "
            "superseded; people hold several at once. Supersede only when the "
            "newer memory is genuinely incompatible with the older.\n"
            "Memories not listed in any group are kept as-is.\n"
            "If nothing should be merged, return {\"merge_groups\": []}.\n"
            "When in doubt, omit — losing information is worse than keeping "
            "a duplicate."
        )
        resp = await llm.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Memories:\n" + items},
            ],
            tools=[],
            max_tokens=2048,
            reasoning_effort=reasoning_effort,
        )
        content = (resp.get("content") or "").strip()
        if not content:
            return None
        if content.startswith("```"):
            stripped = content.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            content = stripped
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            log.warning("memory_dedup: non-JSON response %r", content[:200])
            return None
        return parsed if isinstance(parsed, dict) else None

    async def entries_count(self) -> int:
        """Count of rows usable for retrieval — i.e., embedded with the
        currently-configured model. Rows pending a rebuild are excluded
        (`stale_count()` exposes that bucket separately)."""
        return await self._db.memory_entries_count(self._embed_model)

    # ---- internals ---------------------------------------------------------

    async def _all_rows(self) -> list[dict[str, Any]]:
        """Returns rows embedded with the CURRENT model only. Rows from older
        models are invisible to retrieval + dedup until rebuilt — that's the
        invariant `rebuild_stale_embeddings()` restores."""
        return await self._db.memory_all_rows(self._embed_model)

    async def _insert_row(
        self,
        *,
        text: str,
        embedding: np.ndarray,
        source_turn: str | None,
    ) -> int:
        return await self._db.memory_insert(
            text=text,
            embedding=embedding.tobytes(),
            model=self._embed_model,
            source_turn=source_turn,
        )

    # ---- rebuild on model change ------------------------------------------

    async def stale_count(self) -> int:
        """Rows whose `model` doesn't match the configured embedder. These
        are skipped by retrieval until `rebuild_stale_embeddings()` runs."""
        return await self._db.memory_stale_count(self._embed_model)

    async def rebuild_stale_embeddings(self, *, batch: int = 32) -> dict[str, int]:
        """Re-embed every row whose stored model differs from the configured
        one. Walks in batches so a long-running call doesn't load the whole
        table into memory and so a partial failure leaves the unconverted
        tail recoverable on next startup. Returns counts for the notifier.
        """
        rebuilt = 0
        failed = 0
        while True:
            stale = await self._db.memory_select_stale_batch(
                self._embed_model, batch,
            )
            if not stale:
                break
            texts = [r["text"] for r in stale]
            try:
                vecs = await self._embed.embed(texts)
            except Exception:
                log.exception("rebuild_stale_embeddings: embed call failed (batch of %d)", len(texts))
                failed += len(texts)
                # Don't loop forever on a persistent failure — bail and let
                # the next startup retry.
                break
            updates = [
                (int(r["id"]), np.asarray(v, dtype=np.float32).tobytes())
                for r, v in zip(stale, vecs)
            ]
            await self._db.memory_update_embeddings(updates, self._embed_model)
            rebuilt += len(updates)
        operator_log.info(
            "memory_rebuild " + fmt(
                rebuilt=rebuilt, failed=failed, model=self._embed_model,
            )
        )
        return {"rebuilt": rebuilt, "failed": failed}

    async def _bump_access(self, *row_ids: int) -> None:
        await self._db.memory_bump_access(list(row_ids))

    async def _maybe_evict(self) -> None:
        await self._db.memory_evict_over_capacity(self._capacity, self._embed_model)
