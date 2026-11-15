"""Operator memory, hosted on the Mem0 Platform.

Mem0 owns extraction, deduplication, contradiction handling (its "Dream"
merge/supersede pass) and storage. This module is a thin async REST client
plus the two policies Mem0 has no notion of:

- **Standing instructions** — behavioral rules flagged `standing` in
  metadata, injected into every session instead of retrieved by topic, and
  hard-capped. Only an explicit `store(..., standing=True)` sets the flag;
  automatic extraction never does, so nothing the owner merely mentioned can
  become a rule the agent obeys unasked.
- **Per-session exclusion** — rows already shown in a session are excluded
  inside the search request, not filtered from its output (filtering after
  the fact spends the whole top-k on shown rows and injects nothing).

Write paths: `ingest_turn` hands one owner turn to Mem0's extractor
(`infer=True`, asynchronous on Mem0's side — we poll the event so failures
and results stay visible); `store` writes a fact verbatim (`infer=False`).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .metrics import timed


log = logging.getLogger(__name__)

# Ceiling on one stored fact. Explicit saves are asked to stay ≤200 chars;
# this is defense in depth against a runaway tool argument.
MAX_ENTRY_CHARS = 512
# Bounded extractor inputs so a pasted log can't make every turn expensive.
_PREV_ASSISTANT_CHAR_CAP = 2000
_USER_CHAR_CAP = 4000
# Mem0 rejects longer search queries; the retrieval key puts the turn first,
# so truncating the tail only drops the oldest borrowed context.
_QUERY_CHAR_CAP = 4000


class MemoryUnavailable(RuntimeError):
    """Memory is not configured, or Mem0 answered with an error."""


@dataclass
class Memory:
    id: str
    text: str
    score: float = 0.0


class MemoryStore(Protocol):
    """The slice of memory the Operator depends on. Tests substitute any
    object that satisfies these methods."""

    async def store(
        self, facts: list[str], *, source_turn: str | None = None,
        standing: bool = False,
    ) -> list[str]: ...
    async def ingest_turn(
        self, user_text: str, *, prev_assistant_text: str | None = None,
        source_turn: str | None = None,
    ) -> list[str]: ...
    async def retrieve(
        self, query: str, *, limit: int | None = None,
        exclude_ids: set[str] | None = None,
        include_standing: bool = False,
    ) -> list[Memory]: ...
    async def standing_rows(self) -> list[Memory]: ...
    async def get_by_id(self, memory_id: str) -> Memory | None: ...
    async def delete_by_id(self, memory_id: str) -> bool: ...
    async def entries_count(self) -> int: ...


def _truncate(text: str, limit: int) -> str:
    """Keep head and tail: identifiers cluster at either end of a paste."""
    if len(text) <= limit:
        return text
    half = (limit - 5) // 2
    return f"{text[:half]}\n…\n{text[-half:]}"


def _is_standing(metadata: Any) -> bool:
    # The list endpoint returns metadata booleans as the string "True" while
    # search and get return a real bool; accept both.
    value = (metadata or {}).get("standing") if isinstance(metadata, dict) else None
    return value is True or str(value).lower() == "true"


def _valid_id(memory_id: str) -> bool:
    # Ids reach URL paths from model tool arguments; only real UUIDs pass.
    try:
        uuid.UUID(str(memory_id))
    except ValueError:
        return False
    return True


class Mem0Memory:
    def __init__(
        self,
        *,
        api_key: str,
        user_id: str,
        max_inject: int,
        search_threshold: float,
        standing_cap: int,
        custom_instructions: str = "",
        host: str = "https://api.mem0.ai",
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 20.0,
        event_poll_seconds: float = 2.0,
        event_timeout_seconds: float = 120.0,
        standing_ttl_seconds: float = 60.0,
    ) -> None:
        self._user_id = user_id
        self._max_inject = max_inject
        self._threshold = search_threshold
        self._standing_cap = standing_cap
        self._instructions = custom_instructions.strip()
        self._event_poll = event_poll_seconds
        self._event_timeout = event_timeout_seconds
        self._client = client or httpx.AsyncClient(
            base_url=host,
            timeout=timeout_seconds,
            transport=httpx.AsyncHTTPTransport(retries=2),
        )
        self._client.headers["Authorization"] = f"Token {api_key}"
        # Standing rows are read on every turn and every task spawn. Cached so
        # that is not an HTTP round-trip each time; our own writes invalidate
        # it, and the TTL bounds staleness for edits made in Mem0's dashboard.
        self._standing_ttl = standing_ttl_seconds
        self._standing_cache: tuple[float, list[Memory]] | None = None
        self._standing_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- transport -----------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise MemoryUnavailable(f"mem0 {method} {path}: {type(e).__name__}: {e}") from e
        if resp.status_code >= 400 and resp.status_code != 404:
            raise MemoryUnavailable(
                f"mem0 {method} {path}: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return resp

    def _scope(self, *extra: dict[str, Any]) -> dict[str, Any]:
        return {"AND": [{"user_id": self._user_id}, *extra]}

    async def health(self) -> str | None:
        """None when Mem0 accepts the key, else a one-line reason."""
        try:
            resp = await self._request("GET", "/v1/ping/")
        except MemoryUnavailable as e:
            return str(e)
        return None if resp.status_code == 200 else f"HTTP {resp.status_code}"

    # ---- writes ----------------------------------------------------------------

    async def add_verbatim(
        self, text: str, *, metadata: dict[str, Any] | None = None,
        immutable: bool = False, timestamp: int | None = None,
    ) -> str:
        """Store `text` as-is (no extraction). Returns the new memory id.
        Synchronous on Mem0's side, unlike `infer=True`."""
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": text}],
            "user_id": self._user_id,
            "infer": False,
        }
        if metadata:
            body["metadata"] = metadata
        if immutable:
            body["immutable"] = True
        if timestamp is not None:
            body["timestamp"] = timestamp
        resp = await self._request("POST", "/v3/memories/add/", json=body)
        results = resp.json().get("results") or []
        if resp.status_code != 200 or not results:
            raise MemoryUnavailable(f"mem0 add returned no memory: {resp.text[:300]}")
        return str(results[0]["id"])

    async def store(
        self, facts: list[str], *, source_turn: str | None = None,
        standing: bool = False,
    ) -> list[str]:
        """Write each fact verbatim. Returns the texts actually written; a
        standing write past the cap is refused (not evicted — "least recently
        used" means nothing for rows that are never retrieved), so callers see
        a shorter list."""
        written: list[str] = []
        try:
            for fact in facts:
                text = fact.strip()[:MAX_ENTRY_CHARS]
                if not text:
                    continue
                if standing:
                    current = await self.standing_rows(refresh=True)
                    if len(current) >= self._standing_cap:
                        log.warning(
                            "standing memory cap (%d) reached; refused: %r",
                            self._standing_cap, text[:80],
                        )
                        break
                metadata: dict[str, Any] = {}
                if standing:
                    metadata["standing"] = True
                if source_turn:
                    metadata["source_turn"] = source_turn
                await self.add_verbatim(text, metadata=metadata, immutable=standing)
                written.append(text)
        finally:
            if standing:
                self._standing_cache = None
        return written

    async def ingest_turn(
        self, user_text: str, *, prev_assistant_text: str | None = None,
        source_turn: str | None = None,
    ) -> list[str]:
        """Hand one owner turn to Mem0's extractor and wait for the verdict.
        Returns the facts Mem0 added (empty when it found nothing new). The
        preceding assistant turn rides along as context so references like
        "that one" can be resolved; the custom instructions forbid treating
        it as a source of facts."""
        text = user_text.strip()
        if not text:
            return []
        messages: list[dict[str, str]] = []
        if prev_assistant_text and prev_assistant_text.strip():
            messages.append({
                "role": "assistant",
                "content": _truncate(prev_assistant_text.strip(), _PREV_ASSISTANT_CHAR_CAP),
            })
        messages.append({"role": "user", "content": _truncate(text, _USER_CHAR_CAP)})
        body: dict[str, Any] = {
            "messages": messages, "user_id": self._user_id, "infer": True,
        }
        if self._instructions:
            body["custom_instructions"] = self._instructions
        if source_turn:
            body["metadata"] = {"source_turn": source_turn}
        resp = await self._request("POST", "/v3/memories/add/", json=body)
        event_id = resp.json().get("event_id")
        if not event_id:
            raise MemoryUnavailable(f"mem0 add returned no event_id: {resp.text[:300]}")

        deadline = time.monotonic() + self._event_timeout
        while True:
            ev = (await self._request("GET", f"/v1/event/{event_id}/")).json()
            status = ev.get("status")
            if status == "SUCCEEDED":
                return [
                    str((r.get("data") or {}).get("memory") or "").strip()
                    for r in ev.get("results") or []
                    if r.get("event") == "ADD"
                ]
            if status == "FAILED":
                raise MemoryUnavailable(f"mem0 extraction failed: {ev.get('error')}")
            if time.monotonic() >= deadline:
                raise MemoryUnavailable(
                    f"mem0 extraction still {status} after {self._event_timeout:.0f}s"
                )
            await asyncio.sleep(self._event_poll)

    async def delete_by_id(self, memory_id: str) -> bool:
        if not _valid_id(memory_id):
            return False
        resp = await self._request("DELETE", f"/v1/memories/{memory_id}/")
        self._standing_cache = None
        return resp.status_code == 200

    # ---- reads -----------------------------------------------------------------

    async def retrieve(
        self, query: str, *, limit: int | None = None,
        exclude_ids: set[str] | None = None,
        include_standing: bool = False,
    ) -> list[Memory]:
        """Top matches for `query` above the score threshold. Standing rows
        are excluded unless asked for: they reach every session through
        `standing_rows`, and surfacing them here too would show them twice
        under wording that calls them optional."""
        q = (query or "").strip()[:_QUERY_CHAR_CAP]
        if not q:
            return []
        extra: list[dict[str, Any]] = []
        if not include_standing:
            extra.append({"metadata": {"standing": {"ne": True}}})
        excluded = sorted(i for i in (exclude_ids or ()) if _valid_id(i))
        if excluded:
            extra.append({"NOT": [{"memory_ids": excluded}]})
        body = {
            "query": q,
            "filters": self._scope(*extra),
            "top_k": limit or self._max_inject,
            "threshold": self._threshold,
            "latest_only": True,
        }
        # Rolling "memory" window surfaced in /status.
        with timed("memory"):
            resp = await self._request("POST", "/v3/memories/search/", json=body)
        return [
            Memory(id=str(r["id"]), text=str(r.get("memory") or ""),
                   score=float(r.get("score") or 0.0))
            for r in resp.json().get("results") or []
        ]

    async def standing_rows(self, *, refresh: bool = False) -> list[Memory]:
        """Every standing row, oldest first."""
        async with self._standing_lock:
            cached = self._standing_cache
            if (not refresh and cached is not None
                    and time.monotonic() - cached[0] < self._standing_ttl):
                return list(cached[1])
            resp = await self._request(
                "POST", "/v3/memories/",
                params={"page_size": 200},
                json={
                    "filters": self._scope({"metadata": {"standing": True}}),
                    "latest_only": True,
                },
            )
            rows = sorted(
                (r for r in resp.json().get("results") or []
                 if _is_standing(r.get("metadata"))),
                key=lambda r: str(r.get("created_at") or ""),
            )
            memories = [Memory(id=str(r["id"]), text=str(r.get("memory") or "")) for r in rows]
            self._standing_cache = (time.monotonic(), memories)
            return list(memories)

    async def get_by_id(self, memory_id: str) -> Memory | None:
        if not _valid_id(memory_id):
            return None
        resp = await self._request("GET", f"/v1/memories/{memory_id}/")
        if resp.status_code != 200:
            return None
        row = resp.json()
        # One API key covers the whole Mem0 project; only this owner's rows count.
        if row.get("user_id") != self._user_id:
            return None
        return Memory(id=str(row["id"]), text=str(row.get("memory") or ""))

    async def entries_count(self) -> int:
        resp = await self._request(
            "POST", "/v3/memories/", params={"page_size": 1},
            json={"filters": {"user_id": self._user_id}},
        )
        return int(resp.json().get("count") or 0)

    async def list_all(self) -> list[dict[str, Any]]:
        """Every row for this owner, raw. Used by the legacy import."""
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = await self._request(
                "POST", "/v3/memories/", params={"page": page, "page_size": 200},
                json={"filters": {"user_id": self._user_id}},
            )
            data = resp.json()
            out.extend(data.get("results") or [])
            if not data.get("next"):
                return out
            page += 1


class DisabledMemory:
    """Stand-in when MEM0_API_KEY is unset: the operator still runs, reads
    come back empty, and explicit writes fail loudly instead of pretending."""

    async def aclose(self) -> None:
        return None

    async def store(self, facts, *, source_turn=None, standing=False) -> list[str]:
        raise MemoryUnavailable("memory is disabled: MEM0_API_KEY is not set")

    async def ingest_turn(self, user_text, *, prev_assistant_text=None, source_turn=None) -> list[str]:
        return []

    async def retrieve(self, query, *, limit=None, exclude_ids=None, include_standing=False) -> list[Memory]:
        return []

    async def standing_rows(self) -> list[Memory]:
        return []

    async def get_by_id(self, memory_id) -> Memory | None:
        return None

    async def delete_by_id(self, memory_id) -> bool:
        return False

    async def entries_count(self) -> int:
        return 0
