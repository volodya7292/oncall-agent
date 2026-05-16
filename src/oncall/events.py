"""In-process event bus.

Per-task events are also appended to `task_events` (via db.append_event) so an
SSE subscriber connecting late can replay history from a cursor before tailing
the live stream.

A second, global channel exists for clients (e.g. the `oncall chat` REPL) that
want to see events across all tasks without picking a single task_id. The
global channel is live-only — no replay. Global subscribers receive envelopes
shaped {task_id: str|None, type: str, payload: dict, seq: int}; per-task
subscribers continue to receive the original {seq, type, payload} shape.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, AsyncIterator
from uuid import UUID

from .db import Database


log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._subs: dict[UUID, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._global_subs: set[asyncio.Queue[dict[str, Any]]] = set()

    async def publish(self, task_id: UUID, type_: str, payload: dict[str, Any]) -> None:
        try:
            seq = await self._db.append_event(task_id, type_, payload)
        except Exception:
            log.exception("failed to persist event %s for task %s", type_, task_id)
            seq = -1
        envelope = {"seq": seq, "type": type_, "payload": payload}
        for q in list(self._subs.get(task_id, set())):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                log.warning("subscriber queue full for task %s; dropping event", task_id)
        self._fanout_global({
            "task_id": str(task_id),
            "type": type_,
            "payload": payload,
            "seq": seq,
        })

    async def publish_global(self, type_: str, payload: dict[str, Any]) -> None:
        """Emit an event that isn't tied to a single task — e.g. `messenger.received`.
        Not persisted (no task_id to scope replay to)."""
        self._fanout_global({
            "task_id": None,
            "type": type_,
            "payload": payload,
            "seq": -1,
        })

    def _fanout_global(self, envelope: dict[str, Any]) -> None:
        for q in list(self._global_subs):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                log.warning("global subscriber queue full; dropping event %s", envelope["type"])

    async def subscribe(
        self, task_id: UUID, *, since_seq: int = 0
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay any persisted events newer than `since_seq`, then yield live ones."""
        for e in await self._db.list_events(task_id, since_seq=since_seq):
            yield e

        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self._subs[task_id].add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subs[task_id].discard(q)
            if not self._subs[task_id]:
                del self._subs[task_id]

    async def subscribe_global(
        self, *, types: set[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield live events across all tasks, optionally filtered by type.
        No replay — global stream is live-only."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self._global_subs.add(q)
        try:
            while True:
                envelope = await q.get()
                if types is None or envelope["type"] in types:
                    yield envelope
        finally:
            self._global_subs.discard(q)
