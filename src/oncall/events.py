"""In-process event bus.

Events are also appended to `task_events` (via db.append_event) so that an SSE
subscriber connecting late can replay history from a cursor before tailing the
live stream.
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

    async def subscribe(
        self, task_id: UUID, *, since_seq: int = 0
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay any persisted events newer than `since_seq`, then yield live ones."""
        # Replay historic events first.
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
