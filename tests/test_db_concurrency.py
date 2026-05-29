"""Concurrency regression: shared-connection commit must not crash while
another task holds an active statement.

Bug: inbox-drain / result_delivery intermittently raised
`sqlite3.OperationalError: cannot commit transaction - SQL statements in
progress`. Root cause: every task shares one aiosqlite connection, and
`append_event` runs `INSERT ... RETURNING seq`. Between the `execute()` (which
leaves the RETURNING *write* statement active) and the following `fetchone()`,
the event loop yields. A different task's `commit()` landing in that window
hit the active write statement and raised. aiosqlite serializes individual
ops but not multi-step sequences, so the fix is a per-connection lock that
makes each Database method atomic on the connection.

This test forces the exact interleave deterministically by parking
`append_event` inside its active-RETURNING window (via a thin connection
proxy) and firing a concurrent committer. Without the lock the committer
crashes; with it, the committer simply waits for the lock and succeeds.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from oncall.db import Database
from oncall.models import Task, TaskState


class _PausingResult:
    """Wraps the awaitable/async-CM returned by `Connection.execute`. When
    armed, it pauses *after* the cursor exists (RETURNING statement active)
    but *before* the caller can fetch — reproducing the production yield
    window between append_event's execute() and fetchone()."""

    def __init__(self, inner, *, pause: bool, gate: asyncio.Event, release: asyncio.Event):
        self._inner = inner
        self._pause = pause
        self._gate = gate
        self._release = release

    async def _maybe_pause(self):
        if self._pause:
            self._gate.set()
            await self._release.wait()

    async def __aenter__(self):
        cur = await self._inner.__aenter__()
        await self._maybe_pause()
        return cur

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)

    def __await__(self):
        async def _consume():
            cur = await self._inner
            await self._maybe_pause()
            return cur
        return _consume().__await__()


class _ConnProxy:
    """Delegates everything to the real aiosqlite connection, but the first
    `execute` carrying a RETURNING clause is parked mid-flight."""

    def __init__(self, real, gate: asyncio.Event, release: asyncio.Event):
        self._real = real
        self._gate = gate
        self._release = release
        self._armed = True

    def __getattr__(self, name):
        return getattr(self._real, name)

    def execute(self, sql, *args, **kwargs):
        pause = self._armed and "RETURNING" in sql
        if pause:
            self._armed = False
        return _PausingResult(
            self._real.execute(sql, *args, **kwargs),
            pause=pause, gate=self._gate, release=self._release,
        )


@pytest.mark.asyncio
async def test_concurrent_commit_during_returning_window(tmp_path):
    db = Database(tmp_path / "race.db")
    await db.connect()

    gate = asyncio.Event()      # set once append_event's RETURNING is active
    release = asyncio.Event()   # set to let append_event proceed past the window
    app_task = com_task = None
    com_exc: BaseException | None = None
    try:
        tid = uuid4()
        await db.insert_task(Task(
            id=tid, session_id="s-" + str(tid), state=TaskState.RUNNING,
            prompt="p", model=None, max_turns=None,
        ))
        db._conn = _ConnProxy(db._conn, gate, release)

        async def appender():
            await db.append_event(tid, "assistant.text", {"x": 1})

        async def committer():
            # Lands while the appender is parked with an active RETURNING write.
            await db.mark_inbox_triaged(["inbox-x"])

        app_task = asyncio.create_task(appender())
        await asyncio.wait_for(gate.wait(), timeout=2.0)

        com_task = asyncio.create_task(committer())
        # Give the committer a real chance to attempt its commit in the window.
        await asyncio.sleep(0.05)
        release.set()

        # Capture (don't re-raise) so cleanup always runs — the aiosqlite
        # worker is a non-daemon thread; skipping db.close() wedges teardown.
        try:
            await asyncio.wait_for(com_task, timeout=2.0)
        except BaseException as e:  # noqa: BLE001 — recorded, asserted below
            com_exc = e
        await asyncio.wait_for(app_task, timeout=2.0)

        # The concurrent commit must not crash. Pre-fix it raises
        # OperationalError("cannot commit transaction - SQL statements in
        # progress"); the per-connection lock makes it wait instead.
        assert com_exc is None, f"concurrent commit crashed: {com_exc!r}"

        events = await db.list_events(tid)
        assert len(events) == 1
    finally:
        release.set()
        for t in (com_task, app_task):
            if t is not None and not t.done():
                t.cancel()
        await db.close()
