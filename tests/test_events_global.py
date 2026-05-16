"""EventBus global subscription channel.

Covers:
  * `publish` fans out to BOTH the per-task subscriber and global subscribers.
  * `publish_global` reaches global subscribers without a task_id.
  * `subscribe_global(types=...)` filters by event type.
  * Per-task `subscribe` still works (regression).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from oncall.db import Database
from oncall.events import EventBus


@pytest.fixture
async def bus(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    await db.connect()
    yield EventBus(db)
    await db.close()


async def _collect(agen, n, *, timeout=0.5):
    """Drain n envelopes from an async generator within the timeout."""
    items: list = []

    async def _take():
        async for item in agen:
            items.append(item)
            if len(items) >= n:
                return

    try:
        await asyncio.wait_for(_take(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return items


@pytest.mark.asyncio
async def test_publish_fans_out_to_global_and_per_task(bus):
    # Need a task row before append_event can persist (FK on task_events).
    from oncall.models import Task
    task = Task(session_id=str(uuid4()), prompt="x")
    await bus._db.insert_task(task)

    task_envs: list = []
    global_envs: list = []

    async def per_task():
        async for e in bus.subscribe(task.id):
            task_envs.append(e)
            return

    async def globally():
        async for e in bus.subscribe_global():
            global_envs.append(e)
            return

    consumers = asyncio.gather(per_task(), globally())
    # Give subscribers a tick to attach.
    await asyncio.sleep(0.01)
    await bus.publish(task.id, "state.changed", {"state": "running"})
    await asyncio.wait_for(consumers, timeout=0.5)

    assert task_envs[0]["type"] == "state.changed"
    assert task_envs[0]["payload"] == {"state": "running"}
    assert global_envs[0]["type"] == "state.changed"
    assert global_envs[0]["task_id"] == str(task.id)
    assert global_envs[0]["payload"] == {"state": "running"}


@pytest.mark.asyncio
async def test_publish_global_does_not_require_task(bus):
    collected: list = []

    async def _read():
        async for e in bus.subscribe_global():
            collected.append(e)
            return

    reader = asyncio.create_task(_read())
    await asyncio.sleep(0.01)
    await bus.publish_global("messenger.received", {"body": "hi"})
    await asyncio.wait_for(reader, timeout=0.5)

    assert collected[0]["type"] == "messenger.received"
    assert collected[0]["task_id"] is None
    assert collected[0]["payload"] == {"body": "hi"}


@pytest.mark.asyncio
async def test_global_type_filter_excludes_other_types(bus):
    """Wanted=approval.requested → other events must be skipped."""
    collected: list = []

    async def _read():
        async for e in bus.subscribe_global(types={"approval.requested"}):
            collected.append(e)
            return

    reader = asyncio.create_task(_read())
    await asyncio.sleep(0.01)
    await bus.publish_global("messenger.received", {"body": "hi"})  # filtered out
    await bus.publish_global("approval.requested", {"id": "a1"})    # delivered
    await asyncio.wait_for(reader, timeout=0.5)

    assert len(collected) == 1
    assert collected[0]["type"] == "approval.requested"


@pytest.mark.asyncio
async def test_two_global_subscribers_both_receive(bus):
    """Multiple REPL terminals attached at once both see the same event."""
    a: list = []
    b: list = []

    async def _reader(out):
        async for e in bus.subscribe_global():
            out.append(e)
            return

    ra = asyncio.create_task(_reader(a))
    rb = asyncio.create_task(_reader(b))
    await asyncio.sleep(0.01)
    await bus.publish_global("messenger.received", {"body": "broadcast"})
    await asyncio.wait_for(asyncio.gather(ra, rb), timeout=0.5)

    assert len(a) == 1 and len(b) == 1
    assert a[0]["payload"] == b[0]["payload"] == {"body": "broadcast"}


@pytest.mark.asyncio
async def test_subscribe_global_can_be_aclosed_while_idle(bus):
    """Regression: a subscriber pre-fetching __anext__ then aclose()'ing
    raced and threw 'aclose(): asynchronous generator is already running'.
    Closing an idle generator must work cleanly."""
    agen = bus.subscribe_global()
    pending = asyncio.ensure_future(agen.__anext__())  # mirrors the SSE pattern
    await asyncio.sleep(0.01)  # let it park on the queue
    pending.cancel()
    try:
        await pending
    except BaseException:
        pass
    # This must not raise RuntimeError.
    await agen.aclose()


@pytest.mark.asyncio
async def test_per_task_subscribe_unaffected_by_global(bus):
    """Regression: adding the global channel must not break per-task replay."""
    from oncall.models import Task
    task = Task(session_id=str(uuid4()), prompt="x")
    await bus._db.insert_task(task)
    # Publish first, then subscribe → replay path must surface it.
    await bus.publish(task.id, "result.final", {"reason": "success"})

    collected = []
    async def _read():
        async for e in bus.subscribe(task.id):
            collected.append(e)
            return

    await asyncio.wait_for(_read(), timeout=0.5)
    assert collected[0]["type"] == "result.final"
