"""Lifecycle.recover() — daemon-restart behavior.

With the single-session executor model:
  * RUNNING / AWAITING_APPROVAL tasks left by a prior daemon are marked
    FAILED — we can't safely re-attach a mid-turn into the shared
    claude session.
  * PENDING tasks (queued before the crash) re-enter the FIFO queue in
    submission order.

FakeSupervisor stands in for the real one so no claude binary is invoked.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from oncall import lifecycle as lifecycle_mod
from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.models import Task, TaskState, TerminalReason


class FakeSupervisor:
    instances: list["FakeSupervisor"] = []
    enter_order: list[str] = []

    def __init__(self, **_ignored: Any) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self.release = asyncio.Event()
        FakeSupervisor.instances.append(self)

    async def run(self, task: Task, *, resuming: bool = False) -> TerminalReason:
        self.run_calls.append({"task_id": task.id, "resuming": resuming, "prompt": task.prompt})
        FakeSupervisor.enter_order.append(task.prompt)
        await self.release.wait()
        return TerminalReason.SUCCESS


@pytest.fixture(autouse=True)
def reset_supervisors():
    FakeSupervisor.instances.clear()
    FakeSupervisor.enter_order.clear()
    yield
    FakeSupervisor.instances.clear()
    FakeSupervisor.enter_order.clear()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        oncall_token="t",
        oncall_db_path=tmp_path / "db.sqlite",
        ai_gateway_api_key="x",
    )


@pytest.fixture
async def stack(settings, monkeypatch):
    monkeypatch.setattr(lifecycle_mod, "Supervisor", FakeSupervisor)
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events.publish)
    lc = Lifecycle(
        db=db, broker=broker, approval_client=client,
        events=events, settings=settings, paths=Paths(),
    )
    try:
        yield {"lc": lc, "db": db, "broker": broker}
    finally:
        for s in FakeSupervisor.instances:
            s.release.set()
        await lc.shutdown()
        await db.close()


async def _insert_task(db: Database, *, state: TaskState, prompt: str = "x") -> Task:
    from uuid import uuid4
    task = Task(session_id=str(uuid4()), prompt=prompt, state=state)
    await db.insert_task(task)
    return task


async def _wait_for(predicate, *, timeout: float = 1.0, interval: float = 0.005):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"predicate did not become true within {timeout}s")


@pytest.mark.asyncio
async def test_recover_marks_stale_running_and_awaiting_as_failed(stack):
    """RUNNING / AWAITING_APPROVAL from a prior daemon can't be safely
    re-attached when every executor invocation shares one claude
    session — recover marks them FAILED so the queue starts clean."""
    db = stack["db"]
    t_running = await _insert_task(db, state=TaskState.RUNNING)
    t_awaiting = await _insert_task(db, state=TaskState.AWAITING_APPROVAL)

    await stack["lc"].recover()

    refreshed_run = await db.get_task(t_running.id)
    refreshed_awa = await db.get_task(t_awaiting.id)
    assert refreshed_run is not None and refreshed_run.state == TaskState.FAILED
    assert refreshed_awa is not None and refreshed_awa.state == TaskState.FAILED


@pytest.mark.asyncio
async def test_recover_requeues_pending_tasks_in_order(stack):
    """PENDING tasks (queued before the crash, never started) go back
    onto the FIFO queue in submission order."""
    db = stack["db"]
    t1 = await _insert_task(db, state=TaskState.PENDING, prompt="first")
    t2 = await _insert_task(db, state=TaskState.PENDING, prompt="second")

    await stack["lc"].recover()

    # Worker pulls them one at a time. Release as we observe entries.
    await _wait_for(lambda: len(FakeSupervisor.instances) >= 1)
    await _wait_for(lambda: FakeSupervisor.instances[0].run_calls)
    FakeSupervisor.instances[0].release.set()
    await _wait_for(lambda: len(FakeSupervisor.instances) >= 2)
    await _wait_for(lambda: FakeSupervisor.instances[1].run_calls)

    assert FakeSupervisor.enter_order == ["first", "second"]
    # Re-queued tasks are NOT resuming — they're fresh dequeues.
    for s in FakeSupervisor.instances[:2]:
        for c in s.run_calls:
            assert c["resuming"] is False
    del t1, t2


@pytest.mark.asyncio
async def test_recover_no_op_on_clean_db(stack):
    db = stack["db"]
    await _insert_task(db, state=TaskState.COMPLETED)
    await _insert_task(db, state=TaskState.FAILED)

    await stack["lc"].recover()
    await asyncio.sleep(0.02)

    assert FakeSupervisor.instances == []
