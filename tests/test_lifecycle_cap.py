"""Single-worker FIFO serialization.

Every hand_off lands on one queue drained by one worker, so executor
runs never overlap (they share a global claude --session-id). Order of
arrival = order of execution.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from oncall import lifecycle as lifecycle_mod
from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker

from tests.support import stub_classifier
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.models import TerminalReason


class FakeSupervisor:
    """Replaces the real Supervisor. Each instance blocks on `release`
    so the test controls when the run "finishes"."""

    instances: list["FakeSupervisor"] = []
    enter_order: list[str] = []

    def __init__(self, **_ignored: Any) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.task = None  # populated by run()
        FakeSupervisor.instances.append(self)

    async def run(self, task, *, resuming: bool = False) -> TerminalReason:
        self.task = task
        FakeSupervisor.enter_order.append(task.prompt)
        self.entered.set()
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
async def lc(settings, monkeypatch):
    monkeypatch.setattr(lifecycle_mod, "Supervisor", FakeSupervisor)
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events.publish, classifier=stub_classifier())
    lc = Lifecycle(
        db=db, broker=broker, approval_client=client,
        events=events, settings=settings, paths=Paths(),
    )
    try:
        yield lc
    finally:
        for s in FakeSupervisor.instances:
            s.release.set()
        await lc.shutdown()
        await db.close()


async def _wait_for(predicate, *, timeout: float = 1.0, interval: float = 0.005):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"predicate did not become true within {timeout}s")


@pytest.mark.asyncio
async def test_serialized_one_at_a_time(lc):
    """Three rapid enqueues run strictly one at a time, in submission order."""
    for i in range(3):
        await lc.enqueue_executor(prompt=f"task {i}")

    # First task enters supervisor.run; the other two wait.
    await _wait_for(lambda: len(FakeSupervisor.instances) == 1)
    await _wait_for(lambda: FakeSupervisor.instances[0].entered.is_set())
    await asyncio.sleep(0.02)
    assert len(FakeSupervisor.instances) == 1, "only one supervisor active at a time"

    # Release first → second enters.
    FakeSupervisor.instances[0].release.set()
    await _wait_for(lambda: len(FakeSupervisor.instances) == 2)
    await _wait_for(lambda: FakeSupervisor.instances[1].entered.is_set())
    await asyncio.sleep(0.02)
    assert len(FakeSupervisor.instances) == 2

    # Release second → third enters.
    FakeSupervisor.instances[1].release.set()
    await _wait_for(lambda: len(FakeSupervisor.instances) == 3)
    await _wait_for(lambda: FakeSupervisor.instances[2].entered.is_set())

    # FIFO order preserved.
    assert FakeSupervisor.enter_order == ["task 0", "task 1", "task 2"]


@pytest.mark.asyncio
async def test_busy_flag_reported_to_caller(lc):
    """enqueue_executor returns busy=True when another task is already running."""
    first = await lc.enqueue_executor(prompt="first")
    assert first["busy"] is False  # nothing was running when we enqueued

    # Wait for worker to pick it up.
    await _wait_for(lambda: lc.acting_status()["busy"] is True)

    second = await lc.enqueue_executor(prompt="second")
    assert second["busy"] is True
    assert second["queue_depth"] >= 1
