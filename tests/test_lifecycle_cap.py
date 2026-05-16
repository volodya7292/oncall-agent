"""Concurrency cap on executor tasks.

Lifecycle.submit_task() always returns immediately, but only N tasks may
actually be running in supervisor.run() concurrently. Excess submissions
wait at the semaphore inside _run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from oncall import lifecycle as lifecycle_mod
from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.models import TerminalReason


class FakeSupervisor:
    """Replaces the real Supervisor in tests. Each instance blocks on its
    own `release` event so the test controls when it 'finishes'."""

    # Class-level: every instance registers itself here so tests can
    # find and release them.
    instances: list["FakeSupervisor"] = []

    def __init__(self, **_ignored: Any) -> None:
        self.entered = asyncio.Event()  # set when supervisor.run starts
        self.release = asyncio.Event()  # test must set to let it return
        FakeSupervisor.instances.append(self)

    async def run(self, task, *, resuming: bool = False) -> TerminalReason:
        self.entered.set()
        await self.release.wait()
        return TerminalReason.SUCCESS


@pytest.fixture(autouse=True)
def reset_supervisors():
    FakeSupervisor.instances.clear()
    yield
    FakeSupervisor.instances.clear()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        oncall_token="t",
        oncall_db_path=tmp_path / "db.sqlite",
        oncall_max_concurrent_tasks=2,  # tight cap for tests
        ai_gateway_api_key="x",
    )


@pytest.fixture
async def lc(settings, monkeypatch):
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
        yield lc
    finally:
        # Release any blocked supervisors so cleanup doesn't hang.
        for s in FakeSupervisor.instances:
            s.release.set()
        await lc.shutdown()
        await db.close()


async def _wait_for(predicate, *, timeout: float = 1.0, interval: float = 0.005):
    """Spin-wait helper — checks predicate up to timeout."""
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"predicate did not become true within {timeout}s")


@pytest.mark.asyncio
async def test_cap_holds_excess_tasks_in_pending(lc):
    """With cap=2, the third submitted task must NOT enter supervisor.run
    until one of the first two finishes."""
    cap = lc.settings.oncall_max_concurrent_tasks
    assert cap == 2

    # Submit 3 tasks. submit_task returns immediately.
    tasks = []
    for i in range(3):
        t = await lc.submit_task(prompt=f"task {i}")
        tasks.append(t)

    # Three FakeSupervisors instantiated immediately (one per asyncio runner).
    await _wait_for(lambda: len(FakeSupervisor.instances) == 3)

    # But only the first two are inside supervisor.run; the third is parked
    # at the semaphore inside _run.
    await _wait_for(lambda: sum(s.entered.is_set() for s in FakeSupervisor.instances) == 2)
    await asyncio.sleep(0.02)  # give the 3rd a chance if it were going to start
    assert sum(s.entered.is_set() for s in FakeSupervisor.instances) == 2

    # Release one of the running supervisors; the queued one must now enter.
    FakeSupervisor.instances[0].release.set()
    await _wait_for(lambda: sum(s.entered.is_set() for s in FakeSupervisor.instances) == 3)


@pytest.mark.asyncio
async def test_under_cap_runs_immediately(lc):
    """With cap=2, two tasks both enter run() without queueing."""
    await lc.submit_task(prompt="a")
    await lc.submit_task(prompt="b")
    await _wait_for(lambda: all(s.entered.is_set() for s in FakeSupervisor.instances)
                            and len(FakeSupervisor.instances) == 2)


@pytest.mark.asyncio
async def test_killed_queued_task_releases_slot(lc):
    """Killing a queued task must not consume a slot — the cap math has to
    account for cancellation while waiting on the semaphore."""
    a = await lc.submit_task(prompt="a")
    b = await lc.submit_task(prompt="b")
    c = await lc.submit_task(prompt="c")  # queued behind cap=2
    await _wait_for(lambda: sum(s.entered.is_set() for s in FakeSupervisor.instances) == 2)

    # Kill the queued one. Should resolve cleanly, no semaphore leak.
    assert await lc.kill(c.id, reason="test") is True
    # The other two should still be in their original entered state.
    assert sum(s.entered.is_set() for s in FakeSupervisor.instances[:2]) == 2

    # Now finish one running task; nothing new should try to claim the slot
    # for `c` (it was killed).
    FakeSupervisor.instances[0].release.set()
    await asyncio.sleep(0.02)
    # Still only 2 supervisors that ever entered.
    assert sum(s.entered.is_set() for s in FakeSupervisor.instances) == 2
