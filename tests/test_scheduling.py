"""Scheduling loop: due checks fire, recurring checks re-arm, cancel stops a
fire, and the loop's crash circuit-breaker trips after 3 consecutive failures.

These are the non-obvious flows: the timing gate (fire_at <= now), the recurring
reschedule arithmetic, and the safety invariant that a wedged poll doesn't retry
forever (it dies loudly after 3 strikes, per the background-loop contract in
CLAUDE.md).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from oncall.api import _scheduling_loop, _scheduling_poll
from oncall.config import Settings
from oncall.db import Database
from oncall.models import utcnow


@pytest.fixture
async def db(tmp_path):
    settings = Settings(oncall_token="t", oncall_db_path=tmp_path / "db.sqlite", ai_gateway_api_key="x")
    database = Database(settings.oncall_db_path)
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


class FakeLifecycle:
    """Records submit_task calls instead of spawning a real executor. The
    scheduling loop's job ends at handing the prompt + chat session to the
    executor mechanism; actual result delivery is exercised by the
    result-delivery tests (it keys off the chat_session_id we assert here)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def submit_task(self, *, prompt: str, chat_session_id: str | None = None, **_: Any):
        self.calls.append({"prompt": prompt, "chat_session_id": chat_session_id})
        return None


class SpyEvents:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def publish_global(self, type_: str, payload: dict[str, Any]) -> None:
        self.published.append((type_, payload))


async def test_due_check_fires_and_marks_done(db):
    lc = FakeLifecycle()
    # One-off, due a minute ago.
    await db.create_scheduled_check(
        check_id="c1", chat_session_id="tg-agent-1",
        prompt="check incident X and tell me if it changed",
        fire_at=utcnow() - timedelta(seconds=60), interval_seconds=None,
    )
    fired = await _scheduling_poll(db=db, lifecycle=lc)
    assert fired == 1
    # Handed to the executor with the prompt + the chat that gets notified.
    assert lc.calls == [{
        "prompt": "check incident X and tell me if it changed",
        "chat_session_id": "tg-agent-1",
    }]
    # One-off is retired so it never fires again.
    row = await db.get_scheduled_check("c1")
    assert row["status"] == "done"
    assert row["last_fired_at"] is not None
    # A second poll does nothing.
    assert await _scheduling_poll(db=db, lifecycle=lc) == 0
    assert len(lc.calls) == 1


async def test_future_check_does_not_fire(db):
    lc = FakeLifecycle()
    await db.create_scheduled_check(
        check_id="future", chat_session_id="tg-agent-1", prompt="later",
        fire_at=utcnow() + timedelta(hours=1), interval_seconds=None,
    )
    assert await _scheduling_poll(db=db, lifecycle=lc) == 0
    assert lc.calls == []
    assert (await db.get_scheduled_check("future"))["status"] == "pending"


async def test_recurring_check_rearms_to_now_plus_interval(db):
    lc = FakeLifecycle()
    await db.create_scheduled_check(
        check_id="rec", chat_session_id="tg-agent-1", prompt="poll status",
        fire_at=utcnow() - timedelta(seconds=5), interval_seconds=3600,
    )
    before = utcnow()
    await _scheduling_poll(db=db, lifecycle=lc)
    after = utcnow()

    assert len(lc.calls) == 1
    row = await db.get_scheduled_check("rec")
    # Still pending — it recurs.
    assert row["status"] == "pending"
    assert row["consecutive_failures"] == 0
    # Next fire is interval seconds from the poll's `now`, NOT from the old
    # (overdue) fire_at — so a downtime doesn't storm through missed slots.
    from datetime import datetime
    next_fire = datetime.fromisoformat(row["fire_at"])
    assert before + timedelta(seconds=3600) <= next_fire <= after + timedelta(seconds=3600)


async def test_cancel_prevents_fire(db):
    lc = FakeLifecycle()
    await db.create_scheduled_check(
        check_id="c2", chat_session_id="tg-agent-1", prompt="check X",
        fire_at=utcnow() - timedelta(seconds=30), interval_seconds=None,
    )
    # A different session/chat cannot cancel it.
    assert await db.cancel_scheduled_check("c2", chat_session_id="tg-agent-OTHER") is False
    # The owning chat can.
    assert await db.cancel_scheduled_check("c2", chat_session_id="tg-agent-1") is True

    fired = await _scheduling_poll(db=db, lifecycle=lc)
    assert fired == 0
    assert lc.calls == []
    assert (await db.get_scheduled_check("c2"))["status"] == "cancelled"
    # Cancelling an already-cancelled row is a no-op.
    assert await db.cancel_scheduled_check("c2", chat_session_id="tg-agent-1") is False


class _AlwaysFailDB:
    """Stub whose due-query always raises — drives the loop's crash path."""

    def __init__(self) -> None:
        self.calls = 0

    async def list_due_scheduled_checks(self, now):
        self.calls += 1
        raise RuntimeError("db exploded")


async def test_loop_circuit_breaker_trips_after_three_crashes():
    events = SpyEvents()
    bad_db = _AlwaysFailDB()
    # poll_interval 0 so the three strikes happen without real waiting.
    with pytest.raises(RuntimeError):
        await _scheduling_loop(
            db=bad_db, lifecycle=FakeLifecycle(),
            poll_interval_seconds=0, events=events,
            notify_session_id="tg-agent-1",
        )
    # Exactly three attempts, then it gives up (stays tripped: the coroutine
    # exited by raising rather than looping forever).
    assert bad_db.calls == 3
    # Each crash notified, plus the final "giving up" notice.
    msgs = [p["text"] for t, p in events.published if t == "chat.reply"]
    assert len(msgs) == 4
    assert any("giving up" in m for m in msgs)


class _FlakyThenStopDB:
    """Fails a couple times, succeeds (which must reset the crash counter),
    fails a couple more, then raises CancelledError to end the loop. If the
    reset works the breaker never trips (max 2 consecutive < 3)."""

    def __init__(self) -> None:
        self.calls = 0

    async def list_due_scheduled_checks(self, now):
        self.calls += 1
        if self.calls in (1, 2, 4, 5):
            raise RuntimeError("transient")
        if self.calls == 3:
            return []  # success → resets consecutive_crashes
        raise asyncio.CancelledError  # call 6: stop the loop cleanly


async def test_successful_iteration_resets_the_breaker():
    events = SpyEvents()
    flaky = _FlakyThenStopDB()
    # A success between the two failure pairs means we never hit 3-in-a-row,
    # so the loop survives 4 total failures and only exits on the cancel.
    with pytest.raises(asyncio.CancelledError):
        await _scheduling_loop(
            db=flaky, lifecycle=FakeLifecycle(),
            poll_interval_seconds=0, events=events,
            notify_session_id="tg-agent-1",
        )
    assert flaky.calls == 6
    # Never emitted a "giving up" notice — the breaker did not trip.
    assert not any("giving up" in p["text"] for t, p in events.published if t == "chat.reply")
