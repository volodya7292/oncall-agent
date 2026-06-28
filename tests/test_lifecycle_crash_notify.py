"""Regression: an executor task that crashes before producing any assistant
text must NOT be silent — the user gets a Telegram chat.reply.

Bug it locks down: a missing `claude` binary made the supervisor raise
FileNotFoundError; lifecycle logged it and marked the task FAILED, but the
result-delivery loop skips failed tasks with no assistant text, so nothing
reached the user. _supervise now publishes a chat.reply to the originating
chat session.
"""

from __future__ import annotations

from typing import Any

import pytest

from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.models import Task, TaskState, TerminalReason


@pytest.fixture
def settings(tmp_path):
    return Settings(oncall_token="t", oncall_db_path=tmp_path / "db.sqlite", ai_gateway_api_key="x")


@pytest.fixture
async def env(settings):
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events.publish)
    lc = Lifecycle(db=db, broker=broker, approval_client=client,
                   events=events, settings=settings, paths=Paths())
    try:
        yield db, events, lc
    finally:
        await db.close()


class _BoomSupervisor:
    """Mimics the real bug: spawning `claude` fails before any output."""
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def run(self, task: Task, *, resuming: bool = False) -> TerminalReason:
        raise self._exc


async def test_supervisor_crash_notifies_originating_chat(env):
    db, events, lc = env
    published: list[tuple[str, dict[str, Any]]] = []
    orig = events.publish_global

    async def spy(type_: str, payload: dict[str, Any]) -> None:
        published.append((type_, payload))
        await orig(type_, payload)

    events.publish_global = spy  # type: ignore[method-assign]

    task = Task(session_id="sess-1", prompt="do a thing",
                dispatched_by_chat_session="tg-agent-42")
    await db.insert_task(task)

    res = await lc._supervise(task, _BoomSupervisor(FileNotFoundError(2, "No such file or directory")))

    # Task is marked failed AND the user is told (not silent).
    assert res == TerminalReason.CLI_ERROR
    assert (await db.get_task(task.id)).state == TaskState.FAILED
    replies = [p for t, p in published if t == "chat.reply"]
    assert any(
        p["session_id"] == "tg-agent-42" and "failed" in p["text"].lower()
        for p in replies
    ), f"expected a crash chat.reply to the originating chat, got {published!r}"


async def test_no_chat_session_no_publish(env):
    """A task with no originating chat (e.g. an internal task) must not try to
    publish a reply to a nonexistent session."""
    db, events, lc = env
    published: list[tuple[str, dict[str, Any]]] = []
    orig = events.publish_global

    async def spy(type_: str, payload: dict[str, Any]) -> None:
        published.append((type_, payload))
        await orig(type_, payload)

    events.publish_global = spy  # type: ignore[method-assign]

    task = Task(session_id="sess-2", prompt="x", dispatched_by_chat_session=None)
    await db.insert_task(task)
    await lc._supervise(task, _BoomSupervisor(RuntimeError("boom")))

    assert not [p for t, p in published if t == "chat.reply"]
