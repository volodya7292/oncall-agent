"""Regression: a hand_off'd task that fails before producing any assistant
text must NOT be silent — the user gets a Telegram chat.reply.

Bug it locks down: a missing `claude` binary / session clash made the executor
end FAILED with no output; result-delivery skipped publishing failed-with-no-
text tasks, so the user asked for something and never heard back. Now the
no-text branch notifies on `failed` (but still stays silent for side-effect
`completed` tasks like a lone emoji reaction).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from oncall.config import Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.models import Task
from oncall.result_delivery import deliver_executor_result


@pytest.fixture
async def env(tmp_path):
    settings = Settings(oncall_token="t", oncall_db_path=tmp_path / "db.sqlite", ai_gateway_api_key="x")
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    published: list[tuple[str, dict[str, Any]]] = []
    orig = events.publish_global

    async def spy(type_: str, payload: dict[str, Any]) -> None:
        published.append((type_, payload))
        await orig(type_, payload)

    events.publish_global = spy  # type: ignore[method-assign]
    try:
        yield db, events, published
    finally:
        await db.close()


async def _insert_task(db: Database, chat: str | None) -> Task:
    task = Task(session_id=str(uuid4()), prompt="do a thing", dispatched_by_chat_session=chat)
    await db.insert_task(task)
    return task


async def test_failed_task_with_no_text_notifies_user(env):
    db, events, published = env
    task = await _insert_task(db, "tg-agent-42")
    # No assistant.text events recorded → the silent-failure case.
    await deliver_executor_result(
        db=db, events=events,
        task_id=task.id, chat_session_id="tg-agent-42", terminal_state="failed",
    )
    replies = [p for t, p in published if t == "chat.reply"]
    assert len(replies) == 1
    assert replies[0]["session_id"] == "tg-agent-42"
    assert replies[0]["trigger"] == "executor.failed"
    assert "failed" in replies[0]["text"].lower()


async def test_completed_side_effect_task_stays_silent(env):
    db, events, published = env
    task = await _insert_task(db, "tg-agent-42")
    # e.g. a lone emoji reaction: completed, no assistant text → must NOT spam.
    await deliver_executor_result(
        db=db, events=events,
        task_id=task.id, chat_session_id="tg-agent-42", terminal_state="completed",
    )
    assert not [p for t, p in published if t == "chat.reply"]
