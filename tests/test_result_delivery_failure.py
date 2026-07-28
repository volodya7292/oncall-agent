"""Delivery-path properties that only show up in production.

Two of these lock down a real incident: the Claude CLI hit its weekly quota
and emitted its rejection as a normal assistant turn ("You've hit your weekly
limit · resets 2am (UTC)"). Delivery could not tell that from an answer, so it
forwarded the string to Telegram and wrote it into operator history as
something the operator had said — six times in ninety minutes, while the user
re-asked and got the same line back each time.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from oncall.config import Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.models import Task
from oncall.result_delivery import (
    MAX_TEXT_CHARS,
    MAX_VOICE_CHARS,
    deliver_executor_result,
)


_RATE_LIMIT_EVENT = {
    "type": "rate_limit_event",
    "rate_limit_info": {
        "status": "rejected",
        "resetsAt": 1785290400,  # 02:00 UTC — the real value from the incident
        "rateLimitType": "seven_day",
    },
}


@pytest.fixture
async def env(tmp_path):
    settings = Settings(
        oncall_token="t", oncall_db_path=tmp_path / "db.sqlite", ai_gateway_api_key="x",
    )
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


async def _task_with_text(
    db: Database, events: EventBus, text: str, *, rate_limited: bool = False,
) -> Task:
    task = Task(
        session_id=str(uuid4()), prompt="do a thing",
        dispatched_by_chat_session="tg-agent-42",
    )
    await db.insert_task(task)
    if rate_limited:
        await events.publish(task.id, "cli.rate_limit_event", {"raw": _RATE_LIMIT_EVENT})
    await events.publish(task.id, "assistant.text", {"text": text})
    return task


async def test_failed_task_text_goes_to_the_operator_not_the_user(env):
    """The CLI's own error text must never reach the user or history."""
    db, events, published = env
    cli_meta = "You've hit your weekly limit · resets 2am (UTC)"
    task = await _task_with_text(db, events, cli_meta, rate_limited=True)
    notes: list[tuple[str, str]] = []

    async def on_failure(session_id: str, note: str) -> None:
        notes.append((session_id, note))

    await deliver_executor_result(
        db=db, events=events, task_id=task.id,
        chat_session_id="tg-agent-42", terminal_state="failed",
        on_failure=on_failure,
    )

    assert not [p for t, p in published if t == "chat.reply"], (
        "a failed task must not publish its text as the answer"
    )
    assert await db.load_chat_history("tg-agent-42") == [], (
        "CLI meta-text must not be persisted as an operator turn"
    )
    assert len(notes) == 1
    session_id, note = notes[0]
    assert session_id == "tg-agent-42"
    # The operator needs the cause (so it can say WHY) and a warning not to
    # parrot the excerpt it's shown.
    assert "quota" in note
    assert "02:00 UTC" in note, "resetsAt must be rendered for the operator"
    assert cli_meta in note
    assert "do not relay it verbatim" in note


async def test_failure_falls_back_to_the_banner_when_the_operator_raises(env):
    """The user must hear *something*. If re-invoking the operator fails, the
    canned notice is still better than silence — that silence was itself a
    past bug (see test_lifecycle_crash_notify)."""
    db, events, published = env
    task = await _task_with_text(db, events, "partial work")

    async def on_failure(session_id: str, note: str) -> None:
        raise RuntimeError("operator is down too")

    await deliver_executor_result(
        db=db, events=events, task_id=task.id,
        chat_session_id="tg-agent-42", terminal_state="failed",
        on_failure=on_failure,
    )
    replies = [p for t, p in published if t == "chat.reply"]
    assert len(replies) == 1
    assert replies[0]["trigger"] == "executor.failed"


@pytest.mark.parametrize(
    "spoken, expect_truncated",
    [(False, False), (True, True)],
    ids=["text-keeps-it", "voice-cuts-it"],
)
async def test_ceiling_depends_on_the_channel(env, spoken, expect_truncated):
    """A reply between the two ceilings survives in text and is cut on a call.

    The 600-char limit exists because a spoken reply is TTS'd; it was applied
    to text chat too, which guillotined ordinary Telegram answers mid-word.
    """
    db, events, published = env
    long_reply = "x" * (MAX_VOICE_CHARS + 100)
    assert len(long_reply) < MAX_TEXT_CHARS
    task = await _task_with_text(db, events, long_reply)

    await deliver_executor_result(
        db=db, events=events, task_id=task.id,
        chat_session_id="tg-agent-42", terminal_state="completed",
        spoken=spoken,
    )
    (reply,) = [p for t, p in published if t == "chat.reply"]
    assert reply["text"].endswith("…") is expect_truncated
    assert (len(reply["text"]) < len(long_reply)) is expect_truncated
    # voice_text always carries the tight ceiling, whatever the channel: a
    # call can start between the on_call check and this publish.
    assert len(reply["voice_text"]) <= MAX_VOICE_CHARS
