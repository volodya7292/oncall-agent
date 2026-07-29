"""Delivery-path properties that only show up in production.

Two of these lock down a real incident: the Claude CLI hit its weekly quota
and emitted its rejection as a normal assistant turn ("You've hit your weekly
limit · resets 2am (UTC)"). Delivery could not tell that from an answer, so it
forwarded the string to Telegram and wrote it into operator history as
something the operator had said — six times in ninety minutes, while the user
re-asked and got the same line back each time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from oncall.api import deliver_failure_via_operator
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
    first_pass_answer: str | None = None,
) -> Task:
    task = Task(
        session_id=str(uuid4()), prompt="do a thing",
        dispatched_by_chat_session="tg-agent-42",
        first_pass_answer=first_pass_answer,
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


async def test_operator_answer_to_a_failed_hand_off_reaches_the_user(env):
    """Regression: the operator's answer must be PUBLISHED, not just written.

    `auto_ping` runs the turn and records it; sending is the caller's job.
    Shipped once without the publish — the operator wrote good answers that
    stopped in the DB while the user, holding an ack, heard nothing and asked
    again. Asserting only that `on_failure` fired does not catch this, which
    is why this test goes through the real publish path.
    """
    db, events, published = env

    class StubOperator:
        def __init__(self) -> None:
            self.pings: list[tuple[str, bool]] = []

        async def auto_ping(self, session_id, note, **kwargs):
            self.pings.append((session_id, kwargs.get("allow_hand_off", True)))
            return SimpleNamespace(text="Can't reach my tools — Oakley or Uvex.")

    op = StubOperator()
    await deliver_failure_via_operator(
        operator=op, events=events, chat_session_id="tg-agent-42",
        note="the job you handed off failed",
    )

    assert op.pings == [("tg-agent-42", False)], "must block hand_off for the turn"
    (reply,) = [p for t, p in published if t == "chat.reply"]
    assert reply["session_id"] == "tg-agent-42"
    assert reply["text"] == "Can't reach my tools — Oakley or Uvex."
    # On a live call this is the answer to a spoken question, so it has to cut
    # ahead of chitchat rather than queue behind it.
    assert reply["trigger"] == "executor.done"


async def test_silent_operator_falls_back_instead_of_leaving_an_ack_hanging(env):
    """If the operator writes nothing, the caller must still say something —
    an ack followed by silence is the outcome this path exists to prevent."""
    events = env[1]

    class MuteOperator:
        async def auto_ping(self, session_id, note, **kwargs):
            return SimpleNamespace(text="")

    with pytest.raises(RuntimeError):
        await deliver_failure_via_operator(
            operator=MuteOperator(), events=events,
            chat_session_id="tg-agent-42", note="failed",
        )


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


async def test_first_pass_answer_replaces_verbatim_delivery_with_a_correction(env):
    """When the operator already answered, the executor's finding must reach
    the user through the operator — and only once.

    Both halves matter. Publishing the finding verbatim *as well* would land a
    second answer that quietly disagrees with the first; writing it into
    history as an assistant turn would make the operator believe it had said
    something it never said. The operator's own reply is the only output.
    """
    db, events, published = env
    task = await _task_with_text(
        db, events, "Kerrygold is the pick — grass-fed, and it's on offer.",
        first_pass_answer="Probably the Meggle, it spreads straight from the fridge.",
    )
    notes: list[str] = []

    async def on_reconcile(session_id: str, note: str) -> None:
        notes.append(note)
        await events.publish_global("chat.reply", {
            "session_id": session_id, "text": "Correction — take the Kerrygold.",
            "voice_text": "", "trigger": "executor.done", "task_id": None,
        })

    await deliver_executor_result(
        db=db, events=events, task_id=task.id,
        chat_session_id="tg-agent-42", terminal_state="completed",
        first_pass_answer=task.first_pass_answer, on_reconcile=on_reconcile,
    )

    (reply,) = [p for t, p in published if t == "chat.reply"]
    assert reply["text"] == "Correction — take the Kerrygold."
    assert await db.load_chat_history("tg-agent-42") == [], (
        "the executor's text must not be written as an operator turn on this path"
    )
    # The operator can only reconcile if it is shown both sides of the diff.
    (note,) = notes
    assert "Kerrygold is the pick" in note
    assert "Probably the Meggle" in note


async def test_silent_reconciliation_is_honoured_not_backfilled(env):
    """When the finding only confirms what the operator already said, saying
    nothing is the right output — and must NOT fall back to verbatim.

    This inverts the invariant every other path here holds. Elsewhere an empty
    operator turn means "the user is holding an ack and would hear silence",
    so we backfill. Here they are holding a real answer, and backfilling
    publishes exactly the restatement the reconciliation existed to avoid.
    """
    db, events, published = env
    task = await _task_with_text(
        db, events, "Kerrygold, 1.99 — best value on the shelf.",
        first_pass_answer="Kerrygold at 1.99 is the pick.",
    )

    async def on_reconcile(session_id: str, note: str) -> None:
        return  # nothing to add

    await deliver_executor_result(
        db=db, events=events, task_id=task.id,
        chat_session_id="tg-agent-42", terminal_state="completed",
        first_pass_answer=task.first_pass_answer, on_reconcile=on_reconcile,
    )

    assert not [p for t, p in published if t == "chat.reply"]
    assert await db.load_chat_history("tg-agent-42") == []


async def test_reconciliation_failure_falls_back_to_verbatim_delivery(env):
    """A working answer must not be lost because the follow-up turn broke.

    The operator has an answer sitting in hand and the user is holding a first
    reply that may be wrong; going silent here is strictly worse than shipping
    the executor's text unreconciled.
    """
    db, events, published = env
    task = await _task_with_text(
        db, events, "Kerrygold — grass-fed.", first_pass_answer="Take the Meggle.",
    )

    async def on_reconcile(session_id: str, note: str) -> None:
        raise RuntimeError("operator is down")

    await deliver_executor_result(
        db=db, events=events, task_id=task.id,
        chat_session_id="tg-agent-42", terminal_state="completed",
        first_pass_answer=task.first_pass_answer, on_reconcile=on_reconcile,
    )

    (reply,) = [p for t, p in published if t == "chat.reply"]
    assert reply["text"] == "Kerrygold — grass-fed."


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
