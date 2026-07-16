"""Executor → user result delivery.

When a hand_off'd executor task terminates, we pull its final
assistant text and dual-write it verbatim:

  1. publish `chat.reply` → telegram bot subscriber sends to the user
  2. append to the operator's chat history as an `assistant` row,
     so the operator's next turn naturally sees "what I just told the
     user" without being re-invoked.

The operator is intentionally NOT involved in this path. It said
"Looking…" when it called hand_off; the system delivers the answer.

There is deliberately NO rewrite step here. An LLM compressor used to sit
on this path, and since any chat digest overruns the budget it ran on
nearly every real answer rather than as an edge case. It corrupted the
text it touched: told to "keep first-person voice", it rewrote the
executor's correct "you (the user) advised him" into "I advised" —
stealing the user's own words — and its own overruns then got guillotined
mid-word anyway. The executor is told its budget directly now (see
prompts/executor_system.md) and writes to fit. `_hard_truncate` remains
only as a backstop for a disobedient model, and logs when it fires — if
that warning is anything but rare, fix the executor prompt rather than
reintroducing a rewriter.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from .db import Database
from .events import EventBus
from .voice import to_voice_text


log = logging.getLogger(__name__)


# Hard ceiling on what reaches the user. Bounded by voice: the reply is
# TTS'd, and past this it's a monologue.
MAX_USER_FACING_CHARS = 1000

# What the executor is *asked* to write (injected into its prompt as
# `{{reply_budget_chars}}`). The 50-char gap is slack, not headroom for
# more content: models overshoot a stated limit slightly, and since nothing
# rewrites them anymore, an overshoot lands on the user as a mid-word cut.
# Absorbing it is cheaper than truncating. Keep this BELOW the ceiling.
EXECUTOR_REPLY_BUDGET_CHARS = 300


def latest_executor_text(events: list[dict[str, Any]]) -> str:
    """Pull the most recent `assistant.text` event's text. Empty string
    if the task produced no assistant turns (e.g. killed before output)."""
    for e in reversed(events):
        if e.get("type") == "assistant.text":
            text = ((e.get("payload") or {}).get("text") or "").strip()
            if text:
                return text
    return ""


async def deliver_executor_result(
    *,
    db: Database,
    events: EventBus,
    task_id: UUID,
    chat_session_id: str,
    terminal_state: str,
) -> None:
    """Read the executor's final text and dual-write it verbatim."""
    task_events = await db.list_events(task_id)
    raw = latest_executor_text(task_events)
    if not raw:
        if terminal_state == "failed":
            # The task broke before producing ANY output — claude errored
            # (e.g. session clash, missing binary), the subprocess crashed,
            # etc. Staying silent here is the bug that hid a failure for an
            # hour: the user asked for something and simply never heard back.
            # Tell them it failed; specifics stay in the server logs.
            await _publish_failure_notice(events, chat_session_id, task_id)
            return
        # Otherwise (completed / killed with no text): a pure side-effect task
        # (e.g. a single emoji reaction) or a user-initiated kill. The action
        # itself is the signal — don't fabricate a placeholder that spams the
        # user and pollutes operator history with a fake assistant turn.
        log.info(
            "result_delivery: task %s ended in state=%s with no assistant text; skipping publish",
            task_id, terminal_state,
        )
        return

    final = _hard_truncate(raw).strip()
    if not final:
        log.info("result_delivery: empty final text for task %s; skipping", task_id)
        return

    # 1. operator history — so next operator turn sees the reply as its own.
    try:
        await db.append_chat_message(chat_session_id, "assistant", final)
    except Exception:
        log.exception("result_delivery: failed to persist operator history")

    # 2. user-facing publish.
    try:
        await events.publish_global("chat.reply", {
            "session_id": chat_session_id,
            "text": final,
            "voice_text": to_voice_text(final),
            "trigger": "executor.done",
            "task_id": str(task_id),
        })
    except Exception:
        log.exception("result_delivery: failed to publish chat.reply")


async def _publish_failure_notice(
    events: EventBus, chat_session_id: str, task_id: UUID,
) -> None:
    """Tell the user a hand_off'd task failed before producing any output.

    Without this, a task that errors before its first assistant turn is silent
    on Telegram (the no-text branch used to just skip). We don't have the
    executor's stderr here — that's in the server logs — so keep it terse and
    honest about uncertainty rather than inventing a reason."""
    msg = (
        "SYSTEM: ⚠️ I couldn't complete that — the task failed before it "
        "produced any result. Nothing was confirmed done. (Details are in the "
        "server logs.)"
    )
    try:
        await events.publish_global("chat.reply", {
            "session_id": chat_session_id,
            "text": msg,
            "voice_text": "",
            "trigger": "executor.failed",
            "task_id": str(task_id),
        })
    except Exception:
        log.exception("result_delivery: failed to publish failure notice for %s", task_id)


def _hard_truncate(text: str) -> str:
    """Backstop for an executor that blew past EXECUTOR_REPLY_BUDGET_CHARS by
    more than the slack. Cuts mid-word; that ugliness is the point — it should
    be visible, and rare."""
    if len(text) <= MAX_USER_FACING_CHARS:
        return text
    log.warning(
        "result_delivery: executor returned %d chars, over the %d budget; "
        "truncating. If this is not rare, fix prompts/executor_system.md.",
        len(text), MAX_USER_FACING_CHARS,
    )
    return text[: MAX_USER_FACING_CHARS - 1] + "…"
