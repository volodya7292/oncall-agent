"""Executor → user result delivery.

When a hand_off'd executor task terminates, we pull its final
assistant text, compress to ≤300 chars (passthrough if already short
enough; one-shot LLM summary otherwise), and dual-write:

  1. publish `chat.reply` → telegram bot subscriber sends to the user
  2. append to the operator's chat history as an `assistant` row,
     so the operator's next turn naturally sees "what I just told the
     user" without being re-invoked.

The operator is intentionally NOT involved in this path. It said
"Looking…" when it called hand_off; the system delivers the answer.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from .db import Database
from .events import EventBus
from .voice import to_voice_text


log = logging.getLogger(__name__)


MAX_USER_FACING_CHARS = 300

_SUMMARIZE_SYSTEM_PROMPT = (
    "You compress an on-call worker's reply into a message the operator "
    "sends to the user on Telegram. Output ONLY the compressed message, "
    "no preamble. ≤300 chars total. Keep first-person voice ('I checked', "
    "'looks like…'). Preserve any specific identifiers, numbers, file "
    "paths, error messages, and ANY challenge phrase or quoted prompt "
    "verbatim. Drop process noise ('I ran X, then Y, then Z'). Lead with "
    "the result."
)


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
    llm: Any | None,  # LLMClient — typed loosely to avoid an import cycle
    model: str,
    task_id: UUID,
    chat_session_id: str,
    terminal_state: str,
) -> None:
    """Read the executor's final text, compress if needed, dual-write."""
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

    if len(raw) <= MAX_USER_FACING_CHARS:
        final = raw
    else:
        final = await _summarize(llm, model, raw)

    final = (final or "").strip()
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


async def _summarize(llm: Any | None, model: str, text: str) -> str:
    """Compress with the operator's same LLM (cheap flash-lite by default).
    On any failure, hard-truncate the raw text — fail loud in logs but
    still deliver something."""
    if llm is None:
        log.warning("result_delivery: no llm available; hard-truncating")
        return _hard_truncate(text)
    try:
        resp = await llm.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tools=[],
            max_tokens=512,
            # Without this Gemini's default thinking budget eats most of
            # `max_tokens` and the visible reply gets truncated mid-sentence —
            # producing fragments like ", PointerEventData.InputButton>`".
            # Summarization is mechanical compression; no thinking needed.
            reasoning_effort="minimal",
        )
    except Exception:
        log.exception("result_delivery: summarize call crashed; truncating")
        return _hard_truncate(text)
    out = (resp.get("content") or "").strip()
    if not out:
        log.warning("result_delivery: summarizer returned empty; truncating")
        return _hard_truncate(text)
    if len(out) > MAX_USER_FACING_CHARS:
        out = _hard_truncate(out)
    return out


def _hard_truncate(text: str) -> str:
    if len(text) <= MAX_USER_FACING_CHARS:
        return text
    return text[: MAX_USER_FACING_CHARS - 1] + "…"
