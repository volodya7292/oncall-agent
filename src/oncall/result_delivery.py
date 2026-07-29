"""Executor → user result delivery.

When a hand_off'd executor task terminates, we pull its final
assistant text and dual-write it verbatim:

  1. publish `chat.reply` → telegram bot subscriber sends to the user
  2. append to the operator's chat history as an `assistant` row,
     so the operator's next turn naturally sees "what I just told the
     user" without being re-invoked.

The operator is not involved in that path. It said "Looking…" when it called
hand_off; the system delivers the answer.

That holds only when the ack really was an ack. The operator may instead
answer the user outright AND hand off in the same turn (see "Answer now,
verify in the background" in prompts/operator_system.md) — the user has then
already read an answer, and publishing the executor's text verbatim would
land as a second, unattached answer that silently disagrees with the first.
So when the task carries a `first_pass_answer`, the finding goes back to the
operator (`on_reconcile`) and it writes the follow-up itself: confirm, correct,
or extend what it already said. This is the ONE rewrite on the success path
and it exists because there is something to reconcile against; it is not a
reintroduction of the compressor described below. If it fails for any reason
we fall through to verbatim delivery — an ack (or a first answer) followed by
silence is the one outcome this module must never produce.

The FAILURE path is the opposite: a task that dies produces no answer, and
a canned system banner is a dead end for the user. So a terminal failure is
handed back to the operator (`on_failure`), which replies in its own voice
from what it already knows — or says plainly that it can't. Nothing about
that is remembered: the failure is a per-turn event, and the very next user
message hands off again as normal.

Critically, a failed task's last `assistant.text` is NOT delivered. It is
usually not an answer — when the Claude CLI rejects a run it emits its own
meta-text as an assistant turn ("You've hit your weekly limit · resets 2am
(UTC)"), which this module used to forward verbatim to Telegram *and* write
into operator history as something the operator had said. Whatever text a
failed task produced is passed to the operator as context instead, and the
operator decides what (if anything) the user should hear.

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
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .db import Database
from .events import EventBus
from .voice import to_voice_text


log = logging.getLogger(__name__)

# (chat_session_id, note) -> awaits the operator turn that answers the user.
FailureHandler = Callable[[str, str], Awaitable[None]]
# Same shape, success path: the operator reconciles the executor's finding
# against the answer it already gave. Must raise if it publishes nothing, so
# the caller can fall back to verbatim delivery.
ReconcileHandler = Callable[[str, str], Awaitable[None]]

# How much of a failed task's own output to quote back to the operator. It is
# context for writing a reply, not the reply — a failed run's text is as often
# a CLI error string as a partial answer.
_FAILURE_EXCERPT_CHARS = 300

# Ceiling on each of the two texts quoted into a reconciliation note. Set at the
# text ceiling so a well-behaved executor reply is never clipped, and explicitly
# NOT at the voice one: on a call the operator still needs the whole finding to
# reconcile against, however short what it then says aloud has to be.
_RECONCILE_QUOTE_CHARS = 1500


# Hard ceiling on what reaches the user, per delivery channel.
#
# The 600 limit is bounded by VOICE: the reply is TTS'd on a call, and past
# this it's a monologue. That reasoning never applied to text chat, but the
# ceiling was applied there anyway — so ordinary Telegram answers were being
# guillotined mid-word (measured: 7 truncations in 30h, at 745–1107 chars).
# Text gets its own, looser ceiling; voice keeps the tight one.
MAX_VOICE_CHARS = 600
MAX_TEXT_CHARS = 1500

# What the executor is *asked* to write (injected into its prompt as
# `{{reply_budget_chars}}`). Unchanged and deliberately far below the text
# ceiling: the executor should still answer briefly. The gap is slack, not
# headroom for more content — models overshoot a stated limit, and since
# nothing rewrites them anymore, an overshoot lands on the user as a mid-word
# cut. Absorbing it is cheaper than truncating. Keep this BELOW both ceilings.
EXECUTOR_REPLY_BUDGET_CHARS = 400


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
    spoken: bool = False,
    on_failure: FailureHandler | None = None,
    first_pass_answer: str | None = None,
    on_reconcile: ReconcileHandler | None = None,
) -> None:
    """Read the executor's final text and dual-write it verbatim.

    `spoken`: this session is on a live voice call, so the reply is TTS'd
    and the tight voice ceiling applies. False → text chat.

    `on_failure`: called with a composed `[system note: ...]` body when the
    task failed, so the caller can re-invoke the operator to answer the user
    itself. None falls back to the canned banner (tests / no operator).

    `first_pass_answer`: what the operator itself told the user in the turn
    that handed this task off, if anything. When set (and `on_reconcile` is
    wired) the executor's finding is handed back to the operator to reconcile
    instead of being published verbatim.
    """
    task_events = await db.list_events(task_id)
    raw = latest_executor_text(task_events)

    if terminal_state == "failed":
        # A failed task has no answer to deliver — `raw`, if present, is
        # typically the CLI's own meta-text, not a reply (see module
        # docstring). Hand the whole situation to the operator; it decides
        # what the user hears. Nothing here is written to history: the
        # operator's own turn does that.
        note = _compose_failure_note(task_events, raw)
        if on_failure is None:
            log.info(
                "result_delivery: task %s failed and no on_failure handler is "
                "wired; falling back to the canned notice", task_id,
            )
            await _publish_failure_notice(events, chat_session_id, task_id)
            return
        try:
            await on_failure(chat_session_id, note)
        except Exception:
            log.exception(
                "result_delivery: failure handoff to the operator raised for "
                "task %s; falling back to the canned notice", task_id,
            )
            await _publish_failure_notice(events, chat_session_id, task_id)
        return

    if not raw:
        # completed / killed with no text: a pure side-effect task (e.g. a
        # single emoji reaction) or a user-initiated kill. The action itself
        # is the signal — don't fabricate a placeholder that spams the user
        # and pollutes operator history with a fake assistant turn.
        log.info(
            "result_delivery: task %s ended in state=%s with no assistant text; skipping publish",
            task_id, terminal_state,
        )
        return

    # The operator already answered this turn: its finding is a correction to
    # that answer, not a standalone reply. Let the operator write the follow-up.
    # Falling through on any failure is the point — see the module docstring.
    first_pass = (first_pass_answer or "").strip()
    if first_pass and on_reconcile is not None:
        note = _compose_reconcile_note(first_pass, raw)
        try:
            await on_reconcile(chat_session_id, note)
            return
        except Exception:
            log.exception(
                "result_delivery: reconciliation handoff to the operator raised "
                "for task %s; delivering the executor text verbatim instead",
                task_id,
            )

    final = _hard_truncate(raw, spoken=spoken).strip()
    if not final:
        log.info("result_delivery: empty final text for task %s; skipping", task_id)
        return

    # 1. operator history — so next operator turn sees the reply as its own.
    try:
        await db.append_chat_message(chat_session_id, "assistant", final)
    except Exception:
        log.exception("result_delivery: failed to persist operator history")

    # 2. user-facing publish. `voice_text` keeps the voice ceiling regardless
    # of `spoken`: a call can start between the check above and this publish,
    # and the call subscriber would then TTS a text-length monologue.
    try:
        await events.publish_global("chat.reply", {
            "session_id": chat_session_id,
            "text": final,
            "voice_text": _hard_truncate(to_voice_text(final), spoken=True),
            "trigger": "executor.done",
            "task_id": str(task_id),
        })
    except Exception:
        log.exception("result_delivery: failed to publish chat.reply")


def _rate_limit_reset(task_events: list[dict[str, Any]]) -> str | None:
    """UTC reset time if the run was rejected by a provider rate limit.

    The Claude CLI reports this as a `rate_limit_event` carrying `resetsAt`
    (epoch seconds); the supervisor files anything it doesn't model under
    `cli.<type>`. Reading it here — rather than adding a terminal reason —
    keeps this contained: the value is used for one sentence of prose, not
    to gate anything. None when the failure was something else.
    """
    for e in reversed(task_events):
        if e.get("type") != "cli.rate_limit_event":
            continue
        info = ((e.get("payload") or {}).get("raw") or {}).get("rate_limit_info") or {}
        if info.get("status") != "rejected":
            continue
        resets_at = info.get("resetsAt")
        if not isinstance(resets_at, (int, float)):
            return "an unknown time"
        try:
            return datetime.fromtimestamp(
                float(resets_at), tz=timezone.utc,
            ).strftime("%H:%M UTC on %a %d %b")
        except (OverflowError, OSError, ValueError):
            log.warning("unparseable rate-limit resetsAt %r", resets_at)
            return "an unknown time"
    return None


def _compose_failure_note(
    task_events: list[dict[str, Any]], raw: str,
) -> str:
    """Build the `[system note: ...]` body handed to the operator when a
    hand_off'd task dies.

    States the failure, names the cause when we can (a rate-limit rejection
    reads very differently to the user than a crash), and quotes whatever the
    task did emit as context. The instruction is to answer NOW: the user has
    already seen an ack promising a result, so silence is the one wrong move.
    """
    resets = _rate_limit_reset(task_events)
    if resets:
        cause = (
            f"your acting layer is out of quota and refused the job "
            f"(quota resets at {resets})"
        )
    else:
        cause = "your acting layer crashed before finishing"
    note = (
        f"the job you handed off failed — {cause}. Nothing was done and the "
        f"user is still waiting on the acknowledgement you already sent. "
        f"Answer them yourself now, from what you already know; if the "
        f"request genuinely needs tools you cannot reach, say so plainly."
    )
    excerpt = _clip(raw, _FAILURE_EXCERPT_CHARS)
    if excerpt:
        note += (
            f" All it emitted before dying was: {excerpt!r} — that is "
            f"diagnostic output, not an answer, so do not relay it verbatim."
        )
    return note


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compose_reconcile_note(first_pass: str, report: str) -> str:
    """Build the `[system note: ...]` body handed to the operator when a task
    it handed off — after already answering the user itself — comes back.

    Both texts are quoted because the operator has to diff them: its own
    answer is what the user has read, the report is what turned out to be
    true. The instruction has to hold two edges at once — say what changed
    (or the follow-up is noise) without re-saying what didn't (or it is a
    duplicate answer).
    """
    return (
        f"the job you handed off came back. Your acting layer reports: "
        f"{_clip(report, _RECONCILE_QUOTE_CHARS)!r}. In that same turn you had "
        f"already answered the user yourself: "
        f"{_clip(first_pass, _RECONCILE_QUOTE_CHARS)!r} — they have read that. "
        f"Write the follow-up now, in your own voice: correct yourself plainly "
        f"where the report disagrees with you, and pass on what it adds that "
        f"they still need. Where it only confirms you, say so in a few words. "
        f"Do not restate what they have already read, and claim nothing the "
        f"report does not support."
    )


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


def _hard_truncate(text: str, *, spoken: bool) -> str:
    """Backstop for an executor that blew past EXECUTOR_REPLY_BUDGET_CHARS by
    more than the slack. Cuts mid-word; that ugliness is the point — it should
    be visible, and rare.

    The ceiling depends on the channel: a spoken reply is TTS'd and has to
    stay short, a text reply does not. Applying the voice limit to text is
    what made this fire on ordinary Telegram answers."""
    ceiling = MAX_VOICE_CHARS if spoken else MAX_TEXT_CHARS
    if len(text) <= ceiling:
        return text
    log.warning(
        "result_delivery: executor returned %d chars, over the %d %s ceiling; "
        "truncating. If this is not rare, fix prompts/executor_system.md.",
        len(text), ceiling, "voice" if spoken else "text",
    )
    return text[: ceiling - 1] + "…"
