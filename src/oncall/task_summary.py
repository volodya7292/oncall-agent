"""Per-task result summary.

When the executor finishes a task, we generate a short digest of what it did —
key findings, files touched, errors — via the local `claude` CLI and store it
in `tasks.result_summary`. The operator (and any future executor that picks
the task up by ID) reads this instead of the raw event stream.

Pure function over the DB + runner so it's trivially testable with a fake
runner.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from .db import Database
from .local_claude import OneShotRunner


log = logging.getLogger(__name__)


TASK_SUMMARY_SYSTEM_PROMPT = """\
You are summarizing the execution of one on-call task for the operator.

Output a single block of plain prose under 200 words. Lead with the outcome
(what was found, what was changed, what failed). Include any key data points
the user might ask about (counts, file paths, error messages). Preserve task
IDs and approval ids verbatim. Do NOT include time durations or cost.

No headers, no bullets, no markdown. End with a blank line.
"""


# Hard cap on input size to keep the summarization call snappy.
_MAX_EVENT_PREVIEW_CHARS = 600
_MAX_PROMPT_CHARS = 60_000


async def summarize_task(
    db: Database, runner: OneShotRunner, task_id: UUID, *, model: str = "sonnet",
) -> str | None:
    """Build a summary prompt from the task's event trail, call the runner,
    persist the result. Returns the summary text, or None if summarization
    couldn't proceed (no events, runner failure, etc)."""
    task = await db.get_task(task_id)
    if task is None:
        return None

    events = await db.list_events(task_id)
    if not events:
        # Task was killed before producing any output. Record a stub so we
        # don't keep re-trying summarization on every auto-ping cycle.
        stub = (
            f"Task {task_id} ended in state={task.state.value} with no "
            f"executor output. Prompt was: {task.prompt[:200]}"
        )
        await db.update_task_result_summary(task_id, stub)
        return stub

    formatted = _format_events_for_prompt(task.prompt, task.state.value, events)
    if len(formatted) > _MAX_PROMPT_CHARS:
        formatted = formatted[:_MAX_PROMPT_CHARS] + "\n\n[truncated]"

    text = await runner.one_shot(
        formatted,
        system_prompt=TASK_SUMMARY_SYSTEM_PROMPT,
        model=model,
        timeout_s=60.0,
    )
    if not text:
        log.warning("task_summary: runner returned empty for %s", task_id)
        return None

    await db.update_task_result_summary(task_id, text)
    return text


def _format_events_for_prompt(prompt: str, state: str, events: list[dict[str, Any]]) -> str:
    """Distill the supervisor's event stream into a compact, summarizable form.
    Keeps assistant text in full (truncated per-event), shrinks tool_result
    payloads to a preview, drops state.changed noise."""
    lines: list[str] = [
        f"TASK_STATE: {state}",
        f"USER_PROMPT: {prompt[:500]}",
        "",
        "EVENTS:",
    ]
    for evt in events:
        type_ = evt.get("type", "")
        payload = evt.get("payload") or {}
        if type_ == "assistant.text":
            text = (payload.get("text") or "").strip()
            if text:
                lines.append(f"- assistant said: {text[:_MAX_EVENT_PREVIEW_CHARS]}")
        elif type_ == "tool_use.requested":
            tool = payload.get("tool_name") or payload.get("name") or "?"
            args = payload.get("input") or payload.get("args") or {}
            args_str = json.dumps(args, ensure_ascii=False)[:_MAX_EVENT_PREVIEW_CHARS]
            lines.append(f"- tool_use: {tool} args={args_str}")
        elif type_ == "tool_result":
            preview = (payload.get("preview") or "")[:_MAX_EVENT_PREVIEW_CHARS]
            err = " [ERROR]" if payload.get("is_error") else ""
            lines.append(f"- tool_result{err}: {preview}")
        elif type_ == "approval.requested":
            canonical = (payload.get("canonical_command") or "")[:200]
            lines.append(f"- approval requested: {canonical}")
        elif type_ == "approval.resolved":
            if payload.get("auto"):
                continue  # skip auto-allow noise
            decision = payload.get("decision", "?")
            lines.append(f"- approval resolved: {decision}")
        elif type_ == "result.final":
            if payload.get("is_error"):
                lines.append("- result: ERROR")
            else:
                lines.append("- result: success")
        # state.changed and cli.* events are intentionally dropped — they're
        # bookkeeping noise that doesn't help the summarizer.

    return "\n".join(lines)
