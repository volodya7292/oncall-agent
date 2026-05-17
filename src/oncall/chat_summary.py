"""Telegram chat summarization.

Given a `chat_id`, fetch the recent history via TelegramService, format it as
a compact transcript, and pipe it through Sonnet via the local `claude` CLI
runner. Returns a concise summary string.

Mirrors `task_summary.py` in shape — pure function over a service + runner
so it's straightforwardly testable with a fake runner.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from .local_claude import OneShotRunner

if TYPE_CHECKING:
    from .telegram_service import TelegramService


log = logging.getLogger(__name__)


CHAT_SUMMARY_SYSTEM_PROMPT = """\
You are summarizing one Telegram conversation for the operator.

Output a single block of plain prose under 200 words. Lead with the topic,
then key facts, decisions, and any open questions. Preserve names, dates,
and concrete details (paths, numbers, URLs) verbatim. If a focus is given,
center the summary on it and omit unrelated chatter.

No headers, no bullets, no markdown. End with a blank line.
"""


_MAX_TRANSCRIPT_CHARS = 60_000


async def summarize_chat(
    tg: "TelegramService",
    runner: OneShotRunner,
    chat_id: str,
    *,
    limit: int = 200,
    focus: str | None = None,
    model: str = "sonnet",
) -> str | None:
    """Fetch the last `limit` messages of `chat_id`, format them, and run
    them through `runner` for summarization. Returns the summary text, or
    None if the chat is empty or the runner fails."""
    messages = await tg.get_chat_history(chat_id, limit=limit)
    if not messages:
        return None

    transcript = _format_transcript(messages)
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:_MAX_TRANSCRIPT_CHARS] + "\n\n[truncated]"

    prompt = transcript
    prompt += "\n\nFocus: " + focus.strip() if focus and focus.strip() else "\n\nSummarize."

    text = await runner.one_shot(
        prompt,
        system_prompt=CHAT_SUMMARY_SYSTEM_PROMPT,
        model=model,
        timeout_s=60.0,
    )
    if not text:
        log.warning("chat_summary: runner returned empty for chat %s", chat_id)
        return None
    return text


def _format_transcript(messages: list[dict[str, Any]]) -> str:
    """One line per message, chronological order (oldest first).
    Format: `[YYYY-MM-DD HH:MM] sender_label: text`."""
    rows = list(messages)
    # get_chat_history returns newest-first (iter_messages default); summary
    # reads better in chronological order.
    rows.reverse()
    lines: list[str] = []
    for m in rows:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        date = (m.get("date") or "")[:16].replace("T", " ")
        if m.get("outgoing"):
            who = "me"
        else:
            who = (
                m.get("sender_display_name")
                or (f"@{m['sender_username']}" if m.get("sender_username") else "them")
            )
        lines.append(f"[{date}] {who}: {text}")
    return "\n".join(lines)
