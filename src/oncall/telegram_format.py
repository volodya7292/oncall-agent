"""Pure-function formatting helpers shared by the Telegram agent service.

Carved out of the old Bot API service so they remain available after the
bot is gone. None of these touch the network."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


# Telegram caps a single message at 4096 chars. Stay slightly under for headroom.
TELEGRAM_MSG_LIMIT = 4000

# Long enough for one complete thought, short enough to still look like a
# normal chat bubble rather than a mini-paragraph.
MESSENGER_BUBBLE_TARGET = 45


def chunk_message(text: str, *, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Split a message into ≤limit-char chunks, preferring newline boundaries.
    Pure function — easy to unit-test."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def split_messenger_reply(text: str, *, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Turn an LLM reply into natural-sized Telegram chat bubbles.

    Blank lines are explicit boundaries. Longer prose also splits at sentence
    or word boundaries, never in the middle of a word. Telegram's ordinary
    size limit still applies within each bubble.
    """
    paragraphs = [part.strip() for part in re.split(r"\n[ \t]*\n+", text) if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        # Structured content is clearer as one unit; only Telegram's hard
        # limit may break it up.
        if "```" in paragraph or re.search(r"(?m)^\s*(?:[-*+] |\d+[.)] )", paragraph):
            pieces.extend(chunk_message(paragraph, limit=limit))
            continue
        remaining = paragraph
        target = min(MESSENGER_BUBBLE_TARGET, limit)
        while len(remaining) > target:
            # Prefer finishing a sentence, then use the last complete word
            # that fits the target bubble.
            sentence_ends = [m.end() for m in re.finditer(r"[.!?…)]\s+", remaining[:target])]
            word_ends = [m.end() for m in re.finditer(r"\s+", remaining[:target])]
            cut = (sentence_ends or word_ends or [target])[-1]
            pieces.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            pieces.extend(chunk_message(remaining, limit=limit))
    return pieces


def label_for_chat(chat_id: str, resolved: dict[str, Any] | None) -> str:
    """Render a chat as `Display Name (@username, chat_id)` when resolved,
    or just `chat_id` when not."""
    if not resolved:
        return chat_id
    name = (resolved.get("display_name") or "").strip()
    uname = (resolved.get("username") or "").strip()
    parts: list[str] = []
    if name:
        parts.append(name)
    handle_bits: list[str] = []
    if uname:
        handle_bits.append(f"@{uname}")
    handle_bits.append(chat_id)
    parts.append(f"({', '.join(handle_bits)})")
    return " ".join(parts) if parts and name else f"@{uname} ({chat_id})" if uname else chat_id


def truncate(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def reply_context_note(reply_msg: Any, *, who: str, limit: int = 150) -> str:
    """One-line `[replying to ...]` anchor describing the Telegram message a
    new inbound message replies to. `who` names the quoted message's author
    from the reader's perspective (e.g. "your earlier message"). Telegram's
    reply pointer is otherwise invisible to the operator, which would leave
    it resolving "yes, that one" against whatever is most recent."""
    body = (getattr(reply_msg, "message", None) or "").strip()
    if not body and getattr(reply_msg, "media", None) is not None:
        body = "<media message>"
    return f'[replying to {who}: "{truncate(body, limit)}"]'


def age(when: datetime) -> str:
    """Compact human age: 5s / 12m / 3h / 4d. `when` is timezone-aware UTC."""
    return format_seconds((datetime.now(timezone.utc) - when).total_seconds())


def relative_age(iso_string: str) -> str:
    """Age of an ISO-formatted timestamp string. Returns 'unknown' if it
    can't be parsed."""
    try:
        when = datetime.fromisoformat(iso_string)
    except (TypeError, ValueError):
        return "unknown"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return format_seconds((datetime.now(timezone.utc) - when).total_seconds()) + " ago"


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"
