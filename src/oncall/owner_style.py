"""Learn how the owner writes, and make the agent's chat replies read that way.

`/infer-style` runs `OwnerStyle.infer`: it reads the owner's own replies across
their recent 1:1 chats, has the LLM explain each one (what the owner was
thinking, and how that shows in the writing), then distills those analyses
into one style profile plus a set of representative replies. The result is a
single row in `owner_style`, replaced on the next run.

`OwnerStyle.rewrite` then runs on every chat reply to the owner. Any failure
returns None and the caller sends the original: the style is cosmetic, the
message is not. Quotations never reach the LLM: they are swapped for
placeholders and put back verbatim, so a rewrite can't alter what someone
was told.

All three LLM steps use native structured output (`response_schema`)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from .db import Database

log = logging.getLogger(__name__)

# Incoming messages shown as context for a reply, and their clip length.
_CONTEXT_MESSAGES = 3
_CONTEXT_CLIP = 300
# Longer outgoing messages are usually pasted (logs, addresses, forwards
# retyped), not the owner's voice.
_MAX_OWN_MESSAGE_CHARS = 400
_MAX_BURSTS = 200
_ANALYZE_BATCH = 15
_ANALYZE_CONCURRENCY = 4
_MAX_EXAMPLES = 30
# A quotation, marks included: „…“ / „…” (German), “…”, «…», "…", `…`.
_QUOTE_RE = re.compile(r'„[^“”"\n]*[“”"]|“[^”\n]*”|«[^»\n]*»|"[^"\n]*"|`[^`\n]+`')
_PLACEHOLDER_RE = re.compile(r"⟦(\d+)⟧")


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    # OpenAI-style strict mode needs every property required and no extras.
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_ANALYZE_SCHEMA = _strict_object({
    "analyses": {
        "type": "array",
        "items": _strict_object({
            "index": {"type": "integer"},
            "thoughts": {"type": "string"},
            "writing_style": {"type": "string"},
        }),
    },
})

_PROFILE_SCHEMA = _strict_object({
    "profile": {"type": "string"},
    "example_indices": {"type": "array", "items": {"type": "integer"}},
})

_REWRITE_SCHEMA = _strict_object({
    "messages": {"type": "array", "items": {"type": "string"}},
})

_ANALYZE_PROMPT = """\
You are studying how one person, the owner, writes in their Telegram DMs. Each \
numbered item is one reply of the owner's: the messages they were answering \
(written by someone else, possibly empty) and their reply, which may span \
several consecutive messages.

For every item, infer what went through the owner's mind while writing it, \
and describe the writing style that results: register, length, punctuation, \
capitalization, language choice and mixing, emoji, slang, and how the reply \
is split across messages. Analyze only the owner; the context is there to \
explain their choices. Return one analysis per item, keyed by its index."""

_PROFILE_PROMPT = """\
Below are analyses of how the owner writes in their Telegram DMs, each with \
the reply it describes. Distill them into a style guide that lets someone \
write new messages indistinguishable from the owner's, in any language: what \
holds across all chats, and what changes with the person, the situation or \
the language.

Also pick up to {max_examples} item indices whose replies, together, best \
show the style and its range."""

_REWRITE_PROMPT = """\
You restyle the message an assistant is about to send to its owner on \
Telegram, given inside <message>. The result is still that message, from the \
assistant to the owner and saying what it says, only written the way the \
owner writes: their voice, register, length, punctuation, capitalization and \
way of splitting a thought into messages, as described and shown below. \
Compress as hard as the owner would: drop framing, filler and whatever they \
wouldn't bother to say, keeping what the reader needs to act on.

Each ⟦n⟧ stands for a quotation, marks included, and appears exactly once in \
the result. Identifiers, links, paths and commands stay exact, and the intent \
stays (a question stays a question). {language_rule} Each element of \
`messages` is one Telegram message, sent in order.

# The owner's style

{profile}

# The owner's own replies

{examples}"""


def _language_rule(language: str) -> str:
    target = f"{language} (ISO-639-1)" if language else "the original message's language"
    return f"Write in {target}, whatever language the style notes and examples are in."


def reply_bursts(thread: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a chat (oldest-first {outgoing, text} rows) into the owner's
    replies: each run of consecutive outgoing messages is one reply, carrying
    the incoming messages since the previous reply as its context."""
    bursts: list[dict[str, Any]] = []
    incoming: list[str] = []
    current: dict[str, Any] | None = None
    for message in thread:
        text = (message.get("text") or "").strip()
        if not text:
            continue
        if not message.get("outgoing"):
            current = None
            incoming.append(text[:_CONTEXT_CLIP])
            continue
        if len(text) > _MAX_OWN_MESSAGE_CHARS:
            # A pasted block breaks the run without starting a reply.
            current = None
            incoming = []
            continue
        if current is None:
            current = {"context": incoming[-_CONTEXT_MESSAGES:], "reply": []}
            bursts.append(current)
            incoming = []
        current["reply"].append(text)
    return bursts


def _take_round_robin(per_chat: list[list[dict[str, Any]]], cap: int) -> list[dict[str, Any]]:
    """Up to `cap` bursts, newest first within each chat, alternating chats so
    one busy thread can't dominate the sample."""
    queues = [list(reversed(bursts)) for bursts in per_chat if bursts]
    out: list[dict[str, Any]] = []
    while queues and len(out) < cap:
        for queue in list(queues):
            if len(out) >= cap:
                break
            out.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return out


def _render_item(index: int, burst: dict[str, Any]) -> str:
    lines = [f"## Item {index}"]
    for text in burst["context"]:
        lines.append(f"them: {text}")
    for text in burst["reply"]:
        lines.append(f"owner: {text}")
    return "\n".join(lines)


def _render_example(example: dict[str, Any]) -> str:
    lines = [f"them: {text}" for text in example["context"]]
    lines += [f"owner: {text}" for text in example["reply"]]
    lines.append(f"(thinking: {example['thoughts']})")
    lines.append(f"(style: {example['writing_style']})")
    return "\n".join(lines)


def _mask_quotes(text: str) -> tuple[str, list[str]]:
    quotes: list[str] = []

    def stash(match: re.Match[str]) -> str:
        quotes.append(match.group(0))
        return f"⟦{len(quotes)}⟧"

    return _QUOTE_RE.sub(stash, text), quotes


def _scripts(text: str) -> set[str]:
    """Writing systems of the letters in `text` (LATIN, CYRILLIC, …)."""
    return {unicodedata.name(ch, "?").split(" ", 1)[0] for ch in text if ch.isalpha()}


def _restore_quotes(pieces: list[str], masked: str, quotes: list[str]) -> list[str]:
    """Put the quotations back. Raises ValueError when the rewrite dropped,
    repeated or invented one, or wrote in a script the original used only
    inside quotes or not at all: that is the rewrite passing off its own
    words as a quotation, or drifting into the examples' language."""
    found = sorted(int(n) for piece in pieces for n in _PLACEHOLDER_RE.findall(piece))
    if found != list(range(1, len(quotes) + 1)):
        raise ValueError(f"quote placeholders {found}, expected {len(quotes)}")
    introduced = _scripts(" ".join(pieces)) - _scripts(masked)
    if introduced:
        raise ValueError(f"introduced scripts {sorted(introduced)}")
    return [
        _PLACEHOLDER_RE.sub(lambda m: quotes[int(m.group(1)) - 1], piece)
        for piece in pieces
    ]


def should_rewrite(text: str) -> bool:
    """Code blocks and system notices go out verbatim."""
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("SYSTEM:") and "```" not in stripped


class OwnerStyle:
    def __init__(
        self, *, llm: Any, db: Database, model: str,
        infer_reasoning_effort: str | None = None,
        rewrite_reasoning_effort: str | None = None,
        rewrite_timeout_seconds: float = 10.0,
        max_chats: int = 30, messages_per_chat: int = 60,
        language: str = "",
    ) -> None:
        self._llm = llm
        self._db = db
        self._model = model
        self._infer_effort = infer_reasoning_effort
        self._rewrite_effort = rewrite_reasoning_effort
        self._rewrite_timeout = rewrite_timeout_seconds
        self._max_chats = max_chats
        self._messages_per_chat = messages_per_chat
        # The operator's fixed output language; empty = keep the original's.
        self._language = language
        self._style: dict[str, Any] | None = None
        self._loaded = False

    async def _structured(
        self, *, system: str, user: str, schema: dict[str, Any],
        max_tokens: int, reasoning_effort: str | None,
    ) -> dict[str, Any]:
        resp = await self._llm.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[],
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_schema=schema,
        )
        return json.loads(resp.get("content") or "")

    # ---- inference ----

    async def infer(self, telegram: Any) -> dict[str, Any]:
        """Rebuild and store the style from the owner's recent DMs. Raises when
        there is nothing to learn from or every analysis call failed."""
        threads = await telegram.get_owner_dm_threads(
            max_chats=self._max_chats, per_chat=self._messages_per_chat,
        )
        bursts = _take_round_robin([reply_bursts(t) for t in threads], _MAX_BURSTS)
        if not bursts:
            raise ValueError("no replies of yours found in recent 1:1 chats")

        analyses = await self._analyze_all(bursts)
        if not analyses:
            raise RuntimeError("every style-analysis call failed (see logs)")

        analyzed = [
            {**bursts[i], "thoughts": a["thoughts"], "writing_style": a["writing_style"]}
            for i, a in sorted(analyses.items())
        ]
        rendered = "\n\n".join(
            f"{_render_item(i, ex)}\nthinking: {ex['thoughts']}\nstyle: {ex['writing_style']}"
            for i, ex in enumerate(analyzed)
        )
        result = await self._structured(
            system=_PROFILE_PROMPT.format(max_examples=_MAX_EXAMPLES),
            user=rendered,
            schema=_PROFILE_SCHEMA,
            max_tokens=8192,
            reasoning_effort=self._infer_effort,
        )
        picked = [
            analyzed[i] for i in dict.fromkeys(result.get("example_indices") or [])
            if isinstance(i, int) and 0 <= i < len(analyzed)
        ][:_MAX_EXAMPLES] or analyzed[:_MAX_EXAMPLES]

        style = {
            "profile": result["profile"],
            "examples": picked,
            "chats": len(threads),
            "replies_analyzed": len(analyzed),
            "inferred_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._db.set_owner_style(style)
        self._style, self._loaded = style, True
        log.info(
            "owner style inferred: chats=%d replies=%d examples=%d",
            len(threads), len(analyzed), len(picked),
        )
        return style

    async def _analyze_all(self, bursts: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        gate = asyncio.Semaphore(_ANALYZE_CONCURRENCY)

        async def one(start: int) -> dict[int, dict[str, Any]]:
            batch = range(start, min(start + _ANALYZE_BATCH, len(bursts)))
            async with gate:
                try:
                    result = await self._structured(
                        system=_ANALYZE_PROMPT,
                        user="\n\n".join(_render_item(i, bursts[i]) for i in batch),
                        schema=_ANALYZE_SCHEMA,
                        max_tokens=8192,
                        reasoning_effort=self._infer_effort,
                    )
                except Exception as e:
                    log.warning("style analysis batch at %d failed: %s", start, e)
                    return {}
            return {
                a["index"]: a for a in result.get("analyses") or []
                if isinstance(a, dict) and a.get("index") in batch
            }

        merged: dict[int, dict[str, Any]] = {}
        for part in await asyncio.gather(
            *(one(s) for s in range(0, len(bursts), _ANALYZE_BATCH)),
        ):
            merged.update(part)
        return merged

    async def clear(self) -> bool:
        self._style, self._loaded = None, True
        return await self._db.clear_owner_style()

    async def current(self) -> dict[str, Any] | None:
        if not self._loaded:
            self._style = await self._db.get_owner_style()
            self._loaded = True
        return self._style

    # ---- rewriting ----

    async def rewrite(self, text: str) -> list[str] | None:
        """`text` in the owner's style, one string per Telegram message, or
        None when there is no style yet, the text is exempt, or the call
        fails, times out or returns nothing usable."""
        if not should_rewrite(text):
            return None
        try:
            style = await self.current()
        except Exception:
            log.exception("owner style: loading the stored style failed")
            return None
        if not style:
            return None
        masked, quotes = _mask_quotes(text)
        system = _REWRITE_PROMPT.format(
            profile=style["profile"],
            language_rule=_language_rule(self._language),
            examples="\n\n".join(_render_example(ex) for ex in style["examples"]),
        )
        try:
            result = await asyncio.wait_for(
                self._structured(
                    system=system, user=f"<message>\n{masked}\n</message>",
                    schema=_REWRITE_SCHEMA,
                    max_tokens=2048, reasoning_effort=self._rewrite_effort,
                ),
                timeout=self._rewrite_timeout,
            )
        except Exception as e:
            log.warning("owner style rewrite failed, sending original: %r", e)
            return None
        messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(messages, list):
            log.warning("owner style rewrite returned no message list, sending original")
            return None
        pieces = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
        if not pieces:
            log.warning("owner style rewrite returned only empty messages, sending original")
            return None
        try:
            return _restore_quotes(pieces, masked, quotes)
        except ValueError as e:
            log.warning("owner style rewrite rejected (%s), sending original", e)
            return None
