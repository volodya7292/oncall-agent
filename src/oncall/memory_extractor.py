"""User-turn fact-candidate suggester.

Looks at the user's latest message and the immediately preceding assistant
turn (context only — never a source of facts, though it may supply the
identity of something the user referred to), and asks a cheap LLM for a
JSON list of fact CANDIDATES the operator might want to remember. The
candidates are NOT auto-saved; the operator is auto-pinged with the
suggestions and decides which (if any) to commit via its `save_memory`
tool. This keeps the operator authoritative over what enters memory.

Two things qualify. Standing facts — the ones whose absence next time would
force a clarifying question, so the suggester asks "what makes this terse
intent self-contained?" rather than "what's interesting?". And the user's
own history — what they did, where they were, what happened to them — which
is kept as a record rather than for its effect on the next turn, and so is
judged by whether the occasion is identifiable, not by whether it is still
true.

To avoid re-suggesting things the operator already saved during the same
turn, callers pass `already_saved` so the suggester can exclude near-
duplicates from its output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol


log = logging.getLogger(__name__)


# Bounded inputs so the prompt stays cheap and predictable regardless of
# what the user pasted in. Truncation keeps head + tail so identifiers
# near either end of a long message still surface.
_PREV_ASSISTANT_CHAR_CAP = 2000
_USER_CHAR_CAP = 4000


EXTRACTOR_SYSTEM_PROMPT = """\
You suggest CITATIONS from a single user message to a personal on-call
agent. Your output is advisory — the agent decides which (if any)
citations to actually commit to memory via its `save_memory` tool.

ABSOLUTE RULE — CITE, DO NOT REPHRASE.
A citation is content the user wrote, lifted verbatim. You may:
  - Quote a phrase verbatim and add a minimal lead-in to make it self-contained
    (e.g. user wrote "staging is at api-staging.example.com:8443"
     → citation: 'the user states: "staging is at api-staging.example.com:8443"').
  - Place a reference the user left open — a pronoun, or a deictic standing in
    for something they pointed at, chose, or acted on — against what
    PREVIOUS_ASSISTANT was talking about. The citation MUST name that referent
    — an unresolved "it" is useless a year from now — but only in the lead-in,
    never inside the quoted words, which stay exactly what the user said. A
    referent you cannot place from PREVIOUS_ASSISTANT is not citable at all.
You may NOT:
  - Invent identifiers, handles, URLs, hostnames, IDs, ports, or numbers
    that do not appear in the USER message verbatim.
  - Extrapolate plausible-looking values (e.g. guessing a Telegram handle
    from a person's name — that is a hallucination, not a citation).
  - Restate or "tidy up" what the assistant said in PREVIOUS_ASSISTANT. It is
    never the one asserting: it can say what a reference points at, it can
    never contribute a fact the user did not assert.
  - Describe the contents of an attachment. A `[file attached: ...]` line tells
    you the user sent something, nothing more — what is in it is not available
    to you, and guessing is a hallucination even when the previous turn makes
    it obvious.

You receive three pieces:
  - PREVIOUS_ASSISTANT (may be empty): the assistant's prior reply. CONTEXT
    ONLY, for placing references. Nothing from it may appear in a citation
    except the identity of something the user referred to.
  - USER: the user's latest message. The ONLY source of assertions.
  - ALREADY_SAVED (may be empty): citations the operator already committed
    during this turn. Do NOT re-suggest near-duplicates.

What to cite (when the user introduces them):
  - Identifiers, hostnames, URLs, file paths, service names, project names.
  - People the user references by name/role (coworker, boss, on-call lead).
  - Conventions the user states (where staging lives, which DB is prod).
  - Schedules and preferences ("don't ping me 11pm-7am", "I prefer terse
    replies", "always use lowercase").
  - The user's own history: what they did, where they were, what happened to
    them. Carry whatever time reference they gave, so the occasion stays
    identifiable later.

What NOT to cite:
  - Anything the user did not assert. A lead-in may place a reference they
    left open; it may not add a claim of its own.
  - Anything already in ALREADY_SAVED.
  - Anything quoted from a third party (a DM the user is forwarding).
  - Work in flight: what is running, queued, or being worked on right now.
    The orchestrator tracks that, and it is stale within the hour.
  - Questions or speculation. Only assertions.
  - Secrets: passwords, API tokens/keys, OTP codes, full credit-card numbers.

Mark each citation `behavioral` or not. Behavioral means the user is
directing how the agent itself should act — its manner, timing, defaults,
what it may do unasked. Everything else is not behavioral, including facts
about the user, their world, and their history, however durable those are.
A citation is behavioral because of what it governs, not because it sounds
like a preference or is phrased as an imperative.

Output format — JSON ONLY:
  {"candidates": [{"text": "...", "behavioral": false}]}

Each citation:
  - ≤200 chars.
  - Contains a direct quote from the user message for the content.
  - Self-contained — readable a year from now without the original message.

If the user introduced nothing citable, return {"candidates": []}.
Trivial turns ("ok", "thanks", "hi", short questions, status checks) almost
always produce nothing.
"""


class LLMChat(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Candidate:
    """One suggested citation. `behavioral` means it directs how the agent
    should act, which is what the operator turns into a standing memory —
    injected every session instead of retrieved by relevance."""

    text: str
    behavioral: bool = False


async def extract_candidates(
    llm: LLMChat,
    *,
    model: str,
    user_text: str,
    prev_assistant_text: str | None,
    already_saved: list[str] | None = None,
) -> list[Candidate]:
    """Run the suggester LLM, parse the JSON response, return zero-or-more
    candidate facts the operator may want to save. Raises on LLM transport
    failure (so the caller can surface the failure to the user); a
    malformed/empty model response returns [].

    `already_saved` lists facts the operator committed during this turn —
    passed to the model so it doesn't re-suggest near-duplicates."""
    user_block = _truncate(user_text, _USER_CHAR_CAP)
    prev_block = _truncate(prev_assistant_text or "", _PREV_ASSISTANT_CHAR_CAP)

    parts: list[str] = []
    if prev_block:
        parts.append(f"PREVIOUS_ASSISTANT:\n{prev_block}")
    parts.append(f"USER:\n{user_block}")
    if already_saved:
        saved_block = "\n".join(f"- {s}" for s in already_saved if s.strip())
        if saved_block:
            parts.append(f"ALREADY_SAVED (do not re-suggest):\n{saved_block}")
    body = "\n\n".join(parts)

    resp = await llm.chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ],
        tools=[],
        # Gemini's thinking tokens count against `max_tokens`. Without a
        # bounded thinking_level the default budget swallows most of 512
        # and the JSON response gets truncated mid-object. Extraction
        # benefits from a little reasoning (pronoun resolution, dedup vs.
        # ALREADY_SAVED), so "low" rather than "minimal".
        max_tokens=1536,
        reasoning_effort="low",
    )
    text = (resp.get("content") or "").strip()
    if not text:
        return []
    data = _parse_json_loose(text)
    if not isinstance(data, dict):
        return []
    # Accept both `candidates` (new) and `facts` (old) keys — older
    # extractor responses or models trained on the prior prompt sometimes
    # emit "facts" anyway, and there's no reason to drop them.
    raw = data.get("candidates")
    if not isinstance(raw, list):
        raw = data.get("facts")
    if not isinstance(raw, list):
        return []
    # Bare strings are still accepted: models drift back to the older
    # flat-list shape, and a citation whose flag is missing is worth keeping
    # as a plain (non-behavioral) one. Defaulting the other way would let a
    # formatting slip pin an ordinary fact into every context window.
    out: list[Candidate] = []
    for item in raw:
        if isinstance(item, str):
            text, behavioral = item.strip(), False
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            behavioral = bool(item.get("behavioral"))
        else:
            continue
        if text:
            out.append(Candidate(text=text, behavioral=behavioral))
    return out


# ---- helpers ---------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n…[truncated]…\n{text[-half:]}"


_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```$")
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_loose(text: str) -> Any:
    """Tolerate models that wrap JSON in fences or add prose. Returns the
    parsed value or {} on failure — never raises."""
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s)
        s = _FENCE_CLOSE_RE.sub("", s)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        m = _JSON_OBJ_RE.search(s)
        if m:
            try:
                return json.loads(m.group(0))
            except (ValueError, TypeError):
                pass
        return {}
