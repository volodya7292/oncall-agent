"""User-turn fact-candidate suggester.

Looks at the user's latest message and the immediately preceding assistant
turn (context only — never a source of facts), and asks a cheap LLM for a
JSON list of durable fact CANDIDATES the operator might want to remember.
The candidates are NOT auto-saved; the operator is auto-pinged with the
suggestions and decides which (if any) to commit via its `save_memory`
tool. This keeps the operator authoritative over what enters memory.

The mental model: each candidate, if absent next time, would force a
clarifying question. So the suggester is asking "what makes this terse
intent self-contained?" — not "what's interesting?"

To avoid re-suggesting things the operator already saved during the same
turn, callers pass `already_saved` so the suggester can exclude near-
duplicates from its output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol


log = logging.getLogger(__name__)


# Bounded inputs so the prompt stays cheap and predictable regardless of
# what the user pasted in. Truncation keeps head + tail so identifiers
# near either end of a long message still surface.
_PREV_ASSISTANT_CHAR_CAP = 2000
_USER_CHAR_CAP = 4000


EXTRACTOR_SYSTEM_PROMPT = """\
You suggest durable fact CANDIDATES from a single user message to a personal
on-call agent. Your output is advisory — the agent decides which (if any)
candidates to actually save via its `save_memory` tool. Suggest short,
declarative, reusable facts that would let the user phrase future requests
terser without the agent having to ask clarifying questions.

You receive three pieces:
  - PREVIOUS_ASSISTANT (may be empty): the assistant's prior reply. CONTEXT
    ONLY — never extract facts from it. Use it to disambiguate the user
    message ("use the staging one" means nothing without the prior question).
  - USER: the user's latest message. This is the ONLY source of facts.
  - ALREADY_SAVED (may be empty): facts the operator already committed
    during this turn. Do NOT re-suggest any candidate that's a near-
    duplicate of an entry here.

What to suggest:
  - Identifiers, hostnames, URLs, file paths, service names, project names.
  - People the user references by name/role (coworker, boss, on-call lead).
  - Conventions the user states (where staging lives, which DB is prod).
  - Schedules and preferences ("don't ping me 11pm-7am", "I prefer terse
    replies", "always use lowercase").

What NOT to suggest:
  - Anything already in ALREADY_SAVED (or trivially paraphrasing it).
  - Anything quoted from a third party (a DM the user is forwarding).
  - Anything the assistant said.
  - Task-specific transient state ("the error was X", "T1 is running").
  - Questions or speculation. Only assertions.
  - Secrets: passwords, API tokens/keys, OTP codes, full credit-card numbers,
    anything credentials-shaped.

Output format — JSON ONLY:
  {"candidates": ["...", "..."]}

Each candidate:
  - One declarative sentence, ≤200 chars.
  - Phrased in third person about the user where natural ("the user prefers
    terse replies"; "staging API is at api-staging.example.com:8443").
  - Self-contained — readable a year from now without the original message.

If nothing memorable beyond what's in ALREADY_SAVED, return {"candidates": []}.
Trivial turns ("ok", "thanks", "hi", short questions, status checks) almost
always produce no candidates.
"""


class LLMChat(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


async def extract_candidates(
    llm: LLMChat,
    *,
    model: str,
    user_text: str,
    prev_assistant_text: str | None,
    already_saved: list[str] | None = None,
) -> list[str]:
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
        max_tokens=512,
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
    return [s.strip() for s in raw if isinstance(s, str) and s.strip()]


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
