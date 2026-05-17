"""User-turn fact extraction.

Looks at the user's latest message and the immediately preceding assistant
turn (context only — never a source of facts), and asks a cheap LLM for a
JSON list of durable facts worth remembering. The operator never sees this
prompt or its output; the extracted facts go straight to OperatorMemory.

The mental model: each fact, if absent next time, would force a clarifying
question. So the extractor is asking "what makes this terse intent
self-contained?" — not "what's interesting?"
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
You extract durable facts from a single user message to a personal on-call
agent. Your output IS the agent's long-term memory — short, declarative,
reusable facts that let the user phrase future requests terser without the
agent having to ask clarifying questions.

You receive two pieces:
  - PREVIOUS_ASSISTANT (may be empty): the assistant's prior reply. CONTEXT
    ONLY — never extract facts from it. Use it to disambiguate the user
    message ("use the staging one" means nothing without the prior question).
  - USER: the user's latest message. This is the ONLY source of facts.

What to extract:
  - Identifiers, hostnames, URLs, file paths, service names, project names.
  - People the user references by name/role (coworker, boss, on-call lead).
  - Conventions the user states (where staging lives, which DB is prod).
  - Schedules and preferences ("don't ping me 11pm-7am", "I prefer terse
    replies", "always use lowercase").

What NOT to extract:
  - Anything quoted from a third party (a DM the user is forwarding).
  - Anything the assistant said.
  - Task-specific transient state ("the error was X", "T1 is running").
  - Questions or speculation. Only assertions.
  - Secrets: passwords, API tokens/keys, OTP codes, full credit-card numbers,
    anything credentials-shaped.

Output format — JSON ONLY:
  {"facts": ["...", "..."]}

Each fact:
  - One declarative sentence, ≤200 chars.
  - Phrased in third person about the user where natural ("the user prefers
    terse replies"; "staging API is at api-staging.example.com:8443").
  - Self-contained — readable a year from now without the original message.

If nothing memorable, return {"facts": []}. Trivial turns ("ok", "thanks",
"hi", short questions, status checks) almost always produce no facts.
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


async def extract_facts(
    llm: LLMChat,
    *,
    model: str,
    user_text: str,
    prev_assistant_text: str | None,
) -> list[str]:
    """Run the extractor LLM, parse the JSON response, return zero-or-more
    facts. Raises on LLM transport failure (so the caller can surface the
    failure to the user); a malformed/empty model response returns []."""
    user_block = _truncate(user_text, _USER_CHAR_CAP)
    prev_block = _truncate(prev_assistant_text or "", _PREV_ASSISTANT_CHAR_CAP)

    parts: list[str] = []
    if prev_block:
        parts.append(f"PREVIOUS_ASSISTANT:\n{prev_block}")
    parts.append(f"USER:\n{user_block}")
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
    facts = data.get("facts")
    if not isinstance(facts, list):
        return []
    return [s.strip() for s in facts if isinstance(s, str) and s.strip()]


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
