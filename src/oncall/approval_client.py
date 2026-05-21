"""ApprovalClient abstractions.

The broker awaits an `ApprovalResult` for each mutating tool call. *Where* that
result comes from is pluggable: tests use Auto{Allow,Deny}; production uses the
HTTP long-poll variant where a future is resolved by the FastAPI handler when
the operator (or a direct API caller) submits a response.
"""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from .models import ApprovalRequest, ApprovalResult, utcnow


class ApprovalClient(Protocol):
    """Where the broker blocks waiting for a human decision."""

    async def request_approval(self, req: ApprovalRequest) -> ApprovalResult: ...


class AutoAllowApprovalClient:
    """Allows every mutating call. For tests only."""

    async def request_approval(self, req: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(
            request_id=req.id,
            behavior="allow",
            challenge_matched=True,
            message="auto-allowed (test)",
            responded_at=utcnow(),
        )


class AutoDenyApprovalClient:
    """Denies every mutating call. For tests only."""

    async def request_approval(self, req: ApprovalRequest) -> ApprovalResult:
        return ApprovalResult(
            request_id=req.id,
            behavior="deny",
            challenge_matched=False,
            message="auto-denied (test)",
            responded_at=utcnow(),
        )


class HttpLongPollApprovalClient:
    """Production: the broker awaits a Future; FastAPI resolves it on respond.

    Thread-safety: meant for a single asyncio loop. Not safe across loops/processes.
    All approval state on disk lives in `approvals` (via db.py); this object only
    holds the in-memory wake-up channel.
    """

    def __init__(self) -> None:
        self._pending: dict[UUID, asyncio.Future[ApprovalResult]] = {}

    async def request_approval(self, req: ApprovalRequest) -> ApprovalResult:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ApprovalResult] = loop.create_future()
        self._pending[req.id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=req.timeout_seconds)
        except asyncio.TimeoutError:
            return ApprovalResult(
                request_id=req.id,
                behavior="deny",
                challenge_matched=False,
                message="Approval timed out.",
                responded_at=utcnow(),
            )
        finally:
            self._pending.pop(req.id, None)

    def resolve(self, req_id: UUID, result: ApprovalResult) -> bool:
        """Resolve a pending future. Returns True if a future was waiting."""
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return False
        fut.set_result(result)
        return True

    def has_pending(self, req_id: UUID) -> bool:
        return req_id in self._pending


# ---------------------------------------------------------------------------
# Challenge phrase utilities — owned by the orchestrator, NOT the operator.
# Generation and matching live here; operator only reads them aloud verbatim.
# ---------------------------------------------------------------------------

import random
import re
import unicodedata


# Multilingual affirm / deny vocab. Curated for words people actually type to
# confirm or refuse; not a full dictionary. Diacritic-bearing forms are
# spelled exactly as native speakers write them. We lowercase + NFC-normalize
# input before lookup — we do NOT strip diacritics (matches what someone
# with the right keyboard would type).
_AFFIRM_WORDS = frozenset({
    # English
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    # Russian
    "да", "ага", "угу", "давай",
    # Ukrainian
    "так", "авжеж", "аякже",
    # Polish
    "tak",
    # German / Dutch / Swedish / Danish / Norwegian
    "ja",
    # Spanish
    "sí", "si", "vale", "claro",
    # French
    "oui", "ouais",
    # Italian
    "sì", "certo",
    # Portuguese
    "sim",
    # Finnish
    "joo", "kyllä",
    # Turkish
    "evet", "tamam",
})

_DENY_WORDS = frozenset({
    # English
    "no", "nope", "nah",
    # Russian
    "нет",
    # Ukrainian
    "ні",
    # Polish
    "nie",
    # German
    "nein",
    # French (also colloquial "nan")
    "non", "nan",
    # Portuguese
    "não", "nao",
    # Dutch
    "nee",
    # Swedish / Danish
    "nej",
    # Norwegian
    "nei",
    # Finnish
    "ei",
    # Turkish
    "hayır",
})

# Required affirmative-token count. 3 = deliberate friction; "yes" alone is
# too easy to type by accident. A single "no" is enough to deny.
_AFFIRM_MIN_TOKENS = 3


def generate_challenge_phrase(rng: random.Random | None = None) -> str:
    """Returns the canonical affirm phrase the user types to allow an action.
    `rng` is kept for API compatibility with the old random generator but
    ignored — the phrase is fixed so the prompt to the user is consistent
    and easy to type."""
    del rng
    return "yes yes yes"


def _tokenize(text: str) -> list[str]:
    """Lowercase + NFC-normalize, replace anything that isn't a letter or
    whitespace with a space, split on whitespace. Preserves diacritics."""
    s = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    cur: list[str] = []
    for ch in s:
        if unicodedata.category(ch).startswith("L"):
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def canonicalize_phrase(phrase: str) -> str:
    """Kept for audit-log callers — lowercase + collapse whitespace + strip
    non-letter chars. Don't use for matching logic; use phrases_match /
    is_deny_phrase instead."""
    return " ".join(_tokenize(phrase))


def phrases_match(expected: str, supplied: str) -> bool:
    """True iff `supplied` is an affirmative reply: ≥3 tokens, every one of
    them in the affirmative-word set (any supported language). `expected`
    is ignored — the canonical phrase is fixed; only the supplied text
    matters. Mixed case, commas, trailing punctuation, and mixed
    languages are all tolerated."""
    del expected
    tokens = _tokenize(supplied)
    if len(tokens) < _AFFIRM_MIN_TOKENS:
        return False
    return all(t in _AFFIRM_WORDS for t in tokens)


def is_deny_phrase(supplied: str) -> bool:
    """True iff `supplied` is a deny reply: ≥1 deny tokens, every one of
    them in the deny-word set. A bare 'no' (or 'нет', 'ні', etc.)
    suffices."""
    tokens = _tokenize(supplied)
    if not tokens:
        return False
    return all(t in _DENY_WORDS for t in tokens)


# ---------------------------------------------------------------------------
# Kill phrase
# ---------------------------------------------------------------------------

_KILL_RE = re.compile(r"\bstop\s+everything\b", re.IGNORECASE)


def is_kill_phrase(text: str) -> bool:
    return bool(_KILL_RE.search(text))
