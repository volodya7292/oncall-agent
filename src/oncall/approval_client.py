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


_PHRASE_WORDS: tuple[str, ...] = (
    "amber", "anchor", "apple", "arrow", "atlas", "bamboo", "basin", "beacon",
    "boulder", "branch", "breeze", "candle", "canyon", "cedar", "cipher", "clove",
    "comet", "compass", "copper", "coral", "cotton", "crater", "crystal", "delta",
    "ember", "falcon", "ferret", "flint", "forest", "garnet", "geyser", "ginger",
    "glacier", "granite", "harbor", "harvest", "hollow", "horizon", "iris",
    "ivory", "jasper", "juniper", "kestrel", "kettle", "lantern", "lichen",
    "linen", "lotus", "magnet", "marble", "meadow", "meteor", "mosaic", "nectar",
    "nimbus", "oaken", "obsidian", "orchid", "otter", "paper", "pebble", "penguin",
    "petal", "pewter", "pillar", "pollen", "prairie", "quartz", "quill", "raven",
    "ribbon", "river", "rocket", "saffron", "sapphire", "sequoia", "shoal",
    "silver", "snowfall", "sparrow", "stalk", "summit", "sundial", "syrup",
    "tangent", "tassel", "thicket", "thunder", "topaz", "trellis", "tundra",
    "umber", "valley", "velvet", "viola", "walnut", "willow", "yarrow", "zenith",
)


def generate_challenge_phrase(rng: random.Random | None = None) -> str:
    """Three random words from a low-ambiguity dictionary, space-separated."""
    r = rng or random.SystemRandom()
    return " ".join(r.choice(_PHRASE_WORDS) for _ in range(3))


_NON_ALPHA = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")


def canonicalize_phrase(phrase: str) -> str:
    """Normalize for comparison: lowercase, strip punctuation, collapse whitespace."""
    s = phrase.lower()
    s = _NON_ALPHA.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def phrases_match(expected: str, supplied: str) -> bool:
    return canonicalize_phrase(expected) == canonicalize_phrase(supplied)


# ---------------------------------------------------------------------------
# Kill phrase
# ---------------------------------------------------------------------------

_KILL_RE = re.compile(r"\bstop\s+everything\b", re.IGNORECASE)


def is_kill_phrase(text: str) -> bool:
    return bool(_KILL_RE.search(text))
