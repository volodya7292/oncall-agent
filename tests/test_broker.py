"""Broker state machine tests.

Verify:
  * read-only auto-allows without round-tripping to the approval client
  * catastrophic auto-denies
  * mutating escalates and the supplied phrase is matched in submit_response
  * dedup on (session_id, tool_use_id) on resume
  * consecutive-denial backstop
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from oncall.approval_client import (
    AutoAllowApprovalClient,
    AutoDenyApprovalClient,
    HttpLongPollApprovalClient,
    canonicalize_phrase,
    generate_challenge_phrase,
    is_kill_phrase,
    phrases_match,
)
from oncall.broker import Broker, MAX_CONSECUTIVE_DENIALS

from tests.support import stub_classifier
from oncall.db import Database
from oncall.models import Task, TaskState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def db():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "test.db")
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


@pytest.fixture
def events():
    """A no-op event publisher that records what got published."""
    captured: list[tuple[UUID, str, dict[str, Any]]] = []

    async def publish(task_id, type_, payload):
        captured.append((task_id, type_, payload))

    publish.captured = captured  # type: ignore[attr-defined]
    return publish


async def _make_task(db: Database, prompt: str = "test task") -> Task:
    task = Task(session_id=f"sess-{generate_challenge_phrase().split()[0]}", prompt=prompt)
    await db.insert_task(task)
    return task


# ---------------------------------------------------------------------------
# Challenge phrase utilities
# ---------------------------------------------------------------------------

def test_canonicalize_strips_punct_and_case() -> None:
    assert canonicalize_phrase("Yes Yes Yes.") == "yes yes yes"
    assert canonicalize_phrase("yes  yes  yes") == "yes yes yes"
    assert canonicalize_phrase("YES, YES, YES!") == "yes yes yes"


def test_phrases_match_affirmative() -> None:
    # Tolerates case, commas, trailing punctuation; needs ≥3 affirm tokens.
    assert phrases_match("", "yes yes yes")
    assert phrases_match("", "YES, yes, yes.")
    assert phrases_match("", "так так так")           # Ukrainian
    assert phrases_match("", "да да да")              # Russian
    assert phrases_match("", "yes так sí")            # mixed languages OK
    # Too few tokens — single / double yes is too easy to type by accident.
    assert not phrases_match("", "yes")
    assert not phrases_match("", "yes yes")
    # Mixed with non-affirm tokens — no.
    assert not phrases_match("", "yes yes maybe")


def test_phrases_match_ignores_expected() -> None:
    # `expected` is kept for API compat but ignored — only the supplied
    # text matters.
    assert phrases_match("anything", "yes yes yes")
    assert not phrases_match("yes yes yes", "no no no")


def test_is_deny_phrase() -> None:
    from oncall.approval_client import is_deny_phrase
    assert is_deny_phrase("no")
    assert is_deny_phrase("No.")
    assert is_deny_phrase("нет")
    assert is_deny_phrase("ні")
    assert is_deny_phrase("non")
    assert is_deny_phrase("no no no")                 # repeated still deny
    assert not is_deny_phrase("yes")
    assert not is_deny_phrase("maybe")
    assert not is_deny_phrase("")


def test_generate_phrase_is_canonical_affirm() -> None:
    assert generate_challenge_phrase() == "yes yes yes"


def test_kill_phrase_detection() -> None:
    assert is_kill_phrase("stop everything")
    assert is_kill_phrase("Hey, stop everything now")
    assert not is_kill_phrase("stop the database")
    assert not is_kill_phrase("everything stop")


# ---------------------------------------------------------------------------
# Readonly auto-allow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_readonly_auto_allows(db, events):
    task = await _make_task(db)
    broker = Broker(db, AutoDenyApprovalClient(), events, classifier=stub_classifier("ls "))  # deny would fail if escalated
    result = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_1",
        tool_name="Bash",
        tool_input={"command": "ls /etc"},
    )
    assert result.behavior == "allow"
    # And we recorded a row with auto=1.
    row = await db.get_approval(  # type: ignore[func-returns-value]
        UUID(events.captured[0][2]["approval_id"])  # type: ignore[attr-defined]
    )
    assert row is not None
    assert row["auto"] == 1
    assert row["decision"] == "allow"


# ---------------------------------------------------------------------------
# Catastrophic auto-deny
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catastrophic_auto_denies(db, events):
    task = await _make_task(db)
    broker = Broker(db, AutoAllowApprovalClient(), events, classifier=stub_classifier("ls "))  # allow would fail if escalated
    result = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_x",
        tool_name="Bash",
        tool_input={"command": "rm -rf /"},
    )
    assert result.behavior == "deny"
    assert "catastrophic" in (result.message or "").lower()


# ---------------------------------------------------------------------------
# Unknown tool auto-denies instead of waking the owner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_tool_auto_denies_without_escalating(db, events):
    """A tool the classifier doesn't know must come back to the agent, not go
    to the human.

    The executor is a Claude CLI with built-ins we never classified, so it
    reaches for real-to-it tools like AskUserQuestion. Those fell through to
    the MUTATING default-deny posture and escalated, which asked the owner to
    approve a call that could not have worked either way (the executor runs
    --print, so a built-in prompt has no one to prompt).
    """
    task = await _make_task(db)
    # Auto-ALLOW client: if the broker escalated, this would come back
    # allow. Asserting deny proves the owner was never consulted.
    broker = Broker(db, AutoAllowApprovalClient(), events, classifier=stub_classifier("ls "))
    result = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_unknown",
        tool_name="AskUserQuestion",
        tool_input={"questions": [{"question": "proceed?"}]},
    )
    assert result.behavior == "deny"
    msg = (result.message or "").lower()
    assert "askuserquestion" in msg, "agent must learn which tool was refused"
    assert "mcp__oncall__ask_user" in msg, "point it at the tool that works"
    # Counted, so a model that keeps retrying trips the existing halt backstop.
    refreshed = await db.get_task(task.id)
    assert refreshed.consecutive_denials == 1


# ---------------------------------------------------------------------------
# Mutating escalates; phrase match coerces decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mutating_escalates_and_phrase_matches(db, events):
    task = await _make_task(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events, classifier=stub_classifier("ls "))

    async def respond_when_ready():
        # Poll for a pending approval and resolve via submit_response (the API path).
        for _ in range(200):
            pendings = await db.list_pending_approvals()
            if pendings:
                approval = pendings[0]
                assert approval.challenge_phrase is not None
                await broker.submit_response(
                    approval_id=approval.id,
                    decision="allow",
                    challenge_phrase_supplied=approval.challenge_phrase,
                )
                return
            await asyncio.sleep(0.005)
        raise AssertionError("approval never appeared")

    decide = broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_2",
        tool_name="Bash",
        tool_input={"command": "echo hi >> /tmp/oncall-test.log"},
    )
    result, _ = await asyncio.gather(decide, respond_when_ready())
    assert result.behavior == "allow"


@pytest.mark.asyncio
async def test_phrase_mismatch_coerces_deny(db, events):
    task = await _make_task(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events, classifier=stub_classifier("ls "))

    async def respond_when_ready():
        for _ in range(200):
            pendings = await db.list_pending_approvals()
            if pendings:
                approval = pendings[0]
                await broker.submit_response(
                    approval_id=approval.id,
                    decision="allow",  # user says allow, but...
                    challenge_phrase_supplied="totally wrong words here",  # ...phrase wrong
                )
                return
            await asyncio.sleep(0.005)
        raise AssertionError("approval never appeared")

    decide = broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_3",
        tool_name="Bash",
        tool_input={"command": "echo hi >> /tmp/x.log"},
    )
    result, _ = await asyncio.gather(decide, respond_when_ready())
    assert result.behavior == "deny"


@pytest.mark.asyncio
async def test_explicit_user_deny_names_user_as_source(db, events):
    """Regression: an explicit user deny (decision='deny', not a phrase typo)
    was mislabeled 'Challenge phrase mismatch — coerced to deny'. The executor
    then couldn't tell the user had refused and blamed the DM allowlist. The
    deny reason returned to the executor must name the user as the source."""
    task = await _make_task(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events, classifier=stub_classifier("ls "))

    async def respond_when_ready():
        for _ in range(200):
            pendings = await db.list_pending_approvals()
            if pendings:
                await broker.submit_response(
                    approval_id=pendings[0].id,
                    decision="deny",                  # user explicitly refuses
                    challenge_phrase_supplied="no",   # a deny word, not the phrase
                )
                return
            await asyncio.sleep(0.005)
        raise AssertionError("approval never appeared")

    decide = broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_deny",
        tool_name="Bash",
        tool_input={"command": "echo hi >> /tmp/x.log"},
    )
    result, _ = await asyncio.gather(decide, respond_when_ready())
    assert result.behavior == "deny"
    msg = (result.message or "").lower()
    assert "denied" in msg
    assert "challenge phrase mismatch" not in msg


# ---------------------------------------------------------------------------
# Dedup on (session_id, tool_use_id) — simulates --resume after crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_dedup_returns_cached(db, events):
    task = await _make_task(db)
    broker = Broker(db, AutoAllowApprovalClient(), events, classifier=stub_classifier("ls "))
    first = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_dup",
        tool_name="Bash",
        tool_input={"command": "ls /tmp"},
    )
    assert first.behavior == "allow"
    # Second call with same tool_use_id should be served from the dedup cache —
    # no new approval row, no event published the second time.
    pre_event_count = len(events.captured)  # type: ignore[attr-defined]
    second = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_dup",
        tool_name="Bash",
        tool_input={"command": "ls /tmp"},
    )
    assert second.behavior == "allow"
    assert len(events.captured) == pre_event_count  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Consecutive-denial backstop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consecutive_denial_backstop(db, events):
    task = await _make_task(db)
    broker = Broker(db, AutoDenyApprovalClient(), events, classifier=stub_classifier("ls "), max_consecutive_denials=2)

    async def do_one(tu_id: str):
        return await broker.decide(
            session_id=task.session_id,
            tool_use_id=tu_id,
            tool_name="Bash",
            tool_input={"command": "rm /tmp/foo"},
        )

    r1 = await do_one("tu_a")
    r2 = await do_one("tu_b")
    r3 = await do_one("tu_c")
    assert r1.behavior == "deny"
    assert r2.behavior == "deny"
    assert r3.behavior == "deny"
    assert "halted" in (r3.message or "").lower()
