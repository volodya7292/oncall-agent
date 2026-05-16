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
    assert canonicalize_phrase("Amber Paper Compass.") == "amber paper compass"
    assert canonicalize_phrase("amber  paper  compass") == "amber paper compass"
    assert canonicalize_phrase("AMBER, PAPER, COMPASS!") == "amber paper compass"


def test_phrases_match_tolerates_normalization() -> None:
    assert phrases_match("amber paper compass", "AMBER, paper compass.")
    assert not phrases_match("amber paper compass", "amber paper other")


def test_generate_phrase_is_three_words() -> None:
    p = generate_challenge_phrase()
    assert len(p.split()) == 3


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
    broker = Broker(db, AutoDenyApprovalClient(), events)  # deny would fail if escalated
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
    broker = Broker(db, AutoAllowApprovalClient(), events)  # allow would fail if escalated
    result = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_x",
        tool_name="Bash",
        tool_input={"command": "rm -rf /"},
    )
    assert result.behavior == "deny"
    assert "catastrophic" in (result.message or "").lower()


# ---------------------------------------------------------------------------
# Mutating escalates; phrase match coerces decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mutating_escalates_and_phrase_matches(db, events):
    task = await _make_task(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events)

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
    broker = Broker(db, client, events)

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


# ---------------------------------------------------------------------------
# Dedup on (session_id, tool_use_id) — simulates --resume after crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_dedup_returns_cached(db, events):
    task = await _make_task(db)
    broker = Broker(db, AutoAllowApprovalClient(), events)
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
    broker = Broker(db, AutoDenyApprovalClient(), events, max_consecutive_denials=2)

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
