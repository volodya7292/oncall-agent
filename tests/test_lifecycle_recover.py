"""Lifecycle.recover() — daemon-restart resume path.

On startup we scan the DB for any task left in {running, awaiting_approval}
and re-spawn its supervisor with --resume <session_id>. Combined with the
broker's (session_id, tool_use_id) dedup, the Claude CLI replays its session
JSONL, re-emits the same tool_use_id, and the broker either:
  * returns the cached approval result if the user already responded, or
  * re-attaches to the still-pending row to await a fresh response.

These tests use a FakeSupervisor so no claude binary is involved.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from oncall import lifecycle as lifecycle_mod
from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.models import Task, TaskState, TerminalReason


class FakeSupervisor:
    """Records every constructor + .run() call so the test can verify which
    tasks were rehydrated and whether `resuming` was set."""

    instances: list["FakeSupervisor"] = []

    def __init__(self, **_ignored: Any) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self.release = asyncio.Event()
        FakeSupervisor.instances.append(self)

    async def run(self, task: Task, *, resuming: bool = False) -> TerminalReason:
        self.run_calls.append({"task_id": task.id, "resuming": resuming})
        # Stay parked until the test releases us — keeps the runner task alive
        # long enough to inspect state without a race.
        await self.release.wait()
        return TerminalReason.SUCCESS


@pytest.fixture(autouse=True)
def reset_supervisors():
    FakeSupervisor.instances.clear()
    yield
    FakeSupervisor.instances.clear()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        oncall_token="t",
        oncall_db_path=tmp_path / "db.sqlite",
        ai_gateway_api_key="x",
        oncall_max_concurrent_tasks=10,
    )


@pytest.fixture
async def stack(settings, monkeypatch):
    monkeypatch.setattr(lifecycle_mod, "Supervisor", FakeSupervisor)
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    client = HttpLongPollApprovalClient()
    broker = Broker(db, client, events.publish)
    lc = Lifecycle(
        db=db, broker=broker, approval_client=client,
        events=events, settings=settings, paths=Paths(),
    )
    try:
        yield {"lc": lc, "db": db, "broker": broker}
    finally:
        for s in FakeSupervisor.instances:
            s.release.set()
        await lc.shutdown()
        await db.close()


async def _insert_task(db: Database, *, state: TaskState, prompt: str = "x") -> Task:
    from uuid import uuid4
    task = Task(session_id=str(uuid4()), prompt=prompt, state=state)
    await db.insert_task(task)
    return task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recover_respawns_running_and_awaiting_approval(stack):
    """Both `running` and `awaiting_approval` rows are re-spawned with
    resuming=True. Terminal rows are not."""
    db = stack["db"]
    t_running = await _insert_task(db, state=TaskState.RUNNING)
    t_pending_appr = await _insert_task(db, state=TaskState.AWAITING_APPROVAL)
    # These should NOT be touched.
    await _insert_task(db, state=TaskState.COMPLETED)
    await _insert_task(db, state=TaskState.FAILED)
    await _insert_task(db, state=TaskState.KILLED)
    await _insert_task(db, state=TaskState.PENDING)

    await stack["lc"].recover()
    # Give the runner tasks a tick to call supervisor.run().
    for _ in range(100):
        run_calls = [c for s in FakeSupervisor.instances for c in s.run_calls]
        if len(run_calls) >= 2:
            break
        await asyncio.sleep(0.01)

    recovered_ids = {c["task_id"] for s in FakeSupervisor.instances for c in s.run_calls}
    assert recovered_ids == {t_running.id, t_pending_appr.id}
    # Both must have resuming=True.
    for sup in FakeSupervisor.instances:
        for call in sup.run_calls:
            assert call["resuming"] is True


@pytest.mark.asyncio
async def test_recover_with_no_alive_tasks_is_a_noop(stack):
    """Clean startup → no supervisors spawned."""
    db = stack["db"]
    await _insert_task(db, state=TaskState.COMPLETED)
    await _insert_task(db, state=TaskState.FAILED)

    await stack["lc"].recover()
    await asyncio.sleep(0.02)

    assert FakeSupervisor.instances == []


@pytest.mark.asyncio
async def test_recover_registers_running_dict(stack):
    """Recovered tasks must show up in lifecycle.running so a subsequent
    kill() can reach them."""
    db = stack["db"]
    task = await _insert_task(db, state=TaskState.RUNNING)
    await stack["lc"].recover()
    await asyncio.sleep(0.01)

    assert task.id in stack["lc"].running

    # And kill() unwinds it cleanly.
    for s in FakeSupervisor.instances:
        s.release.set()
    killed = await stack["lc"].kill(task.id, reason="test")
    assert killed is True


@pytest.mark.asyncio
async def test_shutdown_denies_pending_approvals_and_kills_tasks(stack):
    """Safety invariant: when the daemon shuts down with tasks parked at a
    broker approval, every pending approval must be resolved as `deny` AND
    its task must move to `killed` in the DB. Otherwise the next daemon
    boot's recover() would try `claude --resume <session>` against an
    orphan session — exactly the cli_error failure mode we hit on
    2026-05-17.
    """
    from oncall.models import ApprovalRequest, ClassifierVerdict

    db = stack["db"]
    lc = stack["lc"]

    # Submit a task and let the supervisor "spawn" (FakeSupervisor parks
    # on its release event).
    task = await lc.submit_task(prompt="ssh myserver 'docker ps'", model=None)
    await asyncio.sleep(0)  # let the spawned runner register itself

    # Create a pending approval as if the broker had parked one.
    pending = ApprovalRequest(
        task_id=task.id, session_id=task.session_id,
        tool_use_id="tu_shutdown",
        tool_name="Bash",
        tool_input={"command": "ssh myserver 'docker ps'"},
        classifier_verdict=ClassifierVerdict.MUTATING,
        canonical_command="ssh myserver 'docker ps'",
        blast_radius="ssh.",
        challenge_phrase="amber paper compass",
    )
    await db.create_pending_approval(pending)
    # Park a future on the approval client too — kill() resolves it.
    fut = asyncio.get_event_loop().create_future()
    stack["lc"].approval_client._pending[pending.id] = fut

    await lc.shutdown()

    # Future was resolved deny.
    assert fut.done()
    result = fut.result()
    assert result.behavior == "deny", "shutdown must coerce pending approvals to deny"

    # DB approval row is no longer pending.
    assert await db.list_pending_approvals() == []

    # Task ended up in KILLED, so the next boot's recover() skips it
    # instead of trying to --resume an orphan session.
    refreshed = await db.get_task(task.id)
    assert refreshed is not None
    assert refreshed.state == TaskState.KILLED
    assert refreshed.terminal_reason == TerminalReason.KILLED


@pytest.mark.asyncio
async def test_resume_reattach_to_pending_approval(stack):
    """If the daemon crashed BEFORE the user responded, the approval row is
    still `pending`. On --resume, broker.decide must re-attach to it (re-publish
    approval.requested + await) rather than inserting a duplicate row, which
    would violate UNIQUE(session_id, tool_use_id)."""
    from oncall.models import ApprovalRequest, ClassifierVerdict
    from oncall.approval_client import AutoAllowApprovalClient

    db = stack["db"]
    # Build a stack with the auto-allow client so the re-attach await completes.
    events = EventBus(db)
    client = AutoAllowApprovalClient()
    broker = Broker(db, client, events.publish)

    task = await _insert_task(db, state=TaskState.AWAITING_APPROVAL)
    pending_req = ApprovalRequest(
        task_id=task.id, session_id=task.session_id,
        tool_use_id="tu_crash",
        tool_name="Bash",
        tool_input={"command": "mkdir foo"},
        classifier_verdict=ClassifierVerdict.MUTATING,
        canonical_command="mkdir foo",
        blast_radius="creates a directory.",
        challenge_phrase="amber paper compass",
    )
    await db.create_pending_approval(pending_req)

    # Simulate the same tool_use_id surfacing post-resume.
    result = await broker.decide(
        session_id=task.session_id,
        tool_use_id="tu_crash",
        tool_name="Bash",
        tool_input={"command": "mkdir foo"},
    )
    # AutoAllowApprovalClient resolves the await immediately as allow.
    assert result.behavior == "allow"
    # No duplicate row created — still exactly one approval for this dedup key.
    rows = await db.list_pending_approvals()
    # Original was resolved by the auto-allow await → no more pending.
    assert rows == []


@pytest.mark.asyncio
async def test_resume_dedup_returns_cached_approval(stack):
    """The dedup half: if an approval was already resolved BEFORE the crash,
    broker.decide returns the cached result on the re-emitted tool_use_id
    without prompting the user again. (This is what makes recover() safe.)"""
    from oncall.models import ApprovalRequest, ApprovalResult, ClassifierVerdict
    from uuid import uuid4

    db = stack["db"]
    task = await _insert_task(db, state=TaskState.RUNNING)
    req = ApprovalRequest(
        task_id=task.id, session_id=task.session_id,
        tool_use_id="tu_post_crash",
        tool_name="Bash",
        tool_input={"command": "mkdir foo"},
        classifier_verdict=ClassifierVerdict.MUTATING,
        canonical_command="mkdir foo",
        blast_radius="creates a directory.",
        challenge_phrase="amber paper compass",
    )
    await db.create_pending_approval(req)
    await db.append_approval_response(
        req.id,
        ApprovalResult(
            request_id=req.id, behavior="allow",
            challenge_phrase_supplied="amber paper compass",
            challenge_matched=True, message=None,
        ),
    )

    # On --resume the same tool_use_id surfaces again. Broker must short-circuit.
    result = await stack["broker"].decide(
        session_id=task.session_id,
        tool_use_id="tu_post_crash",
        tool_name="Bash",
        tool_input={"command": "mkdir foo"},
    )
    assert result.behavior == "allow"
    # No new approval row created (dedup_hit path).
    rows = await db.list_pending_approvals()
    assert rows == []  # original was already resolved
