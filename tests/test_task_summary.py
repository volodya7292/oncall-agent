"""task_summary.summarize_task — builds a prompt from a task's event trail,
calls a one-shot runner, persists to tasks.result_summary."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from oncall.db import Database
from oncall.models import Task, TaskState
from oncall.task_summary import summarize_task, _format_events_for_prompt


class FakeRunner:
    def __init__(self, *, output: str | None) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def one_shot(self, prompt, *, system_prompt=None, model="sonnet", timeout_s=60.0):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "model": model})
        return self.output


@pytest.fixture
async def db(tmp_path):
    d = Database(tmp_path / "db.sqlite")
    await d.connect()
    yield d
    await d.close()


@pytest.fixture
async def task_with_events(db: Database):
    task = Task(session_id=str(uuid4()), prompt="check staging health", state=TaskState.COMPLETED)
    await db.insert_task(task)
    await db.append_event(task.id, "assistant.text", {"text": "Checking /healthz..."})
    await db.append_event(task.id, "tool_use.requested", {"tool_name": "Bash", "input": {"command": "curl -s staging/healthz"}})
    await db.append_event(task.id, "tool_result", {"is_error": False, "preview": "{\"status\":\"ok\"}"})
    await db.append_event(task.id, "assistant.text", {"text": "Staging is up, /healthz=200."})
    await db.append_event(task.id, "result.final", {"is_error": False, "duration_ms": 1200})
    return task


# ---------------------------------------------------------------------------
# summarize_task
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_task_persists_runner_output(db, task_with_events):
    runner = FakeRunner(output="Checked staging /healthz; it returned 200.")

    result = await summarize_task(db, runner, task_with_events.id)

    assert result == "Checked staging /healthz; it returned 200."
    stored = await db.get_task_result_summary(task_with_events.id)
    assert stored == result


@pytest.mark.asyncio
async def test_summarize_task_passes_compact_event_trail(db, task_with_events):
    """The prompt must include the assistant text + tool uses, but NOT
    state.changed noise."""
    runner = FakeRunner(output="summary")
    await summarize_task(db, runner, task_with_events.id)

    sent = runner.calls[0]["prompt"]
    assert "USER_PROMPT: check staging health" in sent
    assert "/healthz" in sent
    assert "assistant said" in sent
    assert "tool_use" in sent
    # state.changed events shouldn't make it into the prompt.
    assert "state.changed" not in sent


@pytest.mark.asyncio
async def test_summarize_task_uses_sonnet_by_default(db, task_with_events):
    runner = FakeRunner(output="summary")
    await summarize_task(db, runner, task_with_events.id)
    assert runner.calls[0]["model"] == "sonnet"


@pytest.mark.asyncio
async def test_summarize_task_runner_failure_returns_none(db, task_with_events):
    """If the runner returns None (claude not on PATH, etc), summarize_task
    returns None and no result_summary is persisted."""
    runner = FakeRunner(output=None)

    result = await summarize_task(db, runner, task_with_events.id)

    assert result is None
    assert await db.get_task_result_summary(task_with_events.id) is None


@pytest.mark.asyncio
async def test_summarize_task_with_no_events_writes_stub(db):
    """A task killed before producing output still gets a stub summary so the
    auto-ping loop doesn't keep retrying."""
    task = Task(session_id=str(uuid4()), prompt="never ran", state=TaskState.KILLED)
    await db.insert_task(task)
    runner = FakeRunner(output="should not be called")

    result = await summarize_task(db, runner, task.id)

    assert result is not None
    assert "never ran" in result
    assert runner.calls == [], "runner must not be invoked when there are no events"
    stored = await db.get_task_result_summary(task.id)
    assert stored == result


@pytest.mark.asyncio
async def test_summarize_task_returns_none_for_unknown_task(db):
    runner = FakeRunner(output="x")
    result = await summarize_task(db, runner, UUID("00000000-0000-0000-0000-000000000000"))
    assert result is None
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Event formatter — pure function
# ---------------------------------------------------------------------------

def test_format_drops_state_changed_and_cli_events():
    events = [
        {"type": "state.changed", "payload": {"state": "running"}},
        {"type": "assistant.text", "payload": {"text": "hi"}},
        {"type": "cli.api_retry", "payload": {"raw": {}}},
    ]
    out = _format_events_for_prompt("prompt", "completed", events)
    assert "state.changed" not in out
    assert "cli.api_retry" not in out
    assert "assistant said: hi" in out


def test_format_drops_auto_allow_approvals():
    events = [
        {"type": "approval.resolved", "payload": {"auto": True, "decision": "allow"}},
        {"type": "approval.resolved", "payload": {"auto": False, "decision": "allow"}},
    ]
    out = _format_events_for_prompt("p", "completed", events)
    # Auto-allow line should be filtered (noise).
    assert out.count("approval resolved") == 1


def test_format_marks_failure():
    events = [{"type": "result.final", "payload": {"is_error": True}}]
    out = _format_events_for_prompt("p", "failed", events)
    assert "result: ERROR" in out
