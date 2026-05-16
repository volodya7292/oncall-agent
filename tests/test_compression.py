"""Chat history compression via the local-claude one-shot runner.

These tests use a FakeRunner so no `claude` binary is spawned. They cover:
  * No compression below the threshold.
  * Compression triggers above the threshold and inserts one chat_summaries row.
  * Subsequent loads use the summary + only newer messages.
  * Runner failure (returns None) → uncompressed history, no crash.
  * The chosen split point lands on a `user` boundary so tool_call pairs stay intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.operator import Operator


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeRunner:
    """Stand-in for ClaudeCliRunner. Records prompts; returns canned text."""

    def __init__(self, *, output: str | None = "compressed summary") -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def one_shot(
        self, prompt, *, system_prompt=None, model="sonnet", timeout_s=60.0,
    ) -> str | None:
        self.calls.append({
            "prompt": prompt, "system_prompt": system_prompt,
            "model": model, "timeout_s": timeout_s,
        })
        return self.output


class ScriptedLLM:
    """Always emits an empty text turn (no tool calls)."""

    def __init__(self) -> None:
        self.calls_made: list[dict[str, Any]] = []

    async def chat(self, *, model, messages, tools, max_tokens=None):
        self.calls_made.append({"messages": messages})
        return {"role": "assistant", "content": "ok", "tool_calls": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    return Settings(
        oncall_token="t",
        oncall_db_path=tmp_path / "db.sqlite",
        oncall_memory_path=tmp_path / "memory.md",
        oncall_operator_model="openai/test",
        oncall_compression_threshold_tokens=200,  # 800 chars
        oncall_compression_model="sonnet",
        ai_gateway_api_key="x",
    )


@pytest.fixture
async def stack(settings):
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    approval_client = HttpLongPollApprovalClient()
    broker = Broker(db, approval_client, events.publish)
    lifecycle = Lifecycle(
        db=db, broker=broker, approval_client=approval_client,
        events=events, settings=settings, paths=Paths(),
    )
    try:
        yield {"db": db, "settings": settings, "paths": Paths(),
               "broker": broker, "lifecycle": lifecycle}
    finally:
        await db.close()


def _make_operator(stack, runner) -> Operator:
    return Operator(
        db=stack["db"], lifecycle=stack["lifecycle"], broker=stack["broker"],
        settings=stack["settings"], paths=stack["paths"],
        llm=ScriptedLLM(), runner=runner,
    )


async def _populate_history(db: Database, session_id: str, n_user_turns: int, padding: int) -> None:
    """Insert `n_user_turns` (user, assistant) pairs. Each user message is
    `padding` characters of 'A' so we can drive the char-based token estimate
    above the threshold deterministically."""
    await db.ensure_chat_session(session_id)
    for i in range(n_user_turns):
        await db.append_chat_message(session_id, "user", "A" * padding + f" #{i}")
        await db.append_chat_message(session_id, "assistant", f"reply {i}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compression_skipped_under_threshold(stack):
    """Small history (<200 tokens estimated) → no compression call."""
    runner = FakeRunner()
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=3, padding=50)

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is None
    assert len(history) == 6  # 3 user + 3 assistant
    assert runner.calls == [], "runner must not be invoked below threshold"


@pytest.mark.asyncio
async def test_compression_triggers_above_threshold(stack):
    """Big history → one runner call, one chat_summaries row, history shrunk."""
    runner = FakeRunner(output="user wanted X, operator dispatched T1, T1 done.")
    operator = _make_operator(stack, runner)
    # Each user message is 800 chars → ~200 tokens. 10 turns ≈ 2000 tokens,
    # well above the 200-token threshold.
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is not None
    assert summary["summary"] == "user wanted X, operator dispatched T1, T1 done."
    assert summary["through_message_id"] > 0
    assert len(runner.calls) == 1

    # The row was persisted.
    db_summary = await stack["db"].get_latest_chat_summary("s1")
    assert db_summary is not None
    assert db_summary["summary"] == summary["summary"]

    # History returned is strictly newer than the cutoff.
    for row in history:
        assert row["id"] > summary["through_message_id"]


@pytest.mark.asyncio
async def test_compression_uses_configured_sonnet_model(stack):
    """The runner must be called with the compression_model setting, not the
    operator's regular model."""
    runner = FakeRunner(output="summary")
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    await operator._load_and_maybe_compress("s1")

    assert runner.calls[0]["model"] == "sonnet"
    # The system prompt is the compression-specific one, not the operator's.
    assert "summarizing" in (runner.calls[0]["system_prompt"] or "").lower()


@pytest.mark.asyncio
async def test_compression_fail_soft_returns_uncompressed(stack):
    """Runner returning None → no summary row, history returned uncompressed."""
    runner = FakeRunner(output=None)  # simulates claude CLI failure
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is None
    assert len(history) == 20  # all rows retained
    # No DB row inserted.
    assert await stack["db"].get_latest_chat_summary("s1") is None


@pytest.mark.asyncio
async def test_compression_split_lands_on_user_boundary(stack):
    """The summarized older portion must end at a `user` row so the LIVE tail
    starts with a user turn — keeps assistant_tool_calls/tool pairs intact."""
    runner = FakeRunner(output="summary")
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is not None
    # First row in the live tail must be a user turn.
    assert history[0]["role"] == "user", \
        f"split landed on {history[0]['role']}; would tear tool_call pairs"


@pytest.mark.asyncio
async def test_subsequent_load_uses_existing_summary(stack):
    """Second call doesn't re-summarize if total stays under threshold."""
    runner = FakeRunner(output="first summary")
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)
    await operator._load_and_maybe_compress("s1")
    assert len(runner.calls) == 1

    # Add a single short turn — total (summary + new) is still small.
    await stack["db"].append_chat_message("s1", "user", "short follow-up")

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is not None and summary["summary"] == "first summary"
    assert len(runner.calls) == 1, "must not re-summarize when under threshold"
    # History tail contains the new row plus whatever survived the first split.
    assert any(r["content"] == "short follow-up" for r in history)
