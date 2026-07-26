"""Chat history compression via the local-claude one-shot runner.

These tests use a FakeRunner so no `claude` binary is spawned. They cover:
  * No compression below the threshold.
  * Compression triggers above the threshold and inserts one chat_summaries row.
  * Subsequent loads use the summary + only newer messages.
  * Runner failure (returns None) → uncompressed history, no crash.
  * The chosen split point lands on a `user` boundary so tool_call pairs stay intact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker

from tests.support import stub_classifier
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.operator import Operator


# A realistic-length summary. The compression sanity guard (Operator.
# _summarize_older) rejects a summary that is implausibly short relative to its
# input, so fakes that stand in for a *successful* compression must return
# something proportionate, not a 2-word stub.
_LONG_SUMMARY = (
    "The user asked the operator to plan and track the redis migration and a "
    "few follow-up chores. The operator dispatched task T1 to inventory the "
    "current cluster and T2 to draft the cutover runbook, and confirmed the "
    "user prefers changes shipped on weekdays only. They also chatted about "
    "the user's trip to Hamburg and a gelato place worth revisiting. Open "
    "threads: the operator still owes the user the T2 runbook and a decision "
    "on the maintenance window."
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeRunner:
    """Stand-in for ClaudeCliRunner. Records prompts; returns canned text."""

    def __init__(self, *, output: str | None = _LONG_SUMMARY) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def one_shot(
        self, prompt, *, system_prompt=None, model="sonnet", effort=None,
        timeout_s=60.0,
    ) -> str | None:
        self.calls.append({
            "prompt": prompt, "system_prompt": system_prompt,
            "model": model, "effort": effort, "timeout_s": timeout_s,
        })
        return self.output


class ScriptedLLM:
    """Always emits an empty text turn (no tool calls)."""

    def __init__(self) -> None:
        self.calls_made: list[dict[str, Any]] = []

    async def chat(self, *, model, messages, tools, max_tokens=None, reasoning_effort=None):
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
async def stack(settings, tmp_path, monkeypatch):
    # Redirect the executor-session marker files (normally under ~/.oncall)
    # into tmp_path so clear_session()'s executor reset can't touch — or
    # delete — the developer's real session during a test run.
    from oncall import config as _config
    monkeypatch.setattr(_config, "EXECUTOR_SESSION_ID_FILE", tmp_path / "executor_session_id")
    monkeypatch.setattr(
        _config, "EXECUTOR_SESSION_INITIALIZED_FILE",
        tmp_path / "executor_session_initialized",
    )
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    approval_client = HttpLongPollApprovalClient()
    broker = Broker(db, approval_client, events.publish, classifier=stub_classifier())
    lifecycle = Lifecycle(
        db=db, broker=broker, approval_client=approval_client,
        events=events, settings=settings, paths=Paths(),
    )
    try:
        yield {"db": db, "settings": settings, "paths": Paths(),
               "broker": broker, "lifecycle": lifecycle}
    finally:
        await db.close()


class _NullMemory:
    """Tiny stand-in for tests that don't exercise memory."""
    async def store(self, facts, *, source_turn=None): return []
    async def retrieve(self, query, *, limit=None, exclude_ids=None): return []
    async def for_prompt(self, query): return "(no relevant entries this turn)"
    async def entries_count(self): return 0


def _make_operator(stack, runner) -> Operator:
    return Operator(
        db=stack["db"], lifecycle=stack["lifecycle"], broker=stack["broker"],
        settings=stack["settings"], paths=stack["paths"],
        llm=ScriptedLLM(), runner=runner,
        memory=_NullMemory(),
    )


async def _populate_history(db: Database, session_id: str, n_user_turns: int, padding: int) -> None:
    """Insert `n_user_turns` (user, assistant) pairs. Each user message is
    `padding` characters of 'A' so we can drive the char-based token estimate
    above the threshold deterministically."""
    await db.ensure_chat_session(session_id)
    for i in range(n_user_turns):
        await db.append_chat_message(session_id, "user", "A" * padding + f" #{i}")
        await db.append_chat_message(session_id, "assistant", f"reply {i}")


async def _load_settled(operator, session_id):
    """Load, then drain background compression, then re-load.

    Compression is fire-and-forget (Operator._schedule_compression): the turn
    that trips the threshold replies from the uncompressed history and the
    summary lands afterwards. This mirrors what the NEXT turn sees, which is
    the state most of these tests care about.
    """
    await operator._load_and_maybe_compress(session_id)
    while operator._compression_tasks:
        await asyncio.gather(*list(operator._compression_tasks))
    return await operator._load_and_maybe_compress(session_id)


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
    runner = FakeRunner(output=_LONG_SUMMARY)
    operator = _make_operator(stack, runner)
    # Each user message is 800 chars → ~200 tokens. 10 turns ≈ 2000 tokens,
    # well above the 200-token threshold.
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await _load_settled(operator, "s1")

    assert summary is not None
    assert summary["summary"] == _LONG_SUMMARY
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
async def test_compression_does_not_block_the_turn(stack):
    """The turn that trips the threshold must NOT wait on the summarizer.

    Compression is an opus one-shot (tens of seconds); running it inline made
    one turn in every N stall before the operator could say anything. The load
    now returns the uncompressed history immediately and the summary lands
    behind it.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingRunner(FakeRunner):
        async def one_shot(self, prompt, **kw):
            started.set()
            await release.wait()          # summarizer "takes forever"
            return await super().one_shot(prompt, **kw)

    runner = BlockingRunner(output=_LONG_SUMMARY)
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    # Returns while the summarizer is still blocked => it did not await it.
    summary, history = await asyncio.wait_for(
        operator._load_and_maybe_compress("s1"), timeout=1.0,
    )
    assert summary is None, "must reply from the pre-compression checkpoint"
    assert len(history) == 20, "uncompressed history is used for this turn"

    await asyncio.wait_for(started.wait(), timeout=1.0)
    release.set()
    await asyncio.gather(*list(operator._compression_tasks))

    # ...and the summary is there for the NEXT turn.
    assert await stack["db"].get_latest_chat_summary("s1") is not None


@pytest.mark.asyncio
async def test_compression_not_scheduled_twice_for_one_session(stack):
    """Over-budget stays true until the summary lands, so back-to-back turns
    would each queue their own opus one-shot against the same rows without the
    in-flight guard."""
    release = asyncio.Event()

    class BlockingRunner(FakeRunner):
        async def one_shot(self, prompt, **kw):
            await release.wait()
            return await super().one_shot(prompt, **kw)

    runner = BlockingRunner(output=_LONG_SUMMARY)
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    for _ in range(3):                      # three turns, none compressed yet
        await operator._load_and_maybe_compress("s1")
    release.set()
    await asyncio.gather(*list(operator._compression_tasks))

    assert len(runner.calls) == 1, f"expected 1 compression, got {len(runner.calls)}"


@pytest.mark.asyncio
async def test_compression_uses_configured_sonnet_model(stack):
    """The runner must be called with the compression_model setting, not the
    operator's regular model."""
    runner = FakeRunner(output="summary")
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    await _load_settled(operator, "s1")

    assert runner.calls[0]["model"] == "sonnet"
    # The system prompt is the compression-specific one, not the operator's.
    assert "summarizing" in (runner.calls[0]["system_prompt"] or "").lower()


@pytest.mark.asyncio
async def test_compression_fail_soft_returns_uncompressed(stack):
    """Runner returning None → no summary row, history returned uncompressed."""
    runner = FakeRunner(output=None)  # simulates claude CLI failure
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await _load_settled(operator, "s1")

    assert summary is None
    assert len(history) == 20  # all rows retained
    # No DB row inserted.
    assert await stack["db"].get_latest_chat_summary("s1") is None


@pytest.mark.asyncio
async def test_compression_split_lands_on_user_boundary(stack):
    """The summarized older portion must end at a `user` row so the LIVE tail
    starts with a user turn — keeps assistant_tool_calls/tool pairs intact."""
    runner = FakeRunner(output=_LONG_SUMMARY)
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await _load_settled(operator, "s1")

    assert summary is not None
    # First row in the live tail must be a user turn.
    assert history[0]["role"] == "user", \
        f"split landed on {history[0]['role']}; would tear tool_call pairs"


@pytest.mark.asyncio
async def test_subsequent_load_uses_existing_summary(stack):
    """Second call doesn't re-summarize if total stays under threshold."""
    runner = FakeRunner(output=_LONG_SUMMARY)
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)
    await _load_settled(operator, "s1")
    assert len(runner.calls) == 1

    # Add a single short turn — total (summary + new) is still small.
    await stack["db"].append_chat_message("s1", "user", "short follow-up")

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is not None and summary["summary"] == _LONG_SUMMARY
    assert len(runner.calls) == 1, "must not re-summarize when under threshold"
    # History tail contains the new row plus whatever survived the first split.
    assert any(r["content"] == "short follow-up" for r in history)


# ---------------------------------------------------------------------------
# /clear (Operator.clear_session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_session_wipes_messages_and_summaries(stack):
    """/clear must remove every chat_messages and chat_summaries row for the
    session — but ONLY for that session; sibling sessions are untouched."""
    runner = FakeRunner()
    operator = _make_operator(stack, runner)
    await _populate_history(stack["db"], "s1", n_user_turns=4, padding=50)
    await _populate_history(stack["db"], "s2", n_user_turns=2, padding=50)
    # Seed a summary checkpoint on s1 by faking a big history and compressing.
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)
    await _load_settled(operator, "s1")
    s1_summary_before = await stack["db"].get_latest_chat_summary("s1")
    assert s1_summary_before is not None

    out = await operator.clear_session("s1")

    assert out["messages_deleted"] > 0
    assert out["summaries_deleted"] == 1
    # s1 fully wiped.
    s1_after = await stack["db"].load_chat_history("s1")
    assert s1_after == []
    assert await stack["db"].get_latest_chat_summary("s1") is None
    # s2 untouched.
    s2_after = await stack["db"].load_chat_history("s2")
    assert len(s2_after) == 4  # 2 user + 2 assistant


@pytest.mark.asyncio
async def test_clear_session_on_empty_session_is_noop(stack):
    runner = FakeRunner()
    operator = _make_operator(stack, runner)
    out = await operator.clear_session("never-existed")
    assert out["messages_deleted"] == 0
    assert out["summaries_deleted"] == 0


@pytest.mark.asyncio
async def test_clear_session_resets_executor_when_idle(stack):
    """/clear must also forget the shared executor session when no task is in
    flight — the session-id file is deleted so the next spawn starts fresh."""
    from oncall import config
    # Simulate a previously-initialized executor session on disk.
    config.get_global_executor_session_id()
    config.mark_executor_session_initialized()
    assert config.EXECUTOR_SESSION_ID_FILE.exists()

    operator = _make_operator(stack, FakeRunner())
    out = await operator.clear_session("s1")

    assert out["executor_session_reset"] is True
    assert not config.EXECUTOR_SESSION_ID_FILE.exists()
    assert not config.EXECUTOR_SESSION_INITIALIZED_FILE.exists()


@pytest.mark.asyncio
async def test_clear_session_keeps_executor_when_busy(stack):
    """A task in flight must NOT have its session yanked out from under it —
    the reset is refused with reason 'busy' and the files stay put."""
    from uuid import uuid4
    from oncall import config
    config.get_global_executor_session_id()
    config.mark_executor_session_initialized()
    # Mark the lifecycle worker as mid-task.
    stack["lifecycle"]._current_task_id = uuid4()

    operator = _make_operator(stack, FakeRunner())
    out = await operator.clear_session("s1")

    assert out["executor_session_reset"] is False
    assert out["executor_reset_reason"] == "busy"
    assert config.EXECUTOR_SESSION_ID_FILE.exists()


# ---------------------------------------------------------------------------
# /compress (Operator.compress_now)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compress_now_force_compresses_small_history(stack):
    """compress_now bypasses the auto-threshold; it should compress a small
    history that would normally be skipped by _load_and_maybe_compress."""
    runner = FakeRunner(output="forced summary")
    operator = _make_operator(stack, runner)
    # Tiny history — well under the auto-threshold.
    await _populate_history(stack["db"], "s1", n_user_turns=3, padding=20)

    out = await operator.compress_now("s1")

    assert out["compressed"] is True
    assert out["older_rows"] > 0
    # Exactly one runner call (the one we just triggered).
    assert len(runner.calls) == 1
    # A chat_summaries row was persisted with the runner's output.
    sm = await stack["db"].get_latest_chat_summary("s1")
    assert sm is not None and sm["summary"] == "forced summary"


@pytest.mark.asyncio
async def test_compress_now_split_preserves_last_user_turn(stack):
    """The split point must be the most-recent user-role row, so the active
    exchange stays live and only older messages get folded into the summary."""
    db = stack["db"]
    runner = FakeRunner(output="ok")
    operator = _make_operator(stack, runner)
    await db.ensure_chat_session("s1")
    await db.append_chat_message("s1", "user", "first")
    await db.append_chat_message("s1", "assistant", "reply 1")
    await db.append_chat_message("s1", "user", "second")
    await db.append_chat_message("s1", "assistant", "reply 2")
    await db.append_chat_message("s1", "user", "third (most recent)")

    out = await operator.compress_now("s1")
    assert out["compressed"] is True
    # The live tail (rows newer than through_message_id) should start with the
    # most-recent user row and contain just that row.
    summary = await db.get_latest_chat_summary("s1")
    assert summary is not None
    tail = await db.load_chat_history("s1", since_id=summary["through_message_id"])
    assert [r["content"] for r in tail] == ["third (most recent)"]


@pytest.mark.asyncio
async def test_compress_now_returns_reason_when_no_split_possible(stack):
    """Single user message → no older user turn to anchor a split → returns
    `{compressed: False, reason: ...}` instead of fabricating a summary."""
    db = stack["db"]
    runner = FakeRunner()
    operator = _make_operator(stack, runner)
    await db.ensure_chat_session("s1")
    await db.append_chat_message("s1", "user", "only message")
    await db.append_chat_message("s1", "assistant", "reply")

    out = await operator.compress_now("s1")

    assert out["compressed"] is False
    assert "split" in out["reason"] or "anchor" in out["reason"]
    assert runner.calls == [], "runner must not be invoked when no split is possible"
    assert await db.get_latest_chat_summary("s1") is None


# ---------------------------------------------------------------------------
# /context (Operator.export_context)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_context_includes_summary_and_live_messages(stack):
    """export_context must render the latest compression summary AND the
    live message tail. Memory entries are not exported (they're query-
    scoped, not session-scoped) but the count appears in the header."""
    db = stack["db"]
    runner = FakeRunner(output=_LONG_SUMMARY)
    operator = _make_operator(stack, runner)
    # Big history so a summary checkpoint gets persisted.
    await _populate_history(db, "s1", n_user_turns=10, padding=800)
    await operator._load_and_maybe_compress("s1")
    # A couple of fresh messages after the checkpoint.
    await db.append_chat_message("s1", "user", "tail user msg")
    await db.append_chat_message("s1", "assistant", "tail assistant msg")

    dump = await operator.export_context("s1")

    assert "# Operator context — session s1" in dump
    assert "## Compression summary" in dump
    assert _LONG_SUMMARY in dump
    assert "## Live history" in dump
    assert "tail user msg" in dump
    assert "tail assistant msg" in dump
    # Memory entries header shows the count even though no memory body
    # is exported.
    assert "Memory entries" in dump


@pytest.mark.asyncio
async def test_export_context_on_empty_session(stack):
    """Empty session → still produces a valid document, just with an
    (empty) live-history marker and no compression section."""
    runner = FakeRunner()
    operator = _make_operator(stack, runner)
    await stack["db"].ensure_chat_session("s_empty")

    dump = await operator.export_context("s_empty")

    assert "# Operator context — session s_empty" in dump
    assert "## Compression summary" not in dump
    assert "## Live history (0 messages)" in dump
    assert "_(empty)_" in dump


@pytest.mark.asyncio
async def test_compress_now_handles_runner_failure(stack):
    """Runner returns None → compress_now reports failure without persisting
    a bogus summary."""
    db = stack["db"]
    runner = FakeRunner(output=None)  # runner says "can't"
    operator = _make_operator(stack, runner)
    await _populate_history(db, "s1", n_user_turns=3, padding=20)

    out = await operator.compress_now("s1")

    assert out["compressed"] is False
    assert "empty" in out["reason"] or "implausible" in out["reason"]
    assert await db.get_latest_chat_summary("s1") is None


@pytest.mark.asyncio
async def test_compression_rejects_implausibly_short_summary(stack):
    """Regression: the model role-plays / refuses instead of summarizing and
    returns a near-empty string (once observed: a 7-token echo of the
    assistant's last line for a 471-message history, which got persisted and
    gutted the session context). The sanity guard must reject such a result —
    no summary row, history returned uncompressed — so the prior context
    survives instead of being destroyed."""
    runner = FakeRunner(output="На зв'язку!")  # a conversational echo, not a summary
    operator = _make_operator(stack, runner)
    # Large input → a handful of tokens back is implausibly short (< 3%).
    await _populate_history(stack["db"], "s1", n_user_turns=10, padding=800)

    summary, history = await operator._load_and_maybe_compress("s1")

    assert summary is None, "implausibly short summary must be rejected"
    assert len(history) == 20, "all rows retained; context not destroyed"
    assert await stack["db"].get_latest_chat_summary("s1") is None
    # The runner WAS called — rejection happens after, on the result.
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_compression_reopens_memory_injection_for_folded_notes(stack):
    """Regression: a `[memory note: ...]` that compression folds away must
    stop counting as "already shown", or the fact becomes permanently
    unreachable.

    `session_memory_shown` suppresses re-injecting a memory whose verbatim
    text is still in the window. It used to be permanent, so once a note was
    summarized into prose that didn't carry the fact, the memory was neither
    in context nor re-injectable — retrieval scored it every turn and threw
    it away. On the live store that had swallowed 92 of 94 memories in one
    long-running session.

    Only the folded rows may be forgotten: a note still sitting in the live
    tail is in context, and re-injecting it would duplicate text.
    """
    db = stack["db"]
    await db.ensure_chat_session("s1")

    # Two eras, each written the way _run_turn writes one: record the ids
    # first, then append the note row they describe.
    await db.record_memory_shown("s1", [11])
    await db.append_chat_message("s1", "user", "[memory note: - [id=11] old fact]")
    await db.append_chat_message("s1", "user", "an old turn")
    await asyncio.sleep(0.01)
    await db.record_memory_shown("s1", [22])
    await db.append_chat_message("s1", "user", "[memory note: - [id=22] recent fact]")
    await db.append_chat_message("s1", "user", "a recent turn")

    rows = await db.load_chat_history("s1")
    through = rows[1]["id"]  # fold the first era only

    await db.insert_chat_summary(
        session_id="s1", summary=_LONG_SUMMARY,
        through_message_id=through, estimated_token_count=100,
    )

    assert await db.get_shown_memory_ids("s1") == {22}
