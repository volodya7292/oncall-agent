"""chat_summary.summarize_chat — fetches chat history through a tg-like
object, formats a transcript, calls a one-shot runner. Tests use stubs for
both — device-independent, no real telethon / claude binary."""

from __future__ import annotations

from typing import Any

import pytest

from oncall.chat_summary import summarize_chat, _format_transcript


class FakeTelegram:
    def __init__(self, *, history: list[dict[str, Any]] | None = None) -> None:
        self._history = history or []
        self.calls: list[dict[str, Any]] = []

    async def get_chat_history(self, chat_id, *, limit=10):
        self.calls.append({"chat_id": chat_id, "limit": limit})
        return list(self._history)


class FakeRunner:
    def __init__(self, *, output: str | None) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def one_shot(self, prompt, *, system_prompt=None, model="sonnet", timeout_s=60.0):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "model": model})
        return self.output


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_chat_returns_runner_output_with_transcript_prompt():
    tg = FakeTelegram(history=[
        # iter_messages returns newest-first; chat_summary reverses to
        # chronological for the prompt.
        {"text": "ok let's do it monday", "date": "2026-05-15T11:00:00+00:00",
         "outgoing": True, "sender_display_name": None, "sender_username": None},
        {"text": "yeah Monday works", "date": "2026-05-15T10:30:00+00:00",
         "outgoing": False, "sender_display_name": "Alex", "sender_username": "alex_s"},
        {"text": "let's plan the redis migration", "date": "2026-05-15T10:00:00+00:00",
         "outgoing": True, "sender_display_name": None, "sender_username": None},
    ])
    runner = FakeRunner(output="They agreed to start the redis migration on Monday.")

    result = await summarize_chat(tg, runner, "77")

    assert result == "They agreed to start the redis migration on Monday."
    assert tg.calls == [{"chat_id": "77", "limit": 200}]
    assert len(runner.calls) == 1
    call = runner.calls[0]
    # System prompt is the chat-summary one.
    assert "summarizing" in (call["system_prompt"] or "").lower()
    # Transcript is chronological (oldest first) in the user prompt.
    prompt = call["prompt"]
    assert prompt.index("redis migration") < prompt.index("Monday works") < prompt.index("monday")
    # Default tail is "Summarize." (no focus given).
    assert prompt.rstrip().endswith("Summarize.")


@pytest.mark.asyncio
async def test_summarize_chat_focus_appended_to_prompt():
    tg = FakeTelegram(history=[
        {"text": "ping", "date": "2026-05-15T10:00:00+00:00",
         "outgoing": False, "sender_display_name": "X", "sender_username": None},
    ])
    runner = FakeRunner(output="ok")

    await summarize_chat(tg, runner, "77", focus="redis migration")

    prompt = runner.calls[0]["prompt"]
    assert prompt.rstrip().endswith("Focus: redis migration")


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_chat_returns_none_when_history_empty():
    tg = FakeTelegram(history=[])
    runner = FakeRunner(output="should not be called")

    result = await summarize_chat(tg, runner, "77")

    assert result is None
    assert runner.calls == []   # runner never invoked


@pytest.mark.asyncio
async def test_summarize_chat_returns_none_when_runner_fails():
    tg = FakeTelegram(history=[
        {"text": "hi", "date": "2026-05-15T10:00:00+00:00",
         "outgoing": False, "sender_display_name": "X", "sender_username": None},
    ])
    runner = FakeRunner(output=None)   # local claude unavailable

    result = await summarize_chat(tg, runner, "77")

    assert result is None
    assert len(runner.calls) == 1


# ---------------------------------------------------------------------------
# transcript formatting
# ---------------------------------------------------------------------------

def test_format_transcript_uses_me_label_for_outgoing():
    rows = [
        {"text": "hi", "date": "2026-05-15T10:00:00+00:00",
         "outgoing": True, "sender_display_name": None, "sender_username": None},
    ]
    out = _format_transcript(rows)
    assert "[2026-05-15 10:00] me: hi" == out


def test_format_transcript_falls_back_to_username_then_them():
    rows = [
        {"text": "a", "date": "2026-05-15T10:00:00+00:00", "outgoing": False,
         "sender_display_name": None, "sender_username": "alex"},
        {"text": "b", "date": "2026-05-15T10:01:00+00:00", "outgoing": False,
         "sender_display_name": None, "sender_username": None},
    ]
    out = _format_transcript(rows)
    # chronological order is reverse of input (oldest first); inputs are
    # already in newest-first order would reverse, but here we passed them
    # in source-of-history order. Let's verify content membership rather
    # than order.
    assert "@alex: a" in out
    assert "them: b" in out


def test_format_transcript_skips_empty_text():
    rows = [
        {"text": "hi", "date": "2026-05-15T10:00:00+00:00", "outgoing": True,
         "sender_display_name": None, "sender_username": None},
        {"text": "", "date": "2026-05-15T10:01:00+00:00", "outgoing": True,
         "sender_display_name": None, "sender_username": None},
    ]
    out = _format_transcript(rows)
    assert out.count("\n") == 0      # exactly one non-empty line
    assert "hi" in out
