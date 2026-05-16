"""REPL helpers — pure unit tests (no daemon, no terminal, no network).

Covers:
  * format_event renders each known type as a one-liner; unknown types → None.
  * parse_slash recognizes /new /session /help /quit + EOF-style /exit /q.
  * read_session / write_session round-trip; mode 0600 on POSIX.
  * parse_sse_lines handles data: groups, comments, and bad JSON gracefully.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncIterator

import pytest

from oncall.repl import (
    SlashCommand,
    format_event,
    parse_slash,
    parse_sse_lines,
    read_session,
    write_session,
)


# ---------------------------------------------------------------------------
# format_event
# ---------------------------------------------------------------------------

def test_format_approval_requested():
    line = format_event({
        "task_id": "11112222-3333-4444-5555-666677778888",
        "type": "approval.requested",
        "payload": {
            "approval_id": "aaaabbbb-1111-2222-3333-444455556666",
            "canonical_command": "bash 'echo hi >> /tmp/x'",
            "challenge_phrase": "amber paper compass",
        },
    })
    assert line is not None
    assert "approval aaaabbbb" in line
    assert "task 11112222" in line
    assert "bash 'echo hi >> /tmp/x'" in line
    assert 'say "amber paper compass"' in line


def test_format_result_final_success():
    line = format_event({
        "task_id": "abcdefab-0000-0000-0000-000000000000",
        "type": "result.final",
        "payload": {"is_error": False, "duration_ms": 1200},
    })
    assert line == "* task abcdefab done"


def test_format_result_final_failure():
    line = format_event({
        "task_id": "abcdefab-0000-0000-0000-000000000000",
        "type": "result.final",
        "payload": {"is_error": True},
    })
    assert line == "* task abcdefab failed"


def test_format_approval_resolved_auto_allow_silenced():
    """Auto-allowed readonly tools (ls, cat, grep, ...) flood the bus.
    The REPL must NOT print one line per readonly call."""
    line = format_event({
        "task_id": "abcdefab-0000-0000-0000-000000000000",
        "type": "approval.resolved",
        "payload": {"auto": True, "decision": "allow", "tool_name": "Bash"},
    })
    assert line is None


def test_format_approval_resolved_user_decision_shown():
    """User-resolved approvals (auto=False) are user-visible — they confirm
    the user's response was accepted."""
    line = format_event({
        "task_id": "abcdefab-0000-0000-0000-000000000000",
        "type": "approval.resolved",
        "payload": {"auto": False, "decision": "allow"},
    })
    assert line == "~ approval task abcdefab: allow"


def test_format_approval_resolved_auto_deny_shown():
    """Catastrophic auto-denies are rare and security-relevant — surface them."""
    line = format_event({
        "task_id": "abcdefab-0000-0000-0000-000000000000",
        "type": "approval.resolved",
        "payload": {"auto": True, "decision": "deny", "reason": "catastrophic"},
    })
    assert line == "~ approval task abcdefab: deny"


def test_format_chat_reply_for_current_session():
    line = format_event(
        {
            "task_id": None,
            "type": "chat.reply",
            "payload": {"session_id": "s1", "text": "5 projects: alpha, bravo, charlie."},
        },
        session_id="s1",
    )
    assert line == "5 projects: alpha, bravo, charlie."


def test_format_chat_reply_filtered_out_for_other_session():
    line = format_event(
        {
            "task_id": None,
            "type": "chat.reply",
            "payload": {"session_id": "OTHER", "text": "not for me"},
        },
        session_id="s1",
    )
    assert line is None


def test_format_messenger_received_truncates_long_body():
    body = "x" * 200
    line = format_event({
        "task_id": None,
        "type": "messenger.received",
        "payload": {"sender_username": "alex", "body": body},
    })
    assert line is not None
    assert line.startswith("# DM from @alex: ")
    # 120-char window, last 3 chars become "..."
    assert line.endswith("...")
    assert len(line.split(": ", 1)[1]) == 120


def test_format_state_changed_silenced_unless_debug():
    env = {"task_id": "deadbeef-0000-0000-0000-000000000000",
           "type": "state.changed", "payload": {"state": "running"}}
    assert format_event(env, debug=False) is None
    line = format_event(env, debug=True)
    assert line is not None and "running" in line


def test_format_unknown_type_returns_none():
    assert format_event({"task_id": "x", "type": "made.up", "payload": {}}) is None


# ---------------------------------------------------------------------------
# parse_slash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("/quit",          SlashCommand(name="quit",    arg=None)),
    ("/exit",          SlashCommand(name="quit",    arg=None)),
    ("/q",             SlashCommand(name="quit",    arg=None)),
    ("/new",           SlashCommand(name="new",     arg=None)),
    ("/session abc",   SlashCommand(name="session", arg="abc")),
    ("/session",       SlashCommand(name="session", arg=None)),
    ("/help",          SlashCommand(name="help",    arg=None)),
    ("  /HELP  ",      SlashCommand(name="help",    arg=None)),
])
def test_parse_slash_known(line, expected):
    assert parse_slash(line) == expected


def test_parse_slash_unknown_routed_to_help_with_typo_arg():
    cmd = parse_slash("/banana")
    assert cmd is not None
    assert cmd.name == "help"
    assert cmd.arg == "banana"


@pytest.mark.parametrize("line", ["hello", "  ", "", "list tasks", "say /done please"])
def test_parse_slash_returns_none_for_plain_text(line):
    assert parse_slash(line) is None


# ---------------------------------------------------------------------------
# session file
# ---------------------------------------------------------------------------

def test_session_round_trip(tmp_path):
    p = tmp_path / "last_session"
    assert read_session(p) is None
    write_session("abc-123", p)
    assert read_session(p) == "abc-123"


def test_session_file_mode_is_0600(tmp_path):
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits only")
    p = tmp_path / "last_session"
    write_session("xyz", p)
    mode = os.stat(p).st_mode & 0o777
    assert mode == 0o600


def test_session_empty_file_treated_as_absent(tmp_path):
    p = tmp_path / "last_session"
    p.write_text("", encoding="utf-8")
    assert read_session(p) is None


# ---------------------------------------------------------------------------
# parse_sse_lines
# ---------------------------------------------------------------------------

async def _aiter(items):
    for x in items:
        yield x


@pytest.mark.asyncio
async def test_parse_sse_one_event():
    lines = ['data: {"type":"x","payload":{"k":1}}', "", "data: ignored-no-blank"]
    out = []
    async for env in parse_sse_lines(_aiter(lines)):
        out.append(env)
    # Only one event has its terminating blank.
    assert out == [{"type": "x", "payload": {"k": 1}}]


@pytest.mark.asyncio
async def test_parse_sse_ignores_comments_and_other_fields():
    lines = [
        ": ping", "",
        "event: foo", 'data: {"a":1}', "id: 7", "",
        ": another ping", "",
        'data: {"b":2}', "",
    ]
    out = []
    async for env in parse_sse_lines(_aiter(lines)):
        out.append(env)
    assert out == [{"a": 1}, {"b": 2}]


@pytest.mark.asyncio
async def test_parse_sse_drops_bad_json_silently():
    lines = ["data: this is not json", "", 'data: {"ok":true}', ""]
    out = []
    async for env in parse_sse_lines(_aiter(lines)):
        out.append(env)
    assert out == [{"ok": True}]


@pytest.mark.asyncio
async def test_parse_sse_multiline_data_joined_with_newlines():
    """SSE spec: multiple data: lines in one event are joined with \\n. We
    only care about the JSON parse succeeding when the joined payload is valid."""
    lines = ["data: {", 'data: "k":"v"', "data: }", ""]
    out = []
    async for env in parse_sse_lines(_aiter(lines)):
        out.append(env)
    assert out == [{"k": "v"}]
