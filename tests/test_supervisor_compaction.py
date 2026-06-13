"""Executor context-compaction guard.

The supervisor runs ONE long-lived `claude --resume` session shared across
every hand_off, so its context grows unbounded. After each successful task
it reads the live window fill from the `result` event and, once that crosses
`oncall_executor_compact_at_tokens`, runs a `/compact` pass before the next
task resumes (see Supervisor._maybe_compact_session).

These tests pin:
  * `_context_tokens` — the exact stream-json usage fields that count toward
    window occupancy (input + both cache buckets; NOT output). If the CLI's
    usage shape drifts, this fails loudly.
  * `_parse_compact_result` — detecting a real `/compact` success + the
    post-compaction token count from the `compact_boundary` event.
  * The threshold is a `>=` boundary, fires only when armed, and 0 disables.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from oncall import supervisor as sup
from oncall.supervisor import Supervisor, _context_tokens, _parse_compact_result
from oncall.models import TerminalReason


# Real shapes captured from `claude` 2.1.x stream-json output.
ASSISTANT_USAGE = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 6369,
    "cache_read_input_tokens": 10745,
    "output_tokens": 904,  # must NOT count toward the next turn's window
}

COMPACT_STDOUT = b"\n".join([
    json.dumps({"type": "system", "subtype": "status", "status": "compacting"}).encode(),
    json.dumps({"type": "system", "subtype": "status", "status": None,
                "compact_result": "success"}).encode(),
    json.dumps({"type": "system", "subtype": "compact_boundary",
                "compact_metadata": {"trigger": "manual",
                                     "pre_tokens": 17256, "post_tokens": 865}}).encode(),
    json.dumps({"type": "result", "subtype": "success", "is_error": False}).encode(),
])


def test_context_tokens_sums_input_and_cache_excludes_output():
    # 2 + 6369 + 10745 = 17116; the 904 output tokens are excluded.
    assert _context_tokens(ASSISTANT_USAGE) == 17116


def test_context_tokens_tolerates_missing_and_none_fields():
    assert _context_tokens({}) == 0
    assert _context_tokens({"input_tokens": None, "cache_read_input_tokens": 5}) == 5
    assert _context_tokens("not-a-dict") == 0  # defensive: bad payload → 0


def test_parse_compact_result_reads_success_and_post_tokens():
    ok, after = _parse_compact_result(COMPACT_STDOUT)
    assert ok is True
    assert after == 865


def test_parse_compact_result_negative_when_no_boundary():
    noise = b"\n".join([
        json.dumps({"type": "system", "subtype": "init"}).encode(),
        json.dumps({"type": "result", "subtype": "success"}).encode(),
        b"  ",                       # blank line
        b"{not json",                # garbage line must not crash the scan
    ])
    assert _parse_compact_result(noise) == (False, None)


# ---------------------------------------------------------------------------
# Threshold gating in _maybe_compact_session
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, stdout: bytes):
        self._stdout = stdout
        self.returncode = 0

    async def communicate(self, input=None):
        return self._stdout, b""


def _make_supervisor(threshold: int, last_tokens: int):
    published: list[tuple] = []

    class _Events:
        async def publish(self, *a, **k):
            published.append((a, k))

    s = Supervisor(
        db=None,
        events=_Events(),
        settings=SimpleNamespace(oncall_executor_compact_at_tokens=threshold),
        paths=None,
    )
    s._last_context_tokens = last_tokens
    return s, published


@pytest.mark.parametrize("threshold,last,expect_spawn", [
    (200_000, 199_999, False),   # just under → leave it
    (200_000, 200_000, True),    # exactly at the boundary → compact (>=)
    (200_000, 250_000, True),    # over → compact
    (0,       10_000_000, False),  # 0 disables the guard entirely
])
def test_compaction_fires_on_or_above_threshold(monkeypatch, threshold, last, expect_spawn):
    spawned: list[list[str]] = []

    async def fake_exec(*argv, **kwargs):
        spawned.append(list(argv))
        return _FakeProc(COMPACT_STDOUT)

    monkeypatch.setattr(sup.asyncio, "create_subprocess_exec", fake_exec)

    s, _ = _make_supervisor(threshold, last)
    task = SimpleNamespace(id="task-1", model="sonnet")
    asyncio.run(s._maybe_compact_session(task, "sess-abc"))

    assert bool(spawned) == expect_spawn
    if expect_spawn:
        # The pass must target THIS session with /compact-capable argv.
        argv = spawned[0]
        assert "--resume" in argv and "sess-abc" in argv
        assert s._proc is None  # cleaned up in finally even on the happy path
