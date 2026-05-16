"""Unit tests for OperatorMemory.

Covers: add/remove/dedup/over-cap/empty-input/empty-file/for_prompt rendering.
File operations against `tmp_path` so they're reproducible everywhere.
"""

from __future__ import annotations

import pytest

from oncall.operator_memory import MAX_ENTRIES, MAX_ENTRY_CHARS, OperatorMemory


@pytest.fixture
def mem(tmp_path):
    return OperatorMemory(tmp_path / "memory.md")


def test_remember_creates_file_and_appends(mem):
    r = mem.remember("user prefers terse replies", today="2026-05-16")
    assert r == {"added": True, "entries": 1}
    assert mem.path.exists()
    entries = mem.entries()
    assert entries == [("2026-05-16", "user prefers terse replies")]


def test_remember_rejects_empty(mem):
    assert mem.remember("") == {"added": False, "reason": "empty"}
    assert mem.remember("   \n  ") == {"added": False, "reason": "empty"}


def test_remember_rejects_duplicates(mem):
    mem.remember("alex is a coworker", today="2026-05-16")
    r = mem.remember("alex is a coworker", today="2026-05-17")
    assert r == {"added": False, "reason": "duplicate"}
    assert len(mem.entries()) == 1


def test_remember_rejects_too_long(mem):
    big = "x" * (MAX_ENTRY_CHARS + 1)
    r = mem.remember(big)
    assert r["added"] is False
    assert "too_long" in r["reason"]


def test_remember_respects_max_entries(mem):
    for i in range(MAX_ENTRIES):
        mem.remember(f"fact {i}", today="2026-01-01")
    r = mem.remember("one too many", today="2026-01-01")
    assert r["added"] is False
    assert "full" in r["reason"]


def test_forget_removes_matches(mem):
    mem.remember("alex is a coworker", today="2026-05-16")
    mem.remember("boss is named carol", today="2026-05-16")
    mem.remember("alex prefers russian", today="2026-05-16")
    r = mem.forget("alex")
    assert r == {"removed": 2, "entries": 1}
    assert mem.entries() == [("2026-05-16", "boss is named carol")]


def test_forget_case_insensitive(mem):
    mem.remember("ALEX prefers russian", today="2026-05-16")
    r = mem.forget("alex")
    assert r["removed"] == 1


def test_forget_empty_input_is_noop(mem):
    mem.remember("a", today="2026-05-16")
    assert mem.forget("") == {"removed": 0, "reason": "empty"}
    assert len(mem.entries()) == 1


def test_for_prompt_when_empty(mem):
    assert mem.for_prompt() == "(no entries yet)"


def test_for_prompt_shows_entries(mem):
    mem.remember("user is asleep 11pm-7am", today="2026-05-16")
    rendered = mem.for_prompt()
    assert "user is asleep 11pm-7am" in rendered
    assert "2026-05-16" in rendered


def test_newlines_in_input_are_flattened(mem):
    mem.remember("first line\nsecond line", today="2026-05-16")
    # The on-disk entry must stay single-line so the markdown bullet parser
    # remains correct on next read.
    assert mem.entries() == [("2026-05-16", "first line second line")]


def test_round_trip_via_disk(tmp_path):
    """A fresh OperatorMemory instance reads entries written by a prior one."""
    a = OperatorMemory(tmp_path / "memory.md")
    a.remember("durable across instances", today="2026-05-16")
    b = OperatorMemory(tmp_path / "memory.md")
    assert b.entries() == [("2026-05-16", "durable across instances")]
