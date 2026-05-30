"""Turn-taking helpers in voice_call: TTS chunk splitting and the outbound
priority-queue drain.

These pin two non-obvious properties:
  * _split_for_tts must split on sentence boundaries for pipelined TTS but NOT
    inside decimals/abbreviations, and must merge tiny fragments so we don't
    fire a TTS round-trip per "Ok."
  * _drain_chitchat_items is a safety invariant: a user barge-in drops stale
    chitchat but must NEVER drop a queued approval prompt.
"""

from __future__ import annotations

import asyncio

from oncall.voice_call import (
    _PRI_APPROVAL,
    _PRI_CHAT,
    _drain_chitchat_items,
    _split_for_tts,
)


# ---- _split_for_tts ----

def test_split_empty_is_empty():
    assert _split_for_tts("   ") == []


def test_split_short_reply_is_single_chunk():
    # Below min_chars and no boundary to split on → one chunk, spoken as-is.
    assert _split_for_tts("Ok.") == ["Ok."]


def test_split_breaks_on_sentence_boundary():
    chunks = _split_for_tts(
        "Hello there, this is the assistant. I am calling about the deploy. "
        "It failed twice."
    )
    # First chunk ends at the first sentence that clears min_chars; the short
    # trailing sentence stands on its own.
    assert len(chunks) >= 2
    assert chunks[0].endswith("deploy.")
    assert chunks[-1] == "It failed twice."
    # No content lost or duplicated.
    assert " ".join(chunks).count("failed") == 1


def test_split_does_not_break_decimals_or_abbreviations():
    # "3.5" has no space after the dot, so it must not be a split point.
    chunks = _split_for_tts(
        "The error rate hit 3.5 percent on the api service right now."
    )
    assert chunks == [
        "The error rate hit 3.5 percent on the api service right now."
    ]


def test_split_on_hard_newlines():
    chunks = _split_for_tts(
        "First line here is long enough to stand alone.\n"
        "Second line is also long enough to stand alone."
    )
    assert len(chunks) == 2


def test_split_merges_tiny_fragments():
    # A run of tiny sentences shouldn't become a TTS request each — they merge
    # up to the min_chars threshold.
    chunks = _split_for_tts("Yes. No. Maybe. I think so. Probably not.")
    assert len(chunks) < 5
    assert " ".join(chunks).replace("  ", " ").strip()


# ---- _drain_chitchat_items (safety invariant) ----

def _drain(items: list[tuple[int, int, str]]) -> tuple[int, list[str]]:
    """Run the drain over a fresh queue; return (dropped, surviving texts in
    dequeue order)."""
    async def run():
        q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        for it in items:
            q.put_nowait(it)
        dropped = _drain_chitchat_items(q)
        survivors = []
        while not q.empty():
            survivors.append((await q.get())[2])
        return dropped, survivors

    return asyncio.run(run())


def test_drain_keeps_approval_drops_chitchat():
    dropped, survivors = _drain(
        [
            (_PRI_CHAT, 0, "chit A"),
            (_PRI_APPROVAL, 1, "APPROVAL"),
            (_PRI_CHAT, 2, "chit B"),
        ]
    )
    assert dropped == 2
    assert survivors == ["APPROVAL"]  # safety-critical item survives


def test_drain_empty_queue_is_noop():
    dropped, survivors = _drain([])
    assert dropped == 0
    assert survivors == []


def test_drain_all_chitchat_leaves_nothing():
    dropped, survivors = _drain(
        [(_PRI_CHAT, 0, "a"), (_PRI_CHAT, 1, "b")]
    )
    assert dropped == 2
    assert survivors == []


def test_drain_preserves_multiple_approvals_in_order():
    dropped, survivors = _drain(
        [
            (_PRI_APPROVAL, 0, "approval one"),
            (_PRI_CHAT, 1, "chit"),
            (_PRI_APPROVAL, 2, "approval two"),
        ]
    )
    assert dropped == 1
    assert survivors == ["approval one", "approval two"]
