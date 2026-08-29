"""Turn-taking helpers in voice_call: TTS chunk splitting and the outbound
priority-queue drain.

These pin two non-obvious properties:
  * _split_for_tts must split on sentence boundaries for pipelined TTS but NOT
    inside decimals/abbreviations, and must merge tiny fragments so we don't
    fire a TTS round-trip per "Ok."
  * _drain_chitchat_items is a safety invariant: a user barge-in drops stale
    chitchat but must NEVER drop a queued approval prompt.
  * _mirror_voice_note re-uses the spoken bytes and stays detached from the
    call: an unplayable container or a dead upload must cost nothing on the
    line.
"""

from __future__ import annotations

import asyncio

from oncall.voice_call import (
    _PRI_APPROVAL,
    _PRI_CHAT,
    _PRI_HANDOFF,
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


def test_drain_keeps_handoff_result_drops_chitchat():
    # A hand_off result is the answer to something the user asked for; a user
    # barge-in over stale chitchat must NOT throw it away.
    dropped, survivors = _drain(
        [
            (_PRI_CHAT, 0, "chit A"),
            (_PRI_HANDOFF, 1, "HANDOFF RESULT"),
            (_PRI_CHAT, 2, "chit B"),
        ]
    )
    assert dropped == 2
    assert survivors == ["HANDOFF RESULT"]


def test_handoff_result_sorts_ahead_of_chitchat_behind_approval():
    # Queue ordering: approval first (safety), then hand_off result (the
    # answer), then ordinary chitchat — regardless of enqueue order.
    async def run():
        q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        q.put_nowait((_PRI_CHAT, 0, "chit"))
        q.put_nowait((_PRI_HANDOFF, 1, "result"))
        q.put_nowait((_PRI_APPROVAL, 2, "approval"))
        return [(await q.get())[2] for _ in range(3)]

    assert asyncio.run(run()) == ["approval", "result", "chit"]


# ---- voice-note mirror ----

class _FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[dict] = []
        self._fail = fail

    async def send_file(self, peer, buf, **kw):
        if self._fail:
            raise RuntimeError("upload died")
        self.sent.append({"peer": peer, "data": buf.read(), **kw})


def _mirror(ogg: bytes, *, fail: bool = False) -> _FakeClient:
    """Drive _mirror_voice_note on a CallService stripped to the attributes it
    actually touches, and wait for the fire-and-forget upload task."""
    from oncall.voice_call import CallService, _ActiveCall

    svc = CallService.__new__(CallService)
    svc._client = _FakeClient(fail=fail)
    svc._owner_user_id = 42
    svc._mirror_tasks = set()
    svc._active = _ActiveCall.__new__(_ActiveCall)
    svc._active.is_owner = True

    async def run():
        svc._mirror_voice_note(ogg, pcm_len=48_000 * 2 * 3)  # 3 s of PCM16/48k
        for t in list(svc._mirror_tasks):
            await t
        return svc._client

    return asyncio.run(run())


def test_mirror_sends_the_spoken_bytes_as_a_voice_note():
    client = _mirror(b"OggS" + b"\x00" * 64)
    assert len(client.sent) == 1
    sent = client.sent[0]
    assert sent["peer"] == 42
    assert sent["data"] == b"OggS" + b"\x00" * 64  # never re-synthesized
    assert sent["voice_note"] is True
    assert sent["attributes"][0].duration == 3


def test_mirror_skips_tts_output_without_an_ogg_container():
    # Raw-Opus TTS responses decode fine for the call but are unplayable as a
    # Telegram voice note, so nothing is uploaded.
    assert _mirror(b"\x78\x00raw-opus-packet").sent == []


def test_mirror_upload_failure_does_not_escape_into_the_call():
    # The upload runs detached; a dead Telegram connection must not take the
    # audio path down with it.
    assert _mirror(b"OggS" + b"\x00" * 64, fail=True).sent == []
