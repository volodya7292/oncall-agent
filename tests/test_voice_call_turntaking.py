"""Turn-taking helpers in voice_call: TTS chunk splitting and the outbound
priority-queue drain.

These pin two non-obvious properties:
  * _split_for_tts must split on sentence boundaries for pipelined TTS but NOT
    inside decimals/abbreviations, and must merge tiny fragments so we don't
    fire a TTS round-trip per "Ok."
  * _drain_chitchat_items is a safety invariant: a user barge-in drops stale
    chitchat but must NEVER drop a queued approval prompt.
  * _mirror_voice_note encodes the spoken PCM and stays detached from the
    call: a dead upload must cost nothing on the line.
  * Streamed TTS: chunk boundaries that split a sample, and a barge-in that
    lands before the first audio.
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


def _mirror(pcm: bytes, *, fail: bool = False) -> _FakeClient:
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
        svc._mirror_voice_note(pcm, "")
        for t in list(svc._mirror_tasks):
            await t
        return svc._client

    return asyncio.run(run())


def test_mirror_sends_the_spoken_audio_as_a_voice_note():
    from oncall.voice_call import CallService

    pcm = bytes(range(256)) * 1125  # 3 s of PCM16/48k
    client = _mirror(pcm)
    assert len(client.sent) == 1
    sent = client.sent[0]
    assert sent["peer"] == 42
    assert sent["voice_note"] is True
    assert sent["attributes"][0].duration == 3
    # A playable Ogg/Opus container holding the same 3 s (20 ms packets).
    decoded = CallService._opus_decode(None, sent["data"])
    assert sent["data"].startswith(b"OggS")
    assert len(decoded) == len(pcm)


def test_mirror_upload_failure_does_not_escape_into_the_call():
    # The upload runs detached; a dead Telegram connection must not take the
    # audio path down with it.
    assert _mirror(b"\x00\x00" * 48_000, fail=True).sent == []


# ---- streamed TTS ----

def _tts_service(chunks: list[bytes]):
    """A CallService whose TTS endpoint streams `chunks` (PCM16 @ 24 kHz),
    one SSE event each, the way Gemini's streamGenerateContent does."""
    import base64
    import json

    import httpx

    from oncall.voice_call import CallService

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for c in chunks:
                await asyncio.sleep(0)
                part = {"inlineData": {"mimeType": "audio/l16; rate=24000",
                                       "data": base64.b64encode(c).decode()}}
                yield f"data: {json.dumps({'candidates': [{'content': {'parts': [part]}}]})}\r\n\r\n".encode()

    transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=Stream()))
    real = httpx.AsyncClient

    svc = CallService.__new__(CallService)
    svc._gemini_api_key = "k"
    svc._tts_model, svc._tts_voice, svc._tts_instructions = "m", "v", ""
    svc._mirror_tasks = set()
    svc._active = None
    return svc, (lambda **kw: real(transport=transport, **kw))


def test_tts_chunks_that_split_a_sample_resample_like_one_buffer(monkeypatch):
    """A streamed chunk ending mid-sample must carry its odd byte into the
    next, or every later sample is read shifted by one byte — loud noise
    instead of speech."""
    import audioop

    from oncall import voice_call
    from oncall.voice_call import PCM_RATE, TTS_RATE, _TtsStream

    raw = bytes((i * 7) % 256 for i in range(4801 * 2))
    svc, client = _tts_service([raw[:3], raw[3:4], raw[4:1001], raw[1001:]])
    monkeypatch.setattr(voice_call.httpx, "AsyncClient", client)

    stream = _TtsStream()
    asyncio.run(svc._synthesize("hallo", stream))
    expected, _ = audioop.ratecv(raw, 2, 1, TTS_RATE, PCM_RATE, None)
    assert stream.done and stream.error is None
    assert bytes(stream.pcm) == expected


def test_barge_in_before_first_audio_returns_whole_reply_and_stops_synth(monkeypatch):
    """The user can talk over the bot while the reply is still being
    synthesized. Nothing was heard, so the whole reply is the remainder, and
    the synthesis must not keep streaming (or reach the voice-note mirror)."""
    from oncall import voice_call
    from oncall.voice_call import _ActiveCall

    started = asyncio.Event()
    mirrored = []

    async def slow_synth(_text, _stream):
        started.set()
        await asyncio.sleep(30)

    svc, _ = _tts_service([])
    svc._synthesize = slow_synth
    svc._mirror_voice_note = lambda pcm, text: mirrored.append(text)
    active = _ActiveCall.__new__(_ActiveCall)
    active.barge_in_by_user = False

    async def run():
        active.barge_in = asyncio.Event()
        speak = asyncio.create_task(svc._speak(active, "Erster Satz. Zweiter Satz."))
        await started.wait()
        active.barge_in_by_user = True
        active.barge_in.set()
        remainder = await speak
        await asyncio.sleep(0)
        return remainder, active.in_flight_tts

    remainder, producer = asyncio.run(run())
    assert remainder == "Erster Satz. Zweiter Satz."
    assert producer.cancelled()
    assert mirrored == []


# ---- inbound transcription session ----

def _bare_call():
    from oncall.voice_call import CallService, _ActiveCall

    svc = CallService.__new__(CallService)
    active = _ActiveCall.__new__(_ActiveCall)
    active.user_speaking = False
    active.is_speaking = False
    active.barge_in_by_user = False
    return svc, active


def test_inbound_reopens_closed_sessions_and_gives_up_on_repeated_failures(monkeypatch):
    """A session the server closes (its length cap) is reopened without
    counting as a failure; only STT_MAX_FAILURES failures in a row end the
    inbound side. Counting the clean close would let a long call die on its
    third routine reconnect."""
    from oncall import voice_call

    svc, active = _bare_call()
    outcomes = ["fail", "closed", "fail", "fail", "fail"]
    calls = []

    async def fake_session(_active, _gain):
        calls.append(outcomes[len(calls)])
        _active.user_speaking = True  # a turn in flight when the session drops
        if calls[-1] == "fail":
            raise RuntimeError("socket died")

    async def no_sleep(_s):
        return None

    svc._stt_session = fake_session
    monkeypatch.setattr(voice_call.asyncio, "sleep", no_sleep)

    async def run():
        try:
            await svc._inbound_loop(active)
        except RuntimeError:
            return True
        return False

    assert asyncio.run(run())
    assert calls == outcomes
    assert active.user_speaking is False


def test_stt_events_drive_barge_in_and_failed_turns_resume():
    """Speech start while the bot talks is a barge-in by the user; a turn
    whose transcription failed reaches _handle_utterance as None, which
    resumes the cut-off reply instead of leaving dead air."""
    from oncall.realtime_stt import SpeechStarted, SpeechStopped, Transcript, TranscriptFailed

    svc, active = _bare_call()
    active.is_speaking = True
    active.barge_in = asyncio.Event()
    handled = []

    async def fake_handle(_active, text):
        handled.append(text)

    svc._handle_utterance = fake_handle

    class FakeStt:
        async def events(self):
            yield SpeechStarted("a", 0)
            assert active.user_speaking and active.barge_in.is_set()
            assert active.barge_in_by_user
            yield SpeechStopped("a", 900)
            yield TranscriptFailed("a", "boom")
            yield Transcript("b", "status bitte")

    async def run():
        await svc._listen_stt(active, FakeStt())
        await asyncio.sleep(0)  # let the spawned handlers run

    asyncio.run(run())
    assert active.user_speaking is False
    assert handled == [None, "status bitte"]
