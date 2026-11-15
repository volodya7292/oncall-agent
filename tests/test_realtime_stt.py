"""Live test for the Realtime transcription session behind voice calls.

It pins what the fake-free call path depends on and the docs leave open:
the chosen model accepts server VAD in a transcription session, and turns
arrive as speech-start → speech-stop → transcript, tied by item_id.

Skips unless `ONCALL_RUN_OPENAI_TESTS=1` and `OPENAI_API_KEY` are set. The
audio is synthesized by OpenAI TTS, so nothing local is needed.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from oncall.realtime_stt import (
    STT_RATE, RealtimeTranscriber, SpeechStarted, SpeechStopped, Transcript,
)

_LIVE = os.environ.get("ONCALL_RUN_OPENAI_TESTS") == "1" and bool(os.environ.get("OPENAI_API_KEY"))
live = pytest.mark.skipif(not _LIVE, reason="set ONCALL_RUN_OPENAI_TESTS=1 and OPENAI_API_KEY")


async def _speech(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": "gpt-4o-mini-tts", "voice": "cedar", "input": text, "response_format": "pcm"},
        )
        r.raise_for_status()
        return r.content  # PCM16 mono @ 24 kHz, the session's input format


@live
async def test_live_server_vad_segments_turns_and_transcribes():
    second = b"\x00\x00" * STT_RATE
    audio = (
        second + await _speech("Bitte prüfe den Status vom Staging-Server.")
        + second * 2 + await _speech("Danach schick mir eine Zusammenfassung.")
        + second * 2
    )
    stt = RealtimeTranscriber(
        api_key=os.environ["OPENAI_API_KEY"], model="gpt-transcribe", language="de",
    )
    seen: list = []
    async with stt:
        async def feed():
            step = STT_RATE * 2 // 50  # 20 ms
            for i in range(0, len(audio), step):
                await stt.send(audio[i:i + step])
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed())

        async def collect():
            async for ev in stt.events():
                seen.append(ev)
                if sum(isinstance(e, Transcript) for e in seen) == 2:
                    return

        try:
            await asyncio.wait_for(collect(), timeout=30)
        finally:
            feeder.cancel()

    kinds = [type(e) for e in seen]
    assert kinds.count(SpeechStarted) == 2 and kinds.count(SpeechStopped) == 2
    transcripts = {e.item_id: e.text for e in seen if isinstance(e, Transcript)}
    started = [e.item_id for e in seen if isinstance(e, SpeechStarted)]
    assert set(transcripts) == set(started)
    assert "staging" in transcripts[started[0]].lower()
    assert "zusammenfassung" in transcripts[started[1]].lower()
