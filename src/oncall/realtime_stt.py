"""Streaming speech-to-text for voice calls: OpenAI Realtime transcription.

One WebSocket per call carries the caller's audio up. OpenAI's server-side
VAD segments it into turns, so this one connection supplies speech-start
(drives barge-in), speech-stop and the finished transcript of each turn.

Only `gpt-transcribe` and `gpt-4o-transcribe` accept server VAD;
`gpt-live-transcribe` and `gpt-realtime-whisper` reject it outright
("Turn detection is not supported for this transcription model").
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Self

import websockets

log = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
STT_RATE = 24_000  # the session's input format: PCM16 mono @ 24 kHz


@dataclass(frozen=True)
class SpeechStarted:
    item_id: str
    audio_ms: int


@dataclass(frozen=True)
class SpeechStopped:
    item_id: str
    audio_ms: int


@dataclass(frozen=True)
class Transcript:
    item_id: str
    text: str


@dataclass(frozen=True)
class TranscriptFailed:
    item_id: str
    reason: str


SttEvent = SpeechStarted | SpeechStopped | Transcript | TranscriptFailed


class RealtimeTranscriber:
    """One transcription session. `async with` opens it; `send` streams PCM
    at STT_RATE; `events()` yields turn events until the server closes."""

    def __init__(
        self, *, api_key: str, model: str, language: str = "",
        silence_ms: int = 700, vad_threshold: float = 0.5,
        url: str = REALTIME_URL,
    ) -> None:
        self._api_key = api_key
        self._session: dict[str, Any] = {
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": STT_RATE},
                "transcription": {
                    "model": model,
                    **({"languages": [language]} if language else {}),
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": vad_threshold,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": silence_ms,
                },
                "noise_reduction": {"type": "near_field"},
            }},
        }
        self._url = url
        self._ws: Any = None

    async def __aenter__(self) -> Self:
        self._ws = await websockets.connect(
            self._url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=10,
        )
        await self._ws.send(json.dumps({"type": "session.update", "session": self._session}))
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send(self, pcm: bytes) -> None:
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }))

    async def events(self) -> AsyncIterator[SttEvent]:
        """Turn events in arrival order. Transcripts of different turns can
        complete out of order; `item_id` ties them to their speech events.
        A server `error` event raises: the session is unusable after one."""
        async for raw in self._ws:
            msg = json.loads(raw)
            kind = msg.get("type", "")
            if kind == "input_audio_buffer.speech_started":
                yield SpeechStarted(msg["item_id"], int(msg.get("audio_start_ms", 0)))
            elif kind == "input_audio_buffer.speech_stopped":
                yield SpeechStopped(msg["item_id"], int(msg.get("audio_end_ms", 0)))
            elif kind == "conversation.item.input_audio_transcription.completed":
                yield Transcript(msg["item_id"], msg.get("transcript") or "")
            elif kind == "conversation.item.input_audio_transcription.failed":
                err = msg.get("error") or {}
                yield TranscriptFailed(msg["item_id"], str(err.get("message") or err))
            elif kind == "error":
                raise RuntimeError(f"realtime transcription error: {msg.get('error')}")
