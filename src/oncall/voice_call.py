"""Voice calls — owner ↔ agent userbot over Telegram MTProto.

The agent userbot's telethon session ([telegram_agent.py](telegram_agent.py))
is shared with this service; py-tgcalls binds to that same client to accept
1:1 voice calls placed by the OWNER (filtered by chat_id == owner_user_id).

Per call, three asyncio tasks run for the call's lifetime:

  * inbound — pull PCM16 @ 48 kHz from py-tgcalls's stream_frame filter, run
    silero-vad to segment utterances, hand each utterance to STT then to
    operator.chat_turn.
  * outbound — pull text items from a queue, POST TTS, decode Opus, send_frame
    PCM back into the call at 10 ms cadence.
  * chat-reply — subscribe to operator's chat.reply events (same shape as
    [telegram_agent._chat_reply_subscriber](telegram_agent.py)) and push their
    text into the outbound queue. This is how follow-up replies from the
    executor reach the caller mid-call.

Barge-in: when the inbound VAD detects user speech for ≥150 ms while the
outbound loop is playing audio, the outbound loop cancels its in-flight TTS
HTTP request and drops queued audio. The user always wins.

Voice and text chat share the same `chat_sessions` row (agent_session_id),
so a voice call continues the same conversation the user had in text and
vice versa.
"""

from __future__ import annotations

import asyncio
import audioop
import ctypes
import ctypes.util
import logging
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


def _ensure_libopus_discoverable() -> None:
    """opuslib does `ctypes.util.find_library('opus')` at import; on macOS
    that doesn't search Homebrew's /opt/homebrew/lib. Pre-load the dylib
    by absolute path and monkey-patch find_library so opuslib's own lookup
    succeeds. No-op if libopus is already discoverable."""
    if ctypes.util.find_library("opus") is not None:
        return
    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates += [
            "/opt/homebrew/lib/libopus.dylib",   # Apple Silicon Homebrew
            "/usr/local/lib/libopus.dylib",      # Intel Homebrew / MacPorts
        ]
    elif sys.platform.startswith("linux"):
        candidates += ["/usr/lib/x86_64-linux-gnu/libopus.so.0", "/usr/lib/libopus.so.0"]
    found = next((p for p in candidates if Path(p).exists()), None)
    if found is None:
        return
    try:
        ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return
    _orig = ctypes.util.find_library

    def _patched(name: str):
        if name == "opus":
            return found
        return _orig(name)

    ctypes.util.find_library = _patched


_ensure_libopus_discoverable()

from uuid import UUID, uuid4

from .approval_client import is_deny_phrase, phrases_match
from .broker import Broker
from .events import EventBus
from .operator import Operator
from .telegram_agent import agent_session_id

log = logging.getLogger(__name__)


# Telegram MTProto voice carries PCM16 at 48 kHz; py-tgcalls 2.2.x delivers
# 10 ms frames (480 samples, 960 bytes) on the receive side, and we must
# send_frame at the same shape/cadence to avoid pitch/speed artifacts.
PCM_RATE = 48_000
PCM_CHANNELS = 1
FRAME_MS = 10
SAMPLES_PER_FRAME = PCM_RATE * FRAME_MS // 1000  # 480
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2          # 960

# Silero VAD wants 16 kHz mono PCM in 30 ms (512-sample) chunks.
VAD_RATE = 16_000
VAD_CHUNK_SAMPLES = 512
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 2

# Speech-start / speech-end hysteresis.
VAD_START_PROB = 0.6
VAD_END_PROB = 0.35
VAD_SILENCE_END_MS = 500   # silence after speech that ends an utterance
VAD_BARGE_IN_MS = 150      # voice during TTS that cancels playback

# HTTP timeouts. TTS can return seconds of audio; STT can do long files.
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=60.0)

# `send_frame` watchdog. A single 10 ms PCM frame should round-trip into
# pytgcalls instantly; if ntgcalls's native send buffer back-presses, it
# can block our outbound loop for tens of seconds. 1.0 s is generous —
# anything longer means the call is effectively stuck.
SEND_FRAME_TIMEOUT_S = 1.0
# Inactivity watchdog on `_speak()`'s outer loop: if neither a frame
# arrives nor barge-in fires for this many seconds, abandon the current
# TTS so queued items (esp. approval prompts) can play.
SPEAK_IDLE_TIMEOUT_S = 3.0


@dataclass
class _ActiveCall:
    chat_id: int
    # Per-call session id. For inbound (owner) calls this is the shared
    # agent_session_id so voice continues the owner's text chat. For
    # outbound calls to a non-owner, this is a fresh ephemeral id so the
    # operator runs in a clean session with no owner chat history.
    session_id: str
    is_owner: bool                         # drives memory isolation + system note variant
    callee_label: str                      # human-readable; "owner", "Alice", "id=12345"
    reason: str                            # operator's stated purpose; "" for inbound
    outbound_text_queue: asyncio.Queue[str]
    inbound_frames: asyncio.Queue[bytes]
    barge_in: asyncio.Event
    # Which PyTgCalls instance owns this call. Inbound calls use the
    # agent-client instance; outbound calls use the primary-client
    # instance. send_frame / leave_call must be routed to the right one.
    call_py: Any = None
    # Watchdog clocks (monotonic seconds). placed_at is set when the call
    # is initiated; connected_at is set on first inbound audio frame (i.e.
    # actual pickup — for inbound calls we set it equal to placed_at since
    # _on_incoming only fires after acceptance); last_activity_at is bumped
    # on every utterance and TTS playback.
    placed_at: float = 0.0
    connected_at: float | None = None
    last_activity_at: float = 0.0
    tasks: list[asyncio.Task] = field(default_factory=list)
    is_speaking: bool = False              # outbound is currently emitting TTS audio
    in_flight_tts: asyncio.Task | None = None
    # approval_id (str) → challenge phrase. Mirrors telegram_agent's pending
    # dict — populated when broker fires approval.requested for our session,
    # consumed when the user's STT'd utterance matches an affirm/deny phrase.
    pending_approvals: dict[str, str] = field(default_factory=dict)


# Call-lifecycle timeouts (seconds, monotonic). Tuned in CallService below
# but kept here so they sit next to _ActiveCall for easy review.
RING_TIMEOUT_S = 40.0       # outbound call dropped if not picked up by then
IDLE_TIMEOUT_S = 90.0       # any call torn down after this much silence
MAX_CALL_DURATION_S = 600.0  # hard cap on any single call (10 min)
WATCHDOG_TICK_S = 5.0       # how often _call_watchdog re-checks


class CallService:
    def __init__(
        self,
        *,
        client: Any,                       # telethon TelegramClient — AGENT userbot (inbound)
        primary_client: Any = None,        # telethon TelegramClient — OWNER's userbot (outbound)
        operator: Operator,
        events: EventBus,
        broker: Broker,
        owner_user_id: int,
        tts_base_url: str,
        tts_api_key: str,
        tts_voice: str,
        stt_base_url: str,
        stt_api_key: str,
        language: str = "",
        llm: Any = None,                   # operator's LLMClient for paraphrase
        llm_model: str = "",
    ) -> None:
        self._client = client
        # Owner's primary userbot client — used to PLACE outbound calls
        # so the callee sees them coming from the owner's account, not
        # the agent's. When None, place_call refuses.
        self._primary_client = primary_client
        self._operator = operator
        self._events = events
        self._broker = broker
        self._llm = llm
        self._llm_model = llm_model
        self._owner_user_id = int(owner_user_id)
        self._tts_base_url = tts_base_url.rstrip("/")
        self._tts_api_key = tts_api_key
        self._tts_voice = tts_voice
        self._stt_base_url = stt_base_url.rstrip("/")
        self._stt_api_key = stt_api_key
        self._language = language
        # Owner's shared session: voice ↔ text continuity for inbound calls.
        # Outbound calls to non-owners use a fresh per-call ephemeral id.
        self._owner_session_id = agent_session_id(owner_user_id)
        # Two PyTgCalls instances: `_call_py` on the agent client receives
        # inbound owner calls; `_call_py_primary` on the primary client
        # places outbound calls. Both are populated in start(); either may
        # be None when corresponding client wasn't provided.
        self._call_py: Any | None = None
        self._call_py_primary: Any | None = None
        self._started = False
        self._active: _ActiveCall | None = None
        self._vad: Any | None = None
        # Resample state for 48k → 16k.
        self._resample_state: Any = None
        # TTS-health probe cache. Calls (in/outbound) hit a tiny TTS endpoint
        # before connecting; a healthy result stays valid for this many
        # seconds so back-to-back checks don't add latency.
        self._tts_health_until: float = 0.0
        self._tts_health_ttl_s: float = 10.0
        # Strong refs for fire-and-forget post-call brief tasks. Without
        # this the event loop only holds a weak reference and Python GC
        # destroys the coroutine mid-flight (see asyncio.create_task docs).
        self._brief_tasks: set[asyncio.Task] = set()

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def session_id(self) -> str:
        """The owner's session — used by `telegram_agent` to scope events
        even when no call is active. Per-call session ids (which may differ
        for outbound non-owner calls) live on `_ActiveCall.session_id`."""
        return self._owner_session_id

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._started:
            return
        # Lazy import — these pull native deps (libopus, torch) that only
        # matter when voice is enabled.
        from pytgcalls import PyTgCalls, filters as fl  # type: ignore[import-not-found]
        from pytgcalls.types import (  # type: ignore[import-not-found]
            ChatUpdate, Device, Direction, ExternalMedia, MediaStream,
            RecordStream, StreamFrames,
        )
        from pytgcalls.types.raw.audio_parameters import (  # type: ignore[import-not-found]
            AudioParameters,
        )
        from silero_vad import load_silero_vad  # type: ignore[import-not-found]

        self._fl = fl
        self._ChatUpdate = ChatUpdate
        self._Device = Device
        self._Direction = Direction
        self._ExternalMedia = ExternalMedia
        self._MediaStream = MediaStream
        self._RecordStream = RecordStream
        self._StreamFrames = StreamFrames
        self._AudioParameters = AudioParameters

        self._vad = load_silero_vad()

        # --- Agent client: receives INBOUND owner calls. ---
        self._call_py = PyTgCalls(self._client)
        await self._call_py.start()

        @self._call_py.on_update(fl.chat_update(ChatUpdate.Status.INCOMING_CALL))
        async def _on_incoming_agent(_, update):
            await self._on_incoming(update)

        @self._call_py.on_update(fl.chat_update(ChatUpdate.Status.LEFT_CALL))
        async def _on_left_agent(_, update):
            await self._on_left(update)

        @self._call_py.on_update(
            fl.stream_frame(Direction.INCOMING, Device.MICROPHONE),
        )
        async def _on_frames_agent(_, update):
            await self._on_frames(update)

        # --- Primary client: places OUTBOUND calls.
        # No INCOMING_CALL handler on purpose: registering one would
        # intercept the owner's real-person phone calls coming in on
        # their primary account.
        if self._primary_client is not None:
            self._call_py_primary = PyTgCalls(self._primary_client)
            await self._call_py_primary.start()

            @self._call_py_primary.on_update(
                fl.chat_update(ChatUpdate.Status.LEFT_CALL),
            )
            async def _on_left_primary(_, update):
                await self._on_left(update)

            @self._call_py_primary.on_update(
                fl.stream_frame(Direction.INCOMING, Device.MICROPHONE),
            )
            async def _on_frames_primary(_, update):
                await self._on_frames(update)

        self._started = True
        log.info(
            "voice CallService started (owner_user_id=%d, session=%s, primary=%s)",
            self._owner_user_id, self._owner_session_id,
            "yes" if self._primary_client is not None else "no",
        )

    async def stop(self) -> None:
        if not self._started:
            return
        active = self._active
        if active is not None:
            await self._teardown_active(active, reason="service-stop")
        # PyTgCalls 2.2.x exposes no service-level stop() — its only
        # teardown verb is per-call (leave_call), which we've already done
        # via _teardown_active. The PyTgCalls instance is bound to the
        # agent's telethon client; when telegram_agent.stop() disconnects
        # the client (see lifespan cleanup in api.py), the underlying
        # ntgcalls binding goes away with it.
        self._started = False
        log.info("voice CallService stopped")

    # ---- pytgcalls handlers ----

    async def _on_incoming(self, update: Any) -> None:
        caller = int(update.chat_id)
        if caller != self._owner_user_id:
            log.warning(
                "voice: rejecting unauthorized caller=%s (owner=%s)",
                caller, self._owner_user_id,
            )
            try:
                await self._call_py.leave_call(caller)  # type: ignore[union-attr]
            except Exception:
                log.exception("voice: leave_call on rejection failed")
            return
        if self._active is not None:
            log.warning(
                "voice: already in a call with chat_id=%s; rejecting new caller=%s",
                self._active.chat_id, caller,
            )
            try:
                await self._call_py.leave_call(caller)  # type: ignore[union-attr]
            except Exception:
                log.exception("voice: leave_call on busy rejection failed")
            return
        # TTS gates call acceptance: without it the operator's replies can't
        # be spoken and the owner hears nothing back. Dropping the call now
        # is friendlier than connecting and going silent.
        if not await self._tts_health_ok():
            log.warning(
                "voice: TTS unhealthy; dropping incoming call from owner=%s",
                caller,
            )
            try:
                await self._call_py.leave_call(caller)  # type: ignore[union-attr]
            except Exception:
                log.exception("voice: leave_call after TTS-unhealthy failed")
            return
        try:
            params = self._AudioParameters(PCM_RATE, PCM_CHANNELS)
            await self._call_py.play(  # type: ignore[union-attr]
                caller, self._MediaStream(self._ExternalMedia.AUDIO, params),
            )
            await self._call_py.record(  # type: ignore[union-attr]
                caller,
                self._RecordStream(audio=True, audio_parameters=params),
            )
        except Exception:
            log.exception("voice: failed to accept call from %s", caller)
            try:
                await self._call_py.leave_call(caller)  # type: ignore[union-attr]
            except Exception:
                pass
            return

        active = self._attach_active_call(
            chat_id=caller,
            session_id=self._owner_session_id,
            is_owner=True,
            callee_label="owner",
            reason="",
            awaiting_pickup=False,  # owner is on the line by the time we're here
            call_py=self._call_py,
        )

        log.info("voice: call accepted from owner=%s", caller)

        # Tell the operator the call started so it can greet. We use
        # auto_ping (the existing system-event path) rather than chat_turn
        # so memory extraction sees this as a procedural ping, not a user
        # message. The operator's reply lands as chat.reply → TTS via the
        # subscriber → owner hears the greeting.
        asyncio.create_task(self._announce_call_start(active), name="voice-greet")

    # ---- shared setup for inbound and outbound calls ----

    def _attach_active_call(
        self, *,
        chat_id: int,
        session_id: str,
        is_owner: bool,
        callee_label: str,
        reason: str,
        awaiting_pickup: bool,
        call_py: Any,
    ) -> _ActiveCall:
        """Construct the _ActiveCall, install it as `self._active`, and
        spawn the per-call asyncio tasks. Used by both the inbound
        (`_on_incoming`) and outbound (`place_call`) paths so they share
        the same task lifecycle / done-callback wiring.

        `awaiting_pickup=True` for outbound calls — `connected_at` stays
        None until first inbound audio frame arrives, which is when the
        no-answer watchdog stops counting. False for inbound calls (we
        only see them post-acceptance)."""
        now = time.monotonic()
        active = _ActiveCall(
            chat_id=chat_id,
            session_id=session_id,
            is_owner=is_owner,
            callee_label=callee_label,
            reason=reason,
            outbound_text_queue=asyncio.Queue(),
            inbound_frames=asyncio.Queue(maxsize=4096),
            barge_in=asyncio.Event(),
            placed_at=now,
            connected_at=None if awaiting_pickup else now,
            last_activity_at=now,
            call_py=call_py,
        )
        self._active = active
        self._resample_state = None  # reset audioop ratecv state for new call

        active.tasks = [
            asyncio.create_task(self._inbound_loop(active), name="voice-inbound"),
            asyncio.create_task(self._outbound_loop(active), name="voice-outbound"),
            asyncio.create_task(
                self._chat_reply_subscriber(active), name="voice-chat-reply",
            ),
            asyncio.create_task(
                self._approval_subscriber(active), name="voice-approval-req",
            ),
            asyncio.create_task(
                self._approval_resolved_subscriber(active),
                name="voice-approval-resolved",
            ),
            asyncio.create_task(
                self._call_watchdog(active), name="voice-watchdog",
            ),
        ]
        for t in active.tasks:
            t.add_done_callback(self._on_call_task_done)
        return active

    async def _call_watchdog(self, active: _ActiveCall) -> None:
        """Two timeouts:
          * No-answer: outbound calls dropped after RING_TIMEOUT_S without
            inbound audio (callee never picked up or rejected silently).
          * Idle: any call torn down after IDLE_TIMEOUT_S of mutual silence
            (no utterance, no TTS playback). Prevents stale calls from
            hogging the single _active slot if both sides stop talking.

        Ticks every WATCHDOG_TICK_S. Bails when the call is no longer
        active (already torn down)."""
        try:
            while True:
                await asyncio.sleep(WATCHDOG_TICK_S)
                if self._active is not active:
                    return
                now = time.monotonic()
                if active.connected_at is None:
                    waited = now - active.placed_at
                    if waited >= RING_TIMEOUT_S:
                        log.info(
                            "voice: outbound call to %s no-answer after %.0fs; "
                            "tearing down", active.callee_label, waited,
                        )
                        await self._teardown_active(active, reason="no-answer")
                        return
                    continue
                # Hard cap on call duration — prevents the agent from
                # being trapped on a runaway call (held line, stalking,
                # bug). Measured from pickup, not from placement.
                duration = now - active.connected_at
                if duration >= MAX_CALL_DURATION_S:
                    log.info(
                        "voice: max-duration %.0fs reached in call with %s; "
                        "tearing down", duration, active.callee_label,
                    )
                    await self._teardown_active(active, reason="max-duration")
                    return
                # Don't count "currently speaking" as idle — that's the
                # agent actively producing audio.
                if active.is_speaking:
                    continue
                idle = now - active.last_activity_at
                if idle >= IDLE_TIMEOUT_S:
                    log.info(
                        "voice: idle %.0fs in call with %s; tearing down",
                        idle, active.callee_label,
                    )
                    await self._teardown_active(active, reason="idle-timeout")
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice watchdog crashed")
            raise

    # ---- outbound call initiation ----

    async def place_call(self, chat_id: int, *, reason: str) -> None:
        """Place an outbound 1:1 voice call from the OWNER's primary
        userbot to `chat_id`. The callee sees an incoming Telegram call
        from the owner's account, not from the agent userbot.

        Raises a descriptive RuntimeError on any pre-flight failure (not
        started, no primary client, self-call, busy, TTS unhealthy,
        callee's Telegram privacy settings, etc.) so the dispatcher can
        map to a clean tool error. On success, sets up an _ActiveCall
        whose LEFT_CALL / inbound-frame events flow through the primary
        PyTgCalls instance back into the shared lifecycle.

        `reason`: the operator's stated purpose for the call. Surfaced to
        the operator at call-start so it knows what it's permitted to
        discuss."""
        if not self._started:
            raise RuntimeError("voice service not started")
        if self._call_py_primary is None:
            raise RuntimeError(
                "primary userbot not available — cannot place outbound calls",
            )
        if chat_id == self._owner_user_id:
            raise RuntimeError(
                "cannot place a call to the owner from the owner's own account",
            )
        if self._active is not None:
            raise RuntimeError("agent is currently on another call")
        if not await self._tts_health_ok():
            raise RuntimeError("tts_unhealthy")

        # All outbound calls are to non-owners (self-call rejected above),
        # so each gets a fresh ephemeral session — clean chat history,
        # the owner's persistent memory is still accessible.
        session_id = f"voice-out-{uuid4()}"
        callee_label = await self._resolve_callee_label(chat_id)

        params = self._AudioParameters(PCM_RATE, PCM_CHANNELS)
        try:
            await self._call_py_primary.play(
                chat_id, self._MediaStream(self._ExternalMedia.AUDIO, params),
            )
        except Exception as exc:
            # Translate typed exceptions into clear, operator-facing
            # messages. The raw class names (CallDeclined, CallBusy, …)
            # are confusing — the operator was reading "CallDeclined" +
            # 422 status as "broker blocked the request", which is wrong.
            cls_name = type(exc).__name__
            log.warning(
                "voice: outbound play(%s) failed: %s: %s",
                chat_id, cls_name, exc,
            )
            try:
                await self._call_py_primary.leave_call(chat_id)
            except Exception:
                pass
            friendly = {
                "UserPrivacyRestrictedError": (
                    f"{callee_label} did not pick up — their Telegram "
                    f"privacy settings prevent voice calls from this account"
                ),
                "CallDeclined": f"{callee_label} declined the call",
                "CallBusy": f"{callee_label} is on another call",
                "CallDiscarded": f"call to {callee_label} was discarded before connecting",
                "TimedOutAnswer": f"{callee_label} did not answer in time",
            }.get(cls_name)
            if friendly is not None:
                raise RuntimeError(friendly) from exc
            raise RuntimeError(f"failed to place call: {cls_name}: {exc}") from exc
        try:
            await self._call_py_primary.record(
                chat_id,
                self._RecordStream(audio=True, audio_parameters=params),
            )
        except Exception:
            log.exception("voice: outbound record() failed for %s", chat_id)
            try:
                await self._call_py_primary.leave_call(chat_id)
            except Exception:
                pass
            raise RuntimeError("failed to start audio recording") from None

        active = self._attach_active_call(
            chat_id=chat_id,
            session_id=session_id,
            is_owner=False,  # self-call refused above, so this is always a non-owner
            callee_label=callee_label,
            reason=reason,
            awaiting_pickup=True,  # watchdog will count ring time until first frame
            call_py=self._call_py_primary,
        )

        log.info(
            "voice: outbound call (via primary userbot) initiated to %s "
            "(session=%s reason=%r)",
            callee_label, session_id, reason[:80],
        )

        # Prewarm: generate the first message text + TTS while the call
        # is ringing; play it 2 s after pickup is detected. Replaces the
        # inbound-only _announce_call_start path for outbound calls.
        task = asyncio.create_task(
            self._prewarm_and_play_first(active),
            name="voice-prewarm-greet",
        )
        active.tasks.append(task)
        task.add_done_callback(self._on_call_task_done)

    async def _resolve_callee_label(self, chat_id: int) -> str:
        """Best-effort human-readable name for the callee, used in the
        operator's call-start system note and in logs. Falls back to the
        bare id if telethon can't resolve the entity."""
        try:
            entity = await self._client.get_entity(chat_id)
        except Exception:
            log.warning("voice: get_entity(%s) failed; using bare id", chat_id)
            return f"id={chat_id}"
        first = (getattr(entity, "first_name", None) or "").strip()
        last = (getattr(entity, "last_name", None) or "").strip()
        username = (getattr(entity, "username", None) or "").strip()
        if first or last:
            full = (first + " " + last).strip()
            return full if not username else f"{full} (@{username})"
        if username:
            return f"@{username}"
        return f"id={chat_id}"

    # ---- TTS health probe ----

    async def _tts_health_ok(self) -> bool:
        """Quick check that TTS is reachable. Returns True if a recent
        probe succeeded (within `_tts_health_ttl_s`) or if a fresh probe
        succeeds now; False otherwise. A call without working TTS is
        useless — the operator can think but can't speak."""
        now = time.monotonic()
        if now < self._tts_health_until:
            return True
        url = f"{self._tts_base_url}/v1/audio/speech"
        body = {
            "model": "tts-1",
            "voice": self._tts_voice,
            "input": " ",
            "response_format": "opus",
        }
        try:
            # `read` covers the actual TTS synthesis — even input=" " takes
            # 2–3 s on a healthy provider, occasionally up to ~4. 5 s gives
            # headroom for normal variance without letting the daemon hang
            # if the provider is truly stuck.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=5.0),
            ) as client:
                r = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._tts_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                ok = 200 <= r.status_code < 300
        except Exception as exc:
            log.warning("voice: TTS health probe failed: %r", exc)
            return False
        if not ok:
            log.warning("voice: TTS health probe returned status=%d", r.status_code)
            return False
        self._tts_health_until = time.monotonic() + self._tts_health_ttl_s
        return True

    async def _on_left(self, update: Any) -> None:
        chat_id = int(update.chat_id)
        active = self._active
        if active is None or active.chat_id != chat_id:
            return
        log.info("voice: LEFT_CALL chat_id=%s — tearing down", chat_id)
        await self._teardown_active(active, reason="left-call")

    async def _on_frames(self, update: Any) -> None:
        active = self._active
        if active is None:
            return
        # Guard against cross-talk on the primary userbot: it may receive
        # frame events for unrelated calls (e.g. the owner's own real-world
        # voice activity through other PyTgCalls bindings or normal Telegram
        # use). Only handle frames addressed to OUR active call's chat_id.
        frame_chat_id = getattr(update, "chat_id", None)
        if frame_chat_id is not None and int(frame_chat_id) != active.chat_id:
            return
        # First inbound audio frame on an outbound call = the callee picked
        # up. This stops the no-answer watchdog and starts the idle clock.
        if active.connected_at is None:
            active.connected_at = time.monotonic()
            active.last_activity_at = active.connected_at
            log.info(
                "voice: outbound call to %s connected after %.1fs",
                active.callee_label, active.connected_at - active.placed_at,
            )
        for frame in update.frames:
            # py-tgcalls 2.2 attaches a `frame` bytes attribute. Defensive
            # against future renames — fall back to bytes(...).
            data = getattr(frame, "frame", None) or bytes(frame)
            try:
                active.inbound_frames.put_nowait(data)
            except asyncio.QueueFull:
                # Inbound is faster than we can drain — drop the oldest. We
                # care about recent audio for VAD, not historical.
                try:
                    active.inbound_frames.get_nowait()
                    active.inbound_frames.put_nowait(data)
                except asyncio.QueueEmpty:
                    pass

    # ---- inbound loop ----

    async def _inbound_loop(self, active: _ActiveCall) -> None:
        import numpy as np  # local import; numpy is already a top-level dep
        import torch        # silero-vad pulls torch; keep the model on CPU

        vad_buf = bytearray()        # 16 kHz PCM, fed to silero in 512-sample chunks
        utt_buf = bytearray()        # 48 kHz PCM, what we send to STT
        speaking = False
        silent_ms = 0
        voiced_ms = 0
        last_log_t = 0.0

        try:
            while True:
                frame_48k = await active.inbound_frames.get()
                utt_buf += frame_48k
                resampled, self._resample_state = audioop.ratecv(
                    frame_48k, 2, PCM_CHANNELS, PCM_RATE, VAD_RATE,
                    self._resample_state,
                )
                vad_buf += resampled
                # Run VAD on as many 512-sample (32 ms) chunks as we have.
                while len(vad_buf) >= VAD_CHUNK_BYTES:
                    chunk = bytes(vad_buf[:VAD_CHUNK_BYTES])
                    del vad_buf[:VAD_CHUNK_BYTES]
                    pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    with torch.no_grad():
                        prob = float(self._vad(torch.from_numpy(pcm), VAD_RATE).item())
                    chunk_ms = (VAD_CHUNK_SAMPLES * 1000) // VAD_RATE  # 32 ms
                    if speaking:
                        if prob >= VAD_END_PROB:
                            silent_ms = 0
                        else:
                            silent_ms += chunk_ms
                            if silent_ms >= VAD_SILENCE_END_MS:
                                pcm_bytes = bytes(utt_buf)
                                utt_buf.clear()
                                speaking = False
                                silent_ms = 0
                                voiced_ms = 0
                                # Trim trailing silence we already accumulated.
                                trim = (VAD_SILENCE_END_MS * PCM_RATE // 1000) * 2
                                if len(pcm_bytes) > trim:
                                    pcm_bytes = pcm_bytes[:-trim]
                                log.info(
                                    "voice: utterance end %.2fs",
                                    len(pcm_bytes) / 2 / PCM_RATE,
                                )
                                asyncio.create_task(
                                    self._handle_utterance(active, pcm_bytes),
                                    name="voice-handle-utt",
                                )
                    else:
                        if prob >= VAD_START_PROB:
                            voiced_ms += chunk_ms
                            if voiced_ms >= 90:  # 3 consecutive ~32 ms chunks
                                speaking = True
                                log.info("voice: utterance start")
                                # Barge-in: if outbound is currently speaking,
                                # cancel it so the user can talk.
                                if active.is_speaking:
                                    log.info("voice: barge-in detected")
                                    active.barge_in.set()
                        else:
                            voiced_ms = 0
                            # Drop pre-speech buffer so utt_buf doesn't grow
                            # unboundedly while user is silent.
                            if len(utt_buf) > BYTES_PER_FRAME * 50:
                                del utt_buf[:-BYTES_PER_FRAME * 50]
                    now = time.monotonic()
                    if now - last_log_t > 5:
                        last_log_t = now
                        log.debug(
                            "voice vad: p=%.2f speaking=%s silent_ms=%d voiced_ms=%d",
                            prob, speaking, silent_ms, voiced_ms,
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice inbound loop crashed")
            raise

    def _call_start_note(self, active: _ActiveCall) -> str:
        """Procedural system note announcing the call. Phrasing matters:
        this text is read by both the operator (to decide what to do) AND
        the memory extractor (to decide what facts to save). It must be a
        per-call procedural directive, never a user-attributed preference,
        so the extractor finds nothing to keep. The previous attempt —
        chat_turn with "[system] keep replies short and conversational" —
        was paraphrased into memory as "user prefers short replies".
        Don't repeat that mistake.

        For outbound calls to non-owners, this note is also the
        load-bearing place where the operator is told who it's speaking
        to. Memory access isn't gated — the prompt asks for judgement."""
        if active.is_owner:
            return (
                "voice call from the owner just started. greet them briefly "
                "(one sentence, conversational, no markdown). your reply will "
                "be spoken aloud by TTS. audio may be garbled or partly "
                "inaudible, so before you act on anything consequential "
                "(sending a message, placing a call, deleting or changing "
                "something), read back what you understood in one line and "
                "wait for the owner to confirm before you hand off — don't "
                "act on a guess. skip the read-back for chitchat and simple "
                "answers. this note is procedural — do not save anything to "
                "memory based on it."
            )
        return (
            f"you have just placed an outbound voice call from the "
            f"owner's account to {active.callee_label}. you are speaking "
            f"AS the owner's assistant, NOT as the owner. the owner "
            f"authorized this call for: {active.reason!r} — stay on that "
            f"topic. consult your memory for what the owner has documented "
            f"about {active.callee_label} (authorizations, prior context, "
            f"shared facts you may freely use); rely on judgement for "
            f"anything else and don't volunteer owner facts that aren't "
            f"relevant (location, schedule, unrelated contacts). YOU speak "
            f"first — produce a brief one-sentence opener that identifies "
            f"you as the owner's assistant and states why you're calling, "
            f"so the callee knows immediately what this is. your reply "
            f"will be spoken aloud by TTS. this note is procedural — do "
            f"not save anything to memory based on it."
        )

    async def _announce_call_start(self, active: _ActiveCall) -> None:
        """Inbound path: operator generates a greeting which we enqueue
        to outbound TTS. The outbound path uses _prewarm_and_play_first
        instead, so it can synthesize during the ring and start playback
        on a known delay after pickup."""
        note = self._call_start_note(active)
        try:
            result = await self._operator.auto_ping(
                session_id=active.session_id, note=note,
            )
        except Exception:
            log.exception("voice: call-start auto_ping failed")
            return
        text = (result.text or "").strip()
        if text and self._active is active:
            await self._enqueue_text(active, text)

    async def _prewarm_and_play_first(self, active: _ActiveCall) -> None:
        """Outbound path: kick off the operator turn AND TTS synthesis
        immediately (during the ring), then wait POST_PICKUP_DELAY_S
        after pickup before sending the audio. This hides the multi-second
        gen+synth latency that would otherwise be a dead air gap right
        after the callee says 'hello'."""
        POST_PICKUP_DELAY_S = 2.0
        note = self._call_start_note(active)
        try:
            result = await self._operator.auto_ping(
                session_id=active.session_id, note=note,
            )
        except Exception:
            log.exception("voice: prewarm auto_ping failed")
            return
        if self._active is not active:
            return
        text = (result.text or "").strip()
        if not text:
            log.info("voice: prewarm produced no text; nothing to play")
            return
        log.info(
            "voice: prewarm text generated (%d chars); synthesizing TTS",
            len(text),
        )
        # Synthesize while still (likely) ringing — saves seconds vs. the
        # in-line _speak path.
        try:
            opus_bytes = await self._tts_http(text)
            pcm = self._opus_decode(opus_bytes)
        except Exception:
            log.exception(
                "voice: prewarm TTS failed; falling back to enqueue path",
            )
            # Fallback: let _outbound_loop synthesize again post-pickup.
            await self._enqueue_text(active, text)
            return
        log.info(
            "voice: prewarm TTS ready (%d PCM bytes); waiting for pickup",
            len(pcm),
        )
        # Wait for first inbound frame (heuristic for pickup). The
        # watchdog will tear down via no-answer if this never fires.
        while active.connected_at is None:
            if self._active is not active:
                return
            await asyncio.sleep(0.1)
        # Natural pause so we don't talk over the connection beep / the
        # callee's "hello".
        await asyncio.sleep(POST_PICKUP_DELAY_S)
        if self._active is not active:
            return
        # Play the prewarmed audio directly, bypassing the outbound text
        # queue. Mirrors _speak's send_frame loop with barge-in support
        # but without TTS synthesis.
        await self._play_pcm(active, pcm, label="prewarm")

    async def _play_pcm(
        self, active: _ActiveCall, pcm: bytes, *, label: str = "pcm",
    ) -> None:
        """Send pre-rendered PCM out as 10 ms frames. Sets is_speaking,
        respects barge_in, bumps last_activity_at. Shared between the
        prewarm path and any other future direct-playback needs."""
        active.barge_in.clear()
        active.is_speaking = True
        active.last_activity_at = time.monotonic()
        sent = 0
        try:
            for i in range(0, len(pcm), BYTES_PER_FRAME):
                if active.barge_in.is_set():
                    log.info(
                        "voice: %s barge-in after %d frames (%.0fms)",
                        label, sent, sent * FRAME_MS,
                    )
                    return
                if self._active is not active:
                    return
                chunk = pcm[i:i + BYTES_PER_FRAME]
                if len(chunk) < BYTES_PER_FRAME:
                    chunk += b"\x00" * (BYTES_PER_FRAME - len(chunk))
                try:
                    await asyncio.wait_for(
                        active.call_py.send_frame(
                            active.chat_id, self._Device.MICROPHONE, chunk,
                        ),
                        timeout=SEND_FRAME_TIMEOUT_S,
                    )
                    sent += 1
                except asyncio.TimeoutError:
                    log.warning(
                        "voice: %s send_frame timed out at frame %d", label, sent,
                    )
                    return
                except Exception:
                    log.exception(
                        "voice: %s send_frame failed at frame %d", label, sent,
                    )
                    return
                await asyncio.sleep(FRAME_MS / 1000)
            log.info(
                "voice: %s playback complete (%d frames, %.0fms)",
                label, sent, sent * FRAME_MS,
            )
        finally:
            active.is_speaking = False

    async def _handle_utterance(self, active: _ActiveCall, pcm_bytes: bytes) -> None:
        if len(pcm_bytes) < PCM_RATE * 2 * 0.2:  # <200ms ≈ noise
            log.info("voice: dropping <200ms utterance")
            return
        # Real speech resets the idle watchdog. (Frame-level audio doesn't
        # — Telegram sends frames continuously even when nobody speaks.)
        active.last_activity_at = time.monotonic()
        try:
            text = await self._stt(pcm_bytes)
        except Exception:
            log.exception("voice: STT failed; dropping utterance")
            return
        text = (text or "").strip()
        if not text:
            log.info("voice: STT returned empty text")
            return
        log.info("voice: STT -> %r", text[:120])
        # If an approval is pending and the user said yes/no, resolve via
        # the broker — do NOT forward to operator (otherwise the operator
        # would react to the affirm/deny as if it were a new user request).
        if await self._try_resolve_approval(active, text):
            return
        try:
            result = await self._operator.chat_turn(
                session_id=active.session_id, user_text=text,
            )
        except Exception:
            log.exception("voice: operator.chat_turn failed")
            await self._enqueue_text(active, "Sorry — I hit an error.")
            return
        ack = result.user_facing_text()
        if ack:
            await self._enqueue_text(active, ack)

    async def _enqueue_text(self, active: _ActiveCall, text: str) -> None:
        try:
            active.outbound_text_queue.put_nowait(text)
        except asyncio.QueueFull:
            log.warning("voice: outbound queue full; dropping text")

    # ---- outbound loop ----

    async def _outbound_loop(self, active: _ActiveCall) -> None:
        try:
            while True:
                text = await active.outbound_text_queue.get()
                active.barge_in.clear()
                active.is_speaking = True
                active.last_activity_at = time.monotonic()
                try:
                    await self._speak(active, text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("voice: speaking %r failed", text[:80])
                finally:
                    active.is_speaking = False
                    active.in_flight_tts = None
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice outbound loop crashed")
            raise

    async def _speak(self, active: _ActiveCall, text: str) -> None:
        """POST text to TTS, decode the Opus response, push PCM out as 10 ms
        frames. Cancelled mid-stream by barge-in."""
        log.info("voice: TTS speaking %d chars: %r", len(text), text[:80])
        frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        producer = asyncio.create_task(
            self._tts_producer(text, frame_queue),
            name="voice-tts-producer",
        )
        active.in_flight_tts = producer
        sent_frames = 0
        try:
            while True:
                get_task = asyncio.create_task(frame_queue.get())
                barge_task = asyncio.create_task(active.barge_in.wait())
                # Only enforce the idle watchdog AFTER playback has started.
                # Before the first frame, we're waiting on the producer's
                # TTS HTTP call (already bounded by HTTP_TIMEOUT=60s); a
                # long TTS HTTP is normal for long text, not a bug.
                wait_timeout = SPEAK_IDLE_TIMEOUT_S if sent_frames > 0 else None
                done, pending = await asyncio.wait(
                    {get_task, barge_task}, return_when=asyncio.FIRST_COMPLETED,
                    timeout=wait_timeout,
                )
                for t in pending:
                    t.cancel()
                if not done:
                    log.warning(
                        "voice: TTS speak idle watchdog tripped after %.1fs "
                        "(sent %d frames); abandoning current utterance so "
                        "queued items can play",
                        SPEAK_IDLE_TIMEOUT_S, sent_frames,
                    )
                    producer.cancel()
                    return
                if barge_task in done:
                    log.info("voice: barge-in after %d frames (%.0fms)",
                             sent_frames, sent_frames * FRAME_MS)
                    producer.cancel()
                    return
                frame = get_task.result()
                if frame is None:
                    log.info("voice: TTS playback complete, sent %d frames (%.0fms)",
                             sent_frames, sent_frames * FRAME_MS)
                    return
                try:
                    await asyncio.wait_for(
                        active.call_py.send_frame(  # type: ignore[union-attr]
                            active.chat_id, self._Device.MICROPHONE, frame,
                        ),
                        timeout=SEND_FRAME_TIMEOUT_S,
                    )
                    sent_frames += 1
                except asyncio.TimeoutError:
                    log.warning(
                        "voice: send_frame timed out after %.1fs at frame %d; "
                        "abandoning current utterance (ntgcalls back-pressure?)",
                        SEND_FRAME_TIMEOUT_S, sent_frames,
                    )
                    producer.cancel()
                    return
                except Exception:
                    log.exception("voice: send_frame failed at frame %d", sent_frames)
                    producer.cancel()
                    return
                await asyncio.sleep(FRAME_MS / 1000)
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass

    async def _tts_producer(self, text: str, out: asyncio.Queue) -> None:
        """POST TTS, decode Opus stream, push 10 ms PCM frames into `out`,
        terminate with None."""
        try:
            opus_bytes = await self._tts_http(text)
            log.info("voice: TTS HTTP returned %d opus bytes", len(opus_bytes))
            pcm = self._opus_decode(opus_bytes)
            log.info("voice: opus decoded to %d PCM bytes (%.0fms @ 48k mono)",
                     len(pcm), len(pcm) / 2 / PCM_RATE * 1000)
            for i in range(0, len(pcm), BYTES_PER_FRAME):
                chunk = pcm[i:i + BYTES_PER_FRAME]
                if len(chunk) < BYTES_PER_FRAME:
                    chunk += b"\x00" * (BYTES_PER_FRAME - len(chunk))
                await out.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice: TTS producer crashed")
        finally:
            try:
                out.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # ---- chat.reply subscriber ----

    async def _chat_reply_subscriber(self, active: _ActiveCall) -> None:
        """Same shape as TelegramAgentService._chat_reply_subscriber, but
        speaks the text instead of sending it as a message.

        Voice-side rule: TTS only real agent speech. Skip anything starting
        with `SYSTEM:` (memory breadcrumbs, no-output placeholders, error
        notices). Those still land in the text chat as receipts."""
        try:
            async for env in self._events.subscribe_global(types={"chat.reply"}):
                if self._active is not active:
                    return
                payload = env.get("payload") or {}
                if payload.get("session_id") != active.session_id:
                    continue
                text = (payload.get("voice_text") or payload.get("text") or "").strip()
                if not text or text.startswith("SYSTEM:"):
                    continue
                await self._enqueue_text(active, text)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice chat.reply subscriber crashed")
            raise

    # ---- approval subscribers / resolver ----

    async def _approval_subscriber(self, active: _ActiveCall) -> None:
        """Mirror of telegram_agent._approval_subscriber but for voice:
        speak the approval prompt and record the challenge phrase so the
        next user utterance can resolve it."""
        from .audit import telegram_log as _audit
        try:
            async for env in self._events.subscribe_global(
                types={"approval.requested"},
            ):
                if self._active is not active:
                    return
                task_id_str = env.get("task_id")
                payload = env.get("payload") or {}
                approval_id = payload.get("approval_id")
                if not (task_id_str and approval_id):
                    continue
                # Scope to our session: the spawning task's
                # `dispatched_by_chat_session` must match our agent session
                # id. We can read it through the operator's DB handle.
                try:
                    task = await self._operator._db.get_task(UUID(task_id_str))
                except Exception:
                    log.exception("voice approval: get_task %s failed", task_id_str)
                    continue
                if task is None or task.dispatched_by_chat_session != active.session_id:
                    continue
                phrase = (payload.get("challenge_phrase") or "").strip()
                if not phrase:
                    continue
                canonical = (payload.get("canonical_command") or "").strip()
                blast = (payload.get("blast_radius") or "").strip()
                active.pending_approvals[str(approval_id)] = phrase
                prompt = await self._build_voice_approval_prompt(
                    canonical=canonical, blast=blast, phrase=phrase,
                )
                await self._enqueue_text(active, prompt)
                # Approvals trump in-flight chitchat. Set barge_in so any
                # current TTS cancels and the approval plays next. (The
                # outbound loop clears barge_in at the start of each _speak,
                # so the approval prompt itself plays normally.)
                if active.is_speaking:
                    active.barge_in.set()
                _audit.info(
                    "voice approval prompt approval=%s task=%s canonical=%r",
                    approval_id, task_id_str, canonical,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice approval subscriber crashed")
            raise

    async def _build_voice_approval_prompt(
        self, *, canonical: str, blast: str, phrase: str,
    ) -> str:
        """Ask the operator's LLM for a short, speakable, localized version
        of the approval request. Strips IDs / paths / byte counts that are
        useful in text chat but noise when spoken.

        Falls back to a generic minimal phrase if the LLM call fails or
        isn't configured."""
        fallback_generic = (
            f"Потрібен дозвіл. Скажи '{phrase}' щоб підтвердити, "
            f"або 'no' щоб відхилити."
            if (self._language or "").lower() == "uk"
            else f"I need your approval. Say '{phrase}' to allow, "
                 f"or 'no' to deny."
        )
        if self._llm is None or not self._llm_model:
            return fallback_generic
        lang_clause = (
            f"Respond in: {self._language}."
            if self._language else
            "Respond in the user's language (infer from the inputs)."
        )
        system = (
            "Convert a security approval request into a short, conversational "
            "single sentence that will be spoken aloud over a voice call. "
            "Rules:\n"
            "- ONE sentence. Plain spoken language.\n"
            "- Drop chat IDs, full file paths, byte counts, mime types. "
            "  Use 'file <basename>' for paths, 'a contact' for chat IDs.\n"
            "- Drop the blast-radius warning entirely (the user sees that "
            "  in their text chat for the full version).\n"
            f"- {lang_clause}\n"
            f"- End with the exact instruction: «Скажи '{phrase}' щоб "
            f"  підтвердити, або 'no' щоб відхилити.» if you're responding "
            f"  in Ukrainian, or the equivalent in the response language. "
            "Use single quotes around the phrase verbatim.\n"
            "Output ONLY the sentence — no preamble, no quotes around it."
        )
        user = (
            f"Action to confirm: {canonical}\n"
            f"Security note (do NOT include in output): {blast}"
        )
        try:
            resp = await self._llm.chat(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=[],
                max_tokens=200,
            )
            text = (resp.get("content") or "").strip()
            if text:
                return text
        except Exception:
            log.exception("voice: approval-prompt paraphrase failed")
        return fallback_generic

    async def _approval_resolved_subscriber(self, active: _ActiveCall) -> None:
        """Drop pending entries once an approval is resolved (by us, by
        text chat, or by timeout) so a stale phrase can't match a later
        utterance."""
        try:
            async for env in self._events.subscribe_global(
                types={"approval.resolved"},
            ):
                if self._active is not active:
                    return
                payload = env.get("payload") or {}
                aid = payload.get("approval_id")
                if aid:
                    active.pending_approvals.pop(str(aid), None)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice approval-resolved subscriber crashed")
            raise

    async def _try_resolve_approval(
        self, active: _ActiveCall, text: str,
    ) -> bool:
        """Same semantics as telegram_agent._try_resolve_approval: bare
        affirm or deny resolves ALL pending approvals in this call.
        Returns True iff at least one approval was resolved (caller
        should NOT forward the text to operator.chat_turn)."""
        if not active.pending_approvals:
            return False
        affirm = phrases_match("", text)
        deny = (not affirm) and is_deny_phrase(text)
        if not (affirm or deny):
            return False
        decision = "allow" if affirm else "deny"
        any_resolved = False
        for approval_id_str in list(active.pending_approvals.keys()):
            try:
                approval_uuid = UUID(approval_id_str)
            except ValueError:
                active.pending_approvals.pop(approval_id_str, None)
                continue
            approved, matched = await self._broker.submit_response(
                approval_id=approval_uuid,
                decision=decision,
                challenge_phrase_supplied=text,
            )
            active.pending_approvals.pop(approval_id_str, None)
            log.info(
                "voice approval resolve approval=%s decision=%s approved=%s matched=%s",
                approval_id_str, decision, approved, matched,
            )
            any_resolved = True
        if any_resolved:
            await self._enqueue_text(
                active, "Approved." if affirm else "Denied.",
            )
        return any_resolved

    # ---- HTTP: STT + TTS ----

    async def _stt(self, pcm_bytes: bytes) -> str:
        """Opus-encode the PCM, POST to STT, return the transcript text."""
        # Telegram/WebRTC's APM (AGC + noise suppression) leaves us with
        # very low-amplitude PCM (peaks ~1% of int16 full-scale in tests).
        # Normalize each utterance so STT has actual signal to work with.
        pcm_bytes = _normalize_pcm(pcm_bytes, target_peak=0.7)
        ogg = _pcm_to_ogg_opus(pcm_bytes, sample_rate=PCM_RATE, channels=PCM_CHANNELS)
        # Debug dump — lets us listen to what we actually send to STT, and
        # inspect the raw PCM bytes to verify the layout coming from py-tgcalls.
        try:
            ts = int(time.time())
            Path(f"/tmp/oncall_stt_{ts}.ogg").write_bytes(ogg)
            Path(f"/tmp/oncall_stt_{ts}.pcm").write_bytes(pcm_bytes)
        except Exception:
            log.exception("debug stt dump failed")
        url = f"{self._stt_base_url}/v1/audio/transcriptions"
        data = {"model": "whisper-1"}
        if self._language:
            data["language"] = self._language
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {self._stt_api_key}"},
                files={"file": ("utt.ogg", ogg, "audio/ogg")},
                data=data,
            )
            r.raise_for_status()
            payload = r.json()
        return payload.get("text", "") if isinstance(payload, dict) else ""

    async def _tts_http(self, text: str) -> bytes:
        url = f"{self._tts_base_url}/v1/audio/speech"
        body = {
            "model": "tts-1",
            "voice": self._tts_voice,
            "input": text,
            "response_format": "opus",
        }
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._tts_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            return r.content

    def _opus_decode(self, ogg_or_opus: bytes) -> bytes:
        """Decode the TTS Opus response to PCM16 @ 48 kHz mono. Handles both
        raw-Opus and Ogg/Opus container responses by sniffing the OggS magic.
        """
        from opuslib import Decoder  # type: ignore[import-not-found]
        if ogg_or_opus[:4] == b"OggS":
            packets = list(_ogg_opus_packets(ogg_or_opus))
        else:
            packets = [ogg_or_opus]
        dec = Decoder(PCM_RATE, PCM_CHANNELS)
        out = bytearray()
        for p in packets:
            try:
                out += dec.decode(p, SAMPLES_PER_FRAME * 6, decode_fec=False)
            except Exception:
                log.exception("opus decode of one packet failed; continuing")
        return bytes(out)

    # ---- teardown ----

    async def _teardown_active(self, active: _ActiveCall, *, reason: str) -> None:
        if self._active is not active:
            return
        self._active = None
        for t in active.tasks:
            if not t.done():
                t.cancel()
        for t in active.tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await active.call_py.leave_call(active.chat_id)  # type: ignore[union-attr]
        except Exception:
            log.debug("leave_call during teardown raised (often benign): ", exc_info=True)
        log.info("voice: call torn down (reason=%s)", reason)
        # Outbound calls to non-owners ran in an ephemeral session — the
        # owner's text chat saw "call placed" and then silence. Brief the
        # owner now so they know whether it connected and what was said.
        # Inbound (owner) calls share the owner's session, so no extra
        # notification is needed — they already saw the conversation.
        if not active.is_owner:
            task = asyncio.create_task(
                self._brief_owner_on_call_end(active, reason),
                name="voice-post-call-brief",
            )
            self._brief_tasks.add(task)
            task.add_done_callback(self._brief_tasks.discard)

    async def _brief_owner_on_call_end(
        self, active: _ActiveCall, reason: str,
    ) -> None:
        """Post a system note to the owner's session with the call's
        transcript and a directive to summarize the outcome. The
        operator's reply lands in the owner's text chat through the
        normal chat-reply pipeline."""
        log.info(
            "voice: post-call brief starting (callee=%s reason=%s)",
            active.callee_label, reason,
        )
        try:
            rows = await self._operator._db.load_chat_history(
                active.session_id, limit=80,
            )
        except Exception:
            log.exception("voice: failed to load post-call transcript")
            rows = []
        # Compact transcript: skip auto-injected memory/system notes;
        # label `user`=callee, `assistant`=you (the agent on the call).
        lines: list[str] = []
        for row in rows:
            role = row.get("role") or ""
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if role == "user" and (
                content.startswith("[memory note") or
                content.startswith("[system note")
            ):
                continue
            if role == "user":
                lines.append(f"  {active.callee_label}: {content}")
            elif role == "assistant":
                lines.append(f"  you (agent): {content}")
            # tool / assistant_tool_calls rows aren't speech — skip.
        transcript = "\n".join(lines) if lines else "  (no spoken exchange captured)"
        connected = active.connected_at is not None
        connected_clause = (
            "call CONNECTED and ran to completion" if connected and reason == "left-call"
            else f"call CONNECTED but ended with reason={reason!r}" if connected
            else f"call DID NOT CONNECT (reason={reason!r}, no one picked up)"
        )
        note = (
            f"the outbound voice call you placed to {active.callee_label} "
            f"just ended. {connected_clause}. original reason for the call: "
            f"{active.reason!r}. brief the owner in ONE short line: did it "
            f"connect, what was the outcome, anything actionable. don't "
            f"recite the transcript verbatim — summarize.\n\n"
            f"transcript (oldest first; '{active.callee_label}' is what they "
            f"said, 'you (agent)' is what was spoken on your side):\n"
            f"{transcript}"
        )
        try:
            result = await self._operator.auto_ping(
                session_id=self._owner_session_id, note=note,
            )
        except Exception:
            log.exception(
                "voice: failed to brief owner about call end (callee=%s)",
                active.callee_label,
            )
            return
        # auto_ping persists the assistant turn to chat history, but it
        # doesn't fire `chat.reply` — that's the caller's responsibility
        # (cf. dispatch_denied in operator.py). Without this publish, the
        # telegram_agent subscriber never wakes and the owner sees only
        # the original "call placed" ack and silence.
        text = (result.text or "").strip() if result else ""
        if text:
            try:
                await self._events.publish_global("chat.reply", {
                    "session_id": self._owner_session_id,
                    "text": text,
                    "voice_text": text,
                    "trigger": "voice.call_ended",
                    "task_id": None,
                })
                log.info(
                    "voice: post-call brief delivered (callee=%s, reply_len=%d)",
                    active.callee_label, len(text),
                )
            except Exception:
                log.exception("voice: failed to publish post-call chat.reply")
        else:
            log.info(
                "voice: post-call brief generated no text (callee=%s)",
                active.callee_label,
            )

    def _on_call_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        log.error("voice: call task %s exited with: %r", task.get_name(), exc)
        active = self._active
        if active is not None and task in active.tasks:
            asyncio.create_task(
                self._teardown_active(active, reason=f"task-crash:{task.get_name()}"),
            )


# ---- module helpers ----

def _normalize_pcm(pcm: bytes, *, target_peak: float = 0.7) -> bytes:
    """Peak-normalize PCM16 mono so the loudest sample hits target_peak * INT16_MAX.
    No-op when peak is already ≥target_peak. Caps gain at 64× to avoid amplifying
    pure silence into noise."""
    peak = audioop.max(pcm, 2)
    target = int(target_peak * 32767)
    if peak == 0 or peak >= target:
        return pcm
    gain = min(target / peak, 64.0)
    return audioop.mul(pcm, 2, gain)


def _pcm_to_ogg_opus(
    pcm: bytes, *, sample_rate: int = PCM_RATE, channels: int = PCM_CHANNELS,
) -> bytes:
    """Encode PCM16 mono to an Ogg/Opus container suitable for upload to an
    OpenAI-compatible audio.transcriptions endpoint. Frames at 20 ms (the
    encoder's recommended packet size for speech)."""
    from opuslib import Encoder, APPLICATION_VOIP  # type: ignore[import-not-found]
    enc = Encoder(sample_rate, channels, APPLICATION_VOIP)
    enc_frame_samples = sample_rate * 20 // 1000  # 20 ms
    enc_frame_bytes = enc_frame_samples * 2 * channels
    packets: list[bytes] = []
    granules: list[int] = []
    running = 0
    for i in range(0, len(pcm), enc_frame_bytes):
        chunk = pcm[i:i + enc_frame_bytes]
        if len(chunk) < enc_frame_bytes:
            chunk += b"\x00" * (enc_frame_bytes - len(chunk))
        packets.append(enc.encode(chunk, enc_frame_samples))
        running += enc_frame_samples
        granules.append(running)
    return _ogg_opus_write(packets, granules, sample_rate=sample_rate, channels=channels)


def _ogg_opus_write(
    packets: list[bytes], granules: list[int], *,
    sample_rate: int, channels: int,
) -> bytes:
    """Wrap a list of Opus packets into a minimal Ogg container with the
    OpusHead + OpusTags pages, per RFC 7845."""
    import os
    serial = struct.unpack("<I", os.urandom(4))[0]
    head = b"OpusHead" + struct.pack(
        "<BBHIHB",
        1,                 # version
        channels,
        3840,              # pre-skip (recommended for 48k)
        sample_rate,
        0,                 # output gain (Q7.8)
        0,                 # mapping family
    )
    tags = b"OpusTags" + struct.pack("<I", 8) + b"oncall00" + struct.pack("<I", 0)
    pages: list[bytes] = []
    pages.append(_ogg_page(serial, 0, 0, head, header_type=0x02))   # BOS
    pages.append(_ogg_page(serial, 1, 0, tags, header_type=0x00))
    seq = 2
    for i, (pkt, gp) in enumerate(zip(packets, granules, strict=False)):
        header_type = 0x04 if i == len(packets) - 1 else 0x00
        pages.append(_ogg_page(serial, seq, gp, pkt, header_type=header_type))
        seq += 1
    return b"".join(pages)


def _ogg_page(
    serial: int, seq: int, granule: int, payload: bytes, *, header_type: int,
) -> bytes:
    """Build one Ogg page. Single-packet pages only — fine for our packet
    sizes (≤255 bytes per Opus frame at our bitrate)."""
    segments: list[int] = []
    remaining = len(payload)
    while remaining > 255:
        segments.append(255)
        remaining -= 255
    segments.append(remaining)
    if not segments:
        segments = [0]
    header = (
        b"OggS"
        + struct.pack("<B", 0)
        + struct.pack("<B", header_type)
        + struct.pack("<q", granule)
        + struct.pack("<I", serial)
        + struct.pack("<I", seq)
        + struct.pack("<I", 0)            # CRC placeholder
        + struct.pack("<B", len(segments))
        + bytes(segments)
    )
    page = header + payload
    crc = _ogg_crc32(page)
    return page[:22] + struct.pack("<I", crc) + page[26:]


_OGG_CRC_TABLE = None


def _ogg_crc32(data: bytes) -> int:
    """Ogg's variant of CRC-32 (polynomial 0x04c11db7, no input/output XOR
    or reflection). Hot-path-cached table.
    """
    global _OGG_CRC_TABLE
    if _OGG_CRC_TABLE is None:
        table = [0] * 256
        for i in range(256):
            r = i << 24
            for _ in range(8):
                r = ((r << 1) ^ 0x04c11db7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
            table[i] = r
        _OGG_CRC_TABLE = table
    table = _OGG_CRC_TABLE
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
    return crc


def _ogg_opus_packets(ogg: bytes):
    """Iterate raw Opus packets out of an Ogg/Opus container. Skips the
    OpusHead + OpusTags pages and yields each audio packet."""
    i = 0
    page_n = 0
    while i + 27 <= len(ogg):
        if ogg[i:i + 4] != b"OggS":
            return
        n_segs = ogg[i + 26]
        seg_table = ogg[i + 27:i + 27 + n_segs]
        body_start = i + 27 + n_segs
        # Reassemble packets across continuing segments (segments of size 255
        # signal "more to come"; <255 ends the packet).
        packet = bytearray()
        cursor = body_start
        for s in seg_table:
            packet += ogg[cursor:cursor + s]
            cursor += s
            if s < 255:
                if page_n >= 2:  # skip OpusHead (0) and OpusTags (1)
                    yield bytes(packet)
                packet = bytearray()
        i = cursor
        page_n += 1
