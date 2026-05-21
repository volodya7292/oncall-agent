"""Spike: can pytgcalls + telethon receive a 1:1 private voice call and expose
raw PCM frames bidirectionally?

This is the prerequisite check for voice-call support (see the voice-call
plan). Run it on the daemon host with the existing agent userbot session,
then place a 1:1 call from the OWNER's primary Telegram account to the
agent account. Watch the log.

Expected on a successful spike:
  - INCOMING_CALL handler fires when the owner dials.
  - `stream_frame` filter delivers PCM16L chunks (audio you say).
  - A 440 Hz sine wave plays back into the call (you hear it on the
    primary device).

If any of those three doesn't happen, the plan's fallback applies:
swap telethon for a pyrogram client bound to the same session.

Install (one-off):
    uv pip install py-tgcalls

The running daemon holds the agent .session file open. Stop it first:
    oncall service stop

Run:
    uv run python scripts/spike_pytgcalls_p2p.py

Then call the agent account from your primary phone. Hang up + Ctrl-C to
stop the script. When done:
    oncall service start
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
import sys
from pathlib import Path

ENV_PATH = Path.home() / ".oncall" / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("spike")

try:
    from telethon import TelegramClient
except ImportError as exc:
    sys.exit(f"telethon not importable: {exc}")

try:
    from pytgcalls import PyTgCalls, filters as fl  # type: ignore[import-not-found]
    from pytgcalls.types import (  # type: ignore[import-not-found]
        ChatUpdate,
        Device,
        Direction,
        ExternalMedia,
        MediaStream,
        RecordStream,
        StreamFrames,
    )
    from pytgcalls.types.raw.audio_parameters import AudioParameters  # type: ignore[import-not-found]
except ImportError as exc:
    sys.exit(
        f"pytgcalls not importable: {exc}\n"
        "Install with: uv pip install py-tgcalls"
    )


API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
SESSION = str(
    Path(
        os.environ.get(
            "TELEGRAM_AGENT_SESSION_PATH",
            str(Path.home() / ".oncall" / "telegram_agent.session"),
        )
    ).expanduser()
)
OWNER_ID = int(os.environ["TELEGRAM_OWNER_USER_ID"])

SAMPLE_RATE = 48_000
CHANNELS = 1
FRAME_MS = 10  # pytgcalls 2.2.x delivers 10ms PCM frames on the receive side
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480


def sine_frame(freq_hz: float, phase: float) -> tuple[bytes, float]:
    """Generate 20 ms of PCM16L sine at the given frequency."""
    samples = []
    for i in range(SAMPLES_PER_FRAME):
        t = phase + i / SAMPLE_RATE
        v = int(0.25 * 32767 * math.sin(2 * math.pi * freq_hz * t))
        samples.append(v)
    next_phase = phase + SAMPLES_PER_FRAME / SAMPLE_RATE
    return struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples), next_phase


async def main() -> None:
    log.info("loading agent session: %s", SESSION)
    client = TelegramClient(SESSION.removesuffix(".session"), API_ID, API_HASH)
    await client.connect()
    if not await client.is_user_authorized():
        sys.exit(
            "agent session is not authorized. run `oncall telegram-login --agent` "
            "first, or stop the running daemon if it has the file open."
        )
    me = await client.get_me()
    log.info("logged in as agent user_id=%s username=%s", me.id, getattr(me, "username", None))
    log.info("expecting incoming calls from owner_id=%s", OWNER_ID)

    call_py = PyTgCalls(client)
    await call_py.start()
    log.info("pytgcalls started — bind=telethon. waiting for INCOMING_CALL…")

    active_chat: dict[str, int | None] = {"chat_id": None}
    sine_phase = {"v": 0.0}

    @call_py.on_update(fl.chat_update(ChatUpdate.Status.INCOMING_CALL))
    async def on_incoming(_: PyTgCalls, update: ChatUpdate) -> None:
        caller = update.chat_id
        log.info(">>> INCOMING_CALL chat_id=%s", caller)
        if caller != OWNER_ID:
            log.warning("not owner; would reject in prod. accepting anyway for spike.")
        try:
            await call_py.play(
                caller,
                MediaStream(ExternalMedia.AUDIO, AudioParameters(SAMPLE_RATE, CHANNELS)),
            )
            log.info("call accepted, outbound stream prepared")
            active_chat["chat_id"] = caller
        except Exception:
            log.exception("play() failed")
            return
        try:
            await call_py.record(
                caller,
                RecordStream(audio=True, audio_parameters=AudioParameters(SAMPLE_RATE, CHANNELS)),
            )
            log.info("inbound recording started")
        except Exception:
            log.exception("record() failed — inbound frames won't arrive but outbound sine should still play")

    @call_py.on_update(fl.stream_frame(Direction.INCOMING, Device.MICROPHONE))
    async def on_inbound_frames(_: PyTgCalls, update: StreamFrames) -> None:
        try:
            n = len(update.frames)
            first = update.frames[0].frame if n else b""
            log.info("<-- inbound frames n=%d first_len=%d", n, len(first))
        except Exception:
            log.exception("inbound-frame handler crashed")

    @call_py.on_update(fl.chat_update(ChatUpdate.Status.LEFT_CALL))
    async def on_left(_: PyTgCalls, update: ChatUpdate) -> None:
        log.info("<<< LEFT_CALL chat_id=%s — tearing down", update.chat_id)
        active_chat["chat_id"] = None

    async def sine_loop() -> None:
        while True:
            cid = active_chat["chat_id"]
            if cid is None:
                await asyncio.sleep(0.1)
                continue
            try:
                frame, sine_phase["v"] = sine_frame(440.0, sine_phase["v"])
                await call_py.send_frame(cid, Device.MICROPHONE, frame)
                await asyncio.sleep(FRAME_MS / 1000)
            except Exception:
                log.exception("sine_loop send_frame failed")
                await asyncio.sleep(0.5)

    asyncio.create_task(sine_loop())

    log.info("ready. place a 1:1 call from your primary account to the agent now.")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
