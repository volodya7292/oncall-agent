"""Spike: can pytgcalls + telethon INITIATE an outbound 1:1 voice call?

Companion to scripts/spike_pytgcalls_p2p.py (which proved the *inbound*
path). This script proves — or disproves — that calling `PyTgCalls.play()`
cold on a chat_id with no prior INCOMING_CALL event will trigger an MTProto
phone.requestCall and make the target's Telegram client ring.

If it works: the outbound-call plan stands as written.
If it doesn't: fall back to telethon raw MTProto (functions.phone.RequestCallRequest)
+ ntgcalls lower-level API. Plan changes accordingly.

PREREQ — stop the daemon (it holds the agent .session file open):
    oncall service stop

Run:
    TARGET_USER_ID=<int> uv run python scripts/spike_pytgcalls_outbound.py

Expected on success:
  - Target user's Telegram client rings within ~2 seconds.
  - When they pick up, you (running this script) hear a 440 Hz sine wave.
  - When they hang up, LEFT_CALL fires in the log.

Expected on failure modes:
  - play() raises immediately → outbound is unsupported via this surface.
  - play() returns but no ring on target → silent failure, also unsupported.
  - Any MTProto error from ntgcalls → log it; that's the fallback signal.

Ctrl-C to stop. When done:
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
log = logging.getLogger("spike-out")

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
        "Install with: uv pip install py-tgcalls",
    )


API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
SESSION = str(
    Path(
        os.environ.get(
            "TELEGRAM_AGENT_SESSION_PATH",
            str(Path.home() / ".oncall" / "telegram_agent.session"),
        ),
    ).expanduser(),
)
try:
    TARGET = int(os.environ["TARGET_USER_ID"])
except KeyError:
    sys.exit("set TARGET_USER_ID=<int> in env before running")

SAMPLE_RATE = 48_000
CHANNELS = 1
FRAME_MS = 10
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480


def sine_frame(freq_hz: float, phase: float) -> tuple[bytes, float]:
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
            "first, or stop the running daemon if it has the file open.",
        )
    me = await client.get_me()
    log.info(
        "logged in as agent user_id=%s username=%s",
        me.id, getattr(me, "username", None),
    )
    log.info("will attempt outbound call to TARGET=%s", TARGET)

    call_py = PyTgCalls(client)
    await call_py.start()
    log.info("pytgcalls started (bind=telethon)")

    active = {"on": False}
    sine_phase = {"v": 0.0}

    @call_py.on_update(fl.chat_update(ChatUpdate.Status.LEFT_CALL))
    async def on_left(_: PyTgCalls, update: ChatUpdate) -> None:
        log.info("<<< LEFT_CALL chat_id=%s — call ended", update.chat_id)
        active["on"] = False

    @call_py.on_update(fl.stream_frame(Direction.INCOMING, Device.MICROPHONE))
    async def on_inbound_frames(_: PyTgCalls, update: StreamFrames) -> None:
        try:
            n = len(update.frames)
            first_len = len(update.frames[0].frame) if n else 0
            log.info("<-- inbound frames n=%d first_len=%d", n, first_len)
        except Exception:
            log.exception("inbound-frame handler crashed")

    # ---- the actual experiment ----
    params = AudioParameters(SAMPLE_RATE, CHANNELS)
    log.info("calling play(TARGET=%s, MediaStream(...)) — this is the test.", TARGET)
    try:
        await call_py.play(TARGET, MediaStream(ExternalMedia.AUDIO, params))
        log.info("play() returned without raising — checking if target rings")
        active["on"] = True
    except Exception:
        log.exception(
            "play() raised — outbound initiation NOT supported via this surface. "
            "Fallback plan: raw MTProto phone.RequestCallRequest.",
        )
        await client.disconnect()
        return

    try:
        await call_py.record(
            TARGET, RecordStream(audio=True, audio_parameters=params),
        )
        log.info("record() started — will print inbound frames when target speaks")
    except Exception:
        log.exception("record() failed — inbound audio won't be observable")

    async def sine_loop() -> None:
        while active["on"]:
            try:
                frame, sine_phase["v"] = sine_frame(440.0, sine_phase["v"])
                await call_py.send_frame(TARGET, Device.MICROPHONE, frame)
                await asyncio.sleep(FRAME_MS / 1000)
            except Exception:
                log.exception("sine_loop send_frame failed")
                await asyncio.sleep(0.5)

    asyncio.create_task(sine_loop())

    log.info(
        "watching for 60s. target should be ringing now — pick up to hear sine, "
        "or hang up / let it ring out. Ctrl-C also fine.",
    )
    try:
        await asyncio.sleep(60)
    finally:
        try:
            await call_py.leave_call(TARGET)
            log.info("leave_call sent")
        except Exception:
            log.debug("leave_call raised", exc_info=True)
        await client.disconnect()
        log.info("done")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("interrupted")
