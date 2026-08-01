"""Telegram album coalescing.

Telegram has no "one message with N photos" on the wire — an album arrives as
N separate messages sharing a grouped_id, with the caption on only one of
them. Handled as they land, one album became N operator turns: N hand_offs, N
executor tasks, N answers to a question the user asked once. Observed live
with a 2-photo album where the caption rode on the second part, so the first
turn answered a photo it had no question for.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from oncall.telegram_agent import TelegramAgentService


OWNER = 4242


class _RecordingOperator:
    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    async def chat_turn(self, *, session_id, user_text, attachments=None, **kw):
        self.turns.append({"text": user_text, "attachments": attachments or []})
        return SimpleNamespace(text="ok", tool_calls_made=[], user_facing_text=lambda: "ok")


def _agent(operator) -> TelegramAgentService:
    client = SimpleNamespace(
        send_read_acknowledge=_noop, send_message=_noop, get_me=_noop,
    )
    svc = TelegramAgentService(
        client=client, operator=operator,
        events=SimpleNamespace(publish_global=_noop),
        owner_user_id=OWNER, broker=object(), db=object(),
    )
    svc._ALBUM_SETTLE_SECONDS = 0.05  # keep the test fast
    return svc


async def _noop(*a, **kw):
    return None


def _event(*, caption: str, grouped_id: int | None, payload: bytes):
    message = SimpleNamespace(
        message=caption,
        media=object(),
        reply_to=None,
        grouped_id=grouped_id,
        file=SimpleNamespace(mime_type="image/jpeg", name="photo.jpg"),
        download_media=lambda file=None, _p=payload: _resolved(_p),
    )
    return SimpleNamespace(
        is_private=True, chat_id=OWNER, message=message,
        get_sender=lambda: _resolved(SimpleNamespace(id=OWNER, username="owner")),
    )


def _resolved(value):
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    fut.set_result(value)
    return fut


@pytest.mark.asyncio
async def test_album_parts_collapse_into_one_turn(tmp_path, monkeypatch):
    """Two parts of one album → exactly ONE operator turn carrying both
    images, with the caption preserved wherever in the group it rode."""
    monkeypatch.setattr(
        "oncall.telegram_agent._persist_inbound_attachment",
        lambda data, fname: tmp_path / fname,
    )
    op = _RecordingOperator()
    svc = _agent(op)

    # Caption on the SECOND part — the live ordering that made the first
    # turn answer a photo with no question attached to it.
    await svc._handle_inbound(_event(caption="", grouped_id=77, payload=b"a"))
    await svc._handle_inbound(_event(caption="Що за крутилка", grouped_id=77, payload=b"bb"))
    assert op.turns == [], "must not answer before the album settles"

    await asyncio.sleep(0.2)

    assert len(op.turns) == 1, f"one album must be one turn, got {len(op.turns)}"
    turn = op.turns[0]
    assert len(turn["attachments"]) == 2, "both images must reach the operator"
    assert "Що за крутилка" in turn["text"]


@pytest.mark.asyncio
async def test_a_later_part_extends_the_window(tmp_path, monkeypatch):
    """A part arriving while the timer is running restarts it, so a slowly
    delivered album is still answered whole rather than split in two."""
    monkeypatch.setattr(
        "oncall.telegram_agent._persist_inbound_attachment",
        lambda data, fname: tmp_path / fname,
    )
    op = _RecordingOperator()
    svc = _agent(op)

    await svc._handle_inbound(_event(caption="look", grouped_id=9, payload=b"a"))
    await asyncio.sleep(0.04)  # inside the settle window
    await svc._handle_inbound(_event(caption="", grouped_id=9, payload=b"b"))
    await asyncio.sleep(0.04)
    await svc._handle_inbound(_event(caption="", grouped_id=9, payload=b"c"))
    await asyncio.sleep(0.2)

    assert len(op.turns) == 1
    assert len(op.turns[0]["attachments"]) == 3


@pytest.mark.asyncio
async def test_ungrouped_message_still_answers_immediately(tmp_path, monkeypatch):
    """No grouped_id → the settle delay must not apply. A plain photo is the
    common case and must not pay the album latency."""
    monkeypatch.setattr(
        "oncall.telegram_agent._persist_inbound_attachment",
        lambda data, fname: tmp_path / fname,
    )
    op = _RecordingOperator()
    svc = _agent(op)

    await svc._handle_inbound(_event(caption="what is this", grouped_id=None, payload=b"a"))

    assert len(op.turns) == 1, "a lone message must not wait on the album timer"
    assert len(op.turns[0]["attachments"]) == 1
