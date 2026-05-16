"""TelegramService tests with a fake TelegramClientLike.

These cover:
  * NewMessage handler writes inbound DM to messenger_inbox with correct fields
    and triages importance (sender allowlist + keyword match).
  * Private filter rejects non-private events.
  * Duplicate (chat_id, message_id) doesn't insert twice.
  * `get_chat_style` reads only the user's OWN outgoing messages.
  * `send` calls send_message and returns the new message id.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from oncall.db import Database
from oncall.telegram_service import TelegramService


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTelegramClient:
    """In-memory telethon-shaped fake."""

    def __init__(
        self,
        *,
        outgoing: dict[int, list[dict[str, Any]]] | None = None,
        archived_chat_ids: set[int] | None = None,
    ) -> None:
        self.connected = False
        self.disconnected = False
        self.authorized = True
        self.handler: Any = None
        self.event_filter: Any = None
        self.sent: list[dict[str, Any]] = []
        # outgoing[chat_id_int] = [{"id": int, "message": str, "date": dt}]
        self._outgoing = outgoing or {}
        self.archived_chat_ids: set[int] = set(archived_chat_ids or set())
        self.iter_dialogs_calls: int = 0

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def is_user_authorized(self) -> bool:
        return self.authorized

    def add_event_handler(self, callback, event=None) -> None:
        self.handler = callback
        self.event_filter = event

    def remove_event_handler(self, callback, event=None) -> None:
        self.handler = None
        self.event_filter = None

    async def send_message(self, entity, message):
        msg_id = 1000 + len(self.sent)
        self.sent.append({"entity": entity, "message": message, "id": msg_id})
        return SimpleNamespace(id=msg_id)

    def iter_messages(self, entity, *, limit=20, from_user=None, **kwargs):
        chat_id = entity if isinstance(entity, int) else 0
        msgs = list(self._outgoing.get(chat_id, []))
        # Honor `from_user='me'` semantics: we only return what's stored under outgoing.
        return _async_iter([
            SimpleNamespace(id=m["id"], message=m["message"], date=m["date"])
            for m in msgs[:limit]
        ])

    def iter_dialogs(self, *, archived: bool | None = None, **kwargs):
        self.iter_dialogs_calls += 1
        if archived is True:
            dialogs = [
                SimpleNamespace(id=cid, archived=True) for cid in self.archived_chat_ids
            ]
        elif archived is False:
            dialogs = []  # not used in our code path
        else:
            dialogs = [SimpleNamespace(id=cid, archived=True) for cid in self.archived_chat_ids]
        return _async_iter(dialogs)


def _async_iter(items):
    async def gen():
        for it in items:
            yield it
    return gen()


def make_event(
    *, sender_username: str | None, body: str, is_private: bool = True,
    chat_id: int = 12345, message_id: int = 42, is_bot: bool = False,
    is_self: bool = False, out: bool = False,
) -> Any:
    sender = SimpleNamespace(
        username=sender_username, first_name="Alex", last_name=None,
        bot=is_bot, is_self=is_self,
    )
    message = SimpleNamespace(
        message=body, id=message_id, out=out,
        date=datetime.now(timezone.utc), chat_id=chat_id,
    )

    async def get_sender():
        return sender

    return SimpleNamespace(
        is_private=is_private, chat_id=chat_id, message=message,
        get_sender=get_sender, raw_text=body,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def db(tmp_path):
    d = Database(tmp_path / "tg.sqlite")
    await d.connect()
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
async def service(db):
    client = FakeTelegramClient()
    s = TelegramService(
        db=db, client=client,
        important_senders={"alex", "boss"},
        important_keywords={"urgent", "down"},
    )
    await s.start()
    try:
        yield s, client
    finally:
        await s.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inbound_dm_written_to_inbox(service, db):
    s, client = service
    event = make_event(sender_username="someone", body="hey, free tonight?")
    await client.handler(event)

    rows = await db.list_inbox(unread_only=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["body"] == "hey, free tonight?"
    assert row["chat_id"] == "12345"
    assert row["sender_username"] == "someone"
    assert row["is_important"] is False


@pytest.mark.asyncio
async def test_inbound_dm_triaged_important_by_sender(service, db):
    s, client = service
    event = make_event(sender_username="alex", body="just saying hi")
    await client.handler(event)
    rows = await db.list_inbox()
    assert rows[0]["is_important"] is True


@pytest.mark.asyncio
async def test_inbound_dm_triaged_important_by_keyword(service, db):
    s, client = service
    event = make_event(sender_username="rando", body="staging is DOWN")
    await client.handler(event)
    rows = await db.list_inbox()
    assert rows[0]["is_important"] is True


@pytest.mark.asyncio
async def test_non_private_chat_ignored(service, db):
    s, client = service
    event = make_event(sender_username="bot", body="ad spam", is_private=False)
    await client.handler(event)
    rows = await db.list_inbox()
    assert rows == []


@pytest.mark.asyncio
async def test_outgoing_self_message_ignored(service, db):
    s, client = service
    event = make_event(sender_username="me", body="my own msg", out=True, is_self=True)
    await client.handler(event)
    assert await db.list_inbox() == []


@pytest.mark.asyncio
async def test_bot_messages_ignored(service, db):
    s, client = service
    event = make_event(sender_username="some_bot", body="ping", is_bot=True)
    await client.handler(event)
    assert await db.list_inbox() == []


@pytest.mark.asyncio
async def test_duplicate_message_dedup(service, db):
    s, client = service
    e1 = make_event(sender_username="x", body="first", message_id=99)
    await client.handler(e1)
    # Same chat/message_id arrives again — must NOT create a second row.
    await client.handler(e1)
    rows = await db.list_inbox()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_get_chat_style_returns_user_outgoing(db):
    """Style samples must be the USER'S own messages, not the counterparty's."""
    client = FakeTelegramClient(outgoing={
        12345: [
            {"id": 5, "message": "ало", "date": datetime.now(timezone.utc)},
            {"id": 6, "message": "ща приду", "date": datetime.now(timezone.utc)},
            {"id": 7, "message": "+1", "date": datetime.now(timezone.utc)},
        ],
    })
    s = TelegramService(db=db, client=client, important_senders=set(), important_keywords=set())
    await s.start()
    try:
        samples = await s.get_chat_style("12345", limit=10)
    finally:
        await s.stop()
    texts = [m["text"] for m in samples]
    assert texts == ["ало", "ща приду", "+1"]


@pytest.mark.asyncio
async def test_send_calls_underlying_client(db):
    client = FakeTelegramClient()
    s = TelegramService(db=db, client=client, important_senders=set(), important_keywords=set())
    await s.start()
    try:
        out = await s.send("12345", "draft reply")
    finally:
        await s.stop()
    assert client.sent and client.sent[0]["message"] == "draft reply"
    assert client.sent[0]["entity"] == 12345  # coerced to int
    assert out["message_id"] == str(client.sent[0]["id"])


@pytest.mark.asyncio
async def test_mark_read_flag(service, db):
    s, client = service
    await client.handler(make_event(sender_username="x", body="hello", message_id=1))
    rows = await db.list_inbox(unread_only=True)
    assert len(rows) == 1
    inbox_id = rows[0]["id"]
    assert await s.mark_read(inbox_id) is True
    assert await db.list_inbox(unread_only=True) == []
    # Idempotent: second mark_read on an already-read row returns False
    assert await s.mark_read(inbox_id) is False


# ---- archived-chat filtering ----

@pytest.mark.asyncio
async def test_archived_chat_inbound_dropped(db):
    """DMs from a Telegram-archived chat must not be written to the inbox."""
    client = FakeTelegramClient(archived_chat_ids={777})
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        await client.handler(make_event(sender_username="ghost", body="boo", chat_id=777))
        await client.handler(make_event(sender_username="normal", body="hi", chat_id=12345))
    finally:
        await s.stop()
    rows = await db.list_inbox(unread_only=True)
    assert len(rows) == 1
    assert rows[0]["chat_id"] == "12345"


@pytest.mark.asyncio
async def test_list_inbox_filters_archived_after_the_fact(db):
    """A DM persisted before the chat was archived should disappear from
    list_inbox once the archived cache picks up the new state."""
    client = FakeTelegramClient()  # no archived chats yet
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
        archived_cache_ttl=0.0,  # always refresh on access
    )
    await s.start()
    try:
        await client.handler(make_event(sender_username="a", body="msg-a", chat_id=111))
        await client.handler(make_event(sender_username="b", body="msg-b", chat_id=222))
        # Both visible.
        assert {r["chat_id"] for r in await s.list_inbox()} == {"111", "222"}
        # User archives chat 111 in Telegram afterwards.
        client.archived_chat_ids.add(111)
        visible = await s.list_inbox()
        assert {r["chat_id"] for r in visible} == {"222"}
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_archived_cache_ttl_avoids_excessive_refresh(db):
    """Within the TTL window we should NOT keep hitting iter_dialogs on every
    list_inbox call — that'd burn Telegram API quota."""
    client = FakeTelegramClient(archived_chat_ids={999})
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
        archived_cache_ttl=60.0,
    )
    await s.start()  # primes the cache (one iter_dialogs call)
    try:
        for _ in range(5):
            await s.list_inbox()
        # 1 call on start + 0 on subsequent reads (TTL not expired).
        assert client.iter_dialogs_calls == 1
    finally:
        await s.stop()
