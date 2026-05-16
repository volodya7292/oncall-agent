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
        messages: dict[int, list[dict[str, Any]]] | None = None,
        dialogs: list[dict[str, Any]] | None = None,
        contact_search_users: list[dict[str, Any]] | None = None,
    ) -> None:
        self.connected = False
        self.disconnected = False
        self.authorized = True
        self.handler: Any = None
        self.event_filter: Any = None
        self.sent: list[dict[str, Any]] = []
        # outgoing[chat_id_int] = [{"id": int, "message": str, "date": dt}]
        # — only the user's own outgoing, used by get_chat_style tests.
        self._outgoing = outgoing or {}
        # messages[chat_id_int] = [{"id": int, "message": str, "date": dt,
        #                           "out": bool, "sender": SimpleNamespace}]
        # — both sides, used by get_chat_history tests.
        self._messages = messages or {}
        # dialogs: full dialog list with name/username/etc, used by
        # search_chats tests.
        self._dialogs = dialogs or []
        self.archived_chat_ids: set[int] = set(archived_chat_ids or set())
        self.iter_dialogs_calls: int = 0
        # contact_search_users: rows the fake returns when the service issues
        # a contacts.SearchRequest via `client(request)`.
        self._contact_search_users = contact_search_users or []
        self.contact_search_calls: list[str] = []

    async def __call__(self, request):  # telethon-style raw TL dispatch
        q = getattr(request, "q", None)
        if q is not None:
            self.contact_search_calls.append(q)
            users = [
                SimpleNamespace(
                    id=u["id"],
                    first_name=u.get("first_name", ""),
                    last_name=u.get("last_name", ""),
                    username=u.get("username"),
                )
                for u in self._contact_search_users
            ]
            return SimpleNamespace(users=users, chats=[])
        return SimpleNamespace(users=[], chats=[])

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

    def iter_messages(self, entity, *, limit=20, from_user=None, search=None, **kwargs):
        chat_id = entity if isinstance(entity, int) else 0
        if from_user == "me":
            # outgoing-only path (used by get_chat_style).
            msgs = list(self._outgoing.get(chat_id, []))
            return _async_iter([
                SimpleNamespace(id=m["id"], message=m["message"], date=m["date"])
                for m in msgs[:limit]
            ])
        # Both-sides path (get_chat_history + search_messages). Each entry must
        # support .out and .get_sender() for the service to extract direction
        # + sender.
        msgs = list(self._messages.get(chat_id, []))
        if search is not None:
            needle = search.lower()
            msgs = [m for m in msgs if needle in str(m.get("message", "")).lower()]

        def _make(m):
            sender = m.get("sender") or SimpleNamespace(username=None, first_name="", last_name="")

            async def _get_sender():
                return sender

            return SimpleNamespace(
                id=m["id"], message=m["message"], date=m["date"],
                out=bool(m.get("out", False)), get_sender=_get_sender,
            )

        return _async_iter([_make(m) for m in msgs[:limit]])

    def iter_dialogs(self, *, archived: bool | None = None, **kwargs):
        self.iter_dialogs_calls += 1
        if self._dialogs:
            # Full dialog list provided — used by search_chats tests. Filter
            # by archived if requested.
            dlgs = []
            for d in self._dialogs:
                if archived is True and not d.get("archived"):
                    continue
                if archived is False and d.get("archived"):
                    continue
                dlgs.append(SimpleNamespace(
                    id=d["id"], name=d.get("name", ""),
                    entity=SimpleNamespace(username=d.get("username"), id=d["id"]),
                    is_user=d.get("is_user", True),
                    is_group=d.get("is_group", False),
                    is_channel=d.get("is_channel", False),
                    unread_count=d.get("unread_count", 0),
                    archived=d.get("archived", False),
                ))
            return _async_iter(dlgs)
        # Legacy archived-only path for the existing archived-cache tests.
        if archived is True:
            dialogs = [
                SimpleNamespace(id=cid, archived=True) for cid in self.archived_chat_ids
            ]
        elif archived is False:
            dialogs = []
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
    is_self: bool = False, out: bool = False, sender_id: int = 7777,
) -> Any:
    sender = SimpleNamespace(
        id=sender_id, username=sender_username,
        first_name="Alex", last_name=None,
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


# ---------------------------------------------------------------------------
# Ignore filter — drops senders the user doesn't want in the inbox
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inbound_from_ignored_username_is_dropped(db):
    """The own-bot username is added to the ignore set so its outbound
    replies don't re-enter the userbot's inbox. Verify by username."""
    client = FakeTelegramClient()
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
        ignore_usernames={"my_own_bot"},
    )
    await s.start()
    try:
        await client.handler(make_event(
            sender_username="my_own_bot", body="auto-reply from my bot",
        ))
        # Inbox should be empty — the message was filtered.
        rows = await db.list_inbox(unread_only=False)
        assert rows == []
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_inbound_from_ignored_user_id_is_dropped(db):
    """Auto-registration of the own bot uses user_id (which is captured from
    Bot API getMe). Verify by id."""
    client = FakeTelegramClient()
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        # Simulate runtime registration: bot's user_id added after startup.
        s.add_ignore_user_id(424242)
        await client.handler(make_event(
            sender_username=None, sender_id=424242, body="reply",
        ))
        assert await db.list_inbox(unread_only=False) == []
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_get_chat_history_returns_both_sides(db):
    """read_chat: returns last N messages regardless of direction, with the
    `outgoing` flag set per row."""
    now = datetime.now(timezone.utc)
    me = SimpleNamespace(username="me_user", first_name="V", last_name=None)
    them = SimpleNamespace(username="alex_smith", first_name="Alex", last_name=None)
    client = FakeTelegramClient(messages={
        42: [
            {"id": 3, "message": "see you tomorrow", "date": now, "out": True, "sender": me},
            {"id": 2, "message": "sounds good", "date": now, "out": False, "sender": them},
            {"id": 1, "message": "are we still on for 9am?", "date": now, "out": True, "sender": me},
        ],
    })
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.get_chat_history("42", limit=10)
    finally:
        await s.stop()
    assert len(rows) == 3
    # Outgoing flag honored per row.
    outgoing = [r for r in rows if r["outgoing"]]
    incoming = [r for r in rows if not r["outgoing"]]
    assert len(outgoing) == 2 and len(incoming) == 1
    assert incoming[0]["text"] == "sounds good"
    assert incoming[0]["sender_username"] == "alex_smith"


@pytest.mark.asyncio
async def test_search_chats_substring_match_on_name_and_username(db):
    """search_chats: case-insensitive substring match on either name or @username."""
    client = FakeTelegramClient(dialogs=[
        {"id": 1, "name": "Alex Smith",       "username": "alex_smith"},
        {"id": 2, "name": "Boris Johnson",    "username": "boris_j"},
        {"id": 3, "name": "Charlie Browning", "username": "charlie_b"},
        {"id": 4, "name": "Maria Lopez",      "username": "alexandria_m"},  # name miss, username hit
    ])
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_chats("alex")
    finally:
        await s.stop()
    ids = {r["chat_id"] for r in rows}
    # 1 (name match), 4 (username match)
    assert ids == {"1", "4"}


@pytest.mark.asyncio
async def test_search_chats_empty_query_returns_nothing(db):
    client = FakeTelegramClient(dialogs=[{"id": 1, "name": "x"}])
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_chats("")
        rows2 = await s.search_chats("   ")
    finally:
        await s.stop()
    assert rows == [] and rows2 == []


@pytest.mark.asyncio
async def test_search_messages_filters_by_query_and_returns_both_sides(db):
    """search_messages: hand the query to telethon's server-side `search=` and
    return matching messages in get_chat_history shape (both sides, sender
    metadata, skipping empty ones)."""
    now = datetime.now(timezone.utc)
    me = SimpleNamespace(username="me_user", first_name="V", last_name=None)
    artem = SimpleNamespace(username="alex_s", first_name="Алекс", last_name="Смит")
    client = FakeTelegramClient(messages={
        77: [
            {"id": 10, "message": "let's plan the redis migration",
             "date": now, "out": True,  "sender": me},
            {"id": 11, "message": "agreed, redis migration is overdue",
             "date": now, "out": False, "sender": artem},
            {"id": 12, "message": "unrelated weekend plans",
             "date": now, "out": True,  "sender": me},
            {"id": 13, "message": "",
             "date": now, "out": False, "sender": artem},  # media-only
        ],
    })
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_messages("77", "redis migration", limit=10)
    finally:
        await s.stop()
    texts = [r["text"] for r in rows]
    assert texts == [
        "let's plan the redis migration",
        "agreed, redis migration is overdue",
    ]
    incoming = [r for r in rows if not r["outgoing"]]
    assert incoming and incoming[0]["sender_username"] == "alex_s"


@pytest.mark.asyncio
async def test_search_messages_empty_query_returns_nothing(db):
    client = FakeTelegramClient(messages={1: [
        {"id": 1, "message": "x", "date": datetime.now(timezone.utc),
         "out": True, "sender": SimpleNamespace(username=None, first_name="", last_name=None)},
    ]})
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_messages("1", "")
        rows2 = await s.search_messages("1", "   ")
    finally:
        await s.stop()
    assert rows == [] and rows2 == []


@pytest.mark.asyncio
async def test_search_chats_token_and_match_handles_word_order(db):
    """Multi-word queries should match in any order — 'artem sinkovskiy' must
    match a dialog whose display name is 'Smith, Alex'."""
    client = FakeTelegramClient(dialogs=[
        {"id": 7, "name": "Smith, Alex", "username": None},
        {"id": 8, "name": "Alex Volkov",     "username": None},  # only one token
    ])
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_chats("artem sinkovskiy")
    finally:
        await s.stop()
    ids = {r["chat_id"] for r in rows}
    assert ids == {"7"}, f"expected only the row containing both tokens, got {ids}"


@pytest.mark.asyncio
async def test_search_chats_contact_fallback_for_transliterated_name(db):
    """If the local dialog has a Cyrillic display name, a Latin query won't
    substring-match locally. The contacts.SearchRequest fallback must surface
    it (Telegram's server-side search handles transliteration)."""
    client = FakeTelegramClient(
        dialogs=[
            {"id": 1, "name": "Алекс Смит", "username": None},  # Cyrillic
        ],
        contact_search_users=[
            {"id": 1, "first_name": "Алекс", "last_name": "Смит", "username": None},
        ],
    )
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_chats("Alex Smith")
    finally:
        await s.stop()
    assert client.contact_search_calls == ["Alex Smith"]
    assert len(rows) == 1
    assert rows[0]["chat_id"] == "1"
    assert rows[0]["source"] == "contact"
    assert rows[0]["name"] == "Алекс Смит"


@pytest.mark.asyncio
async def test_search_chats_respects_limit(db):
    client = FakeTelegramClient(dialogs=[
        {"id": i, "name": f"alex {i}", "username": None} for i in range(50)
    ])
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.search_chats("alex", limit=5)
    finally:
        await s.stop()
    assert len(rows) == 5


@pytest.mark.asyncio
async def test_get_chat_history_skips_empty_messages(db):
    """Pure-media / system messages have empty .message — skip so the operator
    doesn't see noise."""
    now = datetime.now(timezone.utc)
    me = SimpleNamespace(username="me_user", first_name="V", last_name=None)
    client = FakeTelegramClient(messages={
        99: [
            {"id": 3, "message": "hello", "date": now, "out": True, "sender": me},
            {"id": 2, "message": "",      "date": now, "out": True, "sender": me},  # media-only
        ],
    })
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
    )
    await s.start()
    try:
        rows = await s.get_chat_history("99", limit=10)
    finally:
        await s.stop()
    assert len(rows) == 1
    assert rows[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_ignore_does_not_block_legitimate_senders(db):
    """Sanity: the ignore set must only block matching senders. Others go
    through normally."""
    client = FakeTelegramClient()
    s = TelegramService(
        db=db, client=client,
        important_senders=set(), important_keywords=set(),
        ignore_usernames={"banned_bot"},
    )
    await s.start()
    try:
        await client.handler(make_event(
            sender_username="banned_bot", body="ignored",
        ))
        await client.handler(make_event(
            sender_username="real_person", body="hello",
            message_id=43,
        ))
        rows = await db.list_inbox(unread_only=False)
        assert len(rows) == 1
        assert rows[0]["sender_username"] == "real_person"
    finally:
        await s.stop()
