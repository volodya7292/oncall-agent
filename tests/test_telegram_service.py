"""TelegramService tests with a fake TelegramClientLike.

These cover:
  * NewMessage handler writes inbound DM to messenger_inbox with correct
    fields, gated on the DM allowlist.
  * Private filter rejects non-private events.
  * Duplicate (chat_id, message_id) doesn't insert twice.
  * `get_chat_style` reads only the user's OWN outgoing messages.
  * `send` calls send_message and returns the new message id.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
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
        messages: dict[int, list[dict[str, Any]]] | None = None,
        dialogs: list[dict[str, Any]] | None = None,
        contact_search_users: list[dict[str, Any]] | None = None,
        non_contact_ids: set[int] | None = None,
    ) -> None:
        self.connected = False
        self.disconnected = False
        self.authorized = True
        self.handler: Any = None
        self.event_filter: Any = None
        self.sent: list[dict[str, Any]] = []
        # get_entity(id) reports .contact=True unless the id is listed here.
        # Lets outbound-contact-gate tests simulate a stranger.
        self._non_contact_ids = non_contact_ids or set()
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

    async def get_entity(self, entity):
        # The outbound contacts gate resolves the target and reads .contact.
        try:
            eid = int(entity)
        except (TypeError, ValueError):
            eid = None
        return SimpleNamespace(id=eid, contact=(eid not in self._non_contact_ids))

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
                # `media` / `file` are optional; when present, they let
                # the service detect media-only messages and emit a
                # [photo] / [file: ...] placeholder.
                media=m.get("media"),
                file=m.get("file"),
            )

        return _async_iter([_make(m) for m in msgs[:limit]])

    def iter_dialogs(self, **kwargs):
        # The service enumerates all dialogs and no longer filters on archive
        # state; each dialog just reports its own `archived` flag.
        dlgs = [
            SimpleNamespace(
                id=d["id"], name=d.get("name", ""),
                entity=SimpleNamespace(username=d.get("username"), id=d["id"]),
                is_user=d.get("is_user", True),
                is_group=d.get("is_group", False),
                is_channel=d.get("is_channel", False),
                unread_count=d.get("unread_count", 0),
                archived=d.get("archived", False),
            )
            for d in self._dialogs
        ]
        return _async_iter(dlgs)


def _async_iter(items):
    async def gen():
        for it in items:
            yield it
    return gen()


def make_event(
    *, sender_username: str | None, body: str, is_private: bool = True,
    chat_id: int = 12345, message_id: int = 42, is_bot: bool = False,
    is_self: bool = False, out: bool = False, sender_id: int = 7777,
    reply_to: Any = None, is_contact: bool = True,
) -> Any:
    sender = SimpleNamespace(
        id=sender_id, username=sender_username,
        first_name="Alex", last_name=None,
        bot=is_bot, is_self=is_self, contact=is_contact,
    )
    message = SimpleNamespace(
        message=body, id=message_id, out=out,
        date=datetime.now(timezone.utc), chat_id=chat_id,
    )
    if reply_to is not None:
        message.reply_to = SimpleNamespace(
            reply_to_msg_id=getattr(reply_to, "id", 1),
        )

        async def get_reply_message():
            return reply_to

        message.get_reply_message = get_reply_message

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
    s = TelegramService(db=db, client=client)
    # Inbound triage is gated on the DM allowlist (see _handle_inbound).
    # Allowlist the chat_ids the triage tests exercise so they test triage,
    # not the gate — the gate itself is covered by its own test below.
    await db.allow_dm("12345")
    await db.allow_dm("111")
    await db.allow_dm("222")
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


@pytest.mark.asyncio
async def test_reply_anchor_prepended_to_inbox_body(service, db):
    """A Telegram reply stores a `[replying to ...]` anchor quoting the
    referenced message — without it the reply pointer is invisible to the
    operator and deictic answers lose their anchor."""
    s, client = service
    quoted = SimpleNamespace(
        id=41, message="can you send the report today?", out=True, media=None,
    )
    event = make_event(
        sender_username="rando", body="ok, looking now", reply_to=quoted,
    )
    await client.handler(event)
    rows = await db.list_inbox()
    row = rows[0]
    assert row["body"] == (
        '[replying to the owner\'s earlier message: '
        '"can you send the report today?"]'
        "\nok, looking now"
    )


@pytest.mark.asyncio
async def test_inbound_from_non_contact_is_dropped(service, db):
    """Contacts-only floor: a DM from a sender who is NOT in the owner's
    contacts is dropped even when the chat is on the DM allowlist."""
    s, client = service
    # chat 12345 is allowlisted by the fixture, but the sender is a stranger.
    await client.handler(make_event(
        sender_username="stranger", body="hi, buy my coin", is_contact=False,
    ))
    assert await db.list_inbox(unread_only=False) == []
    # A contact in the same allowlisted chat gets through normally.
    await client.handler(make_event(
        sender_username="friend", body="hey", is_contact=True, message_id=43,
    ))
    rows = await db.list_inbox(unread_only=False)
    assert len(rows) == 1
    assert rows[0]["sender_username"] == "friend"


@pytest.mark.asyncio
async def test_agent_self_chat_dropped_even_if_allowlisted(db):
    """Backstop: the agent account's own chat never enters the primary
    userbot's inbox, even if its chat_id is on the DM allowlist."""
    client = FakeTelegramClient()
    s = TelegramService(db=db, client=client, agent_user_id=999000)
    await db.allow_dm("999000")  # pathological: agent chat allowlisted
    await s.start()
    try:
        await client.handler(make_event(
            sender_username="agentacct", body="[auto] on it",
            chat_id=999000, sender_id=999000,
        ))
        assert await db.list_inbox(unread_only=False) == []
    finally:
        await s.stop()


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
    """Style samples must be the USER'S own messages, not the counterparty's.
    Results are sorted newest-first (combined across the 3 temporal sampling
    windows that get_chat_style fans out across)."""
    base = datetime(2026, 5, 22, 10, 0, tzinfo=timezone.utc)
    client = FakeTelegramClient(outgoing={
        12345: [
            {"id": 5, "message": "ало", "date": base},
            {"id": 6, "message": "ща приду", "date": base + timedelta(minutes=1)},
            {"id": 7, "message": "+1", "date": base + timedelta(minutes=2)},
        ],
    })
    s = TelegramService(db=db, client=client)
    await s.start()
    try:
        samples = await s.get_chat_style("12345", limit=10)
    finally:
        await s.stop()
    texts = [m["text"] for m in samples]
    assert texts == ["+1", "ща приду", "ало"]


@pytest.mark.asyncio
async def test_send_calls_underlying_client(db):
    client = FakeTelegramClient()
    s = TelegramService(db=db, client=client)
    await s.start()
    try:
        out = await s.send("12345", "draft reply")
    finally:
        await s.stop()
    assert client.sent and client.sent[0]["message"] == "draft reply"
    assert client.sent[0]["entity"] == 12345  # coerced to int
    assert out["message_id"] == str(client.sent[0]["id"])


@pytest.mark.asyncio
async def test_send_to_non_contact_is_refused(db):
    """Outbound contacts floor: sending to a chat that isn't in the owner's
    contacts is refused before any message leaves, mirroring the inbound
    drop. Guards against cold-messaging a stranger via a stray chat_id."""
    client = FakeTelegramClient(non_contact_ids={55555})
    s = TelegramService(db=db, client=client)
    await s.start()
    try:
        with pytest.raises(ValueError, match="not in the owner's"):
            await s.send("55555", "unsolicited message")
        assert client.sent == []
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_list_pending_chats_groups_by_chat_with_messages(service, db):
    """Multiple unread DMs in the same chat collapse to ONE pending-chats
    row with `unread_count` and a `messages` list — one entry per unread
    row in chronological order, each carrying message_id/received_at/body.
    A second chat gets its own row."""
    s, client = service
    await client.handler(make_event(
        sender_username="alex", body="ping 1", chat_id=111, message_id=1,
    ))
    await client.handler(make_event(
        sender_username="alex", body="ping 2", chat_id=111, message_id=2,
    ))
    await client.handler(make_event(
        sender_username="bob", body="hey", chat_id=222, message_id=10,
    ))
    rows = await s.list_pending_chats()
    by_chat = {r["chat_id"]: r for r in rows}
    assert set(by_chat.keys()) == {"111", "222"}
    assert by_chat["111"]["unread_count"] == 2
    assert by_chat["111"]["sender_username"] == "alex"
    bodies_111 = [m["body"] for m in by_chat["111"]["messages"]]
    assert bodies_111 == ["ping 1", "ping 2"]
    mids_111 = [m["message_id"] for m in by_chat["111"]["messages"]]
    assert mids_111 == ["1", "2"]
    assert by_chat["222"]["unread_count"] == 1
    assert [m["body"] for m in by_chat["222"]["messages"]] == ["hey"]
    # `inbox_ids` carries the db-side primary keys for every pending row
    # in the chat — the drain uses this exact set for read/triage marking.
    assert len(by_chat["111"]["inbox_ids"]) == 2
    assert len(by_chat["222"]["inbox_ids"]) == 1


@pytest.mark.asyncio
async def test_pending_chats_keys_off_triage_not_read(service, db):
    """Safety invariant: a row marked read but NOT yet triaged still
    appears in `list_pending_chats`. The drain pre-marks the snapshot
    read before invoking the operator; if the daemon crashes (or the
    auto_ping raises) between pre-mark and triage, recovery — and the
    next dirty cycle — must still retry those rows, otherwise the user
    would silently lose a response."""
    s, client = service
    await client.handler(make_event(
        sender_username="alex", body="will crash mid-turn",
        chat_id=111, message_id=1,
    ))
    [row] = await db.list_inbox(unread_only=True)
    # Simulate the pre-turn read-marking only — no triage (the turn
    # never completed).
    flipped = await db.mark_inbox_read([row["id"]])
    assert flipped == 1
    # Row is read, but NOT triaged → must still be pending.
    rows = await s.list_pending_chats()
    by_chat = {r["chat_id"]: r for r in rows}
    assert "111" in by_chat
    assert by_chat["111"]["inbox_ids"] == [row["id"]]
    # Now triage it (as a successful turn would) — and only then does it
    # leave the pending set.
    await db.mark_inbox_triaged([row["id"]])
    rows_after = await s.list_pending_chats()
    assert all(r["chat_id"] != "111" for r in rows_after)


@pytest.mark.asyncio
async def test_reply_to_chat_marks_chat_read_and_records_reply(service, db):
    """`reply_to_chat` sends via the underlying client AND clears the
    chat's unread state. Every unread inbox row for that chat ends up
    marked read; the latest row gets its `replied_message_id` stamped
    for audit."""
    s, client = service
    await client.handler(make_event(
        sender_username="alex", body="m1", chat_id=111, message_id=1,
    ))
    await client.handler(make_event(
        sender_username="alex", body="m2", chat_id=111, message_id=2,
    ))
    out = await s.reply_to_chat("111", "ok")
    assert out["chat_id"] == "111"
    assert out["sender_username"] == "alex"
    assert out["inbound_body"] == "m2"  # the latest unread row's body
    # Both unread rows are now read.
    rows_after = await db.list_inbox(unread_only=True)
    assert [r for r in rows_after if r["chat_id"] == "111"] == []
    # Latest row carries the outbound message id.
    all_rows = await db.list_inbox(unread_only=False)
    for r in all_rows:
        if r["chat_id"] == "111" and r["message_id"] == "2":
            assert r["replied_message_id"] == str(client.sent[-1]["id"])


@pytest.mark.asyncio
async def test_mark_chat_read_bulk(service, db):
    """`mark_chat_read(chat_id)` flips read_at on every unread row in that
    chat in one shot — the chat-keyed replacement for `mark_read(id)`."""
    s, client = service
    for i in range(3):
        await client.handler(make_event(
            sender_username="alex", body=f"m{i}", chat_id=111, message_id=10 + i,
        ))
    n = await s.mark_chat_read("111")
    assert n == 3
    rows = await db.list_inbox(unread_only=True)
    assert [r for r in rows if r["chat_id"] == "111"] == []


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


@pytest.mark.asyncio
async def test_inbound_from_non_allowlisted_chat_is_dropped(db):
    """Triage gate: a DM from a chat that is NOT on the dm_allowlist is
    dropped in _handle_inbound — never recorded, never surfaced. The
    allowlist is absolute."""
    client = FakeTelegramClient()
    s = TelegramService(db=db, client=client)
    await s.start()
    try:
        # Chat 999 isn't allowlisted.
        await client.handler(make_event(
            sender_username="alex", body="urgent: server down", chat_id=999,
        ))
        assert await db.list_inbox(unread_only=False) == []
        # Allowlist it, then the same kind of message gets through.
        await db.allow_dm("999")
        await client.handler(make_event(
            sender_username="alex", body="urgent: server down",
            chat_id=999, message_id=43,
        ))
        rows = await db.list_inbox(unread_only=False)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == "999"
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
    """Multi-word queries should match in any order — 'alex smith' must
    match a dialog whose display name is 'Smith, Alex' (reversed order,
    punctuation between tokens)."""
    client = FakeTelegramClient(dialogs=[
        {"id": 7, "name": "Smith, Alex", "username": None},
        {"id": 8, "name": "Alex Volkov", "username": None},  # only one token
    ])
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.search_chats("alex smith")
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
    )
    await s.start()
    try:
        rows = await s.get_chat_history("99", limit=10)
    finally:
        await s.stop()
    assert len(rows) == 1
    assert rows[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_get_chat_history_surfaces_media_only_messages(db):
    """Regression: a media-only (no caption) message used to be silently
    dropped, leaving the operator to guess at nearby IDs when asked to
    `read_image`. It must now appear with `has_media=True` and a
    placeholder body that names what the file is, so the operator can
    pass the real message_id to `read_image`."""
    now = datetime.now(timezone.utc)
    me = SimpleNamespace(username="me_user", first_name="V", last_name=None)
    image_file = SimpleNamespace(name=None, mime_type="image/jpeg")
    doc_file = SimpleNamespace(name="report.pdf", mime_type="application/pdf")
    client = FakeTelegramClient(messages={
        99: [
            {"id": 30, "message": "hello",  "date": now, "out": True,  "sender": me},
            {"id": 29, "message": "",       "date": now, "out": False, "sender": me,
             "media": object(), "file": image_file},
            {"id": 28, "message": "",       "date": now, "out": False, "sender": me,
             "media": object(), "file": doc_file},
            {"id": 27, "message": "",       "date": now, "out": True,  "sender": me},  # no media, no text → still dropped
        ],
    })
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.get_chat_history("99", limit=10)
    finally:
        await s.stop()
    # The pure-empty row (id=27) is still dropped; the two media rows are
    # surfaced with their real ids + placeholders.
    by_id = {r["message_id"]: r for r in rows}
    assert set(by_id.keys()) == {"30", "29", "28"}
    assert by_id["29"]["text"] == "[photo]"
    assert by_id["29"]["has_media"] is True
    assert by_id["28"]["text"] == "[file: report.pdf]"
    assert by_id["28"]["has_media"] is True
    assert by_id["30"]["has_media"] is False


@pytest.mark.asyncio
async def test_list_chats_returns_recent_dialogs_in_order(db):
    """list_chats: no query, returns telethon's iter_dialogs order verbatim
    (telethon already sorts by last activity)."""
    client = FakeTelegramClient(dialogs=[
        {"id": 1, "name": "Alex Smith", "username": "alex_smith",
         "is_user": True, "unread_count": 0},
        {"id": 2, "name": "Eng",        "username": None,
         "is_user": False, "is_group": True, "unread_count": 3},
        {"id": 3, "name": "News",       "username": "news_channel",
         "is_user": False, "is_channel": True, "unread_count": 0},
    ])
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.list_chats()
    finally:
        await s.stop()
    assert [r["chat_id"] for r in rows] == ["1", "2", "3"]
    assert rows[1] == {
        "chat_id": "2", "name": "Eng", "username": None,
        "is_user": False, "is_group": True, "is_channel": False,
        "unread_count": 3, "archived": False,
    }


@pytest.mark.asyncio
async def test_list_chats_unread_only_filter(db):
    client = FakeTelegramClient(dialogs=[
        {"id": 1, "name": "A", "unread_count": 0, "is_user": True},
        {"id": 2, "name": "B", "unread_count": 5, "is_user": True},
        {"id": 3, "name": "C", "unread_count": 0, "is_user": True},
        {"id": 4, "name": "D", "unread_count": 1, "is_user": True},
    ])
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.list_chats(unread_only=True)
    finally:
        await s.stop()
    assert [r["chat_id"] for r in rows] == ["2", "4"]


@pytest.mark.asyncio
async def test_list_chats_dms_only_filter(db):
    client = FakeTelegramClient(dialogs=[
        {"id": 1, "name": "DM Alice",  "is_user": True},
        {"id": 2, "name": "Eng group", "is_user": False, "is_group": True},
        {"id": 3, "name": "DM Bob",    "is_user": True},
        {"id": 4, "name": "Channel",   "is_user": False, "is_channel": True},
    ])
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.list_chats(dms_only=True)
    finally:
        await s.stop()
    assert [r["chat_id"] for r in rows] == ["1", "3"]
    assert all(r["is_user"] for r in rows)


@pytest.mark.asyncio
async def test_list_chats_includes_archived(db):
    """Archive state is no longer a filter — archived dialogs are surfaced like
    any other, with their `archived` flag reported as-is. Guards against
    re-introducing the old "hide archived" behavior, which silently dropped
    DMs from a chat the user had merely archived (not blocked/muted)."""
    client = FakeTelegramClient(dialogs=[
        {"id": 1, "name": "Active",   "is_user": True, "archived": False},
        {"id": 2, "name": "Archived", "is_user": True, "archived": True},
        {"id": 3, "name": "Active2",  "is_user": True, "archived": False},
    ])
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.list_chats()
    finally:
        await s.stop()
    assert [r["chat_id"] for r in rows] == ["1", "2", "3"]
    assert {r["chat_id"]: r["archived"] for r in rows} == {
        "1": False, "2": True, "3": False,
    }


@pytest.mark.asyncio
async def test_list_chats_respects_limit(db):
    client = FakeTelegramClient(dialogs=[
        {"id": i, "name": f"chat {i}", "is_user": True} for i in range(50)
    ])
    s = TelegramService(
        db=db, client=client,
    )
    await s.start()
    try:
        rows = await s.list_chats(limit=7)
    finally:
        await s.stop()
    assert len(rows) == 7


