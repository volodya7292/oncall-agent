"""TelegramBotService tests with a fake BotApi + a fake operator.

The bot now talks to Telegram over the HTTP Bot API, so tests don't touch
telethon at all. They cover:

  * `getMe` runs at startup and captures the bot's user_id / username.
  * Stranger messages are silently ignored — operator NOT called.
  * Owner message → operator.chat_turn → bot replies via sendMessage.
  * /start, /help slash commands answered locally without touching operator.
  * chat.reply events for THIS bot's session_id are delivered to the owner.
  * chat.reply events for OTHER sessions are filtered out.
  * Long replies are chunked under Telegram's per-message limit.
  * chunk_message respects newline boundaries.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from uuid import UUID, uuid4

from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.db import Database
from oncall.events import EventBus
from oncall.models import ApprovalRequest, ClassifierVerdict, Task
from oncall.operator import OperatorTurnResult
from oncall.telegram_bot import (
    TelegramBotService,
    bot_session_id,
    chunk_message,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBotApi:
    """In-memory stand-in for HttpxBotApi. `responses` lets each test script
    what `getUpdates` returns on successive calls."""

    def __init__(self, *, getme: dict | None = None, getupdates_script: list | None = None) -> None:
        self._getme = getme or {"id": 100200, "username": "oncall_bot", "is_bot": True}
        # Each entry is the list of updates for one getUpdates call.
        self._getupdates_script = list(getupdates_script or [])
        self.calls: list[tuple[str, dict]] = []
        self.sent: list[dict] = []
        self.closed = False

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, payload or {}))
        if method == "getMe":
            return self._getme
        if method == "getUpdates":
            if self._getupdates_script:
                return self._getupdates_script.pop(0)
            # No more scripted updates → behave like Telegram on idle: return
            # nothing, but don't churn the poll loop. Sleep briefly to yield.
            await asyncio.sleep(0.5)
            return []
        if method == "sendMessage":
            self.sent.append(payload or {})
            return {"message_id": len(self.sent)}
        raise AssertionError(f"unexpected bot API method: {method}")

    async def aclose(self) -> None:
        self.closed = True


class FakeOperator:
    def __init__(self, reply_text: str = "got it") -> None:
        self.reply_text = reply_text
        self.calls: list[dict[str, Any]] = []

    async def chat_turn(self, *, session_id: str, user_text: str, language=None) -> OperatorTurnResult:
        self.calls.append({"session_id": session_id, "user_text": user_text, "language": language})
        return OperatorTurnResult(text=self.reply_text, tool_calls_made=[])


def _msg(*, sender_id: int, text: str, chat_id: int = 999, update_id: int = 1) -> dict:
    """Construct a Bot API Update payload for an incoming text message."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender_id, "is_bot": False, "first_name": "X"},
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OWNER_ID = 12345


@pytest.fixture
async def bus(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    await db.connect()
    try:
        yield EventBus(db)
    finally:
        await db.close()


async def _make_bot(bus, api: FakeBotApi, operator: FakeOperator) -> TelegramBotService:
    svc = TelegramBotService(
        api=api, operator=operator, events=bus, owner_user_id=OWNER_ID,
    )
    await svc.start()
    return svc


# ---------------------------------------------------------------------------
# Startup / getMe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_getme_captures_user_id_and_username(bus):
    api = FakeBotApi(getme={"id": 100200, "username": "oncall_bot"})
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        assert svc.bot_user_id == 100200
        assert svc.bot_username == "oncall_bot"
        # getMe was the first call.
        assert api.calls[0][0] == "getMe"
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_stop_closes_transport(bus):
    api = FakeBotApi()
    svc = await _make_bot(bus, api, FakeOperator())
    await svc.stop()
    assert api.closed is True


# ---------------------------------------------------------------------------
# Direct handler dispatch (bypasses the poll loop for determinism)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stranger_is_ignored(bus):
    """Non-owner sender → no operator call, no reply."""
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=99999, text="hi")["message"])
        assert operator.calls == []
        assert api.sent == []
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_owner_message_routes_through_operator(bus):
    api = FakeBotApi()
    operator = FakeOperator(reply_text="staging is up.")
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="check staging")["message"])
        assert operator.calls == [{
            "session_id": bot_session_id(OWNER_ID),
            "user_text": "check staging",
            "language": None,
        }]
        assert api.sent == [{"chat_id": 999, "text": "staging is up."}]
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_start_command_answered_locally(bus):
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/start")["message"])
        assert operator.calls == []
        assert len(api.sent) == 1
        assert "on-call operator" in api.sent[0]["text"].lower()
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_help_command_answered_locally(bus):
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/help")["message"])
        assert operator.calls == []
        assert "/start" in api.sent[0]["text"]
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_empty_text_is_ignored(bus):
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="   ")["message"])
        assert operator.calls == []
        assert api.sent == []
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# Long-poll integration: getUpdates → dispatch path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_polling_loop_consumes_updates_and_advances_offset(bus):
    api = FakeBotApi(getupdates_script=[
        [_msg(sender_id=OWNER_ID, text="hi", update_id=7)],
        [_msg(sender_id=OWNER_ID, text="and again", update_id=8)],
        [],
    ])
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        # Wait for two updates to be dispatched.
        for _ in range(100):
            if len(operator.calls) >= 2:
                break
            await asyncio.sleep(0.01)

        assert len(operator.calls) >= 2
        # The second getUpdates call must have been made with offset > 7.
        getupdates_calls = [c for c in api.calls if c[0] == "getUpdates"]
        assert getupdates_calls[1][1]["offset"] >= 8
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# chat.reply auto-ping delivery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_reply_for_this_session_delivered(bus):
    api = FakeBotApi()
    svc = await _make_bot(bus, api, FakeOperator())
    try:
        # Yield so the subscriber task reaches `await q.get()` and registers
        # itself in EventBus._global_subs before we publish.
        for _ in range(10):
            await asyncio.sleep(0)
        await bus.publish_global("chat.reply", {
            "session_id": bot_session_id(OWNER_ID),
            "text": "Found 5 projects.",
            "task_id": "abcdef12",
        })
        for _ in range(50):
            if api.sent:
                break
            await asyncio.sleep(0.01)
        assert api.sent == [{"chat_id": OWNER_ID, "text": "Found 5 projects."}]
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_chat_reply_for_other_session_filtered(bus):
    api = FakeBotApi()
    svc = await _make_bot(bus, api, FakeOperator())
    try:
        await bus.publish_global("chat.reply", {
            "session_id": "tg-bot-99999",
            "text": "not for me",
            "task_id": "x",
        })
        await asyncio.sleep(0.05)
        assert api.sent == []
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# chunk_message — pure function
# ---------------------------------------------------------------------------

def test_chunk_short_passthrough():
    assert chunk_message("hi") == ["hi"]


def test_chunk_hard_cuts_when_no_safe_boundary():
    chunks = chunk_message("a" * 8000, limit=4000)
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks) == "a" * 8000


def test_chunk_splits_on_newline_boundary_past_midpoint():
    text = "x" * 3000 + "\n" + "y" * 2000
    chunks = chunk_message(text, limit=4000)
    assert len(chunks) >= 2
    assert chunks[0] == "x" * 3000
    assert chunks[1].startswith("y")


# ---------------------------------------------------------------------------
# Long reply chunking through _send
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_operator_reply_is_chunked(bus):
    api = FakeBotApi()
    operator = FakeOperator(reply_text="x" * 5000)
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="long please")["message"])
        assert len(api.sent) >= 2
        assert sum(len(s["text"]) for s in api.sent) >= 5000
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# Inline Yes/No keyboard for approvals
# ---------------------------------------------------------------------------

@pytest.fixture
async def db(tmp_path):
    d = Database(tmp_path / "bot.sqlite")
    await d.connect()
    try:
        yield d
    finally:
        await d.close()


async def _make_bot_with_approvals(
    db: Database, bus: EventBus, api: FakeBotApi,
) -> tuple[TelegramBotService, Broker]:
    approval_client = HttpLongPollApprovalClient()
    broker = Broker(db, approval_client, bus.publish)
    svc = TelegramBotService(
        api=api, operator=FakeOperator(), events=bus, owner_user_id=OWNER_ID,
        broker=broker, db=db,
    )
    await svc.start()
    return svc, broker


async def _seed_pending_approval(db: Database, *, session_id: str, phrase: str) -> UUID:
    """Insert a task tied to `session_id` and a pending mutating approval."""
    task = Task(
        session_id=str(uuid4()), prompt="mkdir test",
        dispatched_by_chat_session=session_id,
    )
    await db.insert_task(task)
    req = ApprovalRequest(
        task_id=task.id, session_id=task.session_id, tool_use_id="tu_1",
        tool_name="Bash",
        tool_input={"command": "mkdir foo"},
        classifier_verdict=ClassifierVerdict.MUTATING,
        canonical_command="mkdir foo",
        blast_radius="creates a directory on this host.",
        challenge_phrase=phrase,
    )
    await db.create_pending_approval(req)
    return req.id


@pytest.mark.asyncio
async def test_approval_event_sends_inline_keyboard(bus, db):
    api = FakeBotApi()
    svc, _broker = await _make_bot_with_approvals(db, bus, api)
    try:
        approval_id = await _seed_pending_approval(
            db, session_id=svc.session_id, phrase="orchid cipher quartz",
        )
        await bus.publish(
            # task id from get_latest pending — round-trip through db to find it
            UUID((await db.get_approval(approval_id))["task_id"]),
            "approval.requested",
            {
                "approval_id": str(approval_id),
                "tool_name": "Bash",
                "canonical_command": "mkdir foo",
                "blast_radius": "creates a directory on this host.",
                "challenge_phrase": "orchid cipher quartz",
            },
        )
        # Wait for the subscriber task to fire.
        for _ in range(100):
            send_calls = [c for c in api.calls if c[0] == "sendMessage"]
            if send_calls:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("approval prompt was never sent")

        msg = send_calls[0][1]
        assert msg["chat_id"] == OWNER_ID
        assert "mkdir foo" in msg["text"]
        kb = msg["reply_markup"]["inline_keyboard"]
        assert len(kb) == 1 and len(kb[0]) == 2
        labels = {btn["text"] for btn in kb[0]}
        assert any("Yes" in l for l in labels)
        assert any("No" in l for l in labels)
        cbs = {btn["callback_data"] for btn in kb[0]}
        assert f"appr:{approval_id}:allow" in cbs
        assert f"appr:{approval_id}:deny" in cbs
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_approval_event_for_other_session_ignored(bus, db):
    """Approval for a task dispatched in some OTHER chat session should not
    spawn a button message for this bot."""
    api = FakeBotApi()
    svc, _broker = await _make_bot_with_approvals(db, bus, api)
    try:
        # Seed against a different session_id.
        approval_id = await _seed_pending_approval(
            db, session_id="some-other-session", phrase="x y z",
        )
        await bus.publish(
            UUID((await db.get_approval(approval_id))["task_id"]),
            "approval.requested",
            {
                "approval_id": str(approval_id), "tool_name": "Bash",
                "canonical_command": "x", "blast_radius": "y",
                "challenge_phrase": "x y z",
            },
        )
        await asyncio.sleep(0.1)
        assert [c for c in api.calls if c[0] == "sendMessage"] == []
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_callback_yes_tap_resolves_approval(bus, db):
    """A Yes callback from the owner: looks up the phrase, calls broker,
    answers the callback, edits the message to remove buttons."""
    api = FakeBotApi()
    svc, broker = await _make_bot_with_approvals(db, bus, api)
    try:
        approval_id = await _seed_pending_approval(
            db, session_id=svc.session_id, phrase="amber paper compass",
        )
        # Stand a waiter on the broker's approval future so submit_response
        # has something to resolve.
        request_future = asyncio.create_task(
            broker._client.request_approval(
                await _model_request(db, approval_id),
            )
        )
        await asyncio.sleep(0)  # let it register

        cq = {
            "id": "cb1",
            "from": {"id": OWNER_ID},
            "data": f"appr:{approval_id}:allow",
            "message": {"chat": {"id": 999}, "message_id": 42, "text": "Approve?"},
        }
        await svc._dispatch_callback(cq)

        result = await asyncio.wait_for(request_future, timeout=0.5)
        assert result.behavior == "allow"
        assert result.challenge_matched is True

        methods = [c[0] for c in api.calls]
        assert "answerCallbackQuery" in methods
        assert "editMessageText" in methods
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_callback_no_tap_denies_approval(bus, db):
    api = FakeBotApi()
    svc, broker = await _make_bot_with_approvals(db, bus, api)
    try:
        approval_id = await _seed_pending_approval(
            db, session_id=svc.session_id, phrase="amber paper compass",
        )
        request_future = asyncio.create_task(
            broker._client.request_approval(
                await _model_request(db, approval_id),
            )
        )
        await asyncio.sleep(0)

        cq = {
            "id": "cb2",
            "from": {"id": OWNER_ID},
            "data": f"appr:{approval_id}:deny",
            "message": {"chat": {"id": 999}, "message_id": 42, "text": "Approve?"},
        }
        await svc._dispatch_callback(cq)

        result = await asyncio.wait_for(request_future, timeout=0.5)
        # Phrase matched but the user denied, so server-side decision is deny.
        assert result.behavior == "deny"
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_callback_from_non_owner_rejected(bus, db):
    api = FakeBotApi()
    svc, _broker = await _make_bot_with_approvals(db, bus, api)
    try:
        approval_id = await _seed_pending_approval(
            db, session_id=svc.session_id, phrase="x y z",
        )
        cq = {
            "id": "cb3",
            "from": {"id": 99999},   # not owner
            "data": f"appr:{approval_id}:allow",
            "message": {"chat": {"id": 1}, "message_id": 1, "text": "?"},
        }
        await svc._dispatch_callback(cq)
        # Bot must still answer the callback (so client UI doesn't hang) but
        # never call editMessageText or submit anything.
        methods = [c[0] for c in api.calls]
        assert "answerCallbackQuery" in methods
        assert "editMessageText" not in methods
        # Approval still pending.
        row = await db.get_approval(approval_id)
        assert row["state"] == "pending"
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_chat_reply_with_approval_trigger_is_suppressed(bus, db):
    """When auto-ping for approval.requested fires, the operator's verbose
    text comes through as chat.reply — we must NOT send it (the button
    message is the bot's approval UI). Otherwise the user sees both."""
    api = FakeBotApi()
    svc, _broker = await _make_bot_with_approvals(db, bus, api)
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        await bus.publish_global("chat.reply", {
            "session_id": svc.session_id,
            "text": "Task wants to run mkdir... say orchid cipher quartz",
            "trigger": "approval.requested",
            "task_id": "abc",
        })
        await asyncio.sleep(0.05)
        # No sendMessage from the chat.reply path.
        assert [c for c in api.calls if c[0] == "sendMessage"] == []
    finally:
        await svc.stop()


# Pulls an ApprovalRequest object from a DB row (for use with the broker's
# in-memory client which only knows the ApprovalRequest shape).
async def _model_request(db: Database, approval_id: UUID) -> ApprovalRequest:
    row = await db.get_approval(approval_id)
    assert row is not None
    import json as _json
    return ApprovalRequest(
        id=UUID(row["id"]),
        task_id=UUID(row["task_id"]),
        session_id=row["session_id"],
        tool_use_id=row["tool_use_id"],
        tool_name=row["tool_name"],
        tool_input=_json.loads(row["tool_input_json"]),
        classifier_verdict=ClassifierVerdict(row["classifier_verdict"]),
        canonical_command=row["canonical_command"],
        blast_radius=row["blast_radius"],
        challenge_phrase=row["challenge_phrase"],
    )
