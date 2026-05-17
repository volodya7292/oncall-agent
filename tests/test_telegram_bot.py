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
from datetime import datetime, timezone
from typing import Any

import pytest

from uuid import UUID, uuid4

from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.db import Database
from oncall.events import EventBus
from oncall.models import ApprovalRequest, ClassifierVerdict, Task
from oncall.operator import OperatorTurnResult
from oncall.models import TaskState
from oncall.telegram_bot import (
    TelegramBotService,
    bot_session_id,
    chunk_message,
    _format_seconds,
    _relative_age,
    _truncate,
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
        self.documents: list[dict] = []
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
        if method == "setMyCommands":
            return True
        raise AssertionError(f"unexpected bot API method: {method}")

    async def send_document(self, *, chat_id, filename, content, caption=None):
        self.documents.append({
            "chat_id": chat_id, "filename": filename,
            "content": content, "caption": caption,
        })
        return {"message_id": 9000 + len(self.documents)}

    async def download_file(self, file_id: str) -> tuple[bytes, str, str]:
        self.calls.append(("download_file", {"file_id": file_id}))
        # Tests pre-stamp `_attachment_payload` if they want non-default bytes.
        if hasattr(self, "_attachment_payload"):
            return self._attachment_payload  # type: ignore[return-value]
        return b"fake-png-bytes", "image/png", "photo.jpg"

    async def aclose(self) -> None:
        self.closed = True


class FakeOperator:
    def __init__(
        self, reply_text: str = "got it",
        status: dict[str, Any] | None = None,
        clear_result: dict[str, int] | None = None,
        compress_result: dict[str, Any] | None = None,
        export_payload: str = "# Operator context\n\n_(stub)_\n",
    ) -> None:
        self.reply_text = reply_text
        self.calls: list[dict[str, Any]] = []
        self.status_payload = status
        self.status_calls: list[str] = []
        self.clear_calls: list[str] = []
        self.compress_calls: list[str] = []
        self.export_calls: list[str] = []
        self._clear_result = clear_result or {"messages_deleted": 0, "summaries_deleted": 0}
        self._compress_result = compress_result or {"compressed": False, "reason": "not enough history"}
        self._export_payload = export_payload

    async def chat_turn(
        self, *, session_id: str, user_text: str,
        language=None, attachments=None,
    ) -> OperatorTurnResult:
        self.calls.append({
            "session_id": session_id, "user_text": user_text,
            "language": language, "attachments": attachments,
        })
        return OperatorTurnResult(text=self.reply_text, tool_calls_made=[])

    async def clear_session(self, session_id: str) -> dict[str, int]:
        self.clear_calls.append(session_id)
        return dict(self._clear_result)

    async def compress_now(self, session_id: str) -> dict[str, Any]:
        self.compress_calls.append(session_id)
        return dict(self._compress_result)

    async def export_context(self, session_id: str) -> str:
        self.export_calls.append(session_id)
        return self._export_payload

    async def get_status(self, session_id: str) -> dict[str, Any]:
        self.status_calls.append(session_id)
        if self.status_payload is not None:
            return self.status_payload
        return {
            "model": "openai/test-model",
            "memory_entries": 0,
            "compression_threshold_tokens": 64000,
            "session_id": session_id,
            "session_messages_since_summary": 0,
            "estimated_context_tokens": 0,
            "latest_summary": None,
        }


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
            "attachments": None,
        }]
        assert len(api.sent) == 1
        assert api.sent[0]["chat_id"] == 999
        assert "staging is up" in api.sent[0]["text"]
        assert api.sent[0].get("parse_mode") == "MarkdownV2"
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_start_command_answered_locally(bus):
    """/start must reply without routing to the operator. We don't pin the
    greeting text — only the local-handling property matters here."""
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/start")["message"])
        assert operator.calls == []
        assert len(api.sent) == 1
        assert api.sent[0]["text"].strip()
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
        assert len(api.sent) == 1
        assert api.sent[0]["chat_id"] == OWNER_ID
        assert "Found 5 projects" in api.sent[0]["text"]
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


# ---------------------------------------------------------------------------
# /status — render & dispatch
# ---------------------------------------------------------------------------

async def _make_bot_with_db(
    db: Database, bus: EventBus, api: FakeBotApi, operator: FakeOperator,
) -> TelegramBotService:
    """Bot wired with a real DB (so /status can query state). No broker
    needed — these tests don't exercise the approval path."""
    svc = TelegramBotService(
        api=api, operator=operator, events=bus, owner_user_id=OWNER_ID, db=db,
    )
    await svc.start()
    return svc


@pytest.mark.asyncio
async def test_status_all_quiet(db, bus):
    """No tasks, no approvals, no unread DMs → terse 'all quiet' line."""
    api = FakeBotApi()
    # Operator returns None so all-quiet branch fires.
    operator = FakeOperator()
    svc = await _make_bot_with_db(db, bus, api, operator)
    try:
        # Force the "no operator" branch so it returns the all-quiet string.
        operator.status_payload = None
        # Override: cause get_status to raise so op stays None.
        async def boom(_sid):
            raise RuntimeError("nope")
        operator.get_status = boom  # type: ignore[method-assign]
        text = await svc._render_status()
    finally:
        await svc.stop()
    assert text == "All quiet. No tasks, no pending approvals, no unread DMs."


@pytest.mark.asyncio
async def test_status_summarizes_tasks_approvals_inbox_and_operator(db, bus):
    api = FakeBotApi()
    operator = FakeOperator(status={
        "model": "openai/gpt-oss-20b",
        "memory_entries": 7,
        "compression_threshold_tokens": 64000,
        "session_id": bot_session_id(OWNER_ID),
        "session_messages_since_summary": 12,
        "estimated_context_tokens": 16000,
        "latest_summary": None,
    })
    svc = await _make_bot_with_db(db, bus, api, operator)
    try:
        await db.insert_task(Task(
            session_id=str(uuid4()), prompt="check staging health",
            state=TaskState.RUNNING,
        ))
        await db.insert_task(Task(
            session_id=str(uuid4()), prompt="refactor auth handler",
            state=TaskState.PENDING,
        ))
        await db.insert_task(Task(
            session_id=str(uuid4()), prompt="anything",
            state=TaskState.AWAITING_APPROVAL,
        ))
        # Seed an unread inbox row.
        await db.record_inbox(
            inbox_id="i1", platform="telegram", chat_id="999",
            message_id="1", sender_username="alex",
            sender_display_name="Alex", body="hi",
            is_important=False,
            received_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        )
        text = await svc._render_status()
    finally:
        await svc.stop()
    # Top-line summary counts.
    assert "1 running, 1 queued, 1 awaiting approval" in text
    assert "Unread DMs: 1" in text
    # Task previews surface.
    assert "check staging health" in text
    assert "refactor auth handler" in text
    # Operator section is rendered.
    assert "openai/gpt-oss-20b" in text
    assert "7 entries" in text
    assert "16000 tokens" in text and "64000 threshold" in text
    assert "25%" in text                # 16000 / 64000
    assert "12 msgs since last compression" in text
    assert "none yet" in text           # latest_summary was None
    # get_status was called with the bot's session_id.
    assert operator.status_calls == [bot_session_id(OWNER_ID)]


@pytest.mark.asyncio
async def test_status_command_ignored_from_non_owner(db, bus):
    """Slash commands inherit the owner gate — a stranger sending /status
    must get nothing back and must NOT touch the DB/operator path."""
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot_with_db(db, bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=99999, text="/status")["message"])
        assert api.sent == []
        assert operator.status_calls == []
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_status_command_dispatches_locally_without_operator_turn(db, bus):
    """`/status` must NOT route to operator.chat_turn — it's a local readout."""
    api = FakeBotApi()
    operator = FakeOperator()
    svc = await _make_bot_with_db(db, bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/status")["message"])
        # /status is local; chat_turn must not have been invoked.
        assert operator.calls == []
        # Exactly one outbound message (the status snapshot).
        assert len(api.sent) == 1
        body = api.sent[0]["text"]
        assert "oncall status" in body or "All quiet" in body
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# /clear and /compress commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_command_calls_operator_and_reports(bus):
    api = FakeBotApi()
    operator = FakeOperator(
        clear_result={"messages_deleted": 7, "summaries_deleted": 1},
    )
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/clear")["message"])
        assert operator.clear_calls == [svc.session_id]
        # Bot must NOT route /clear through chat_turn.
        assert operator.calls == []
        assert len(api.sent) == 1
        body = api.sent[0]["text"]
        assert "Context cleared" in body
        assert "7 messages" in body and "1 summaries" in body
        assert "Memory preserved" in body
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_compress_command_reports_success(bus):
    api = FakeBotApi()
    operator = FakeOperator(compress_result={
        "compressed": True, "older_rows": 12,
        "summary_tokens": 84, "through_message_id": 99,
    })
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/compress")["message"])
        assert operator.compress_calls == [svc.session_id]
        assert len(api.sent) == 1
        body = api.sent[0]["text"]
        assert "Compressed 12 messages" in body
        assert "84 tokens" in body
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_compress_command_reports_nothing_to_compress(bus):
    api = FakeBotApi()
    operator = FakeOperator(compress_result={
        "compressed": False, "reason": "not enough history",
    })
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/compress")["message"])
        assert len(api.sent) == 1
        assert "Nothing to compress" in api.sent[0]["text"]
        assert "not enough history" in api.sent[0]["text"]
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_heavy_ops_are_serialized_per_bot(bus):
    """While /compress (or /context) is mid-flight, a second long-running
    command must fast-fail with a busy message instead of queueing. This
    matters because the operator's session lock would otherwise let the
    second call merely WAIT, leaving the user typing into a black hole
    with no signal that the first is still working.

    Asserts the cross-command interaction: /context-while-/compress and
    vice versa, plus that the gate releases on completion."""
    api = FakeBotApi()

    # An operator whose compress_now blocks until released — lets us
    # exercise the "another command arrived while one was running" branch.
    compress_started = asyncio.Event()
    compress_release = asyncio.Event()

    class GatedOperator(FakeOperator):
        async def compress_now(self, session_id):  # type: ignore[override]
            self.compress_calls.append(session_id)
            compress_started.set()
            await compress_release.wait()
            return {"compressed": True, "older_rows": 1, "summary_tokens": 1}

    operator = GatedOperator()
    svc = await _make_bot(bus, api, operator)
    try:
        # Kick off /compress and let it park.
        first = asyncio.create_task(
            svc._dispatch(_msg(sender_id=OWNER_ID, text="/compress")["message"])
        )
        await compress_started.wait()

        # Second /compress arrives — must NOT start the operator method again.
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/compress")["message"])
        assert len(operator.compress_calls) == 1, "second /compress should be rejected"
        assert "still running" in api.sent[-1]["text"].lower()

        # /context while /compress is in flight — also rejected, no
        # export_context call.
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/context")["message"])
        assert operator.export_calls == [], "/context should not run during /compress"
        assert api.documents == []
        assert "still running" in api.sent[-1]["text"].lower()

        # Release the first /compress; gate clears.
        compress_release.set()
        await first

        # Now /context works again.
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/context")["message"])
        assert operator.export_calls == [svc.session_id]
        assert len(api.documents) == 1
    finally:
        compress_release.set()
        await svc.stop()


@pytest.mark.asyncio
async def test_help_lists_clear_compress_context(bus):
    api = FakeBotApi()
    svc = await _make_bot(bus, api, FakeOperator())
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/help")["message"])
        body = api.sent[0]["text"]
        assert "/clear" in body
        assert "/compress" in body
        assert "/context" in body
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# /context — operator context export as a Telegram document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_registers_slash_commands(bus):
    """Booting the bot must call setMyCommands so the user sees the menu.

    Telegram clients pick the most-specific scope; for an owner-DM-only
    bot, `all_private_chats` is the scope that actually drives the
    autocomplete in the chat. We also publish under `default` as a
    fallback for clients that key off it. If either call is missing the
    menu may silently fail to populate."""
    api = FakeBotApi()
    svc = await _make_bot(bus, api, FakeOperator())
    try:
        set_calls = [c for c in api.calls if c[0] == "setMyCommands"]
        assert len(set_calls) == 2, api.calls
        scopes = {c[1]["scope"]["type"] for c in set_calls}
        assert scopes == {"all_private_chats", "default"}
        for _, payload in set_calls:
            names = {c["command"] for c in payload["commands"]}
            assert {"context", "clear", "compress", "status", "help"} <= names
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_context_command_uploads_markdown_document(bus):
    payload = "# Operator context — session tg-bot-99\n\nfoo bar"
    api = FakeBotApi()
    operator = FakeOperator(export_payload=payload)
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/context")["message"])
        # Routed to the operator, not chat_turn.
        assert operator.export_calls == [svc.session_id]
        assert operator.calls == []
        # No inline sendMessage for /context — the payload goes via sendDocument.
        assert api.sent == []
        assert len(api.documents) == 1
        doc = api.documents[0]
        assert doc["filename"].startswith(f"oncall-context-{svc.session_id}-")
        assert doc["filename"].endswith(".md")
        assert doc["content"] == payload.encode("utf-8")
        assert doc["caption"]
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_context_command_handles_export_failure(bus):
    api = FakeBotApi()

    class BoomOperator(FakeOperator):
        async def export_context(self, session_id):  # type: ignore[override]
            raise RuntimeError("db locked")

    svc = await _make_bot(bus, api, BoomOperator())
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="/context")["message"])
        # User got an inline error, no document uploaded.
        assert api.documents == []
        assert len(api.sent) == 1
        assert "Failed to export" in api.sent[0]["text"]
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# Formatting helpers — pure functions
# ---------------------------------------------------------------------------

def test_truncate_short_passthrough():
    assert _truncate("hello", 10) == "hello"


def test_truncate_collapses_newlines_and_trims():
    assert _truncate("  hi\nworld  ", 20) == "hi world"


def test_truncate_long_gets_ellipsis():
    out = _truncate("x" * 100, 10)
    assert len(out) == 10 and out.endswith("…")


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"),
    (45, "45s"),
    (60, "1m"),
    (3599, "59m"),
    (3600, "1h"),
    (47 * 3600, "47h"),
    (48 * 3600, "2d"),
])
def test_format_seconds_buckets(seconds, expected):
    assert _format_seconds(seconds) == expected


def test_relative_age_unparseable_returns_unknown():
    assert _relative_age("not a date") == "unknown"
    assert _relative_age("") == "unknown"


# ---------------------------------------------------------------------------
# Owner-sent media (photo / document) → chat_turn(attachments=[...])
# ---------------------------------------------------------------------------


def _photo_msg(*, sender_id: int, caption: str = "", chat_id: int = 999) -> dict:
    """Minimal Telegram `message` payload carrying a photo. Telegram sends
    `photo` as an array of size variants — the bot picks the LAST one."""
    return {
        "message_id": 5,
        "date": 0,
        "from": {"id": sender_id, "is_bot": False, "first_name": "Owner"},
        "chat": {"id": chat_id, "type": "private"},
        "photo": [
            {"file_id": "tiny",  "file_unique_id": "u_tiny",  "width": 64,  "height": 64,  "file_size": 1234},
            {"file_id": "small", "file_unique_id": "u_small", "width": 320, "height": 240, "file_size": 5678},
            {"file_id": "large", "file_unique_id": "u_large", "width": 1280, "height": 960, "file_size": 90000},
        ],
        "caption": caption,
    }


@pytest.mark.asyncio
async def test_owner_photo_with_caption_routes_attachment_to_chat_turn(bus):
    """Photo + caption: caption becomes user_text; download_file fires on
    the LARGEST variant; bytes are passed as a single base64-encoded
    attachment to operator.chat_turn."""
    api = FakeBotApi()
    api._attachment_payload = (b"\x89PNG\r\n\x1a\nlive-bytes", "image/png", "shot.jpg")
    operator = FakeOperator(reply_text="looks ok.")
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_photo_msg(sender_id=OWNER_ID, caption="what's wrong?"))
    finally:
        await svc.stop()

    dl_calls = [c for c in api.calls if c[0] == "download_file"]
    assert dl_calls and dl_calls[0][1]["file_id"] == "large"

    assert len(operator.calls) == 1
    call = operator.calls[0]
    assert call["user_text"] == "what's wrong?"
    assert call["attachments"] is not None and len(call["attachments"]) == 1
    att = call["attachments"][0]
    import base64
    assert base64.b64decode(att["data_b64"]) == b"\x89PNG\r\n\x1a\nlive-bytes"
    assert att["mime_type"] == "image/png"
    assert att["size_bytes"] == len(b"\x89PNG\r\n\x1a\nlive-bytes")
    assert "telegram bot" in att["source"]


@pytest.mark.asyncio
async def test_owner_photo_without_caption_gets_default_prompt(bus):
    """Pure photo (no caption): user_text falls back to a generic 'please
    look at the image' prompt so the operator has SOMETHING to react to
    instead of an empty string."""
    api = FakeBotApi()
    operator = FakeOperator(reply_text="i see a graph.")
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_photo_msg(sender_id=OWNER_ID, caption=""))
    finally:
        await svc.stop()
    call = operator.calls[0]
    assert "please look at the image" in call["user_text"]
    assert call["attachments"] is not None and len(call["attachments"]) == 1


@pytest.mark.asyncio
async def test_owner_text_only_still_passes_attachments_none(bus):
    """Plain text message (no photo / document) → attachments=None.
    Regression: don't manufacture an empty list."""
    api = FakeBotApi()
    operator = FakeOperator(reply_text="ok")
    svc = await _make_bot(bus, api, operator)
    try:
        await svc._dispatch(_msg(sender_id=OWNER_ID, text="hi")["message"])
    finally:
        await svc.stop()
    assert operator.calls[0]["attachments"] is None
    assert [c[0] for c in api.calls if c[0] == "download_file"] == []
