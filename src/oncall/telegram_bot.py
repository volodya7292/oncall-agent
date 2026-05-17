"""Telegram bot front-end for the operator.

Talks to Telegram via the HTTP Bot API (`https://api.telegram.org/bot<TOKEN>/...`),
NOT MTProto. That means the bot is fully decoupled from the userbot path —
the only env var required is `TELEGRAM_BOT_TOKEN` (and `TELEGRAM_BOT_OWNER_ID`
for the allowlist). No api_id/api_hash needed.

Distinct from telegram_service.py:
  * telegram_service.py = USERBOT (acts as the user's own account via MTProto).
    Reads inbound DMs from arbitrary senders for triage + reply-by-proposal.
  * telegram_bot.py     = BOT (a separate account via the HTTP Bot API). The
    only thing the user explicitly talks to. Only OWNER_ID can DM it.

Auto-ping replies (`chat.reply` events) for this bot's session_id are
delivered automatically so the user gets a follow-up DM when a dispatched
task terminates — no need to ask.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

import httpx

from .audit import fmt, telegram_log
from .broker import Broker
from .db import Database
from .events import EventBus
from .models import TaskState
from .operator import Operator


log = logging.getLogger(__name__)


# Telegram caps a single message at 4096 chars. Stay slightly under for headroom.
_TELEGRAM_MSG_LIMIT = 4000
# Long-poll: how long Telegram holds the request open if no updates. The Bot
# API caps timeout at ~50s; 25s is a comfortable middle ground.
_LONG_POLL_SECONDS = 25
# Backoff between failed getUpdates calls (network issues, 5xx, rate limits).
_RETRY_DELAY_SECONDS = 3.0


def bot_session_id(owner_user_id: int) -> str:
    """Deterministic chat-session id for the bot's conversation with the owner.
    One owner ↔ one session, persistent across daemon restarts."""
    return f"tg-bot-{owner_user_id}"


def chunk_message(text: str, *, limit: int = _TELEGRAM_MSG_LIMIT) -> list[str]:
    """Split a message into ≤limit-char chunks, preferring newline boundaries.
    Pure function — easy to unit-test."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


# ---------------------------------------------------------------------------
# BotApi: thin transport for the HTTP Bot API
# ---------------------------------------------------------------------------

class BotApi(Protocol):
    """Minimal slice of the Telegram Bot API we use. Tests inject a fake;
    production uses HttpxBotApi (real network)."""

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> Any: ...
    async def aclose(self) -> None: ...


class HttpxBotApi:
    """HTTP transport for the Bot API, backed by httpx.AsyncClient."""

    def __init__(self, token: str, *, timeout: float = 60.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=f"https://api.telegram.org/bot{token}",
            timeout=timeout,
        )

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        r = await self._http.post(f"/{method}", json=payload or {})
        # Telegram returns {"ok": bool, "result"?: ..., "description"?: str}.
        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"bot API {method}: non-JSON response (HTTP {r.status_code})") from e
        if not data.get("ok"):
            raise RuntimeError(
                f"bot API {method} failed: {data.get('description')} (HTTP {r.status_code})"
            )
        return data.get("result")

    async def aclose(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TelegramBotService:
    def __init__(
        self,
        *,
        api: BotApi,
        operator: Operator,
        events: EventBus,
        owner_user_id: int,
        broker: Broker | None = None,
        db: Database | None = None,
    ) -> None:
        self._api = api
        self._operator = operator
        self._events = events
        self._owner_user_id = owner_user_id
        # broker + db are optional only because tests that don't exercise the
        # approval path can omit them. In production both are required for
        # the inline-keyboard Yes/No flow.
        self._broker = broker
        self._db = db
        self._session_id = bot_session_id(owner_user_id)
        self._poll_task: asyncio.Task | None = None
        self._reply_task: asyncio.Task | None = None
        self._approval_task: asyncio.Task | None = None
        self._update_offset: int = 0
        self._bot_username: str | None = None
        self._bot_user_id: int | None = None
        self._started = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def bot_username(self) -> str | None:
        return self._bot_username

    @property
    def bot_user_id(self) -> int | None:
        """The bot's own numeric Telegram user_id, captured from `getMe` at
        startup. Used so the userbot can filter the bot's replies out of its
        inbox stream — otherwise every bot reply would show up as an
        'incoming DM' from the user's own bot account."""
        return self._bot_user_id

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._started:
            return
        # Fail fast if the token is invalid — better than discovering on first
        # poll. Also captures the bot's own username + numeric id so the
        # userbot can auto-add this bot to its ignore filter.
        me = await self._api.call("getMe")
        if isinstance(me, dict):
            self._bot_username = me.get("username")
            try:
                self._bot_user_id = int(me["id"]) if me.get("id") is not None else None
            except (TypeError, ValueError):
                self._bot_user_id = None
        self._poll_task = asyncio.create_task(self._poll_loop(), name="tg-bot-poll")
        self._reply_task = asyncio.create_task(
            self._chat_reply_subscriber(), name="tg-bot-reply",
        )
        # Approval-request subscriber: send inline Yes/No buttons whenever a
        # task dispatched in this bot's session needs approval.
        if self._broker is not None and self._db is not None:
            self._approval_task = asyncio.create_task(
                self._approval_subscriber(), name="tg-bot-approval",
            )
        self._started = True
        log.info(
            "telegram bot started (owner_user_id=%d, username=@%s, session=%s)",
            self._owner_user_id, self._bot_username, self._session_id,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        for task in (self._poll_task, self._reply_task, self._approval_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        try:
            await self._api.aclose()
        except Exception:
            log.exception("error closing bot api transport")
        self._started = False
        log.info("telegram bot stopped")

    # ---- polling ----

    async def _poll_loop(self) -> None:
        """Long-poll `getUpdates` and dispatch each update. Messages → operator;
        callback_query (button taps) → approval resolver. The polling loop
        must keep pulling so we don't miss updates, so handlers run in spawned
        tasks."""
        while True:
            try:
                updates = await self._api.call("getUpdates", {
                    "offset": self._update_offset,
                    "timeout": _LONG_POLL_SECONDS,
                    "allowed_updates": ["message", "callback_query"],
                })
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("getUpdates failed; backing off")
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
                continue
            if not isinstance(updates, list):
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
                continue
            for upd in updates:
                try:
                    self._update_offset = max(self._update_offset, int(upd["update_id"]) + 1)
                except (KeyError, ValueError, TypeError):
                    continue
                if msg := upd.get("message"):
                    asyncio.create_task(self._handle_message(msg))
                elif cq := upd.get("callback_query"):
                    asyncio.create_task(self._handle_callback(cq))

    # ---- handler ----

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        try:
            await self._dispatch(msg)
        except Exception:
            log.exception("telegram bot message handler crashed")

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        sender = msg.get("from") or {}
        sender_id = sender.get("id")
        chat_id = (msg.get("chat") or {}).get("id")
        if sender_id != self._owner_user_id:
            telegram_log.warning("bot inbound from non-owner " + fmt(
                sender_id=sender_id, owner_id=self._owner_user_id,
                username=sender.get("username"),
            ))
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return

        # Slash commands handled locally — don't burn an operator turn on them.
        if text.startswith("/start"):
            await self._send(chat_id, (
                "Hi. I'm your on-call operator. Tell me what you need — "
                "I'll dispatch tasks and ping you back when they're done."
            ))
            return
        if text.startswith("/help"):
            await self._send(chat_id, (
                "/start — greeting\n"
                "/status — snapshot of running tasks, queue, approvals, unread DMs\n"
                "/help — this\n"
                "Anything else is a chat turn."
            ))
            return
        if text.startswith("/status"):
            await self._send(chat_id, await self._render_status())
            return

        telegram_log.info("bot inbound " + fmt(
            session=self._session_id, len=len(text),
        ))
        try:
            result = await self._operator.chat_turn(
                session_id=self._session_id, user_text=text,
            )
        except Exception:
            log.exception("operator.chat_turn failed for telegram bot")
            await self._send(chat_id, "Internal error. Try again in a moment.")
            return

        reply = result.text or "(empty reply)"
        await self._send(chat_id, reply)
        telegram_log.info("bot reply " + fmt(
            session=self._session_id, len=len(reply),
            tool_calls=len(result.tool_calls_made),
        ))

    # ---- /status renderer ----

    async def _render_status(self) -> str:
        """Build a one-message snapshot of the orchestrator's state. Queries
        the DB directly (no operator turn) so /status stays cheap and never
        waits behind a busy chat session."""
        if self._db is None:
            return "Status unavailable (DB not wired)."

        running = await self._db.list_tasks_in_states(TaskState.RUNNING)
        queued = await self._db.list_tasks_in_states(TaskState.PENDING)
        awaiting = await self._db.list_tasks_in_states(TaskState.AWAITING_APPROVAL)
        approvals = await self._db.list_pending_approvals()
        unread = await self._db.list_inbox(unread_only=True, limit=200)

        # Sort lists deterministically: running by age (oldest first — longest
        # in flight deserves attention), queued by created_at (FIFO).
        running.sort(key=lambda t: t.created_at)
        queued.sort(key=lambda t: t.created_at)

        lines = [
            "oncall status",
            "",
            f"Tasks: {len(running)} running, {len(queued)} queued, "
            f"{len(awaiting)} awaiting approval",
            f"Approvals pending: {len(approvals)}",
            f"Unread DMs: {len(unread)}",
        ]

        if running:
            lines += ["", "Running:"]
            for t in running[:5]:
                lines.append(f"- {str(t.id)[:6]} ({_age(t.created_at)}): {_truncate(t.prompt, 70)}")
            if len(running) > 5:
                lines.append(f"- ...and {len(running) - 5} more")

        if queued:
            lines += ["", "Queued:"]
            for t in queued[:5]:
                lines.append(f"- {str(t.id)[:6]}: {_truncate(t.prompt, 70)}")
            if len(queued) > 5:
                lines.append(f"- ...and {len(queued) - 5} more")

        # Operator-side state: model, memory size, this session's context
        # usage, last compression checkpoint. Cheap — DB reads only.
        try:
            op = await self._operator.get_status(self._session_id)
        except Exception:
            log.exception("operator.get_status failed for /status")
            op = None

        if op is not None:
            lines += ["", "Operator:"]
            lines.append(f"- model: {op['model']}")
            lines.append(f"- memory: {op['memory_entries']} entries")
            est = op["estimated_context_tokens"]
            thr = op["compression_threshold_tokens"]
            pct = (100 * est // thr) if thr else 0
            lines.append(
                f"- context: ~{est} tokens / {thr} threshold ({pct}%, "
                f"{op['session_messages_since_summary']} msgs since last compression)"
            )
            last = op["latest_summary"]
            if last:
                created = last.get("created_at") or ""
                lines.append(
                    f"- last compression: {_relative_age(created)} "
                    f"(through msg #{last['through_message_id']}, "
                    f"~{last['estimated_token_count']} summary tokens)"
                )
            else:
                lines.append("- last compression: none yet")

        if not (running or queued or approvals or unread) and op is None:
            return "All quiet. No tasks, no pending approvals, no unread DMs."

        return "\n".join(lines)

    # ---- chat.reply auto-ping subscriber ----

    async def _chat_reply_subscriber(self) -> None:
        """Push `chat.reply` events tagged with this bot's session_id to the
        owner — that's how an auto-ping (operator's follow-up after a task
        terminates) reaches the user without them having to ask.

        For approval triggers, we DROP the operator's text and let
        `_approval_subscriber` send the button message instead — otherwise the
        user would see the operator's verbose canonical-command readback AND
        the [Yes][No] keyboard for the same approval."""
        async for env in self._events.subscribe_global(types={"chat.reply"}):
            payload = env.get("payload") or {}
            if payload.get("session_id") != self._session_id:
                continue
            if payload.get("trigger") == "approval.requested":
                continue  # button UI takes over
            text = payload.get("text") or ""
            if not text:
                continue
            try:
                await self._send(self._owner_user_id, text)
                telegram_log.info("bot auto-ping " + fmt(
                    session=self._session_id, len=len(text),
                    task_id=payload.get("task_id"),
                ))
            except Exception:
                log.exception("failed to deliver chat.reply via telegram bot")

    async def _approval_subscriber(self) -> None:
        """Send an inline-keyboard Yes/No message when a task in this bot's
        session needs approval. Only runs if the bot was constructed with a
        broker + db."""
        assert self._broker is not None and self._db is not None
        async for env in self._events.subscribe_global(types={"approval.requested"}):
            task_id_str = env.get("task_id")
            payload = env.get("payload") or {}
            approval_id = payload.get("approval_id")
            if not (task_id_str and approval_id):
                continue
            try:
                task = await self._db.get_task(UUID(task_id_str))
            except Exception:
                log.exception("approval subscriber: load task %s", task_id_str)
                continue
            if task is None or task.dispatched_by_chat_session != self._session_id:
                continue
            canonical = (payload.get("canonical_command") or "").strip()
            blast = (payload.get("blast_radius") or "").strip()
            body = f"Approve this command?\n\n`{canonical}`"
            if blast:
                body += f"\n\n{blast}"
            try:
                await self._api.call("sendMessage", {
                    "chat_id": self._owner_user_id,
                    "text": body,
                    "parse_mode": "Markdown",
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": "✅ Yes", "callback_data": f"appr:{approval_id}:allow"},
                            {"text": "❌ No",  "callback_data": f"appr:{approval_id}:deny"},
                        ]],
                    },
                })
                telegram_log.info("bot approval prompt " + fmt(
                    session=self._session_id, approval=approval_id,
                    task=task_id_str, canonical=canonical,
                ))
            except Exception:
                log.exception("failed to send approval prompt for %s", approval_id)

    # ---- callback (Yes/No tap) handler ----

    async def _handle_callback(self, cq: dict[str, Any]) -> None:
        try:
            await self._dispatch_callback(cq)
        except Exception:
            log.exception("telegram bot callback handler crashed")

    async def _dispatch_callback(self, cq: dict[str, Any]) -> None:
        cq_id = cq.get("id")
        sender_id = (cq.get("from") or {}).get("id")
        data = cq.get("data") or ""
        if sender_id != self._owner_user_id:
            telegram_log.warning("bot callback from non-owner " + fmt(
                sender_id=sender_id, owner_id=self._owner_user_id,
            ))
            # Always answer the callback so the client doesn't spin forever.
            if cq_id:
                await self._safe_answer_callback(cq_id, "Not authorized.")
            return
        # Parse `appr:<approval_id>:<decision>`.
        parts = data.split(":", 2)
        if len(parts) != 3 or parts[0] != "appr":
            if cq_id:
                await self._safe_answer_callback(cq_id, "Unknown action.")
            return
        approval_id_str = parts[1]
        decision = parts[2]
        if decision not in {"allow", "deny"}:
            if cq_id:
                await self._safe_answer_callback(cq_id, "Bad decision.")
            return

        if self._broker is None or self._db is None:
            if cq_id:
                await self._safe_answer_callback(cq_id, "Broker not wired.")
            return

        try:
            approval_uuid = UUID(approval_id_str)
        except ValueError:
            if cq_id:
                await self._safe_answer_callback(cq_id, "Bad approval id.")
            return

        row = await self._db.get_approval(approval_uuid)
        if row is None:
            if cq_id:
                await self._safe_answer_callback(cq_id, "Unknown approval.")
            return
        if row["state"] != "pending":
            if cq_id:
                await self._safe_answer_callback(cq_id, "Already resolved.")
            await self._maybe_edit_resolved(cq, row["decision"] or "?")
            return

        phrase = row["challenge_phrase"] or ""
        approved, matched = await self._broker.submit_response(
            approval_id=approval_uuid,
            decision=decision,
            challenge_phrase_supplied=phrase,
        )
        outcome = "allow" if approved else "deny"
        telegram_log.info("bot approval resolve " + fmt(
            approval=approval_id_str, decision=decision,
            approved=approved, matched=matched,
        ))
        if cq_id:
            await self._safe_answer_callback(
                cq_id,
                "Approved ✓" if approved else "Denied ✗",
            )
        await self._maybe_edit_resolved(cq, outcome)

    async def _safe_answer_callback(self, callback_id: str, text: str) -> None:
        try:
            await self._api.call("answerCallbackQuery", {
                "callback_query_id": callback_id, "text": text,
            })
        except Exception:
            log.exception("answerCallbackQuery failed")

    async def _maybe_edit_resolved(self, cq: dict[str, Any], outcome: str) -> None:
        """Strip the buttons off the original prompt + annotate the outcome
        so the chat scrollback reflects what happened."""
        msg = cq.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        message_id = msg.get("message_id")
        original = msg.get("text") or ""
        if chat_id is None or message_id is None:
            return
        marker = "✅ Approved" if outcome == "allow" else "❌ Denied"
        new_text = f"{original}\n\n_{marker}._"
        try:
            await self._api.call("editMessageText", {
                "chat_id": chat_id, "message_id": message_id,
                "text": new_text, "parse_mode": "Markdown",
                "reply_markup": {"inline_keyboard": []},
            })
        except Exception:
            log.exception("editMessageText failed for resolved approval")

    # ---- send ----

    async def _send(self, chat_id: Any, text: str) -> None:
        if chat_id is None or not text:
            return
        for piece in chunk_message(text):
            await self._api.call("sendMessage", {
                "chat_id": chat_id, "text": piece,
            })


# ---------------------------------------------------------------------------
# Formatting helpers (module-scope so tests can exercise them directly)
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _age(when: datetime) -> str:
    """Compact human age: 5s / 12m / 3h / 4d. `when` is timezone-aware UTC
    (Task.created_at)."""
    return _format_seconds((datetime.now(timezone.utc) - when).total_seconds())


def _relative_age(iso_string: str) -> str:
    """Age of an ISO-formatted timestamp string. Returns 'unknown' if it
    can't be parsed."""
    try:
        when = datetime.fromisoformat(iso_string)
    except (TypeError, ValueError):
        return "unknown"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _format_seconds((datetime.now(timezone.utc) - when).total_seconds()) + " ago"


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"
