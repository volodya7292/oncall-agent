"""Telegram userbot — long-lived listener + send/style read.

This is a *userbot* (MTProto), not a bot account. Bot accounts can't see DMs
from arbitrary people; only userbots reading on behalf of the user's own
account can. That's exactly what's needed for the reply-by-proposal flow.

The service lives in the orchestrator process (single long-lived asyncio task,
not per-Claude-task) because:
  * Telegram requires a persistent connection for inbound updates.
  * The MCP server is a stdio child of `claude` and dies between tasks.

Operations exposed:
  * inbound: NewMessage handler writes to messenger_inbox + publishes a global event.
  * `list_inbox` / `get_message` / `mark_read`     — pure DB.
  * `get_chat_style(chat_id, limit)`               — reads the USER'S OWN recent
    outgoing messages in that chat, so the agent can mimic the user's voice
    (length, tone, emoji, language) before drafting a reply.
  * `send(chat_id, text)`                          — telethon send_message;
    routed through the broker by the time it's reached (via MCP messenger_inbox
    tool + --permission-prompt-tool flow).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from .audit import fmt, telegram_log
from .db import Database


ARCHIVED_CACHE_TTL_SECONDS = 1800.0  # 30 min — archive state changes rarely


log = logging.getLogger(__name__)


# Type for the optional callback the service fires on each new inbound DM.
NewMessageCallback = Callable[[dict[str, Any]], Awaitable[None]]


class TelegramClientLike(Protocol):
    """The slice of telethon.TelegramClient we depend on. Lets tests inject a fake."""
    async def connect(self) -> Any: ...
    async def disconnect(self) -> Any: ...
    async def is_user_authorized(self) -> bool: ...
    async def send_message(self, entity: Any, message: str) -> Any: ...
    def iter_messages(self, entity: Any, **kwargs: Any) -> Any: ...
    def iter_dialogs(self, **kwargs: Any) -> Any: ...
    def add_event_handler(self, callback: Any, event: Any = None) -> Any: ...
    def remove_event_handler(self, callback: Any, event: Any = None) -> Any: ...


class TelegramService:
    """Owns the long-lived telethon client + the inbound DM pipeline."""

    def __init__(
        self,
        *,
        db: Database,
        client: TelegramClientLike,
        important_senders: set[str],
        important_keywords: set[str],
        on_new_message: NewMessageCallback | None = None,
        archived_cache_ttl: float = ARCHIVED_CACHE_TTL_SECONDS,
    ) -> None:
        self._db = db
        self._client = client
        self._important_senders = {s.lstrip("@").lower() for s in important_senders}
        self._important_keywords = {k.lower() for k in important_keywords}
        self._on_new_message = on_new_message
        self._handler_ref: Any = None
        self._started = False
        # Archived-chats cache: { chat_id_str }. Telegram users archive chats
        # they want hidden from main view without muting/blocking; we treat
        # archived chats as "do not surface" unless the user explicitly asks.
        self._archived: set[str] = set()
        self._archived_refreshed_at: float = 0.0
        self._archived_cache_ttl = archived_cache_ttl

    @property
    def is_started(self) -> bool:
        return self._started

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._started:
            return
        await self._client.connect()
        if not await self._client.is_user_authorized():
            raise RuntimeError(
                "Telegram session not authorized — run `oncall telegram-login` first."
            )
        # Import telethon's NewMessage filter lazily so tests don't need it.
        try:
            from telethon import events  # type: ignore
            event_filter: Any = events.NewMessage(incoming=True)
        except ImportError:
            event_filter = None  # tests pass a fake client that accepts None
        self._handler_ref = self._build_handler()
        self._client.add_event_handler(self._handler_ref, event_filter)
        # Prime the archived cache so the very first inbound DM gets filtered.
        await self._refresh_archived(force=True)
        self._started = True
        log.info("telegram listener started (archived chats cached: %d)", len(self._archived))

    async def stop(self) -> None:
        if not self._started:
            return
        try:
            self._client.remove_event_handler(self._handler_ref)
        except Exception:
            log.exception("error removing telegram handler")
        try:
            await self._client.disconnect()
        except Exception:
            log.exception("error disconnecting telegram client")
        self._started = False
        log.info("telegram listener stopped")

    # ---- archived-chats cache ----

    async def _refresh_archived(self, *, force: bool = False) -> None:
        """Repopulate the archived-chat-id set from telethon. Best-effort —
        on failure we keep the previous cache (better to over-surface than
        to silently drop after an API blip)."""
        if not force and (time.monotonic() - self._archived_refreshed_at) < self._archived_cache_ttl:
            return
        try:
            fresh: set[str] = set()
            async for dlg in self._client.iter_dialogs(archived=True):
                cid = _dialog_chat_id(dlg)
                if cid is not None:
                    fresh.add(cid)
            self._archived = fresh
            self._archived_refreshed_at = time.monotonic()
        except TypeError:
            # Older telethon may not accept `archived` kwarg — fall back to
            # iterating all dialogs and filtering on .archived.
            try:
                fresh = set()
                async for dlg in self._client.iter_dialogs():
                    if getattr(dlg, "archived", False):
                        cid = _dialog_chat_id(dlg)
                        if cid is not None:
                            fresh.add(cid)
                self._archived = fresh
                self._archived_refreshed_at = time.monotonic()
            except Exception:
                log.exception("failed to refresh archived chats (fallback path)")
        except Exception:
            log.exception("failed to refresh archived chats")

    async def _is_archived(self, chat_id: str, event: Any | None = None) -> bool:
        await self._refresh_archived()
        if chat_id in self._archived:
            return True
        # Cache miss: ask the event for its dialog. Cheap because telethon
        # caches dialog state on the event object. Defensive: any failure
        # → treat as non-archived (over-surface, never under-).
        if event is not None and hasattr(event, "get_dialog"):
            try:
                dlg = await _maybe_await(event.get_dialog())
                if dlg is not None and getattr(dlg, "archived", False):
                    self._archived.add(chat_id)
                    return True
            except Exception:
                pass
        return False

    # ---- handler ----

    def _build_handler(self) -> Callable[[Any], Awaitable[None]]:
        async def _on_new_message(event: Any) -> None:
            try:
                await self._handle_inbound(event)
            except Exception:
                log.exception("telegram inbound handler failed")
        return _on_new_message

    async def _handle_inbound(self, event: Any) -> None:
        # MVP: private chats only. Telethon: event.is_private is True for 1:1 DMs.
        if not getattr(event, "is_private", False):
            return
        sender = await _maybe_await(event.get_sender()) if hasattr(event, "get_sender") else None
        if sender is None:
            return
        # Don't loop on our own outgoing messages.
        if getattr(sender, "is_self", False) or getattr(event.message, "out", False):
            return
        # Skip bots.
        if getattr(sender, "bot", False):
            return

        body = getattr(event.message, "message", None) or getattr(event, "raw_text", None) or ""
        if not isinstance(body, str) or not body.strip():
            return

        username = (getattr(sender, "username", None) or "").lower() or None
        display = _display_name(sender)
        chat_id = str(getattr(event, "chat_id", None) or getattr(event.message, "chat_id", ""))
        message_id = str(getattr(event.message, "id", ""))
        received_at = getattr(event.message, "date", None) or datetime.now(timezone.utc)

        # Archived = user has deliberately hidden this chat. Don't surface.
        if await self._is_archived(chat_id, event=event):
            log.debug("skipping inbound from archived chat %s", chat_id)
            return

        important = self._triage(username, body)
        inbox_id = str(uuid4())
        inserted = await self._db.record_inbox(
            inbox_id=inbox_id,
            platform="telegram",
            chat_id=chat_id,
            message_id=message_id,
            sender_username=username,
            sender_display_name=display,
            body=body,
            is_important=important,
            received_at=received_at,
        )
        if not inserted:
            return  # duplicate, skip

        telegram_log.info("inbound " + fmt(
            inbox=inbox_id, chat=chat_id, sender=username or display,
            important=important, body_len=len(body), body=body,
        ))

        if self._on_new_message is not None:
            row = await self._db.get_inbox_message(inbox_id)
            if row is not None:
                await self._on_new_message(row)

    def _triage(self, username: str | None, body: str) -> bool:
        if username and username in self._important_senders:
            return True
        body_l = body.lower()
        return any(kw in body_l for kw in self._important_keywords)

    # ---- DB-backed queries (no telethon needed) ----

    async def list_inbox(
        self, *, unread_only: bool = True, limit: int = 20,
    ) -> list[dict[str, Any]]:
        # Pull more than `limit` from DB so the archived filter doesn't starve
        # the result. 4× is a cheap heuristic; bump if archived ratio is high.
        await self._refresh_archived()
        rows = await self._db.list_inbox(unread_only=unread_only, limit=limit * 4)
        filtered = [r for r in rows if r["chat_id"] not in self._archived]
        return filtered[:limit]

    async def get_message(self, inbox_id: str) -> dict[str, Any] | None:
        return await self._db.get_inbox_message(inbox_id)

    async def mark_read(self, inbox_id: str) -> bool:
        return await self._db.mark_inbox_read(inbox_id)

    # ---- telethon-backed reads/writes ----

    async def get_chat_style(
        self, chat_id: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the USER'S OWN recent outgoing messages in this chat — the
        source material for mimicking the user's voice. Each entry is
        {'message_id', 'text', 'date'}.

        We deliberately filter to the user's own side (from_user='me'); reading
        the counterparty's messages would teach the model the wrong voice."""
        entity = _entity_arg(chat_id)
        samples: list[dict[str, Any]] = []
        # telethon's iter_messages is async-iterable
        async for msg in self._client.iter_messages(
            entity, limit=limit, from_user="me",
        ):
            text = getattr(msg, "message", None)
            if not text:
                continue
            samples.append({
                "message_id": str(getattr(msg, "id", "")),
                "text": text,
                "date": _iso_or_none(getattr(msg, "date", None)),
            })
        return samples

    async def send(self, chat_id: str, text: str) -> dict[str, Any]:
        entity = _entity_arg(chat_id)
        sent = await self._client.send_message(entity, text)
        sent_id = str(getattr(sent, "id", ""))
        telegram_log.info("send " + fmt(
            chat=chat_id, message_id=sent_id, len=len(text), text=text,
        ))
        return {"message_id": sent_id, "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display_name(sender: Any) -> str | None:
    first = getattr(sender, "first_name", None) or ""
    last = getattr(sender, "last_name", None) or ""
    name = f"{first} {last}".strip()
    return name or None


def _dialog_chat_id(dlg: Any) -> str | None:
    """Best-effort: pull the chat id off a telethon Dialog. Telethon exposes
    `.id` directly on Dialog; older shapes route through `.entity.id`."""
    cid = getattr(dlg, "id", None)
    if cid is None:
        entity = getattr(dlg, "entity", None)
        cid = getattr(entity, "id", None) if entity is not None else None
    return str(cid) if cid is not None else None


def _entity_arg(chat_id: str) -> Any:
    """Telethon accepts ints, strings, or username-like strings as the entity.
    Stored chat_id is a numeric string for personal chats; pass through as int
    when possible, else as the raw string."""
    s = chat_id.strip()
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Telethon client factory + login (interactive CLI)
# ---------------------------------------------------------------------------

def make_telethon_client(
    *, api_id: int, api_hash: str, session_path: Path,
) -> Any:
    """Construct a telethon.TelegramClient. Lazy import keeps tests free of
    the telethon dependency."""
    from telethon import TelegramClient  # type: ignore
    session_path.parent.mkdir(parents=True, exist_ok=True)
    return TelegramClient(str(session_path), api_id, api_hash)


async def login_interactive(
    *, api_id: int, api_hash: str, session_path: Path,
) -> None:
    """One-shot interactive login. Prompts for phone, code, optional 2FA password."""
    import getpass

    client = make_telethon_client(api_id=api_id, api_hash=api_hash, session_path=session_path)
    await client.start(
        phone=lambda: input("Telegram phone (e.g. +14155551234): ").strip(),
        code_callback=lambda: input("Code sent to Telegram: ").strip(),
        password=lambda: getpass.getpass("2FA password (blank if none): ") or None,
    )
    me = await client.get_me()
    log.info("logged in as @%s (%s)", getattr(me, "username", None), getattr(me, "id", None))
    await client.disconnect()
