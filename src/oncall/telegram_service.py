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
        ignore_usernames: set[str] | None = None,
        ignore_user_ids: set[int] | None = None,
    ) -> None:
        self._db = db
        self._client = client
        self._important_senders = {s.lstrip("@").lower() for s in important_senders}
        self._important_keywords = {k.lower() for k in important_keywords}
        self._on_new_message = on_new_message
        self._handler_ref: Any = None
        self._started = False
        # Senders whose messages should never reach the inbox. Two channels:
        #   * ignore_usernames: lowercased @handles (from env). Useful for
        #     blocking third-party bots that auto-DM you (e.g. @userinfobot).
        #   * ignore_user_ids: numeric ids (populated at runtime when the
        #     own-bot front-end starts and tells us its user_id, so the bot's
        #     own replies don't show up in your DM inbox stream).
        self._ignore_usernames: set[str] = {
            s.lstrip("@").lower() for s in (ignore_usernames or set())
        }
        self._ignore_user_ids: set[int] = set(ignore_user_ids or set())
        # Archived-chats cache: { chat_id_str }. Telegram users archive chats
        # they want hidden from main view without muting/blocking; we treat
        # archived chats as "do not surface" unless the user explicitly asks.
        self._archived: set[str] = set()
        self._archived_refreshed_at: float = 0.0
        self._archived_cache_ttl = archived_cache_ttl

    @property
    def is_started(self) -> bool:
        return self._started

    def add_ignore_user_id(self, user_id: int) -> None:
        """Add a numeric user_id whose messages should be dropped from the
        inbox. Safe to call any time — checked on each inbound."""
        self._ignore_user_ids.add(int(user_id))

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
        sender_id = getattr(sender, "id", None)
        if sender_id is not None and sender_id in self._ignore_user_ids:
            return
        if username is not None and username in self._ignore_usernames:
            return
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

    async def list_pending_chats(
        self, *, body_tail_chars: int = 500,
    ) -> list[dict[str, Any]]:
        """One entry per chat with unread DMs. Archived chats are excluded
        — they're the same things `list_inbox` filters out. Used by the
        inbox-drain triage path and by the operator's `read_inbox` tool
        in the new chat-centric flow (the operator sees the dirty chat
        and then calls `read_chat` for full context if it wants it)."""
        await self._refresh_archived()
        rows = await self._db.list_pending_chats(body_tail_chars=body_tail_chars)
        return [r for r in rows if r["chat_id"] not in self._archived]

    async def get_message(self, inbox_id: str) -> dict[str, Any] | None:
        return await self._db.get_inbox_message(inbox_id)

    async def mark_read(self, inbox_id: str) -> bool:
        return await self._db.mark_inbox_read(inbox_id)

    async def mark_chat_read(self, chat_id: str) -> int:
        """Mark every unread inbox row in this chat as read. Returns the
        rowcount affected. The operator calls this after the user says
        'skip / ignore / dismiss' a chat's pending DMs."""
        return await self._db.mark_chat_read(chat_id)

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

    async def get_chat_history(
        self, chat_id: str, *, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the last N messages of a chat, BOTH sides. Each row:
            message_id, text, date, outgoing, sender_username, sender_display_name, has_media.

        Media-only messages (photo / document / etc. with no caption) get
        a synthetic body like `[photo]` / `[file: name.pdf]` so the
        operator sees the message_id and can call `read_image(chat_id,
        message_id)` to load the bytes. Without this they were invisible
        and the operator would guess (wrongly) at nearby IDs.

        Unlike `get_chat_style` (which filters to from_user='me'), this
        includes the counterparty's messages — use it when the user asks
        'what did X say?'."""
        entity = _entity_arg(chat_id)
        out: list[dict[str, Any]] = []
        async for msg in self._client.iter_messages(entity, limit=limit):
            text = getattr(msg, "message", None) or ""
            has_media = bool(getattr(msg, "media", None))
            if not text and not has_media:
                continue  # system events, reactions, etc.
            if not text and has_media:
                text = _media_placeholder(msg)
            sender = (
                await _maybe_await(msg.get_sender())
                if hasattr(msg, "get_sender") else None
            )
            out.append({
                "message_id": str(getattr(msg, "id", "")),
                "text": text,
                "date": _iso_or_none(getattr(msg, "date", None)),
                "outgoing": bool(getattr(msg, "out", False)),
                "sender_username": getattr(sender, "username", None) if sender else None,
                "sender_display_name": _display_name(sender) if sender else None,
                "has_media": has_media,
            })
        return out

    async def list_chats(
        self, *, unread_only: bool = False, dms_only: bool = False, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Enumerate the user's recent Telegram dialogs (no name query).
        Telethon returns dialogs in last-activity order. Archived chats are
        filtered out (matches read_inbox / search_chats behaviour).
        - `unread_only=True` keeps only dialogs with unread_count > 0.
        - `dms_only=True` keeps only private chats (1:1)."""
        await self._refresh_archived()
        out: list[dict[str, Any]] = []
        async for dlg in self._client.iter_dialogs():
            cid = _dialog_chat_id(dlg)
            if cid is None or cid in self._archived:
                continue
            is_user = bool(getattr(dlg, "is_user", False))
            unread = int(getattr(dlg, "unread_count", 0) or 0)
            if unread_only and unread <= 0:
                continue
            if dms_only and not is_user:
                continue
            entity = getattr(dlg, "entity", None)
            username = (getattr(entity, "username", None) or "") if entity else ""
            out.append({
                "chat_id": cid,
                "name": getattr(dlg, "name", None) or "",
                "username": username or None,
                "is_user": is_user,
                "is_group": bool(getattr(dlg, "is_group", False)),
                "is_channel": bool(getattr(dlg, "is_channel", False)),
                "unread_count": unread,
                "archived": False,
            })
            if len(out) >= limit:
                break
        return out

    async def search_messages(
        self, chat_id: str, query: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Server-side full-text search across messages in one chat. Returns
        message rows in the same shape as `get_chat_history` (both sides,
        media-only messages surfaced via `[photo]`/`[file: ...]` placeholders
        with `has_media: True`). Use for 'did we talk about X with Y'.
        Telegram's search handles stemming/case better than a local substring
        scan would."""
        q = query.strip()
        if not q:
            return []
        entity = _entity_arg(chat_id)
        out: list[dict[str, Any]] = []
        async for msg in self._client.iter_messages(entity, search=q, limit=limit):
            text = getattr(msg, "message", None) or ""
            has_media = bool(getattr(msg, "media", None))
            if not text and not has_media:
                continue
            if not text and has_media:
                text = _media_placeholder(msg)
            sender = (
                await _maybe_await(msg.get_sender())
                if hasattr(msg, "get_sender") else None
            )
            out.append({
                "message_id": str(getattr(msg, "id", "")),
                "text": text,
                "date": _iso_or_none(getattr(msg, "date", None)),
                "outgoing": bool(getattr(msg, "out", False)),
                "sender_username": getattr(sender, "username", None) if sender else None,
                "sender_display_name": _display_name(sender) if sender else None,
                "has_media": has_media,
            })
        return out

    async def search_chats(
        self, query: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Token-AND match the user's recent dialogs against `dialog.name`
        (every whitespace-split token must appear in the lowercased name) or a
        plain substring against `@username`. Falls back to Telegram's
        server-side `contacts.SearchRequest`, which handles transliteration
        (e.g. "Alex" -> "Алекс") and surfaces contacts not yet in local
        dialogs. Returns rows with a `source` field ("dialog" or "contact").
        """
        q_raw = query.strip()
        if not q_raw:
            return []
        q = q_raw.lower()
        tokens = [t for t in q.split() if t]

        def _match(name: str, username: str) -> bool:
            nl = name.lower()
            if username and q in username.lower():
                return True
            return bool(tokens) and all(tok in nl for tok in tokens)

        await self._refresh_archived()
        seen_ids: set[str] = set()
        out: list[dict[str, Any]] = []
        async for dlg in self._client.iter_dialogs():
            name = (getattr(dlg, "name", None) or "")
            entity = getattr(dlg, "entity", None)
            username = (getattr(entity, "username", None) or "") if entity else ""
            if not _match(name, username):
                continue
            cid = _dialog_chat_id(dlg)
            if cid is None or cid in seen_ids:
                continue
            seen_ids.add(cid)
            out.append({
                "chat_id": cid,
                "name": name,
                "username": username or None,
                "is_user": bool(getattr(dlg, "is_user", False)),
                "is_group": bool(getattr(dlg, "is_group", False)),
                "is_channel": bool(getattr(dlg, "is_channel", False)),
                "unread_count": int(getattr(dlg, "unread_count", 0) or 0),
                "archived": cid in self._archived,
                "source": "dialog",
            })
            if len(out) >= limit:
                return out

        # Fallback: Telegram-side contact search. Catches transliteration
        # (Cyrillic <-> Latin) and contacts not yet present in iter_dialogs.
        try:
            from telethon.tl.functions.contacts import SearchRequest
            res = await self._client(SearchRequest(q=q_raw, limit=limit))
        except Exception:
            log.exception("contacts.search fallback failed")
            return out
        for user in (getattr(res, "users", None) or []):
            uid = str(getattr(user, "id", "") or "")
            if not uid or uid in seen_ids:
                continue
            full_name = " ".join(
                p for p in (
                    getattr(user, "first_name", None),
                    getattr(user, "last_name", None),
                ) if p
            ).strip()
            seen_ids.add(uid)
            out.append({
                "chat_id": uid,
                "name": full_name,
                "username": getattr(user, "username", None) or None,
                "is_user": True,
                "is_group": False,
                "is_channel": False,
                "unread_count": 0,
                "archived": False,
                "source": "contact",
            })
            if len(out) >= limit:
                break
        return out

    async def resolve_chat_name(
        self, chat_id: str,
    ) -> dict[str, str | None] | None:
        """Resolve `chat_id` → `{display_name, username, is_user}` via telethon's
        `get_entity`. Used to surface human-readable labels for chats stored
        only by id (e.g. the DM allowlist). Returns None on any failure —
        unknown ids, network blips, or a dropped client connection. The
        caller falls back to showing just the id."""
        try:
            entity = await self._client.get_entity(_entity_arg(chat_id))
        except Exception:
            return None
        if entity is None:
            return None
        username = getattr(entity, "username", None) or None
        # _display_name handles first+last; for non-user entities (groups,
        # channels) the `.title` attribute is what they expose, so fall
        # back to that.
        name = _display_name(entity) or getattr(entity, "title", None) or None
        is_user = bool(getattr(entity, "first_name", None)) or bool(
            getattr(entity, "last_name", None)
        )
        return {
            "chat_id": chat_id,
            "display_name": name,
            "username": username,
            "is_user": is_user,
        }

    async def send(self, chat_id: str, text: str) -> dict[str, Any]:
        entity = _entity_arg(chat_id)
        sent = await self._client.send_message(entity, text)
        sent_id = str(getattr(sent, "id", ""))
        telegram_log.info("send " + fmt(
            chat=chat_id, message_id=sent_id, len=len(text), text=text,
        ))
        return {"message_id": sent_id, "chat_id": chat_id}

    async def download_attachment(
        self, chat_id: str, message_id: str, *, max_bytes: int = 10 * 1024 * 1024,
    ) -> tuple[bytes, str, str]:
        """Re-fetch one Telegram message by (chat_id, message_id) and
        return (data, mime_type, file_name) for its attachment.

        Used by the operator's `read_image` tool to feed an inbound
        attachment back to the LLM as inline_data. Raises ValueError on
        a missing message, missing media, or oversize file. Mime type
        falls back to 'application/octet-stream' when telethon doesn't
        report one (e.g. some old documents)."""
        try:
            msg_id_int = int(str(message_id).strip())
        except (TypeError, ValueError):
            raise ValueError(f"invalid message_id: {message_id!r}")
        entity = _entity_arg(chat_id)
        msg = None
        async for m in self._client.iter_messages(entity, ids=[msg_id_int]):
            msg = m
            break
        if msg is None:
            raise ValueError(f"message {message_id} not found in chat {chat_id}")
        if not getattr(msg, "media", None):
            raise ValueError(f"message {message_id} has no attachment")
        f = getattr(msg, "file", None)
        declared_size = int(getattr(f, "size", 0) or 0) if f else 0
        if declared_size and declared_size > max_bytes:
            raise ValueError(
                f"attachment too large: {declared_size} bytes (cap {max_bytes})"
            )
        data = await msg.download_media(file=bytes)
        if not data:
            raise ValueError(f"download returned empty for message {message_id}")
        if len(data) > max_bytes:
            raise ValueError(
                f"attachment too large: {len(data)} bytes (cap {max_bytes})"
            )
        mime = (getattr(f, "mime_type", None) if f else None) or "application/octet-stream"
        name = (getattr(f, "name", None) if f else None) or ""
        return data, mime, name

    async def transcribe_voice(
        self, chat_id: str, message_id: str, *, max_wait_s: int = 20,
    ) -> dict[str, Any]:
        """Transcribe a voice / audio message via Telegram's premium
        transcription API (`messages.transcribeAudio`). Requires the
        userbot account to have Telegram Premium.

        Returns `{text, pending}`. `pending=True` means the server is
        still working — we already polled up to `max_wait_s` seconds in
        ~2s intervals; the caller can choose to retry later or work
        with the partial text we have (Telegram streams partial
        transcripts as they finalize). Raises ValueError on a bad
        message id or RPC error (e.g. message has no voice/audio,
        Premium not active, peer not found)."""
        from telethon.tl.functions.messages import TranscribeAudioRequest  # type: ignore
        try:
            msg_id_int = int(str(message_id).strip())
        except (TypeError, ValueError):
            raise ValueError(f"invalid message_id: {message_id!r}")
        entity = _entity_arg(chat_id)
        deadline = time.monotonic() + max_wait_s
        text = ""
        pending = True
        while True:
            try:
                resp = await self._client(
                    TranscribeAudioRequest(peer=entity, msg_id=msg_id_int),
                )
            except Exception as e:
                raise ValueError(f"{type(e).__name__}: {e}") from e
            text = getattr(resp, "text", "") or ""
            pending = bool(getattr(resp, "pending", False))
            if not pending:
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(2.0)
        return {"text": text, "pending": pending}

    async def reply_to_chat(
        self, chat_id: str, text: str,
    ) -> dict[str, Any]:
        """Send `text` to `chat_id`, then clear that chat's unread state
        (mark every unread inbox row read and stamp `replied_message_id`
        on the latest one for audit). Used by the operator's `reply_to_dm`
        tool for memory-authorized autonomous replies under the new
        chat-centric drain flow.

        Returns a payload the caller surfaces to the model: outbound
        message id, the sender metadata pulled from the latest unread
        row (best-effort; may be None if the chat had no unread row
        when this was called — e.g. an out-of-band reply), and the
        latest inbound body for the audit notice."""
        # Snapshot the latest unread row BEFORE we mark it read — that's
        # the sender/body we'll cite in the audit notice. If there's no
        # unread row at all (e.g. operator decided to reply on a chat
        # that's already been triaged), fall back to chat-level lookup.
        latest = await self._latest_unread_for_chat(chat_id)
        sent = await self.send(chat_id, text)
        await self._db.record_chat_reply(chat_id, sent["message_id"])
        return {
            "chat_id": chat_id,
            "message_id": sent["message_id"],
            "sender_username": (latest or {}).get("sender_username"),
            "sender_display_name": (latest or {}).get("sender_display_name"),
            "inbound_body": (latest or {}).get("body"),
        }

    async def _latest_unread_for_chat(
        self, chat_id: str,
    ) -> dict[str, Any] | None:
        """Most-recent unread inbox row in `chat_id`. Returns None if the
        chat is fully read. Used to attribute a `reply_to_chat` call to a
        specific inbound message for audit purposes."""
        rows = await self._db.list_inbox(unread_only=True, limit=200)
        for r in rows:
            if r["chat_id"] == chat_id:
                return r
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _media_placeholder(msg: Any) -> str:
    """Synthetic body for a media-only Telegram message (no caption). The
    operator sees this in `op=history` results alongside the real
    `message_id` and routes accordingly: `read_image` for photos,
    `transcribe` for voice/voice-notes. Voice notes get a duration in
    the placeholder so the model can decide whether to bother
    transcribing (a 1-second blip is usually noise; a 30-second message
    almost always carries content)."""
    # Voice notes first — telethon exposes `.voice` only for proper voice
    # messages (DocumentAttributeAudio.voice=True), not for music files.
    voice = getattr(msg, "voice", None)
    if voice is not None:
        duration = _audio_duration(voice)
        return f"[voice: {duration}s]" if duration is not None else "[voice]"
    # Non-voice audio (music / forwarded files).
    audio = getattr(msg, "audio", None)
    if audio is not None:
        duration = _audio_duration(audio)
        return f"[audio: {duration}s]" if duration is not None else "[audio]"
    f = getattr(msg, "file", None)
    mime = getattr(f, "mime_type", None) if f else None
    name = getattr(f, "name", None) if f else None
    if mime:
        if mime.startswith("image/"):
            return "[photo]"
        if mime.startswith("video/"):
            return "[video]"
        if mime.startswith("audio/"):
            return "[audio]"
    if name:
        return f"[file: {name}]"
    if mime:
        return f"[file: {mime}]"
    return "[attachment]"


def _audio_duration(doc: Any) -> int | None:
    """Pull the duration (seconds) from a Telegram Document's audio
    attribute. Returns None if the document has no audio attribute or
    no duration field."""
    try:
        from telethon.tl.types import DocumentAttributeAudio  # type: ignore
    except Exception:
        return None
    for attr in getattr(doc, "attributes", None) or []:
        if isinstance(attr, DocumentAttributeAudio):
            d = getattr(attr, "duration", None)
            return int(d) if d is not None else None
    return None


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


async def login_qr_interactive(
    *, api_id: int, api_hash: str, session_path: Path,
) -> None:
    """QR-code login. Sidesteps SMS/in-app code delivery entirely — telethon
    asks the server for a single-use login token, we render it as a QR code
    in the terminal, the user scans it from an already-logged-in Telegram
    app (Settings → Devices → Link Desktop Device), and the session is
    authorized server-side. 2FA password is still requested after the scan."""
    import getpass
    import qrcode
    from telethon import errors  # type: ignore

    client = make_telethon_client(api_id=api_id, api_hash=api_hash, session_path=session_path)
    print(f"Connecting to Telegram (session={session_path}, api_id={api_id})...")
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Already logged in as @{getattr(me, 'username', None)} (id={getattr(me, 'id', None)}).")
        await client.disconnect()
        return

    print("Open Telegram on your phone/desktop. Go to:")
    print("  Settings → Devices → Link Desktop Device")
    print("Then scan the QR code shown below.")
    print()

    qr_login = await client.qr_login()
    while True:
        # Re-render every cycle (the URL changes on .recreate()).
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data(qr_login.url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)  # invert=True renders well on dark terminals
        print(f"URL (paste if your terminal mangles QR): {qr_login.url}")
        print("Waiting for scan...")
        try:
            await qr_login.wait()
            break  # scanned
        except asyncio.TimeoutError:
            # QR codes expire after ~30s; telethon raises TimeoutError so we
            # re-issue a fresh token and re-render.
            print("(QR expired, generating a new one...)\n")
            await qr_login.recreate()
            continue
        except errors.SessionPasswordNeededError:
            # User has 2FA — telethon raises this after a successful scan.
            print("\n2FA is enabled on this account.")
            password = getpass.getpass("2FA password: ")
            try:
                await client.sign_in(password=password)
                break
            except errors.PasswordHashInvalidError:
                print("Wrong 2FA password. Re-run `oncall telegram-login --qr`.")
                await client.disconnect()
                return

    me = await client.get_me()
    print(f"Logged in as @{getattr(me, 'username', None)} "
          f"(id={getattr(me, 'id', None)}, name={getattr(me, 'first_name', None)}).")
    await client.disconnect()


async def login_interactive(
    *, api_id: int, api_hash: str, session_path: Path,
) -> None:
    """One-shot interactive login. Prompts for phone, code, optional 2FA.

    Implemented with the lower-level telethon calls (send_code_request +
    sign_in) instead of client.start() so we can show the user exactly which
    channel Telegram chose to deliver the code (in-app message vs SMS vs
    voice call) and give clear error messages per exception type. This is
    the path that needs to debug well — it's the one the user runs once and
    swears at if anything goes wrong."""
    import getpass
    from telethon import errors  # type: ignore

    client = make_telethon_client(api_id=api_id, api_hash=api_hash, session_path=session_path)
    print(f"Connecting to Telegram (session={session_path}, api_id={api_id})...")
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Already logged in as @{getattr(me, 'username', None)} (id={getattr(me, 'id', None)}).")
        await client.disconnect()
        return

    phone = input("Telegram phone (E.164, e.g. +14155551234): ").strip()
    if not phone.startswith("+"):
        # Telethon accepts both, but the canonical E.164 form is +<country><number>.
        # Warn the user — typo'd country code is one of the most common reasons
        # the code goes to a stranger.
        print(f"Note: phone has no leading '+'. Treating as +{phone}. Hit Ctrl-C to abort if wrong.")

    print(f"Sending code request to {phone}...")
    try:
        sent = await client.send_code_request(phone)
    except errors.PhoneNumberInvalidError:
        print(f"ERROR: Telegram rejected '{phone}' as not a valid phone number.")
        await client.disconnect()
        raise
    except errors.PhoneNumberBannedError:
        print(f"ERROR: Telegram says {phone} is banned. Contact recover@telegram.org.")
        await client.disconnect()
        raise
    except errors.FloodWaitError as e:
        print(f"ERROR: rate-limited. Wait {e.seconds}s and retry.")
        await client.disconnect()
        raise
    except errors.PhoneNumberFloodError:
        print(f"ERROR: too many sign-in attempts on this phone. Wait several hours.")
        await client.disconnect()
        raise

    code_type = type(sent.type).__name__
    next_type = type(sent.next_type).__name__ if sent.next_type else None
    print(f"Code sent. channel={code_type}, length={getattr(sent.type, 'length', '?')}, "
          f"timeout={getattr(sent, 'timeout', '?')}s, fallback_next={next_type or 'none'}")
    if code_type == "SentCodeTypeApp":
        print("  → Look in your Telegram app (any logged-in device) for a chat with")
        print("    the official 'Telegram' account (blue ✓). The 5-digit code is there.")
        print("    Note: Telegram chooses the delivery channel server-side; there is no")
        print("    longer a reliable way to force SMS via the API.")
    elif code_type == "SentCodeTypeSms":
        print("  → SMS to your phone.")
    elif code_type == "SentCodeTypeCall":
        print("  → Automated voice call to your phone with the digits.")
    elif code_type == "SentCodeTypeFlashCall":
        print("  → A missed call; the code is the last digits of the calling number.")

    while True:
        code = input("Code: ").strip()
        if not code:
            print("Empty code — Telegram codes are always 5 digits. Try again.")
            continue
        try:
            await client.sign_in(phone=phone, code=code, phone_code_hash=sent.phone_code_hash)
            break
        except errors.SessionPasswordNeededError:
            print("2FA is enabled on this account.")
            password = getpass.getpass("2FA password: ")
            try:
                await client.sign_in(password=password)
                break
            except errors.PasswordHashInvalidError:
                print("Wrong 2FA password. Try again from the start (Ctrl-C, re-run).")
                await client.disconnect()
                return
        except errors.PhoneCodeInvalidError:
            print(f"Telegram says the code is wrong. Got {len(code)} digits; should be 5.")
            continue
        except errors.PhoneCodeExpiredError:
            print("Code expired (~5 min lifetime). Re-run `oncall telegram-login`.")
            await client.disconnect()
            return
        except errors.PhoneCodeEmptyError:
            print("Empty code rejected; try again.")
            continue

    me = await client.get_me()
    print(f"Logged in as @{getattr(me, 'username', None)} "
          f"(id={getattr(me, 'id', None)}, name={getattr(me, 'first_name', None)}).")
    await client.disconnect()
