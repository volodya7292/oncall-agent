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
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from .audit import fmt, telegram_log
from .db import Database
from .telegram_format import reply_context_note


_ONE_SECOND = timedelta(seconds=1)

# Allowed reactions for the executor's `op=react` tool. Telegram's free
# reaction set is larger; we expose only this curated 4 to keep prompt
# guidance tight and prevent the model from probing arbitrary emojis.
# Stored in canonical form (heart includes VS-16) — passed verbatim to
# Telegram's SendReactionRequest.
_ALLOWED_REACTIONS = {"👍", "🔥", "❤️", "😁"}


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
        on_new_message: NewMessageCallback | None = None,
        agent_user_id: int | None = None,
    ) -> None:
        self._db = db
        self._client = client
        self._on_new_message = on_new_message
        self._handler_ref: Any = None
        self._started = False
        # Hard guard: the agent account's chat with the owner must never
        # enter the primary userbot's inbox — otherwise the agent's own
        # replies would loop back as inbound DMs. The DM allowlist normally
        # keeps it out (the owner just doesn't /allowdm it), but this is an
        # unconditional backstop that holds even if that chat_id somehow
        # lands on the allowlist. None until the agent account is known.
        self._agent_user_id = agent_user_id

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
        self._started = True
        log.info("telegram listener started")

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

    # ---- handler ----

    def _build_handler(self) -> Callable[[Any], Awaitable[None]]:
        async def _on_new_message(event: Any) -> None:
            try:
                await self._handle_inbound(event)
            except Exception:
                log.exception("telegram inbound handler failed")
        return _on_new_message

    async def _handle_inbound(self, event: Any) -> None:
        # Single entry-log so we can prove whether telethon is delivering
        # NewMessage events at all when no inbound shows up downstream.
        chat_id_raw = str(
            getattr(event, "chat_id", None)
            or getattr(getattr(event, "message", None), "chat_id", "")
        )
        body_raw = getattr(getattr(event, "message", None), "message", None) or ""
        log.info(
            "telethon NewMessage chat=%s is_private=%s body_preview=%r",
            chat_id_raw, getattr(event, "is_private", None),
            str(body_raw)[:60],
        )
        # MVP: private chats only. Telethon: event.is_private is True for 1:1 DMs.
        if not getattr(event, "is_private", False):
            log.info("inbound skipped reason=not_private chat=%s", chat_id_raw)
            return
        sender = await _maybe_await(event.get_sender()) if hasattr(event, "get_sender") else None
        if sender is None:
            log.info("inbound skipped reason=no_sender chat=%s", chat_id_raw)
            return
        # Don't loop on our own outgoing messages.
        if getattr(sender, "is_self", False) or getattr(event.message, "out", False):
            log.info(
                "inbound skipped reason=self_or_outgoing chat=%s sender_id=%s",
                chat_id_raw, getattr(sender, "id", None),
            )
            return
        # Skip bots.
        if getattr(sender, "bot", False):
            log.info(
                "inbound skipped reason=bot chat=%s sender_id=%s",
                chat_id_raw, getattr(sender, "id", None),
            )
            return

        body = getattr(event.message, "message", None) or getattr(event, "raw_text", None) or ""
        # Voice / photo / document messages with no caption arrive with an
        # empty text body but a populated `media`. Substitute a synthetic
        # `[voice: 12s]` / `[photo]` / `[file: …]` placeholder so the row
        # still lands in the inbox — the operator can then transcribe or
        # read_image via the message_id. Without this, voice-only messages
        # were silently dropped at the empty-body filter below.
        if (not isinstance(body, str) or not body.strip()) and getattr(event.message, "media", None) is not None:
            body = _media_placeholder(event.message)
        if not isinstance(body, str) or not body.strip():
            log.info(
                "inbound skipped reason=empty_body chat=%s sender_id=%s",
                chat_id_raw, getattr(sender, "id", None),
            )
            return

        username = (getattr(sender, "username", None) or "").lower() or None
        display = _display_name(sender)
        chat_id = str(getattr(event, "chat_id", None) or getattr(event.message, "chat_id", ""))

        # Unconditional backstop: never surface the agent account's own chat,
        # even if it somehow ends up on the DM allowlist. In a 1:1 DM,
        # chat_id == the other party's user_id.
        if self._agent_user_id is not None and chat_id == str(self._agent_user_id):
            log.info(
                "inbound skipped reason=agent_self_chat chat=%s", chat_id,
            )
            return

        # Contacts-only floor: the agent may only ever communicate with
        # people in the owner's Telegram address book. The primary userbot
        # runs on the owner's account, so the sender User's `.contact` flag
        # is exactly "is this person an owner contact". Strangers (spam,
        # cold DMs, injection attempts) are dropped before triage. This sits
        # beneath the allowlist: allowlisting a non-contact still won't
        # surface them.
        if not getattr(sender, "contact", False):
            log.info(
                "inbound skipped reason=not_a_contact chat=%s sender=%s",
                chat_id, username or display,
            )
            return

        # Triage gate: only chats the owner has explicitly allowlisted (via
        # /allowdm, shown by /dmlist) are triaged. DMs from everyone else are
        # dropped here so they never reach the inbox-drain / operator. The
        # real conversation still lives in Telegram; we just don't surface
        # it. This is the same allowlist that gates autonomous DM replies.
        if not await self._db.is_dm_allowed(chat_id):
            log.info(
                "inbound skipped reason=not_allowlisted chat=%s sender=%s",
                chat_id, username or display,
            )
            return

        message_id = str(getattr(event.message, "id", ""))
        received_at = getattr(event.message, "date", None) or datetime.now(timezone.utc)

        if getattr(event.message, "reply_to", None) is not None:
            try:
                reply = await _maybe_await(event.message.get_reply_message())
            except Exception:
                log.warning(
                    "fetching replied-to message failed chat=%s msg=%s",
                    chat_id, message_id, exc_info=True,
                )
                reply = None
            if reply is not None:
                # Primary userbot: out=True → the owner sent the quoted
                # message; else the sender quotes themself (or a bot/service
                # message in this 1:1 chat).
                who = (
                    "the owner's earlier message"
                    if getattr(reply, "out", False)
                    else "their own earlier message"
                )
                body = f"{reply_context_note(reply, who=who)}\n{body}"
        inbox_id = str(uuid4())
        inserted = await self._db.record_inbox(
            inbox_id=inbox_id,
            platform="telegram",
            chat_id=chat_id,
            message_id=message_id,
            sender_username=username,
            sender_display_name=display,
            body=body,
            received_at=received_at,
        )
        if not inserted:
            return  # duplicate, skip

        telegram_log.info("inbound " + fmt(
            inbox=inbox_id, chat=chat_id, sender=username or display,
            body_len=len(body), body=body,
        ))

        if self._on_new_message is not None:
            row = await self._db.get_inbox_message(inbox_id)
            if row is not None:
                await self._on_new_message(row)

    # ---- DB-backed queries (no telethon needed) ----

    async def list_inbox(
        self, *, unread_only: bool = True, limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = await self._db.list_inbox(unread_only=unread_only, limit=limit)
        return rows

    async def list_pending_chats(self) -> list[dict[str, Any]]:
        """One entry per chat with unread DMs. Used by the inbox-drain triage
        path and by the operator's `read_inbox` tool in the chat-centric flow
        (the operator sees the dirty chat and then calls `read_chat` for full
        context if it wants it)."""
        return await self._db.list_pending_chats()

    async def get_message(self, inbox_id: str) -> dict[str, Any] | None:
        return await self._db.get_inbox_message(inbox_id)

    async def mark_read(self, inbox_id: str) -> bool:
        return await self._db.mark_inbox_read([inbox_id]) > 0

    async def mark_chat_read(self, chat_id: str) -> int:
        """Mark every unread inbox row in this chat as read. Returns the
        rowcount affected. The operator calls this after the user says
        'skip / ignore / dismiss' a chat's pending DMs."""
        return await self._db.mark_chat_read(chat_id)

    async def get_chat_unread_count(self, chat_id: str) -> int | None:
        """Telegram-side unread count for one chat. Returns None when
        the chat isn't resolvable or the API call errors — caller
        should treat None as "don't know, proceed as if unread."

        Used by the inbox-drain so we don't bother handing the chat off
        to the executor when the user has already read the messages on
        their phone (Telegram-side read state isn't pushed into our
        `messenger_inbox.read_at` automatically)."""
        try:
            from telethon.tl.functions.messages import GetPeerDialogsRequest
            entity = _entity_arg(chat_id)
            result = await self._client(GetPeerDialogsRequest(peers=[entity]))
            dialogs = getattr(result, "dialogs", None) or []
            if not dialogs:
                return None
            return int(getattr(dialogs[0], "unread_count", 0) or 0)
        except Exception as e:
            log.warning("get_chat_unread_count failed for %s: %s", chat_id, e)
            return None

    # ---- telethon-backed reads/writes ----

    async def get_chat_style(
        self, chat_id: str, *, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the USER'S OWN outgoing messages in this chat, sampled
        from three temporal windows (latest, ~2 days ago, ~30 days ago) so
        the voice profile reflects how the user writes across different
        conversational moods rather than just the latest exchange. Each
        entry is {'message_id', 'text', 'date'}. Deduplicated across
        windows by message_id; returned newest-first.

        We deliberately filter to the user's own side (from_user='me');
        reading the counterparty's messages would teach the wrong voice.
        We also exclude messages ≥200 chars — long messages tend to be
        structured/informational (lists, briefings, code) and are bad
        examples of the user's casual voice. `limit` applies PER WINDOW;
        the caller may receive up to ~3×limit samples in total."""
        entity = _entity_arg(chat_id)
        now = datetime.now(timezone.utc)
        windows: list[datetime | None] = [
            None,                          # 1) latest messages
            now - timedelta(days=2),       # 2) ~2 days ago
            now - timedelta(days=30),      # 3) ~30 days ago
        ]
        seen: set[str] = set()
        samples: list[dict[str, Any]] = []
        for offset_date in windows:
            kwargs: dict[str, Any] = {"limit": limit, "from_user": "me"}
            if offset_date is not None:
                kwargs["offset_date"] = offset_date
            async for msg in self._client.iter_messages(entity, **kwargs):
                text = getattr(msg, "message", None)
                if not text or len(text) >= 200:
                    continue
                mid = str(getattr(msg, "id", ""))
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                samples.append({
                    "message_id": mid,
                    "date": _iso_or_none(getattr(msg, "date", None)),
                    "text": text,
                })
        samples.sort(key=lambda s: s["date"] or "", reverse=True)
        return samples

    async def get_chat_history(
        self, chat_id: str, *, limit: int = 10, since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` messages of a chat, BOTH sides. Each row:
            message_id, text, date, outgoing, sender_username, sender_display_name, has_media.

        When `since` is None, returns the most recent `limit` messages,
        newest first. When `since` is set, returns up to `limit` messages
        at or after that moment, oldest first (chronological forward) —
        used for "what was happening starting at T".

        Media-only messages (photo / document / etc. with no caption) get
        a synthetic body like `[photo]` / `[file: name.pdf]` so the
        operator sees the message_id and can call `read_image(chat_id,
        message_id)` to load the bytes. Without this they were invisible
        and the operator would guess (wrongly) at nearby IDs.

        Unlike `get_chat_style` (which filters to from_user='me'), this
        includes the counterparty's messages — use it when the user asks
        'what did X say?'."""
        entity = _entity_arg(chat_id)
        iter_kwargs: dict[str, Any] = {"limit": limit}
        if since is not None:
            # telethon: in default direction, offset_date returns messages
            # STRICTLY OLDER than the date; with reverse=True, returns
            # messages STRICTLY NEWER. We want at-or-after, so subtract
            # 1 second to make the "strictly newer" boundary inclusive.
            iter_kwargs["offset_date"] = since.replace(microsecond=0) - _ONE_SECOND
            iter_kwargs["reverse"] = True
        out: list[dict[str, Any]] = []
        async for msg in self._client.iter_messages(entity, **iter_kwargs):
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
            # `date` placed right after `message_id` so the model reads
            # recency before content — easy to miss when buried mid-record.
            out.append({
                "message_id": str(getattr(msg, "id", "")),
                "date": _iso_or_none(getattr(msg, "date", None)),
                "text": text,
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
        Telethon returns dialogs in last-activity order. Each dialog's
        `archived` state is reported as-is — nothing is filtered on it.
        - `unread_only=True` keeps only dialogs with unread_count > 0.
        - `dms_only=True` keeps only private chats (1:1)."""
        out: list[dict[str, Any]] = []
        async for dlg in self._client.iter_dialogs():
            cid = _dialog_chat_id(dlg)
            if cid is None:
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
                "archived": bool(getattr(dlg, "archived", False)),
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
            # `date` placed right after `message_id` so the model reads
            # recency before content — easy to miss when buried mid-record.
            out.append({
                "message_id": str(getattr(msg, "id", "")),
                "date": _iso_or_none(getattr(msg, "date", None)),
                "text": text,
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
                "archived": bool(getattr(dlg, "archived", False)),
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

    async def is_owner_contact(self, chat_id: str) -> bool:
        """Whether `chat_id` resolves to a user in the owner's Telegram
        address book. The primary userbot runs on the owner's account, so a
        resolved User's `.contact` flag is exactly "owner contact". Fails
        closed (False) when the entity can't be resolved or isn't a user."""
        try:
            entity = await self._client.get_entity(_entity_arg(chat_id))
        except Exception as e:
            log.warning(
                "contact check: get_entity failed for chat=%s: %s", chat_id, e,
            )
            return False
        return bool(getattr(entity, "contact", False))

    async def _require_owner_contact(self, chat_id: str) -> None:
        """Refuse to message a chat that isn't in the owner's Telegram
        address book. The agent may only ever communicate with the owner's
        known contacts — a hard boundary against cold-messaging strangers,
        mirroring the inbound contacts-only floor. Fails closed: if the
        entity can't be resolved, the send is refused rather than allowed."""
        if not await self.is_owner_contact(chat_id):
            raise ValueError(
                f"refusing to message chat_id={chat_id}: not in the owner's "
                f"Telegram contacts (or the contact could not be verified)"
            )

    async def send(self, chat_id: str, text: str) -> dict[str, Any]:
        await self._require_owner_contact(chat_id)
        entity = _entity_arg(chat_id)
        text = re.sub(r"[ \t]*[–—][ \t]*", " - ", text)
        sent = await self._client.send_message(entity, text)
        sent_id = str(getattr(sent, "id", ""))
        telegram_log.info("send " + fmt(
            chat=chat_id, message_id=sent_id, len=len(text), text=text,
        ))
        return {"message_id": sent_id, "chat_id": chat_id}

    async def send_file(
        self, chat_id: str, file_path: str, *, caption: str | None = None,
    ) -> dict[str, Any]:
        """Upload a local file to a Telegram chat. Distinct from `send`
        (which sends a text body) — this attaches the file as a Telegram
        document. Caption (optional) is the accompanying text shown
        alongside the file."""
        from pathlib import Path
        p = Path(file_path).expanduser()
        if not p.is_file():
            raise ValueError(f"file not found or not a regular file: {p}")
        await self._require_owner_contact(chat_id)
        entity = _entity_arg(chat_id)
        normalized_caption = (
            re.sub(r"[ \t]*[–—][ \t]*", " - ", caption) if caption else None
        )
        sent = await self._client.send_file(
            entity, str(p), caption=normalized_caption,
        )
        sent_id = str(getattr(sent, "id", ""))
        size_bytes = p.stat().st_size
        telegram_log.info("send_file " + fmt(
            chat=chat_id, message_id=sent_id, file=str(p), bytes=size_bytes,
            caption=(caption or ""),
        ))
        return {
            "message_id": sent_id, "chat_id": chat_id,
            "file_name": p.name, "size_bytes": size_bytes,
        }

    async def react(
        self, chat_id: str, message_id: str, emoji: str,
    ) -> dict[str, Any]:
        if emoji not in _ALLOWED_REACTIONS:
            raise ValueError(
                f"emoji {emoji!r} not in allowed set "
                f"{sorted(_ALLOWED_REACTIONS)}"
            )
        try:
            msg_id_int = int(str(message_id).strip())
        except (TypeError, ValueError):
            raise ValueError(f"invalid message_id: {message_id!r}")
        await self._require_owner_contact(chat_id)
        from telethon.tl.functions.messages import SendReactionRequest  # type: ignore
        from telethon.tl.types import ReactionEmoji  # type: ignore
        entity = _entity_arg(chat_id)
        # Telethon RPCError (MessageIdInvalidError, ChatWriteForbidden,
        # FloodWait, …) bubbles up to the top-level catch in
        # api.messenger_op, which logs to the audit channel and converts
        # to 422. We no longer translate locally — the executor's
        # tool_use record carries chat_id/message_id/emoji right next to
        # the error result, so the per-react context isn't lost.
        await self._client(SendReactionRequest(
            peer=entity, msg_id=msg_id_int,
            reaction=[ReactionEmoji(emoticon=emoji)],
        ))
        telegram_log.info("react " + fmt(
            chat=chat_id, message_id=message_id, emoji=emoji,
        ))
        return {"chat_id": chat_id, "message_id": message_id, "emoji": emoji}

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
        # Snapshot the latest pending row BEFORE we record the reply —
        # that's the sender/body we'll cite in the audit notice. Keyed
        # off triaged-ness (NOT `read_at`): the inbox-drain pre-marks a
        # turn's snapshot read before invoking the operator, so by the
        # time `reply_to_chat` fires from within that turn, the rows are
        # already read — only the triaged set reliably distinguishes
        # "we've finished handling this row" from "still in flight".
        latest = await self._latest_pending_for_chat(chat_id)
        sent = await self.send(chat_id, text)
        await self._db.record_chat_reply(chat_id, sent["message_id"])
        return {
            "chat_id": chat_id,
            "message_id": sent["message_id"],
            "sender_username": (latest or {}).get("sender_username"),
            "sender_display_name": (latest or {}).get("sender_display_name"),
            "inbound_body": (latest or {}).get("body"),
        }

    async def _latest_pending_for_chat(
        self, chat_id: str,
    ) -> dict[str, Any] | None:
        """Most-recent not-yet-triaged inbox row in `chat_id`. Returns
        None if the chat is fully triaged. Used to attribute a
        `reply_to_chat` call to a specific inbound message for audit
        purposes. NOT keyed off `read_at` — see `reply_to_chat`."""
        return await self._db.latest_pending_for_chat(chat_id)


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
    the telethon dependency.

    `connection_retries=None` makes telethon retry the reconnect loop
    forever instead of giving up after the default 5 attempts. Without
    this, a laptop going through a string of maintenance-sleep cycles +
    Wi-Fi drops can exhaust the budget, after which every outbound call
    raises `ConnectionError: Cannot send requests while disconnected`
    until something explicitly re-runs `connect()`. `retry_delay=60`
    spaces those attempts out so we don't hammer Telegram while the
    network is genuinely down."""
    from telethon import TelegramClient  # type: ignore
    session_path.parent.mkdir(parents=True, exist_ok=True)
    return TelegramClient(
        str(session_path), api_id, api_hash,
        connection_retries=None,
        retry_delay=60,
    )


async def login_qr_interactive(
    *, api_id: int, api_hash: str, session_path: Path,
) -> int | None:
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
        return int(getattr(me, "id", 0)) or None

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
                return None

    me = await client.get_me()
    print(f"Logged in as @{getattr(me, 'username', None)} "
          f"(id={getattr(me, 'id', None)}, name={getattr(me, 'first_name', None)}).")
    await client.disconnect()
    return int(getattr(me, "id", 0)) or None


async def login_interactive(
    *, api_id: int, api_hash: str, session_path: Path,
) -> int | None:
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
        return int(getattr(me, "id", 0)) or None

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
                return None
        except errors.PhoneCodeInvalidError:
            print(f"Telegram says the code is wrong. Got {len(code)} digits; should be 5.")
            continue
        except errors.PhoneCodeExpiredError:
            print("Code expired (~5 min lifetime). Re-run `oncall telegram-login`.")
            await client.disconnect()
            return None
        except errors.PhoneCodeEmptyError:
            print("Empty code rejected; try again.")
            continue

    me = await client.get_me()
    print(f"Logged in as @{getattr(me, 'username', None)} "
          f"(id={getattr(me, 'id', None)}, name={getattr(me, 'first_name', None)}).")
    await client.disconnect()
    return int(getattr(me, "id", 0)) or None
