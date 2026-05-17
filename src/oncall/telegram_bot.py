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
import base64
import logging
import re
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
from . import service


log = logging.getLogger(__name__)


# Telegram caps a single message at 4096 chars. Stay slightly under for headroom.
_TELEGRAM_MSG_LIMIT = 4000
# Long-poll: how long Telegram holds the request open if no updates. The Bot
# API caps timeout at ~50s; 25s is a comfortable middle ground.
_LONG_POLL_SECONDS = 25
# Backoff between failed getUpdates calls (network issues, 5xx, rate limits).
_RETRY_DELAY_SECONDS = 3.0

# Max bytes for an inbound photo/document we'll pull into a chat turn. The
# operator's `read_image` cap is identical; sized for screenshots, small PDFs,
# etc. — well within Gemini's inline-data limits without blowing up the
# prompt.
_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024


def _pick_attachment_file(msg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull the best `file_id` (and declared mime, if any) out of a
    Telegram `message` update. Returns `(None, None)` when the message
    carries no media we know how to ingest.

    For photos Telegram sends an array of size variants — we pick the
    LAST entry (largest, highest-quality) so the model gets the best
    image it can. Documents and stickers carry a single file_id; voice/
    audio/video also work but are passed through as-is since Gemini
    accepts those mime types too."""
    photos = msg.get("photo")
    if isinstance(photos, list) and photos:
        last = photos[-1]
        fid = last.get("file_id") if isinstance(last, dict) else None
        if fid:
            return str(fid), None  # photos don't carry mime; let the sniffer guess
    for key in ("document", "audio", "voice", "video", "video_note", "sticker"):
        m = msg.get(key)
        if isinstance(m, dict):
            fid = m.get("file_id")
            if fid:
                return str(fid), m.get("mime_type")
    return None, None

# Bot menu definition. Pushed to Telegram via setMyCommands at startup so
# the user gets autocomplete + the slash-command menu without having to
# configure anything via @BotFather. Keep descriptions ≤256 chars (Telegram
# limit) and start lowercase per Telegram convention.
_BOT_COMMANDS: list[dict[str, str]] = [
    {"command": "start",    "description": "greeting"},
    {"command": "status",   "description": "running tasks, queue, approvals, unread DMs"},
    {"command": "context",  "description": "export this session's history + summary as a markdown file"},
    {"command": "clear",    "description": "wipe this session's history (memory preserved)"},
    {"command": "compress", "description": "force-compress older messages into a summary"},
    {"command": "allowdm",  "description": "allowlist a chat_id for autonomous DM replies"},
    {"command": "denydm",   "description": "remove a chat_id from the DM allowlist"},
    {"command": "dmlist",   "description": "show which chats are allowlisted for autonomous DM replies"},
    {"command": "setownername", "description": "set your display name used in the operator's system prompt"},
    {"command": "restart",  "description": "restart the oncall service (brief downtime)"},
    {"command": "stop",     "description": "stop the oncall service entirely (bot goes silent)"},
    {"command": "help",     "description": "list commands"},
]


def bot_session_id(owner_user_id: int) -> str:
    """Deterministic chat-session id for the bot's conversation with the owner.
    One owner ↔ one session, persistent across daemon restarts."""
    return f"tg-bot-{owner_user_id}"


# --- MarkdownV2 escaping ----------------------------------------------------
# Telegram MarkdownV2 demands every special char be escaped UNLESS it's part
# of a recognized formatting pair. We preserve four pairs:
#   ```fenced```, `inline`, *bold*, _italic_
# and escape every other special char as `\X` so the message parses cleanly.
# The operator's model output is GFM-flavored (uses `**bold**`); we normalize
# to V2's single-asterisk form before escaping.

_V2_SPECIAL_CHARS = set("_*[]()~`>#+-=|{}.!")
_V2_SPECIAL_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!])")
_GFM_BOLD_RE = re.compile(r"\*\*([^\n*]+?)\*\*")


def _escape_code(s: str) -> str:
    """V2 inside-code escape: only backslash and backtick."""
    return s.replace("\\", "\\\\").replace("`", "\\`")


def escape_v2(text: str) -> str:
    """Render `text` as Telegram MarkdownV2 with proper escaping. Best-effort:
    if the caller writes anything weirder than the four supported pairs above,
    the formatting won't render — but the message will still parse and reach
    the user. `_send` falls back to plain text on any parse error.
    """
    # GFM `**bold**` → V2 `*bold*` (V2 uses single asterisks).
    text = _GFM_BOLD_RE.sub(r"*\1*", text)
    out: list[str] = []
    n = len(text)
    pos = 0
    while pos < n:
        ch = text[pos]
        # Fenced code block ```...```
        if text.startswith("```", pos):
            end = text.find("```", pos + 3)
            if end != -1:
                out.append("```")
                out.append(_escape_code(text[pos + 3:end]))
                out.append("```")
                pos = end + 3
                continue
        # Inline code `...` (single line, non-empty body)
        if ch == "`":
            end = text.find("`", pos + 1)
            if end != -1 and end > pos + 1 and "\n" not in text[pos + 1:end]:
                out.append("`")
                out.append(_escape_code(text[pos + 1:end]))
                out.append("`")
                pos = end + 1
                continue
        # Bold pair *...* (single line, non-empty body)
        if ch == "*":
            end = text.find("*", pos + 1)
            if end != -1 and end > pos + 1 and "\n" not in text[pos + 1:end]:
                out.append("*")
                out.append(_V2_SPECIAL_RE.sub(r"\\\1", text[pos + 1:end]))
                out.append("*")
                pos = end + 1
                continue
        # Italic pair _..._ (single line, non-empty body)
        if ch == "_":
            end = text.find("_", pos + 1)
            if end != -1 and end > pos + 1 and "\n" not in text[pos + 1:end]:
                out.append("_")
                out.append(_V2_SPECIAL_RE.sub(r"\\\1", text[pos + 1:end]))
                out.append("_")
                pos = end + 1
                continue
        # Literal char: escape if special.
        if ch in _V2_SPECIAL_CHARS:
            out.append("\\")
        out.append(ch)
        pos += 1
    return "".join(out)


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
    async def send_document(
        self,
        *,
        chat_id: Any,
        filename: str,
        content: bytes,
        caption: str | None = None,
    ) -> Any: ...
    async def download_file(self, file_id: str) -> tuple[bytes, str, str]: ...
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

    async def send_document(
        self,
        *,
        chat_id: Any,
        filename: str,
        content: bytes,
        caption: str | None = None,
    ) -> Any:
        """Multipart upload via /sendDocument. Used to ship things that are
        too long or too noisy for an inline message — e.g. `/context`
        exports. Telegram caps documents at 50 MB for bots; we're well
        under that for any plausible chat history."""
        data: dict[str, str] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        files = {"document": (filename, content, "text/markdown")}
        r = await self._http.post("/sendDocument", data=data, files=files)
        try:
            body = r.json()
        except Exception as e:
            raise RuntimeError(
                f"sendDocument: non-JSON response (HTTP {r.status_code})"
            ) from e
        if not body.get("ok"):
            raise RuntimeError(
                f"sendDocument failed: {body.get('description')} "
                f"(HTTP {r.status_code})"
            )
        return body.get("result")

    async def download_file(self, file_id: str) -> tuple[bytes, str, str]:
        """Fetch the bytes for a `file_id` produced by a Telegram update.

        Two round-trips: (1) `getFile` resolves `file_id` to a server-side
        `file_path`; (2) GET against the file endpoint (NOTE: a different
        base URL than the bot API itself — `/file/bot<token>/...` instead
        of `/bot<token>/...`) returns the raw bytes.

        Returns `(data, mime_type, filename)`. mime_type is guessed from
        the file_path suffix (Telegram doesn't return it in getFile);
        falls back to `application/octet-stream`. filename is the basename
        of `file_path` so the caller can show the user something sensible
        in audit logs.
        """
        import mimetypes
        from os.path import basename

        info = await self.call("getFile", {"file_id": file_id})
        file_path = (info or {}).get("file_path") or ""
        if not file_path:
            raise RuntimeError(f"getFile returned no file_path for {file_id}")
        # `self._http.base_url` is `https://api.telegram.org/bot<token>`.
        # The file fetch uses `https://api.telegram.org/file/bot<token>/<path>`.
        # Build the absolute URL once instead of swapping base URLs on the
        # client — simpler and keeps `call` correct for everything else.
        bot_base = str(self._http.base_url)  # ".../bot<token>"
        file_url = bot_base.replace("/bot", "/file/bot", 1) + "/" + file_path
        r = await self._http.get(file_url)
        r.raise_for_status()
        data = r.content
        mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        return data, mime, basename(file_path) or file_path

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
        telegram: Any | None = None,
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
        # Optional userbot handle; used by `/dmlist` to resolve chat_ids to
        # human-readable names. None → /dmlist still works but shows ids only.
        # Typed as `Any` to avoid an import cycle with telegram_service.
        self._telegram = telegram
        self._session_id = bot_session_id(owner_user_id)
        self._poll_task: asyncio.Task | None = None
        self._reply_task: asyncio.Task | None = None
        self._approval_task: asyncio.Task | None = None
        self._dispatch_approval_task: asyncio.Task | None = None
        self._update_offset: int = 0
        self._bot_username: str | None = None
        self._bot_user_id: int | None = None
        self._started = False
        # Mutual-exclusion flag for long-running owner-initiated commands
        # (/compress, /context). asyncio is single-threaded, so a plain bool
        # checked-then-set across an event-loop turn boundary is sufficient
        # — no Lock required. When set, holds the user-facing name of the
        # in-flight op so we can name it in the "busy" reply.
        self._heavy_op_in_flight: str | None = None

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
        # Register the slash-command menu so the user gets autocomplete in
        # the Telegram client without having to set anything up via BotFather.
        # Scope must be explicit — `default` is a fallback the client may
        # ignore in favor of more-specific scopes. This bot is owner-DM only
        # (single-user), so `all_private_chats` is the right canonical scope.
        # We also publish under `default` so clients that read the fallback
        # have something to display. Fail-soft: a transient network blip
        # must not block polling.
        for scope in (
            {"type": "all_private_chats"},
            {"type": "default"},
        ):
            try:
                await self._api.call("setMyCommands", {
                    "commands": _BOT_COMMANDS,
                    "scope": scope,
                })
                telegram_log.info("bot commands registered " + fmt(
                    count=len(_BOT_COMMANDS), scope=scope["type"],
                ))
            except Exception:
                log.exception(
                    "setMyCommands failed for scope=%s; menu may be stale",
                    scope.get("type"),
                )
        self._poll_task = asyncio.create_task(self._poll_loop(), name="tg-bot-poll")
        self._poll_task.add_done_callback(self._on_bg_task_done)
        self._reply_task = asyncio.create_task(
            self._chat_reply_subscriber(), name="tg-bot-reply",
        )
        self._reply_task.add_done_callback(self._on_bg_task_done)
        # Approval-request subscriber: send inline Yes/No buttons whenever a
        # task dispatched in this bot's session needs approval.
        if self._broker is not None and self._db is not None:
            self._approval_task = asyncio.create_task(
                self._approval_subscriber(), name="tg-bot-approval",
            )
            self._approval_task.add_done_callback(self._on_bg_task_done)
        # Deferred-dispatch subscriber: operator-initiated dispatch_task
        # calls made during an autonomous-reply turn land here. We send
        # Yes/No buttons; on tap, `operator.resolve_dispatch_approval`
        # spawns (or denies) the task.
        self._dispatch_approval_task = asyncio.create_task(
            self._dispatch_approval_subscriber(),
            name="tg-bot-dispatch-approval",
        )
        self._dispatch_approval_task.add_done_callback(self._on_bg_task_done)
        self._started = True
        log.info(
            "telegram bot started (owner_user_id=%d, username=@%s, session=%s)",
            self._owner_user_id, self._bot_username, self._session_id,
        )

    # ---- notifications ----

    async def notify_owner(self, text: str) -> None:
        """Send an out-of-band plain-text message to the owner. Used by the
        daemon to surface startup status + background-task crashes without
        going through the operator. No parse_mode — markdown failures must
        not silently drop the message. Errors are swallowed: a failed
        notification must never crash whatever was trying to surface state.
        """
        if not self._started or self._owner_user_id is None:
            return
        try:
            await self._api.call("sendMessage", {
                "chat_id": self._owner_user_id, "text": text,
            })
        except Exception:
            log.exception("notify_owner failed")

    def _on_bg_task_done(self, task: asyncio.Task) -> None:
        """Done-callback for long-lived background subscribers. If a task
        exits with an exception (NOT cancellation), notify the owner so the
        daemon isn't silently degraded — those tasks are how approvals,
        auto-pings, and inbound messages reach the user."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        log.error("background task %r exited: %r", task.get_name(), exc)
        # Schedule the notify in the running loop; can't await here.
        asyncio.create_task(self.notify_owner(
            f"⚠️ background task '{task.get_name()}' crashed: "
            f"{type(exc).__name__}: {exc}"
        ))

    async def stop(self) -> None:
        if not self._started:
            return
        for task in (self._poll_task, self._reply_task, self._approval_task,
                     self._dispatch_approval_task):
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

    # ---- long-op concurrency gate ----

    async def _claim_heavy_op(self, chat_id: Any, name: str) -> bool:
        """Reserve the heavy-op slot for `name`. Returns True if the caller
        owns the slot now (proceed); False if another long op is already
        running (the user has been told to wait). Pair every True return
        with a `_release_heavy_op()` call in a `finally`.

        The check-then-set is non-racy because asyncio is single-threaded
        and we don't `await` between reading `_heavy_op_in_flight` and
        writing it."""
        if self._heavy_op_in_flight is not None:
            await self._send(chat_id, (
                f"{self._heavy_op_in_flight} is still running; "
                f"try {name} again once it finishes."
            ))
            return False
        self._heavy_op_in_flight = name
        return True

    def _release_heavy_op(self) -> None:
        self._heavy_op_in_flight = None

    async def _do_service_action(self, *, is_restart: bool) -> None:
        """Background task that fires `service.start()` (kickstart -k) or
        `service.stop()` (bootout) on a thread, after a brief delay so the
        ack message to the user has time to flush. Both actions kill this
        very daemon (launchd SIGTERMs the process), so we don't expect to
        return from `to_thread` — control flow ends when launchd takes us
        down. Any exception is logged before that happens."""
        try:
            await asyncio.sleep(0.5)
            if is_restart:
                await asyncio.to_thread(service.start)
            else:
                await asyncio.to_thread(service.stop)
        except Exception:
            log.exception(
                "service-%s failed", "restart" if is_restart else "stop",
            )

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
        # Photo / document fallback: when the owner sends an image or a
        # file (with or without a caption), Telegram puts the text on
        # `msg.caption` instead of `msg.text`, and the media metadata on
        # `msg.photo` / `msg.document`. We fetch the bytes eagerly so the
        # operator can see the image on the SAME turn the user asks about
        # it — no extra `read_image` round-trip needed.
        attachments: list[dict[str, Any]] = []
        if not text:
            caption = (msg.get("caption") or "").strip()
            file_id, declared_mime = _pick_attachment_file(msg)
            if file_id is not None:
                try:
                    data, sniffed_mime, fname = await self._api.download_file(file_id)
                except Exception:
                    log.exception("download_file failed for owner attachment")
                    await self._send(
                        chat_id,
                        "Couldn't download that attachment. Try again or send it "
                        "as a smaller file.",
                    )
                    return
                if len(data) > _ATTACHMENT_MAX_BYTES:
                    await self._send(
                        chat_id,
                        f"Attachment too large ({len(data)} bytes; cap "
                        f"{_ATTACHMENT_MAX_BYTES}).",
                    )
                    return
                mime = declared_mime or sniffed_mime
                attachments.append({
                    "data_b64": base64.b64encode(data).decode("ascii"),
                    "mime_type": mime,
                    "size_bytes": len(data),
                    "source": f"telegram bot ({fname or 'attachment'})",
                })
                # Default text when the user sent only the image, so the
                # operator has SOMETHING in the user turn instead of an
                # empty string (which would confuse the LLM).
                text = caption or "(attachment — please look at the image)"
        if not text:
            return

        # Slash commands handled locally — don't burn an operator turn on them.
        if text.startswith("/start"):
            await self._send(chat_id, (
                "Hi. Tell me what you need — "
                "I'll dispatch tasks and ping you back when they're done."
            ))
            return
        if text.startswith("/help"):
            await self._send(chat_id, (
                "/start — greeting\n"
                "/status — snapshot of running tasks, queue, approvals, unread DMs\n"
                "/context — export this session's chat history + latest summary as a markdown file\n"
                "/clear — wipe this chat session's history (memory is preserved)\n"
                "/compress — force-compress older messages into a summary now\n"
                "/allowdm <chat_id> — allowlist a chat for autonomous DM replies (empty by default)\n"
                "/denydm <chat_id> — remove a chat from the DM allowlist\n"
                "/dmlist — show allowlisted chats\n"
                "/setownername <name> — set your display name used in the operator's system prompt\n"
                "/restart — restart the oncall daemon via launchctl (brief downtime)\n"
                "/stop — stop the oncall daemon via launchctl (bot goes silent until manual start)\n"
                "/help — this\n"
                "Anything else is a chat turn."
            ))
            return
        if text.startswith("/allowdm") or text.startswith("/denydm"):
            await self._handle_allowlist(chat_id, text)
            return
        if text.startswith("/dmlist"):
            await self._send(chat_id, await self._render_dmlist())
            return
        if text.startswith("/setownername"):
            from .config import write_owner_name, read_owner_name
            arg = text[len("/setownername"):].strip()
            if not arg:
                current = read_owner_name()
                await self._send(chat_id, (
                    f"Usage: /setownername <name>\nCurrent: {current}"
                ))
                return
            try:
                write_owner_name(arg)
            except OSError as e:
                log.warning("write_owner_name failed: %s", e)
                await self._send(chat_id, f"Failed to write owner name: {e}")
                return
            saved = read_owner_name()
            telegram_log.info("bot setownername " + fmt(name=saved))
            await self._send(chat_id, f"Owner name set to: {saved}")
            return
        if text.startswith("/status"):
            await self._send(chat_id, await self._render_status())
            return
        if text.startswith("/context"):
            if not await self._claim_heavy_op(chat_id, "/context"):
                return
            try:
                try:
                    dump = await self._operator.export_context(self._session_id)
                except Exception:
                    log.exception("operator.export_context failed")
                    await self._send(chat_id, "Failed to export context. Check logs.")
                    return
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                filename = f"oncall-context-{self._session_id}-{stamp}.md"
                try:
                    await self._api.send_document(
                        chat_id=chat_id,
                        filename=filename,
                        content=dump.encode("utf-8"),
                        caption="Operator context for this session.",
                    )
                except Exception:
                    log.exception("send_document failed for /context")
                    await self._send(chat_id, "Failed to upload context file. Check logs.")
                    return
                telegram_log.info("bot context " + fmt(
                    session=self._session_id, bytes=len(dump),
                ))
            finally:
                self._release_heavy_op()
            return
        if text.startswith("/clear"):
            out = await self._operator.clear_session(self._session_id)
            await self._send(chat_id, (
                f"Context cleared ({out['messages_deleted']} messages, "
                f"{out['summaries_deleted']} summaries). Memory preserved."
            ))
            telegram_log.info("bot clear " + fmt(
                session=self._session_id, **out,
            ))
            return
        if text.startswith("/restart") or text.startswith("/stop"):
            # /restart → `oncall service start` (which kickstart -k's when
            # already loaded; launchd kills + respawns the daemon).
            # /stop    → `oncall service stop` (bootout; daemon dies, no
            # auto-restart until the user runs `oncall service start`).
            # Both kill the current daemon (us). We send the ack first,
            # then schedule the actual launchctl call on a background
            # thread with a brief sleep so the ack flushes to Telegram
            # before launchd SIGTERMs the process.
            is_restart = text.startswith("/restart")
            verb = "restart" if is_restart else "stop"
            ack = (
                "Restarting service. Brief downtime; you'll get the "
                "startup ping when I'm back."
                if is_restart else
                "Stopping service. Bot will go silent until "
                "`oncall service start` is run manually."
            )
            await self._send(chat_id, ack)
            telegram_log.info(f"bot service-{verb} requested " + fmt(
                session=self._session_id,
            ))
            asyncio.create_task(self._do_service_action(is_restart=is_restart))
            return
        if text.startswith("/compress"):
            if not await self._claim_heavy_op(chat_id, "/compress"):
                return
            try:
                out = await self._operator.compress_now(self._session_id)
                if out.get("compressed"):
                    await self._send(chat_id, (
                        f"Compressed {out['older_rows']} messages into "
                        f"~{out['summary_tokens']} tokens of summary."
                    ))
                else:
                    await self._send(chat_id, f"Nothing to compress: {out.get('reason')}.")
                telegram_log.info("bot compress " + fmt(
                    session=self._session_id, **out,
                ))
            finally:
                self._release_heavy_op()
            return

        telegram_log.info("bot inbound " + fmt(
            session=self._session_id, len=len(text),
            attachments=len(attachments),
        ))
        try:
            result = await self._operator.chat_turn(
                session_id=self._session_id, user_text=text,
                attachments=attachments or None,
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

    # ---- DM allowlist commands ----

    async def _handle_allowlist(self, chat_id: Any, text: str) -> None:
        """Handle `/allowdm <chat_id>` and `/denydm <chat_id>`.

        The argument is treated as an opaque string — Telegram chat ids for
        users are positive ints, but supergroups/channels use negative ids
        with the `-100…` prefix and we don't want to silently coerce or
        reject those. We DO require non-empty + no whitespace; anything else
        is a user typo.
        """
        if self._db is None:
            await self._send(chat_id, "DB not wired; allowlist unavailable.")
            return
        parts = text.split(None, 1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if not arg:
            verb = "allowdm" if cmd.startswith("/allowdm") else "denydm"
            await self._send(chat_id, f"Usage: /{verb} <chat_id>")
            return
        if any(c.isspace() for c in arg):
            await self._send(chat_id, "chat_id must not contain whitespace.")
            return
        if cmd.startswith("/allowdm"):
            added = await self._db.allow_dm(arg)
            msg = (
                f"Allowlisted chat_id={arg} for autonomous DM replies."
                if added else
                f"chat_id={arg} was already on the allowlist."
            )
            telegram_log.info("bot allowdm " + fmt(chat_id=arg, newly_added=added))
        else:
            removed = await self._db.deny_dm(arg)
            msg = (
                f"Removed chat_id={arg} from the DM allowlist."
                if removed else
                f"chat_id={arg} was not on the allowlist."
            )
            telegram_log.info("bot denydm " + fmt(chat_id=arg, was_present=removed))
        await self._send(chat_id, msg)

    async def _render_dmlist(self) -> str:
        if self._db is None:
            return "DB not wired; allowlist unavailable."
        rows = await self._db.list_dm_allowed()
        if not rows:
            return (
                "DM allowlist is empty. No chat may receive an autonomous "
                "reply. Use /allowdm <chat_id> to add one."
            )
        lines = ["DM allowlist:"]
        for r in rows:
            chat_id = r["chat_id"]
            label = _label_for_chat(chat_id, await self._resolve_label(chat_id))
            lines.append(f"- {label} (added {_relative_age(r['added_at'])})")
        return "\n".join(lines)

    async def _resolve_label(self, chat_id: str) -> dict[str, Any] | None:
        """Resolve `chat_id` via the userbot, if available. Returns the
        raw dict from `TelegramService.resolve_chat_name`, or None on any
        failure / when the userbot isn't wired."""
        if self._telegram is None:
            return None
        try:
            return await self._telegram.resolve_chat_name(chat_id)
        except Exception:
            log.exception("resolve_chat_name failed for %s", chat_id)
            return None

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
                    "text": escape_v2(body),
                    "parse_mode": "MarkdownV2",
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

    async def _dispatch_approval_subscriber(self) -> None:
        """Listen for `dispatch.approval_requested` events. These are
        operator-initiated dispatch_task calls made during an autonomous-
        reply turn — we send the user a Yes/No keyboard. On tap, the
        callback handler calls `operator.resolve_dispatch_approval` which
        either spawns the task (locked to the same chat) or drops it."""
        async for env in self._events.subscribe_global(
            types={"dispatch.approval_requested"},
        ):
            payload = env.get("payload") or {}
            # Filter by the bot's own session — there may be multiple chat
            # sessions in the DB but only one bot per owner.
            if payload.get("chat_session_id") != self._session_id:
                continue
            dispatch_id = payload.get("dispatch_id")
            if not dispatch_id:
                continue
            prompt = (payload.get("prompt") or "").strip()
            model = (payload.get("model") or "?").strip()
            locked = payload.get("restricted_to_chat") or "?"
            preview = _truncate(prompt, 400)
            body = (
                f"Approve autonomous dispatch_task?\n\n"
                f"_Locked to chat {locked}; spawned task will be too._\n"
                f"_Model: {model}_\n\n"
                f"```\n{preview}\n```"
            )
            try:
                await self._api.call("sendMessage", {
                    "chat_id": self._owner_user_id,
                    "text": escape_v2(body),
                    "parse_mode": "MarkdownV2",
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": "✅ Yes", "callback_data": f"disp:{dispatch_id}:allow"},
                            {"text": "❌ No",  "callback_data": f"disp:{dispatch_id}:deny"},
                        ]],
                    },
                })
                telegram_log.info("bot dispatch approval prompt " + fmt(
                    session=self._session_id, dispatch_id=dispatch_id,
                    locked_to=locked, model=model,
                ))
            except Exception:
                log.exception(
                    "failed to send dispatch approval prompt for %s", dispatch_id,
                )

    async def _handle_dispatch_callback(
        self, cq: dict[str, Any], dispatch_id: str, decision: str,
    ) -> None:
        cq_id = cq.get("id")
        try:
            outcome = await self._operator.resolve_dispatch_approval(
                dispatch_id, decision,
            )
        except Exception:
            log.exception("resolve_dispatch_approval crashed for %s", dispatch_id)
            if cq_id:
                await self._safe_answer_callback(cq_id, "Internal error.")
            return
        status = outcome.get("status")
        if status == "approved":
            answer = "Approved ✓ — task dispatched"
            edit = "allow"
        elif status == "denied":
            answer = "Denied ✗"
            edit = "deny"
        elif status == "already_resolved":
            answer = f"Already resolved ({outcome.get('resolution')})"
            edit = outcome.get("resolution") or "?"
        else:
            answer = f"Error: {outcome.get('error', 'unknown')}"
            edit = "?"
        telegram_log.info("bot dispatch approval resolve " + fmt(
            dispatch_id=dispatch_id, decision=decision, status=status,
        ))
        if cq_id:
            await self._safe_answer_callback(cq_id, answer)
        await self._maybe_edit_resolved(cq, edit)

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
        # Two callback shapes, distinguished by prefix:
        #   appr:<approval_id>:<decision>  — executor-side tool approval
        #   disp:<dispatch_id>:<decision>  — operator-initiated deferred
        #                                    dispatch_task approval
        parts = data.split(":", 2)
        if len(parts) != 3 or parts[0] not in ("appr", "disp"):
            if cq_id:
                await self._safe_answer_callback(cq_id, "Unknown action.")
            return
        decision = parts[2]
        if decision not in {"allow", "deny"}:
            if cq_id:
                await self._safe_answer_callback(cq_id, "Bad decision.")
            return
        if parts[0] == "disp":
            await self._handle_dispatch_callback(cq, parts[1], decision)
            return
        approval_id_str = parts[1]

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
                "text": escape_v2(new_text), "parse_mode": "MarkdownV2",
                "reply_markup": {"inline_keyboard": []},
            })
        except Exception:
            log.exception("editMessageText failed for resolved approval")

    # ---- send ----

    async def _send(self, chat_id: Any, text: str) -> None:
        """Send `text` as MarkdownV2 with auto-escape; fall back to plain text
        if Telegram rejects the parse (returns ok=false). The fallback path is
        the safety net for anything our V2 escaper doesn't anticipate — e.g.
        an unmatched bracket inside a long code block, or formatting the model
        invented that we don't yet recognize."""
        if chat_id is None or not text:
            return
        for piece in chunk_message(text):
            try:
                await self._api.call("sendMessage", {
                    "chat_id": chat_id, "text": escape_v2(piece),
                    "parse_mode": "MarkdownV2",
                })
            except Exception as e:
                log.warning("MarkdownV2 send failed (%s); falling back to plain text", e)
                await self._api.call("sendMessage", {
                    "chat_id": chat_id, "text": piece,
                })


# ---------------------------------------------------------------------------
# Formatting helpers (module-scope so tests can exercise them directly)
# ---------------------------------------------------------------------------

def _label_for_chat(chat_id: str, resolved: dict[str, Any] | None) -> str:
    """Render a chat as `Display Name (@username, chat_id)` when resolved,
    or just `chat_id` when not. Either or both of display_name / username
    may be missing — fall back gracefully."""
    if not resolved:
        return chat_id
    name = (resolved.get("display_name") or "").strip()
    uname = (resolved.get("username") or "").strip()
    parts: list[str] = []
    if name:
        parts.append(name)
    handle_bits: list[str] = []
    if uname:
        handle_bits.append(f"@{uname}")
    handle_bits.append(chat_id)
    parts.append(f"({', '.join(handle_bits)})")
    return " ".join(parts) if parts and name else f"@{uname} ({chat_id})" if uname else chat_id


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
