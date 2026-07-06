"""Telegram agent — user-facing chat surface, running as a second userbot.

This is a userbot (MTProto via telethon), NOT a Bot API account. It runs on
a dedicated Telegram account separate from the user's primary account. The
user DMs this account to talk to the operator; approvals, chat.reply
auto-pings, and (future) voice calls land here.

Distinct from [telegram_service.py](telegram_service.py): that runs on the
user's PRIMARY account and surfaces inbound DMs from third parties for
triage and reply-on-behalf-of. The agent userbot only ever exchanges
messages with the owner.

Why a userbot instead of a Bot API account: bots cannot place or receive
voice calls, and 1:1 Telegram voice calls between userbots are E2EE. We
also gain a uniform MTProto transport for both directions of the agent's
work.

Approvals are text-only (DESIGN §8): the broker emits a prompt with the
canonical command, blast radius, and challenge phrase; the user replies
with the phrase to allow. There is no inline-keyboard variant — that's a
Bot-API-only feature.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re as _re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .approval_client import is_deny_phrase, phrases_match
from .audit import fmt, telegram_log
from .broker import Broker
from .db import Database
from .events import EventBus
from .models import TaskState
from .operator import Operator
from .telegram_format import (
    age,
    chunk_message,
    label_for_chat,
    relative_age,
    reply_context_note,
    truncate,
)
from . import service


log = logging.getLogger(__name__)


# Per-attachment cap. Mirrors the bot's old _ATTACHMENT_MAX_BYTES — sized for
# screenshots, small PDFs, etc.; well within Gemini's inline-data limit.
_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024

# Where inbound attachments are written so the executor can `Read` / `Bash`
# them by absolute path. One subdir per attachment so collisions on
# `filename` (multiple users sending "image.png") can't clobber each other.
_INBOUND_DIR = Path("~/.oncall/inbound").expanduser()
_FILENAME_SAFE = _re.compile(r"[^A-Za-z0-9._-]+")


def _persist_inbound_attachment(data: bytes, filename: str | None) -> Path:
    """Write `data` to ~/.oncall/inbound/<uuid>/<safe-filename> and return
    the absolute path. Returned path is included in the operator's user
    message so the executor can find and read the file."""
    _INBOUND_DIR.mkdir(parents=True, exist_ok=True)
    subdir = _INBOUND_DIR / uuid4().hex
    subdir.mkdir(parents=True, exist_ok=False)
    raw = (filename or "attachment").strip() or "attachment"
    safe = _FILENAME_SAFE.sub("_", raw)[:120] or "attachment"
    out = subdir / safe
    out.write_bytes(data)
    return out


def agent_session_id(owner_user_id: int) -> str:
    """Deterministic chat-session id for the agent's conversation with the
    owner. One owner ↔ one session, persistent across daemon restarts."""
    return f"tg-agent-{owner_user_id}"


_SLASH_HELP = (
    "/start — greeting\n"
    "/status — snapshot of running tasks, queue, approvals, pending DMs\n"
    "/context — export this session's chat history + latest summary as a markdown file\n"
    "/clear — wipe this chat's history and reset the executor session (memory is preserved)\n"
    "/compact — force-compact older messages into a summary now\n"
    "/allowdm <chat_id> — allowlist a chat for triage + autonomous DM replies (empty by default; non-allowlisted DMs are dropped, never surfaced)\n"
    "/denydm <chat_id> — remove a chat from the DM allowlist (stops triaging it)\n"
    "/dmlist — show allowlisted chats\n"
    "/setownername <name> — set your display name used in the operator's system prompt\n"
    "/yes <id> — approve a pending deferred dispatch\n"
    "/no <id> — deny a pending deferred dispatch (or approval)\n"
    "/restart — restart the oncall daemon via launchctl (brief downtime)\n"
    "/stop — stop the oncall daemon via launchctl (agent goes silent until manual start)\n"
    "/help — this\n"
    "Anything else is a chat turn. To approve a tool call, reply with the challenge phrase."
)


class TelegramAgentService:
    def __init__(
        self,
        *,
        client: Any,
        operator: Operator,
        events: EventBus,
        owner_user_id: int,
        broker: Broker,
        db: Database,
        telegram: Any | None = None,
    ) -> None:
        self._client = client
        self._operator = operator
        self._events = events
        self._owner_user_id = int(owner_user_id)
        self._broker = broker
        self._db = db
        # Optional primary-userbot handle; used by `/dmlist` to resolve
        # chat_ids to human-readable names. None → /dmlist shows ids only.
        self._telegram = telegram
        self._session_id = agent_session_id(owner_user_id)
        self._started = False
        self._handler_ref: Any = None
        # Pending approvals for this session: approval_id → challenge_phrase.
        # Populated on approval.requested; drained on approval.resolved.
        # The user resolves an approval by typing the phrase as a normal
        # chat message; _try_resolve_approval matches against this dict.
        self._pending_approvals: dict[str, str] = {}
        self._reply_task: asyncio.Task | None = None
        self._approval_task: asyncio.Task | None = None
        self._approval_resolved_task: asyncio.Task | None = None
        self._dispatch_approval_task: asyncio.Task | None = None
        # Mutual-exclusion flag for /compact, /context — single-threaded
        # asyncio makes a plain attribute sufficient.
        self._heavy_op_in_flight: str | None = None
        # Cached at startup via get_me() so /help / logs can reference us.
        self._me_user_id: int | None = None
        self._me_username: str | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def agent_user_id(self) -> int | None:
        return self._me_user_id

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._started:
            return
        await self._client.connect()
        if not await self._client.is_user_authorized():
            raise RuntimeError(
                "Telegram agent session not authorized — run "
                "`oncall telegram-login --agent` first."
            )
        me = await self._client.get_me()
        self._me_user_id = int(getattr(me, "id", 0)) or None
        self._me_username = getattr(me, "username", None)
        try:
            from telethon import events  # type: ignore
            event_filter: Any = events.NewMessage(incoming=True)
        except ImportError:
            event_filter = None
        self._handler_ref = self._build_handler()
        self._client.add_event_handler(self._handler_ref, event_filter)

        self._reply_task = asyncio.create_task(
            self._chat_reply_subscriber(), name="tg-agent-reply",
        )
        self._reply_task.add_done_callback(self._on_bg_task_done)
        self._approval_task = asyncio.create_task(
            self._approval_subscriber(), name="tg-agent-approval",
        )
        self._approval_task.add_done_callback(self._on_bg_task_done)
        self._approval_resolved_task = asyncio.create_task(
            self._approval_resolved_subscriber(),
            name="tg-agent-approval-resolved",
        )
        self._approval_resolved_task.add_done_callback(self._on_bg_task_done)
        self._dispatch_approval_task = asyncio.create_task(
            self._dispatch_approval_subscriber(),
            name="tg-agent-dispatch-approval",
        )
        self._dispatch_approval_task.add_done_callback(self._on_bg_task_done)
        self._started = True
        log.info(
            "telegram agent started (owner_user_id=%d, agent=@%s id=%s, session=%s)",
            self._owner_user_id, self._me_username, self._me_user_id, self._session_id,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        for task in (
            self._reply_task, self._approval_task,
            self._approval_resolved_task, self._dispatch_approval_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        try:
            self._client.remove_event_handler(self._handler_ref)
        except Exception:
            log.exception("error removing telegram agent handler")
        try:
            await self._client.disconnect()
        except Exception:
            log.exception("error disconnecting telegram agent client")
        self._started = False
        log.info("telegram agent stopped")

    # ---- notifications ----

    async def notify_owner(self, text: str) -> None:
        """Send an out-of-band plain-text message to the owner. Used by the
        daemon for startup status + background-task crash surfacing."""
        if not self._started:
            return
        try:
            for piece in chunk_message(text):
                await self._client.send_message(self._owner_user_id, piece)
        except Exception:
            log.exception("notify_owner failed")

    def _on_bg_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        log.error("background task %r exited: %r", task.get_name(), exc)
        asyncio.create_task(self.notify_owner(
            f"⚠️ background task '{task.get_name()}' crashed: "
            f"{type(exc).__name__}: {exc}"
        ))

    # ---- inbound handler ----

    def _build_handler(self):
        async def _on_new_message(event: Any) -> None:
            try:
                await self._handle_inbound(event)
            except Exception:
                log.exception("telegram agent inbound handler crashed")
        return _on_new_message

    async def _handle_inbound(self, event: Any) -> None:
        if not getattr(event, "is_private", False):
            return  # MVP: 1:1 DMs only
        sender = await event.get_sender() if hasattr(event, "get_sender") else None
        sender_id = getattr(sender, "id", None) if sender else None
        if sender_id != self._owner_user_id:
            telegram_log.warning("agent inbound from non-owner " + fmt(
                sender_id=sender_id, owner_id=self._owner_user_id,
                username=getattr(sender, "username", None),
            ))
            return

        # In telethon, `event.message.message` is the message body — for
        # pure text it's the text; for a media message with caption it's
        # the caption; for media without caption it's empty.
        text = (getattr(event.message, "message", None) or "").strip()
        # Telegram reply pointer → explicit anchor for the operator. Without
        # it, a reply to a specific agent message is indistinguishable from
        # a plain message and deictic answers ("yes, that one") get resolved
        # against whatever is most recent in history.
        reply_note: str | None = None
        if getattr(event.message, "reply_to", None) is not None:
            try:
                reply = await event.message.get_reply_message()
            except Exception:
                log.warning(
                    "agent: fetching replied-to message failed", exc_info=True,
                )
                reply = None
            if reply is not None:
                # In the agent userbot's client, out=True → the agent's own
                # message (i.e. the operator's), else the owner's.
                who = (
                    "your earlier message"
                    if getattr(reply, "out", False)
                    else "their own earlier message"
                )
                reply_note = reply_context_note(reply, who=who)
        attachments: list[dict[str, Any]] = []
        if getattr(event.message, "media", None) is not None:
            try:
                data = await event.message.download_media(file=bytes)
            except Exception:
                log.exception("agent download_media failed")
                await self._send(
                    "Couldn't download that attachment. Try again or send it "
                    "as a smaller file.",
                )
                return
            if data and len(data) > _ATTACHMENT_MAX_BYTES:
                await self._send(
                    f"Attachment too large ({len(data)} bytes; cap "
                    f"{_ATTACHMENT_MAX_BYTES})."
                )
                return
            if data:
                f = getattr(event.message, "file", None)
                mime = (getattr(f, "mime_type", None) if f else None) \
                    or "application/octet-stream"
                fname = (getattr(f, "name", None) if f else None) or "attachment"
                # Persist to disk so a follow-on executor turn can Read/Bash
                # the file by path. We KEEP the inline base64 too — the
                # operator (multimodal Gemini) reads images/PDFs directly
                # from bytes; the disk path is for the executor.
                local_path: Path | None = None
                try:
                    local_path = _persist_inbound_attachment(data, fname)
                except Exception:
                    log.exception("agent: persist inbound attachment failed")
                attachments.append({
                    "data_b64": base64.b64encode(data).decode("ascii"),
                    "mime_type": mime,
                    "size_bytes": len(data),
                    "source": f"telegram agent ({fname})",
                    **({"local_path": str(local_path)} if local_path else {}),
                })
                if local_path:
                    note = (
                        f"[file attached: {local_path} "
                        f"({mime}, {len(data)} bytes)]"
                    )
                    text = f"{text}\n{note}".strip() if text else note
                elif not text:
                    text = "(attachment — please look at the image)"
        if not text:
            return

        # Approval phrase match BEFORE slash commands / operator routing.
        # The phrase is freeform text the broker chose; it could conflict
        # with a slash command word, but in practice the phrase is 3
        # BIP39-shaped words.
        if await self._try_resolve_approval(text):
            return

        if text.startswith("/"):
            handled = await self._handle_slash(text)
            if not handled:
                cmd = text.split(None, 1)[0]
                telegram_log.info("agent unknown slash " + fmt(
                    session=self._session_id, cmd=cmd,
                ))
                await self._send(
                    f"Unknown command: {cmd}. Send /help for the list."
                )
            return

        # Prepend the reply anchor only for the operator path — approval
        # phrases and slash commands above must match the raw text even
        # when sent as a Telegram reply.
        if reply_note:
            text = f"{reply_note}\n{text}"

        telegram_log.info("agent inbound " + fmt(
            session=self._session_id, len=len(text),
            attachments=len(attachments),
        ))
        try:
            result = await self._operator.chat_turn(
                session_id=self._session_id, user_text=text,
                attachments=attachments or None,
            )
        except Exception:
            log.exception("operator.chat_turn failed for telegram agent")
            await self._send("Internal error. Try again in a moment.")
            return

        reply = result.user_facing_text()
        if not reply:
            telegram_log.info("agent reply suppressed (empty) " + fmt(
                session=self._session_id,
                tool_calls=len(result.tool_calls_made),
            ))
            return
        await self._send(reply)
        telegram_log.info("agent reply " + fmt(
            session=self._session_id, len=len(reply),
            tool_calls=len(result.tool_calls_made),
        ))
        # If an owner voice call is active for this same session, the voice
        # subscriber TTSes any `chat.reply` event with non-empty `voice_text`.
        # We publish with `text=""` so the telegram-side chat.reply subscriber
        # (which filters empty text) doesn't double-send what we already sent
        # above. No call active → no voice subscriber → harmless no-op.
        try:
            await self._events.publish_global("chat.reply", {
                "session_id": self._session_id,
                "text": "",
                "voice_text": reply,
                "trigger": "agent.chat_turn",
                "task_id": None,
            })
        except Exception:
            log.exception(
                "agent: publish chat.reply for voice-tts failed (session=%s)",
                self._session_id,
            )

    # ---- slash commands ----

    async def _handle_slash(self, text: str) -> bool:
        """Returns True if `text` matched a slash command (handled). False
        means the caller should fall through to the operator."""
        cmd = text.split(None, 1)[0]
        if cmd == "/start":
            await self._send(
                "Hi. Tell me what you need — I'll dispatch tasks and ping "
                "you back when they're done."
            )
            return True
        if cmd == "/help":
            await self._send(_SLASH_HELP)
            return True
        if cmd in ("/allowdm", "/denydm"):
            await self._handle_allowlist(text)
            return True
        if cmd == "/dmlist":
            await self._send(await self._render_dmlist())
            return True
        if cmd == "/setownername":
            from .config import write_owner_name, read_owner_name
            arg = text[len("/setownername"):].strip()
            if not arg:
                current = read_owner_name()
                await self._send(
                    f"Usage: /setownername <name>\nCurrent: {current}"
                )
                return True
            try:
                write_owner_name(arg)
            except OSError as e:
                log.warning("write_owner_name failed: %s", e)
                await self._send(f"Failed to write owner name: {e}")
                return True
            saved = read_owner_name()
            telegram_log.info("agent setownername " + fmt(name=saved))
            await self._send(f"Owner name set to: {saved}")
            return True
        if cmd == "/status":
            await self._send(await self._render_status())
            return True
        if cmd == "/context":
            if not await self._claim_heavy_op("/context"):
                return True
            try:
                try:
                    dump = await self._operator.export_context(self._session_id)
                except Exception:
                    log.exception("operator.export_context failed")
                    await self._send("Failed to export context. Check logs.")
                    return True
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                filename = f"oncall-context-{self._session_id}-{stamp}.md"
                try:
                    await self._client.send_file(
                        self._owner_user_id,
                        file=dump.encode("utf-8"),
                        attributes=None,
                        force_document=True,
                        file_name=filename,
                        caption="Operator context for this session.",
                    )
                except Exception:
                    log.exception("send_file failed for /context")
                    await self._send("Failed to upload context file. Check logs.")
                    return True
                telegram_log.info("agent context " + fmt(
                    session=self._session_id, bytes=len(dump),
                ))
            finally:
                self._release_heavy_op()
            return True
        if cmd == "/clear":
            out = await self._operator.clear_session(self._session_id)
            if out.get("executor_session_reset"):
                exec_note = " Executor session reset."
            elif out.get("executor_reset_reason") == "busy":
                exec_note = (
                    " Executor session kept — a task is in flight; "
                    "re-run /clear once it's idle to reset it."
                )
            else:
                exec_note = ""
            await self._send(
                f"Context cleared ({out['messages_deleted']} messages, "
                f"{out['summaries_deleted']} summaries).{exec_note}"
            )
            telegram_log.info("agent clear " + fmt(
                session=self._session_id, **out,
            ))
            return True
        if cmd == "/compact":
            if not await self._claim_heavy_op("/compact"):
                return True
            try:
                out = await self._operator.compress_now(self._session_id)
                if out.get("compressed"):
                    await self._send(
                        f"Compacted {out['older_rows']} messages into "
                        f"~{out['summary_tokens']} tokens of summary."
                    )
                else:
                    await self._send(f"Nothing to compact: {out.get('reason')}.")
                telegram_log.info("agent compact " + fmt(
                    session=self._session_id, **out,
                ))
            finally:
                self._release_heavy_op()
            return True
        if cmd in ("/restart", "/stop"):
            is_restart = cmd == "/restart"
            ack = (
                "Restarting service. Brief downtime; you'll get the "
                "startup ping when I'm back."
                if is_restart else
                "Stopping service. Agent will go silent until "
                "`oncall service start` is run manually."
            )
            await self._send(ack)
            telegram_log.info(
                f"agent service-{'restart' if is_restart else 'stop'} requested "
                + fmt(session=self._session_id),
            )
            asyncio.create_task(self._do_service_action(is_restart=is_restart))
            return True
        if cmd in ("/yes", "/no"):
            await self._handle_dispatch_decision(text)
            return True
        return False

    async def _handle_allowlist(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if not arg:
            verb = "allowdm" if cmd.startswith("/allowdm") else "denydm"
            await self._send(f"Usage: /{verb} <chat_id>")
            return
        if any(c.isspace() for c in arg):
            await self._send("chat_id must not contain whitespace.")
            return
        if cmd.startswith("/allowdm"):
            added = await self._db.allow_dm(arg)
            msg = (
                f"Allowlisted chat_id={arg} — its DMs will now be triaged, and autonomous replies are permitted."
                if added else
                f"chat_id={arg} was already on the allowlist."
            )
            telegram_log.info("agent allowdm " + fmt(chat_id=arg, newly_added=added))
        else:
            removed = await self._db.deny_dm(arg)
            msg = (
                f"Removed chat_id={arg} from the DM allowlist."
                if removed else
                f"chat_id={arg} was not on the allowlist."
            )
            telegram_log.info("agent denydm " + fmt(chat_id=arg, was_present=removed))
        await self._send(msg)

    async def _render_dmlist(self) -> str:
        rows = await self._db.list_dm_allowed()
        if not rows:
            return (
                "DM allowlist is empty. No DMs are triaged and no chat may "
                "receive an autonomous reply. Use /allowdm <chat_id> to add one."
            )
        lines = ["DM allowlist:"]
        for r in rows:
            chat_id = r["chat_id"]
            label = label_for_chat(chat_id, await self._resolve_label(chat_id))
            lines.append(f"- {label} (added {relative_age(r['added_at'])})")
        return "\n".join(lines)

    async def _resolve_label(self, chat_id: str) -> dict[str, Any] | None:
        if self._telegram is None:
            return None
        try:
            return await self._telegram.resolve_chat_name(chat_id)
        except Exception:
            log.exception("resolve_chat_name failed for %s", chat_id)
            return None

    async def _render_status(self) -> str:
        running = await self._db.list_tasks_in_states(TaskState.RUNNING)
        queued = await self._db.list_tasks_in_states(TaskState.PENDING)
        awaiting = await self._db.list_tasks_in_states(TaskState.AWAITING_APPROVAL)
        approvals = await self._db.list_pending_approvals()
        # "Pending" = not-yet-triaged: matches the drain's notion of work
        # the bot still owes the user. `read_at` is just a recipient-side
        # indicator and isn't a signal for our agentic logic — a triaged
        # chat may still be unread, and a read chat may be mid-flight.
        pending_chats = await self._db.list_pending_chats()
        pending = sum(r["unread_count"] for r in pending_chats)

        running.sort(key=lambda t: t.created_at)
        queued.sort(key=lambda t: t.created_at)

        lines = [
            "oncall status",
            "",
            f"Tasks: {len(running)} running, {len(queued)} queued, "
            f"{len(awaiting)} awaiting approval",
            f"Approvals pending: {len(approvals)}",
            f"Pending DMs: {pending}",
        ]

        if running:
            lines += ["", "Running:"]
            for t in running[:5]:
                lines.append(f"- {str(t.id)[:6]} ({age(t.created_at)}): {truncate(t.prompt, 70)}")
            if len(running) > 5:
                lines.append(f"- ...and {len(running) - 5} more")

        if queued:
            lines += ["", "Queued:"]
            for t in queued[:5]:
                lines.append(f"- {str(t.id)[:6]}: {truncate(t.prompt, 70)}")
            if len(queued) > 5:
                lines.append(f"- ...and {len(queued) - 5} more")

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
                    f"- last compression: {relative_age(created)} "
                    f"(through msg #{last['through_message_id']}, "
                    f"~{last['estimated_token_count']} summary tokens)"
                )
            else:
                lines.append("- last compression: none yet")

        if not (running or queued or approvals or pending) and op is None:
            return "All quiet. No tasks, no pending approvals, no pending DMs."

        return "\n".join(lines)

    # ---- heavy-op gate ----

    async def _claim_heavy_op(self, name: str) -> bool:
        if self._heavy_op_in_flight is not None:
            await self._send(
                f"{self._heavy_op_in_flight} is still running; "
                f"try {name} again once it finishes."
            )
            return False
        self._heavy_op_in_flight = name
        return True

    def _release_heavy_op(self) -> None:
        self._heavy_op_in_flight = None

    async def _do_service_action(self, *, is_restart: bool) -> None:
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

    # ---- approval subscribers ----

    async def _approval_subscriber(self) -> None:
        """When the broker emits an `approval.requested` event for a task
        dispatched by THIS session, send a text prompt to the owner with
        the canonical command, blast radius, and challenge phrase. The
        user resolves it by typing the phrase as a normal message
        (handled in `_try_resolve_approval`)."""
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
            phrase = (payload.get("challenge_phrase") or "").strip()
            if not phrase:
                continue
            self._pending_approvals[str(approval_id)] = phrase
            body_lines = [
                "Approve this command?",
                "",
                f"`{canonical}`" if canonical else "",
            ]
            if blast:
                body_lines += ["", blast]
            body_lines += [
                "",
                f"Reply `{phrase}` to allow, or `no` to deny.",
            ]
            body = "\n".join(b for b in body_lines if b is not None)
            try:
                await self._send(body)
                telegram_log.info("agent approval prompt " + fmt(
                    session=self._session_id, approval=approval_id,
                    task=task_id_str, canonical=canonical,
                ))
                # Mirror the prompt into operator chat history as an
                # assistant message so the operator's next turn sees that
                # an approval was asked. No LLM inference — just a DB
                # write. Runs regardless of voice state; the operator
                # benefits from this context in both modes.
                try:
                    await self._db.append_chat_message(
                        self._session_id, "assistant", body,
                    )
                except Exception:
                    log.exception(
                        "agent approval: append_chat_message failed",
                    )
            except Exception:
                log.exception("failed to send approval prompt for %s", approval_id)

    async def _approval_resolved_subscriber(self) -> None:
        """Drop entries from the in-memory pending dict once an approval is
        resolved (by us, by timeout, or by anyone else). Keeps the
        try-resolve path from matching against stale phrases."""
        async for env in self._events.subscribe_global(types={"approval.resolved"}):
            payload = env.get("payload") or {}
            approval_id = payload.get("approval_id")
            if approval_id:
                self._pending_approvals.pop(str(approval_id), None)

    async def _try_resolve_approval(self, text: str) -> bool:
        """If `text` reads as an affirmative ("yes yes yes" / multilingual)
        OR a bare deny ("no" / multilingual), resolve every pending
        approval in this chat accordingly and return True. Returns False
        if `text` is neither — caller forwards to the operator as usual."""
        if not self._pending_approvals:
            return False
        affirm = phrases_match("", text)
        deny = (not affirm) and is_deny_phrase(text)
        if not (affirm or deny):
            return False
        decision = "allow" if affirm else "deny"
        # Snapshot — the dict can mutate as approval.resolved events come
        # in concurrently. Resolve every pending approval in this chat with
        # the same decision; in practice there's usually one at a time, but
        # if multiple are open the user's single "no" denies them all
        # (predictable, no ambiguity).
        any_resolved = False
        for approval_id_str in list(self._pending_approvals.keys()):
            try:
                approval_uuid = UUID(approval_id_str)
            except ValueError:
                self._pending_approvals.pop(approval_id_str, None)
                continue
            approved, matched = await self._broker.submit_response(
                approval_id=approval_uuid,
                decision=decision,
                challenge_phrase_supplied=text,
            )
            self._pending_approvals.pop(approval_id_str, None)
            telegram_log.info("agent approval resolve " + fmt(
                approval=approval_id_str, decision=decision,
                approved=approved, matched=matched,
            ))
            any_resolved = True
        if any_resolved:
            await self._send("Approved ✓" if affirm else "Denied ✗")
        return any_resolved

    async def _dispatch_approval_subscriber(self) -> None:
        """Listen for `dispatch.approval_requested` events. The operator
        initiated a `dispatch_task` during an autonomous-reply turn and
        is asking us to confirm before spawning. User resolves via
        `/yes <dispatch_id>` or `/no <dispatch_id>`."""
        async for env in self._events.subscribe_global(
            types={"dispatch.approval_requested"},
        ):
            payload = env.get("payload") or {}
            if payload.get("chat_session_id") != self._session_id:
                continue
            dispatch_id = payload.get("dispatch_id")
            if not dispatch_id:
                continue
            prompt = (payload.get("prompt") or "").strip()
            model = (payload.get("model") or "?").strip()
            locked = payload.get("restricted_to_chat") or "?"
            preview = truncate(prompt, 400)
            body = (
                f"Approve autonomous dispatch_task?\n\n"
                f"Locked to chat {locked}; spawned task will be too.\n"
                f"Model: {model}\n\n"
                f"```\n{preview}\n```\n\n"
                f"Reply `/yes {dispatch_id}` to approve, "
                f"`/no {dispatch_id}` to deny."
            )
            try:
                await self._send(body)
                telegram_log.info("agent dispatch approval prompt " + fmt(
                    session=self._session_id, dispatch_id=dispatch_id,
                    locked_to=locked, model=model,
                ))
            except Exception:
                log.exception(
                    "failed to send dispatch approval prompt for %s", dispatch_id,
                )

    async def _handle_dispatch_decision(self, text: str) -> None:
        """`/yes <id>` and `/no <id>` resolve either a deferred dispatch
        OR a pending tool approval (denial path), depending on which kind
        of id is supplied. We try dispatch first, then approval."""
        parts = text.split(None, 1)
        verb = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if not arg:
            await self._send(f"Usage: {verb} <id>")
            return
        decision_dispatch = "allow" if verb == "/yes" else "deny"

        # Try deferred-dispatch resolution first (the only place /yes is
        # meaningful — tool approvals require the challenge phrase, not /yes).
        try:
            outcome = await self._operator.resolve_dispatch_approval(
                arg, decision_dispatch,
            )
        except Exception:
            log.exception("resolve_dispatch_approval crashed for %s", arg)
            outcome = {"status": "error", "error": "internal error"}
        status = outcome.get("status")
        if status == "approved":
            await self._send("Approved ✓ — task dispatched.")
            return
        if status == "denied":
            await self._send("Denied ✗.")
            return
        if status == "already_resolved":
            await self._send(f"Already resolved ({outcome.get('resolution')}).")
            return

        # Not a known dispatch — fall back to "is it a tool approval id?"
        # Only meaningful for /no (deny). /yes for tool approvals is wrong
        # input — the user is supposed to supply the challenge phrase.
        if verb != "/no":
            await self._send(
                f"No pending deferred dispatch with id {arg}. "
                f"To approve a tool call, reply with the challenge phrase."
            )
            return
        try:
            approval_uuid = UUID(arg)
        except ValueError:
            await self._send(f"Unknown id: {arg}")
            return
        row = await self._db.get_approval(approval_uuid)
        if row is None:
            await self._send(f"No pending approval with id {arg}.")
            return
        if row["state"] != "pending":
            await self._send(f"Approval {arg} is already resolved.")
            return
        # Submit deny — phrase mismatch coerces to deny anyway, so anything
        # non-empty works. Use the canonical phrase so the audit log shows
        # an intentional deny rather than a typo.
        await self._broker.submit_response(
            approval_id=approval_uuid,
            decision="deny",
            challenge_phrase_supplied=row["challenge_phrase"] or "",
        )
        self._pending_approvals.pop(arg, None)
        telegram_log.info("agent approval explicit deny " + fmt(
            approval=arg, session=self._session_id,
        ))
        await self._send("Denied ✗.")

    # ---- chat.reply auto-ping subscriber ----

    async def _chat_reply_subscriber(self) -> None:
        """Deliver `chat.reply` events to the owner. With text-only approvals
        there's no separate button UI, so we drop the bot's old filter
        for approval.requested triggers — the operator's text IS the
        message."""
        async for env in self._events.subscribe_global(types={"chat.reply"}):
            payload = env.get("payload") or {}
            if payload.get("session_id") != self._session_id:
                continue
            text = payload.get("text") or ""
            if not text:
                continue
            try:
                await self._send(text)
                telegram_log.info("agent auto-ping " + fmt(
                    session=self._session_id, len=len(text),
                    task_id=payload.get("task_id"),
                ))
            except Exception:
                log.exception("failed to deliver chat.reply via telegram agent")

    # ---- send ----

    async def _send(self, text: str) -> None:
        if not text:
            return
        for piece in chunk_message(text):
            try:
                await self._client.send_message(
                    self._owner_user_id, piece, parse_mode="md",
                )
            except Exception as e:
                log.warning(
                    "markdown send failed (%s); falling back to plain text", e,
                )
                try:
                    await self._client.send_message(self._owner_user_id, piece)
                except Exception:
                    log.exception("plain-text send also failed")
