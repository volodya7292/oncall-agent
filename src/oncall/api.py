"""FastAPI HTTP surface.

Milestone 1 endpoints:
  POST   /tasks
  GET    /tasks
  GET    /tasks/{id}
  GET    /tasks/{id}/events            (SSE)
  POST   /tasks/{id}/kill
  GET    /approvals/pending
  GET    /approvals/{id}
  POST   /approvals/{id}/respond
  POST   /internal/broker/decide       (loopback only — called by mcp_server.py)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
from typing import Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .approval_client import HttpLongPollApprovalClient, is_kill_phrase
from .broker import Broker
from .config import get_paths, get_settings
from .db import Database
from .embeddings import GatewayEmbeddingClient, OllamaEmbeddingClient, is_ollama_model
from .events import EventBus
from .lifecycle import Lifecycle
from .local_claude import ClaudeCliRunner
from .operator import GatewayLLMClient, GenAILLMClient, Operator
from .operator_memory import OperatorMemory
from .task_summary import summarize_task
from .telegram_bot import HttpxBotApi, TelegramBotService
from .telegram_service import TelegramService, make_telethon_client
from .voice import to_voice_text


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SubmitTaskBody(BaseModel):
    prompt: str
    model: str | None = None
    max_turns: int | None = None
    chat_session_id: str | None = None


class TaskOut(BaseModel):
    id: UUID
    session_id: str
    state: str
    prompt: str
    model: str | None
    created_at: str
    updated_at: str
    terminal_reason: str | None


class ApprovalRespondBody(BaseModel):
    decision: Literal["allow", "deny"]
    challenge_phrase_supplied: str
    message: str | None = None


class KillBody(BaseModel):
    phrase: str


class ChatBody(BaseModel):
    session_id: str | None = None  # if absent, a new session id is minted
    text: str
    # BCP-47 / common code (e.g. "en", "ru", "en-US"). The operator gets it
    # as a hint at the bottom of its system prompt. Optional; if absent the
    # operator infers from the conversation history.
    language: str | None = None


class BrokerDecideBody(BaseModel):
    session_id: str
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


class MessengerOpBody(BaseModel):
    op: Literal[
        "list", "read", "mark_read", "style", "send",
        "history", "search", "search_messages", "list_chats",
    ]
    chat_id: str | None = None
    message_id: str | None = None
    inbox_id: str | None = None
    text: str | None = None
    query: str | None = None
    # Per-op default applied at the router. `read_inbox` reads as True
    # if unset; `list_chats` reads as False.
    unread_only: bool | None = None
    dms_only: bool = False
    limit: int = 20


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_token(x_oncall_token: str = Header(default="")) -> None:
    expected = get_settings().oncall_token
    if not x_oncall_token or x_oncall_token != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid X-Oncall-Token")


def verify_loopback(request: Request, x_oncall_token: str = Header(default="")) -> None:
    """Stricter check for /internal/* — token AND loopback origin."""
    verify_token(x_oncall_token)
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"loopback-only endpoint (got {host})")


# ---------------------------------------------------------------------------
# App factory + lifespan
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()
    paths = get_paths()

    db = Database(settings.oncall_db_path)
    events = EventBus(db)
    approval_client = HttpLongPollApprovalClient()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.connect()
        broker = Broker(
            db, approval_client, events.publish,
            approval_timeout_seconds=settings.oncall_approval_timeout_seconds,
        )
        lifecycle = Lifecycle(
            db=db, broker=broker, approval_client=approval_client,
            events=events, settings=settings, paths=paths,
        )
        # Operator LLM backend choice. "gemini" uses the native AI Studio
        # API and is the default — it preserves ack-first (text + tool_call
        # in the same response) which the Vercel gateway's gemma routing
        # silently strips. "vercel" keeps the OpenAI-compatible gateway path.
        # The operator is only set up if the backend's auth is configured;
        # otherwise /chat returns a clear 503.
        operator: Operator | None = None
        llm: GenAILLMClient | GatewayLLMClient | None = None
        if settings.oncall_operator_backend == "gemini" and settings.gemini_api_key:
            llm = GenAILLMClient(api_key=settings.gemini_api_key)
            log.info("operator LLM backend: gemini (AI Studio)")
        elif settings.oncall_operator_backend == "vercel" and settings.gateway_key:
            llm = GatewayLLMClient(
                base_url=settings.ai_gateway_base_url,
                api_key=settings.gateway_key,
            )
            log.info("operator LLM backend: vercel (AI Gateway)")
        elif settings.gateway_key:
            # Fallback: backend was set to "gemini" but key missing — and
            # vercel key happens to be there. Use vercel so the daemon still
            # boots with a working operator.
            llm = GatewayLLMClient(
                base_url=settings.ai_gateway_base_url,
                api_key=settings.gateway_key,
            )
            log.warning(
                "ONCALL_OPERATOR_BACKEND=%s but its key is unset; "
                "falling back to vercel gateway",
                settings.oncall_operator_backend,
            )
        # Telegram userbot — only set up if api_id/hash + session file are
        # all present. Inbound DMs from senders listed in
        # `telegram_userbot_ignore_usernames` are skipped at the handler.
        telegram: TelegramService | None = await _maybe_start_telegram(
            settings, db, events,
            ignore_usernames=settings.userbot_ignore_usernames,
        )
        # Shared one-shot Claude CLI runner — used by the operator for chat
        # context compression and by the auto-ping loop for per-task summaries.
        # Single instance: it's a fresh subprocess per call, no shared state.
        cli_runner = ClaudeCliRunner()
        # Operator construction requires (a) an LLM client and (b) an
        # embedder. Embedder backend is chosen by the model name shape:
        # ollama tags ("nomic-embed-text:...") route to the local daemon,
        # vendor-prefixed slugs ("alibaba/...") go through the Vercel
        # gateway. Local Ollama is the default — ~30× lower latency.
        embedder: GatewayEmbeddingClient | OllamaEmbeddingClient | None = None
        if llm is not None:
            if is_ollama_model(settings.oncall_memory_embed_model):
                embedder = OllamaEmbeddingClient(
                    host=settings.oncall_ollama_host,
                    model=settings.oncall_memory_embed_model,
                )
                log.info("memory embedder: ollama / %s",
                         settings.oncall_memory_embed_model)
            elif settings.gateway_key:
                embedder = GatewayEmbeddingClient(
                    base_url=settings.ai_gateway_base_url,
                    api_key=settings.gateway_key,
                    model=settings.oncall_memory_embed_model,
                )
                log.info("memory embedder: vercel gateway / %s",
                         settings.oncall_memory_embed_model)
            else:
                log.warning(
                    "no embedder configured: model %r looks like a gateway "
                    "slug but AI_GATEWAY_API_KEY is unset; operator memory "
                    "will be disabled",
                    settings.oncall_memory_embed_model,
                )
        if embedder is not None:
            memory = OperatorMemory(
                db, embedder,
                embed_model=settings.oncall_memory_embed_model,
                capacity=settings.oncall_memory_capacity,
                max_inject=settings.oncall_memory_max_inject,
                relevance_floor=settings.oncall_memory_relevance_floor,
                hybrid_alpha=settings.oncall_memory_hybrid_alpha,
                hybrid_beta=settings.oncall_memory_hybrid_beta,
            )
            operator = Operator(
                db=db, lifecycle=lifecycle, broker=broker,
                settings=settings, paths=paths, llm=llm,
                memory=memory,
                telegram=telegram,
                events=events,
                extract_llm=llm,  # share the gateway client; extractor uses
                                  # the configured cheap model via Settings
                runner=cli_runner,
            )
        # Telegram bot front-end (optional, separate from userbot above).
        telegram_bot: TelegramBotService | None = None
        if operator is not None:
            telegram_bot = await _maybe_start_telegram_bot(
                settings, operator, events, broker=broker, db=db,
            )
        # Tell the userbot to ignore the bot's own replies — otherwise every
        # outbound the bot front-end sends would re-enter the user's inbox.
        if telegram is not None and telegram_bot is not None and telegram_bot.bot_user_id:
            telegram.add_ignore_user_id(telegram_bot.bot_user_id)
            log.info(
                "userbot will ignore inbound from bot (@%s, id=%d)",
                telegram_bot.bot_username, telegram_bot.bot_user_id,
            )

        app.state.db = db
        app.state.events = events
        app.state.approval_client = approval_client
        app.state.broker = broker
        app.state.lifecycle = lifecycle
        app.state.operator = operator
        app.state.telegram = telegram
        app.state.telegram_bot = telegram_bot
        # Recover any tasks left running / awaiting_approval from a prior
        # orchestrator process. The CLI's session JSONL is on disk; we
        # re-spawn with --resume and the broker's (session_id, tool_use_id)
        # dedup re-attaches to existing pending approval rows.
        await lifecycle.recover()
        # Background: when a task dispatched from a chat session reaches a
        # terminal state, auto-ping the operator so it can summarize the result
        # for the user without the user having to ask.
        auto_ping_task: asyncio.Task | None = None
        if operator is not None:
            auto_ping_task = asyncio.create_task(
                _auto_ping_loop(
                    events=events, operator=operator, db=db,
                    runner=cli_runner,
                    summary_model=settings.oncall_compression_model,
                )
            )
        # Inbox drain: when the userbot lands an *important* inbound DM, push
        # it into the bot front-end's session as an auto-ping so the user
        # finds out immediately. Non-important DMs sit silently in
        # messenger_inbox and are picked up later via /status or
        # `read_inbox`. Requires the bot front-end (we ping its session).
        inbox_drain_task: asyncio.Task | None = None
        if operator is not None and telegram_bot is not None:
            inbox_drain_task = asyncio.create_task(
                _inbox_drain_loop(
                    events=events, operator=operator, db=db,
                    target_session_id=telegram_bot.session_id,
                ),
                name="inbox-drain",
            )
        # Memory dedup: write time always INSERTs (no heuristic merge).
        # Every 5 minutes we ask the operator LLM to consolidate clusters
        # of near-duplicates so paraphrase merges and same-template-
        # different-entity cases are decided by reading the texts, not by
        # cosine alone.
        memory_dedup_task: asyncio.Task | None = None
        if operator is not None and operator.memory is not None and llm is not None:
            memory_dedup_task = asyncio.create_task(
                _memory_dedup_loop(
                    memory=operator.memory, llm=llm,
                    model=settings.oncall_operator_model,
                ),
                name="memory-dedup",
            )
        # Memory-embedding rebuild: if the configured embed model differs
        # from what stored rows were last embedded with, kick off a
        # background re-embed pass. Retrieval is already filtering stale
        # rows out, so the operator just sees fewer memories until this
        # task completes — no data loss either way.
        stale_before = 0
        if operator is not None and operator.memory is not None:
            stale_before = await operator.memory.stale_count()
            if stale_before > 0:
                log.info(
                    "scheduling memory rebuild: %d row(s) embedded with a "
                    "different model than the configured %s",
                    stale_before, settings.oncall_memory_embed_model,
                )
                # Fire-and-forget — the task notifies on completion via the
                # bot; nothing in the lifespan waits on it.
                asyncio.create_task(
                    _rebuild_memory_then_notify(
                        operator.memory, telegram_bot,
                        stale=stale_before,
                        model=settings.oncall_memory_embed_model,
                    ),
                    name="memory-rebuild",
                )

        # Startup notification to the owner — single message summarizing what
        # came up cleanly and what's degraded. Sent only if the bot is up
        # (no bot → no way to notify). The probes are best-effort: a failed
        # probe means "couldn't verify", not "definitely broken".
        if telegram_bot is not None:
            status = await _build_startup_status(
                settings=settings, operator=operator,
                telegram_userbot=telegram is not None,
                lifecycle=lifecycle,
                stale_memories=stale_before,
            )
            await telegram_bot.notify_owner(status)
        try:
            yield
        finally:
            for bg_task in (auto_ping_task, inbox_drain_task, memory_dedup_task):
                if bg_task is None:
                    continue
                bg_task.cancel()
                try:
                    await bg_task
                except (asyncio.CancelledError, Exception):
                    pass
            await lifecycle.shutdown()
            if telegram_bot is not None:
                await telegram_bot.stop()
            if telegram is not None:
                await telegram.stop()
            await db.close()

    app = FastAPI(title="oncall-agent", lifespan=lifespan)
    _register_routes(app)
    return app


_TERMINAL_STATES = {"completed", "failed", "killed"}


async def _auto_ping_loop(
    *, events: EventBus, operator: Operator, db: Database,
    runner: ClaudeCliRunner, summary_model: str,
) -> None:
    """Re-engage the operator on two kinds of triggers, so the user sees
    follow-ups via the chat UI (REPL or Telegram bot) without having to ask:

      * state.changed → terminal (completed/failed/killed):
          1. Summarize the task's event trail into tasks.result_summary.
          2. auto_ping with `task X just terminated`.
          3. Publish the operator's reply as chat.reply.

      * approval.requested:
          auto_ping with `task X needs approval, approval_id=Y, tool=Z`.
          The operator's prompt directs it to call present_pending_approval
          and read back the canonical command + challenge phrase verbatim.

    Each step is fail-soft; the loop only exits on cancel."""
    from uuid import UUID
    async for env in events.subscribe_global(
        types={"state.changed", "approval.requested"},
    ):
        type_ = env.get("type")
        task_id_str = env.get("task_id")
        if not task_id_str:
            continue
        try:
            task_uuid = UUID(task_id_str)
            task = await db.get_task(task_uuid)
        except Exception:
            log.exception("auto-ping: failed to load task %s", task_id_str)
            continue
        if task is None or not task.dispatched_by_chat_session:
            continue
        session_id = task.dispatched_by_chat_session
        short = task_id_str[:8]
        payload = env.get("payload") or {}

        if type_ == "state.changed":
            new_state = payload.get("state")
            if new_state not in _TERMINAL_STATES:
                continue
            try:
                await summarize_task(db, runner, task_uuid, model=summary_model)
            except Exception:
                log.exception("auto-ping: summarize_task failed for %s", task_id_str)
            terminal = (task.terminal_reason.value if task.terminal_reason else new_state)
            note = f"task {short} just terminated, state={new_state}, reason={terminal}"
            trigger = "task.terminal"
            approval_id = None
        elif type_ == "approval.requested":
            approval_id = payload.get("approval_id") or ""
            tool_name = payload.get("tool_name") or "?"
            note = (
                f"task {short} needs approval. approval_id={approval_id}, "
                f"tool={tool_name}. Call present_pending_approval with that id, "
                f"then read the canonical command, blast radius, and challenge "
                f"phrase to the user VERBATIM. Do not paraphrase."
            )
            trigger = "approval.requested"
        else:
            continue

        try:
            result = await operator.auto_ping(session_id=session_id, note=note)
        except Exception:
            log.exception("auto-ping: operator.auto_ping failed for session %s", session_id)
            continue
        if not result.text:
            continue

        chat_reply_payload: dict[str, Any] = {
            "session_id": session_id,
            "text": result.text,
            "voice_text": to_voice_text(result.text),
            "trigger": trigger,
            "task_id": task_id_str,
        }
        if approval_id:
            chat_reply_payload["approval_id"] = approval_id
        await events.publish_global("chat.reply", chat_reply_payload)


_INBOX_BATCH_SIZE = 10
_INBOX_IDLE_FLUSH_SECONDS = 120.0  # 2 minutes


async def _inbox_drain_loop(
    *, events: EventBus, operator: Operator, db: Database,
    target_session_id: str,
) -> None:
    """Triage inbound DMs through the operator in per-chat batches.

    Each flush carries messages from EXACTLY ONE chat (no cross-chat mixing).
    A chat's batch flushes when whichever of these fires first for that chat:
      * `_INBOX_BATCH_SIZE` messages have accumulated, OR
      * `_INBOX_IDLE_FLUSH_SECONDS` elapses since its last new DM.

    Round-robin fairness: when multiple chats are ready to flush at once,
    drain in least-recently-flushed order so a chatty sender can't starve
    quieter ones. A chat with >10 backlog flushes 10 now and waits its turn
    again — the leftover doesn't jump the queue.

    Why this isn't a hard `is_important` gate: the heuristic in telegram_service
    (sender-in-allowlist OR keyword-match) is coarse — it misses "your sister
    just messaged" if she isn't in the allowlist and over-fires on the
    literal word 'urgent'. The operator has memory at hand and decides in
    context. The heuristic verdict is threaded into the note as a hint.

    Silence contract: the operator prompt instructs it to emit empty text
    when no DM in the batch is worth interrupting the user. The bot's
    chat.reply subscriber drops empty text, so nothing reaches Telegram in
    that case. The DMs still live in messenger_inbox for later inspection."""
    # Per-chat state. dict ordering is irrelevant — we always sort by
    # last_flush_at when picking who flushes next.
    pending: dict[str, list[dict[str, Any]]] = {}
    last_msg_at: dict[str, float] = {}
    last_flush_at: dict[str, float] = {}
    sub_iter = events.subscribe_global(types={"messenger.received"}).__aiter__()
    loop = asyncio.get_event_loop()

    # Recovery: pick up unread DMs that arrived while the daemon was down or
    # that were sitting in pending when the previous process exited. The
    # subscribe_global() iterator only yields FUTURE events, so without this
    # rows already in messenger_inbox would never auto-triage. We exclude
    # rows the drain has already shown to the operator (silent or replied)
    # via `messenger_inbox_triaged` so a restart can't re-burn LLM calls
    # on previously-decided rows.
    try:
        unread = await db.list_inbox(
            unread_only=True, exclude_triaged=True, limit=200,
        )
    except Exception:
        log.exception("inbox-drain: recovery query failed; starting empty")
        unread = []
    if unread:
        now_t = loop.time()
        # list_inbox returns DESC by received_at — reverse so the per-chat
        # queue is oldest-first.
        for row in reversed(unread):
            chat_id = str(row.get("chat_id") or "")
            pending.setdefault(chat_id, []).append(row)
            last_msg_at[chat_id] = now_t
        log.info(
            "inbox-drain: recovered %d unread DM(s) across %d chat(s)",
            len(unread), len(pending),
        )

    def _next_idle_deadline() -> float | None:
        """Earliest time at which some non-empty chat hits the idle limit.
        None means no chat is waiting — block on the next event indefinitely."""
        deadlines = [
            last_msg_at[c] + _INBOX_IDLE_FLUSH_SECONDS
            for c, msgs in pending.items() if msgs
        ]
        return min(deadlines) if deadlines else None

    while True:
        # Flush any chats already ready (at capacity or idle past deadline)
        # BEFORE blocking on the next event. This handles recovery (29
        # DMs preloaded on boot need to flush without waiting for a 30th
        # message) and the general "11 DMs in a chat → flush 10 now, the
        # 11th waits in the next iteration" case.
        now = loop.time()
        ready = [
            c for c, msgs in pending.items()
            if msgs and (
                len(msgs) >= _INBOX_BATCH_SIZE
                or now - last_msg_at[c] >= _INBOX_IDLE_FLUSH_SECONDS
            )
        ]
        ready.sort(key=lambda c: last_flush_at.get(c, 0.0))
        for chat_id in ready:
            batch = pending[chat_id][:_INBOX_BATCH_SIZE]
            remainder = pending[chat_id][_INBOX_BATCH_SIZE:]
            if remainder:
                pending[chat_id] = remainder
            else:
                pending.pop(chat_id, None)
                last_msg_at.pop(chat_id, None)
            await _flush_inbox_batch(events, operator, target_session_id, batch)
            last_flush_at[chat_id] = loop.time()
            # Mark every row in this batch as triaged so a restart's recovery
            # doesn't re-queue them. Silent outcomes don't set read_at — only
            # this mark distinguishes "operator has seen it" from "user has
            # read it themselves".
            try:
                await db.mark_inbox_triaged(
                    [str(r["id"]) for r in batch if r.get("id")]
                )
            except Exception:
                log.exception(
                    "inbox-drain: mark_inbox_triaged failed for batch %s",
                    [r.get("id") for r in batch],
                )

        # Now wait for the next event, or for the earliest idle deadline.
        deadline = _next_idle_deadline()
        timeout = max(0.0, deadline - loop.time()) if deadline is not None else None
        try:
            env = await asyncio.wait_for(sub_iter.__anext__(), timeout=timeout)
            payload = env.get("payload") or {}
            chat_id = str(payload.get("chat_id") or "")
            pending.setdefault(chat_id, []).append(payload)
            last_msg_at[chat_id] = loop.time()
        except asyncio.TimeoutError:
            pass  # next loop iteration will flush idle chats
        except StopAsyncIteration:
            break


async def _flush_inbox_batch(
    events: EventBus, operator: Operator,
    target_session_id: str, batch: list[dict[str, Any]],
) -> None:
    """Format a single chat's batch into one auto-ping note and hand it to
    the operator. All messages share `chat_id` (the loop guarantees no
    cross-chat mixing). No-op on empty input."""
    if not batch:
        return
    head = batch[0]
    chat_id = head.get("chat_id") or ""
    sender = (
        head.get("sender_username")
        or head.get("sender_display_name")
        or "unknown"
    )
    lines: list[str] = []
    for i, row in enumerate(batch, start=1):
        inbox_id = row.get("id") or ""
        body = (row.get("body") or "").replace("\n", " ").strip()
        body_preview = body[:200] + ("…" if len(body) > 200 else "")
        heuristic = "yes" if row.get("is_important") else "no"
        lines.append(
            f"  {i}. inbox_id={inbox_id} "
            f"heuristic_important={heuristic} body={body_preview!r}"
        )
    note = (
        f"{len(batch)} inbound DM(s) from @{sender} (chat_id={chat_id}) "
        f"since the last triage:\n"
        + "\n".join(lines)
        + "\n\nYou have exactly TWO options for this batch: AUTO-REPLY or "
          "STAY SILENT. No heads-up to the user — the user reads their own "
          "Telegram.\n"
          "AUTO-REPLY: if the memory entries loaded into your system prompt "
          "(possibly via JOINT inference across multiple entries) authorize "
          "you to reply on the user's behalf for THIS sender on THIS topic, "
          "execute the instruction (dispatch tasks if needed for gathering) "
          "then call `reply_to_dm` with the controlling memory's id. One "
          "reply may address the whole batch; you do not need to reply per DM.\n"
          "STAY SILENT: otherwise. Emit ZERO assistant content. Do not "
          "narrate non-events; the user already sees their inbox.\n"
          "heuristic_important is a hint, not a gate."
    )
    # Retrieval key: sender name + joined bodies (capped) so memory hits cover
    # any topic in the batch and may key off the contact's name.
    bodies = " ".join((r.get("body") or "") for r in batch)
    retrieval_query = (sender + " " + bodies).strip()[:1200] or None
    inbox_ids = [r.get("id") for r in batch]
    try:
        result = await operator.auto_ping(
            session_id=target_session_id,
            note=note,
            retrieval_query=retrieval_query,
        )
    except Exception:
        log.exception(
            "inbox-drain: auto_ping failed for batch %s", inbox_ids,
        )
        return
    # Empty text == operator triaged "nothing important here". Skip the
    # chat.reply publish so the bot doesn't relay anything; the DMs stay
    # in messenger_inbox for later inspection.
    if not result.text:
        return
    await events.publish_global("chat.reply", {
        "session_id": target_session_id,
        "text": result.text,
        "voice_text": to_voice_text(result.text),
        "trigger": "inbox.batch",
        "task_id": None,
    })


_MEMORY_DEDUP_INTERVAL_SECONDS = 300.0


async def _memory_dedup_loop(
    *, memory, llm, model: str,
    interval_seconds: float = _MEMORY_DEDUP_INTERVAL_SECONDS,
) -> None:
    """Periodic intelligent dedup of stored memories. Sleeps first so a
    fresh-booted daemon doesn't fire a pass before any memory has accrued.
    Failures only log — the next tick retries."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await memory.dedup_pass(
                llm, model=model, reasoning_effort="medium",
            )
        except Exception:
            log.exception("memory-dedup: pass crashed")


async def _probe_ollama(host: str = "http://localhost:11434") -> str | None:
    """Return the Ollama version string if reachable, None otherwise. Best-
    effort, 1s timeout — we use this in the startup notification only."""
    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            r = await c.get(f"{host}/api/version")
            r.raise_for_status()
            return r.json().get("version", "?")
    except Exception:
        return None


async def _build_startup_status(
    *, settings, operator: Operator | None,
    telegram_userbot: bool, lifecycle: Lifecycle,
    stale_memories: int = 0,
) -> str:
    """Compose the startup ping. On a fully clean boot it's just one line —
    `✅ oncall up`. Anything degraded gets its own ⚠️ line; informational
    follow-ups (recovered tasks, pending rebuild) get a ↻ line. The signal
    a user scans for is *whether there's anything below the headline*."""
    lines: list[str] = ["✅ oncall up"]
    if operator is None:
        lines.append("⚠️ operator: NOT configured (no LLM key)")
    if is_ollama_model(settings.oncall_memory_embed_model):
        if await _probe_ollama(settings.oncall_ollama_host) is None:
            lines.append(
                f"⚠️ ollama: unreachable at {settings.oncall_ollama_host} "
                f"(memory embedder won't work)"
            )
    if not telegram_userbot:
        lines.append("⚠️ telegram userbot: disabled (DM triage unavailable)")
    recovered = len(lifecycle.running)
    if recovered:
        lines.append(f"↻ recovered {recovered} in-flight task(s)")
    if stale_memories:
        lines.append(
            f"↻ re-embedding {stale_memories} memory rows in the background"
        )
    return "\n".join(lines)


async def _rebuild_memory_then_notify(
    memory, telegram_bot, *, stale: int, model: str,
) -> None:
    """Background task: re-embed all rows whose stored model differs from
    `model`, then ping the owner with the result. Errors are surfaced as a
    notification so the user isn't left wondering why memory looks empty."""
    try:
        result = await memory.rebuild_stale_embeddings()
    except Exception as e:
        log.exception("memory rebuild crashed")
        if telegram_bot is not None:
            await telegram_bot.notify_owner(
                f"⚠️ memory rebuild crashed: {type(e).__name__}: {e}"
            )
        return
    if telegram_bot is None:
        return
    rebuilt = result.get("rebuilt", 0)
    failed = result.get("failed", 0)
    if failed:
        await telegram_bot.notify_owner(
            f"⚠️ memory rebuild partial: {rebuilt}/{stale} rebuilt, "
            f"{failed} failed (model={model}). Will retry next boot."
        )
    else:
        await telegram_bot.notify_owner(
            f"✅ memory rebuilt: {rebuilt} row(s) re-embedded with {model}"
        )


async def _maybe_start_telegram_bot(
    settings, operator: Operator, events: EventBus,
    *, broker, db: Database,
) -> TelegramBotService | None:
    """Boot the Telegram bot front-end if a token + owner_id are set. Uses
    the HTTP Bot API, so api_id/api_hash are NOT required. Logs and returns
    None on misconfiguration / start failure — the rest of the API stays up."""
    if not settings.telegram_bot_token:
        log.info("telegram bot disabled: TELEGRAM_BOT_TOKEN not set")
        return None
    if not settings.telegram_bot_owner_id:
        log.warning(
            "telegram bot disabled: TELEGRAM_BOT_OWNER_ID not set. "
            "Get your numeric user id from @userinfobot."
        )
        return None
    try:
        owner_id = int(settings.telegram_bot_owner_id)
    except (TypeError, ValueError):
        log.warning("telegram bot disabled: TELEGRAM_BOT_OWNER_ID must be an integer")
        return None
    try:
        api = HttpxBotApi(settings.telegram_bot_token)
        service = TelegramBotService(
            api=api, operator=operator, events=events, owner_user_id=owner_id,
            broker=broker, db=db,
        )
        await service.start()
        return service
    except Exception:
        log.exception("telegram bot failed to start; continuing without it")
        return None


async def _maybe_start_telegram(
    settings, db: Database, events: EventBus,
    *, ignore_usernames: set[str] | None = None,
) -> TelegramService | None:
    """Boot the telethon listener if credentials and a session file are present.
    Failures are logged but never crash the API — `/chat` and tasks still work
    without messenger integration."""
    if not (settings.telegram_api_id and settings.telegram_api_hash):
        log.info("telegram disabled: TELEGRAM_API_ID/HASH not set")
        return None
    session_path = settings.telegram_session_path
    if not session_path.exists():
        log.warning("telegram disabled: session not found at %s; run `oncall telegram-login`", session_path)
        return None
    try:
        client = make_telethon_client(
            api_id=int(settings.telegram_api_id),
            api_hash=settings.telegram_api_hash,
            session_path=session_path,
        )

        async def _emit_received(row: dict[str, Any]) -> None:
            await events.publish_global("messenger.received", row)

        service = TelegramService(
            db=db,
            client=client,
            important_senders=settings.important_senders,
            important_keywords=settings.important_keywords,
            on_new_message=_emit_received,
            ignore_usernames=ignore_usernames or set(),
        )
        await service.start()
        return service
    except Exception:
        log.exception("telegram failed to start; continuing without it")
        return None


def _register_routes(app: FastAPI) -> None:

    def _lc(request: Request) -> Lifecycle:
        return request.app.state.lifecycle

    def _db(request: Request) -> Database:
        return request.app.state.db

    def _broker(request: Request) -> Broker:
        return request.app.state.broker

    def _events(request: Request) -> EventBus:
        return request.app.state.events

    # ---- Tasks ----

    @app.post("/tasks", dependencies=[Depends(verify_token)])
    async def submit_task(body: SubmitTaskBody, request: Request) -> dict[str, str]:
        task = await _lc(request).submit_task(
            prompt=body.prompt,
            model=body.model,
            max_turns=body.max_turns,
            chat_session_id=body.chat_session_id,
        )
        return {"task_id": str(task.id), "session_id": task.session_id}

    @app.get("/tasks", dependencies=[Depends(verify_token)])
    async def list_tasks(request: Request) -> list[dict[str, Any]]:
        tasks = await _db(request).list_tasks(limit=100)
        return [_task_out(t) for t in tasks]

    @app.get("/tasks/{task_id}", dependencies=[Depends(verify_token)])
    async def get_task(task_id: UUID, request: Request) -> dict[str, Any]:
        t = await _db(request).get_task(task_id)
        if t is None:
            raise HTTPException(404, "no such task")
        events = await _db(request).list_events(task_id)
        return {"task": _task_out(t), "events": events}

    @app.get("/tasks/{task_id}/events", dependencies=[Depends(verify_token)])
    async def stream_events(task_id: UUID, request: Request, since: int = 0):
        ev = _events(request)

        async def gen():
            async for evt in ev.subscribe(task_id, since_seq=since):
                yield {"data": json.dumps(evt)}

        return EventSourceResponse(gen())

    @app.get("/events", dependencies=[Depends(verify_token)])
    async def stream_global_events(
        request: Request,
        types: str = "approval.requested,approval.resolved,result.final,messenger.received,state.changed",
    ):
        """Live global SSE feed (public API surface — third-party clients,
        future voice gateway). `types` is a comma-separated filter; pass
        empty to receive everything."""
        ev = _events(request)
        wanted = {t.strip() for t in types.split(",") if t.strip()} or None

        async def gen():
            agen = ev.subscribe_global(types=wanted)
            next_event: asyncio.Task | None = None
            try:
                next_event = asyncio.ensure_future(agen.__anext__())
                while True:
                    if await request.is_disconnected():
                        return
                    done, _ = await asyncio.wait(
                        {next_event}, timeout=15.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        # Idle keepalive — SSE comments are ignored by clients.
                        yield {"comment": "ping"}
                        continue
                    try:
                        evt = next_event.result()
                    except StopAsyncIteration:
                        return
                    yield {"data": json.dumps(evt)}
                    next_event = asyncio.ensure_future(agen.__anext__())
            finally:
                # CRITICAL: cancel AND await the pre-fetched task before
                # calling aclose(). aclose() raises "generator already running"
                # if its underlying __anext__ coroutine hasn't finished
                # settling the cancellation yet.
                if next_event is not None and not next_event.done():
                    next_event.cancel()
                    try:
                        await next_event
                    except BaseException:
                        pass
                try:
                    await agen.aclose()
                except Exception:
                    log.debug("global events aclose raised", exc_info=True)

        return EventSourceResponse(gen())

    @app.post("/tasks/{task_id}/kill", dependencies=[Depends(verify_token)])
    async def kill_task(task_id: UUID, body: KillBody, request: Request) -> dict[str, Any]:
        if not is_kill_phrase(body.phrase):
            raise HTTPException(400, "kill phrase did not match 'stop everything'")
        ok = await _lc(request).kill(task_id, reason="kill_phrase")
        if not ok:
            raise HTTPException(404, "task not running")
        return {"killed": True}

    # ---- Approvals ----

    @app.get("/approvals/pending", dependencies=[Depends(verify_token)])
    async def list_pending(request: Request) -> list[dict[str, Any]]:
        rows = await _db(request).list_pending_approvals()
        return [_approval_summary(r) for r in rows]

    @app.get("/approvals/{approval_id}", dependencies=[Depends(verify_token)])
    async def get_approval(approval_id: UUID, request: Request) -> dict[str, Any]:
        row = await _db(request).get_approval(approval_id)
        if row is None:
            raise HTTPException(404, "no such approval")
        # Surface tool input + canonical + blast_radius + challenge.
        return {
            "id": row["id"],
            "task_id": row["task_id"],
            "tool_name": row["tool_name"],
            "tool_input": json.loads(row["tool_input_json"]),
            "classifier_verdict": row["classifier_verdict"],
            "canonical_command": row["canonical_command"],
            "blast_radius": row["blast_radius"],
            "challenge_phrase": row["challenge_phrase"],
            "state": row["state"],
            "decision": row["decision"],
            "challenge_supplied": row["challenge_supplied"],
            "challenge_matched": bool(row["challenge_matched"]) if row["challenge_matched"] is not None else None,
            "requested_at": row["requested_at"],
            "responded_at": row["responded_at"],
            "auto": bool(row["auto"]),
        }

    @app.post("/approvals/{approval_id}/respond", dependencies=[Depends(verify_token)])
    async def respond_approval(
        approval_id: UUID,
        body: ApprovalRespondBody,
        request: Request,
    ) -> dict[str, Any]:
        approved, matched = await _broker(request).submit_response(
            approval_id=approval_id,
            decision=body.decision,
            challenge_phrase_supplied=body.challenge_phrase_supplied,
            message=body.message,
        )
        # If submit_response found no pending approval, both flags are False.
        # Distinguish "no such pending" from "phrase mismatch" via DB lookup.
        row = await _db(request).get_approval(approval_id)
        if row is None:
            raise HTTPException(404, "no such approval")
        return {"approved": approved, "matched": matched}

    # ---- Internal (loopback only) ----

    @app.post("/internal/broker/decide", dependencies=[Depends(verify_loopback)])
    async def broker_decide(body: BrokerDecideBody, request: Request) -> dict[str, Any]:
        result = await _broker(request).decide(
            session_id=body.session_id,
            tool_use_id=body.tool_use_id,
            tool_name=body.tool_name,
            tool_input=body.tool_input,
        )
        return result.to_cli_payload()

    @app.post("/internal/messenger", dependencies=[Depends(verify_loopback)])
    async def messenger_op(body: MessengerOpBody, request: Request) -> dict[str, Any]:
        tg: TelegramService | None = request.app.state.telegram
        if tg is None:
            raise HTTPException(503, "telegram service not configured")
        if body.op == "list":
            unread_only = True if body.unread_only is None else body.unread_only
            return {"messages": await tg.list_inbox(unread_only=unread_only, limit=body.limit)}
        if body.op == "read":
            if not body.inbox_id:
                raise HTTPException(400, "inbox_id required")
            msg = await tg.get_message(body.inbox_id)
            if msg is None:
                raise HTTPException(404, "no such inbox message")
            return msg
        if body.op == "mark_read":
            if not body.inbox_id:
                raise HTTPException(400, "inbox_id required")
            ok = await tg.mark_read(body.inbox_id)
            return {"marked_read": ok}
        if body.op == "style":
            if not body.chat_id:
                raise HTTPException(400, "chat_id required")
            return {"samples": await tg.get_chat_style(body.chat_id, limit=body.limit)}
        if body.op == "history":
            if not body.chat_id:
                raise HTTPException(400, "chat_id required")
            return {"messages": await tg.get_chat_history(body.chat_id, limit=body.limit)}
        if body.op == "search":
            if not body.query:
                raise HTTPException(400, "query required")
            return {"chats": await tg.search_chats(body.query, limit=body.limit)}
        if body.op == "search_messages":
            if not body.chat_id or not body.query:
                raise HTTPException(400, "chat_id and query required")
            return {"messages": await tg.search_messages(
                body.chat_id, body.query, limit=body.limit,
            )}
        if body.op == "list_chats":
            unread_only = False if body.unread_only is None else body.unread_only
            return {"chats": await tg.list_chats(
                unread_only=unread_only,
                dms_only=body.dms_only,
                limit=body.limit,
            )}
        if body.op == "send":
            if not body.chat_id or not body.text:
                raise HTTPException(400, "chat_id and text required")
            return await tg.send(body.chat_id, body.text)
        raise HTTPException(400, f"unknown op {body.op!r}")

    # ---- Operator / chat ----

    @app.post("/chat", dependencies=[Depends(verify_token)])
    async def chat(body: ChatBody, request: Request) -> dict[str, Any]:
        operator: Operator | None = request.app.state.operator
        if operator is None:
            raise HTTPException(503, "operator not configured: set AI_GATEWAY_API_KEY")
        session_id = body.session_id or str(__import__("uuid").uuid4())
        result = await operator.chat_turn(
            session_id=session_id, user_text=body.text, language=body.language,
        )
        return {
            "session_id": session_id,
            "text": result.text,
            "voice_text": to_voice_text(result.text, language=body.language),
            "language": body.language,
            "tool_calls": result.tool_calls_made,
        }

    @app.get("/chat/{session_id}", dependencies=[Depends(verify_token)])
    async def get_chat(session_id: str, request: Request) -> dict[str, Any]:
        history = await _db(request).load_chat_history(session_id, limit=200)
        return {"session_id": session_id, "messages": history}

    # ---- Misc ----

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}


def _task_out(t) -> dict[str, Any]:
    return {
        "id": str(t.id),
        "session_id": t.session_id,
        "state": t.state.value,
        "prompt": t.prompt,
        "model": t.model,
        "created_at": t.created_at.isoformat(),
        "updated_at": t.updated_at.isoformat(),
        "terminal_reason": t.terminal_reason.value if t.terminal_reason else None,
    }


def _approval_summary(r) -> dict[str, Any]:
    return {
        "id": str(r.id),
        "task_id": str(r.task_id),
        "tool_name": r.tool_name,
        "canonical_command": r.canonical_command,
        "blast_radius": r.blast_radius,
        "challenge_phrase": r.challenge_phrase,
        "classifier_verdict": r.classifier_verdict.value,
        "requested_at": r.requested_at.isoformat(),
    }
