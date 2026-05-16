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

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .approval_client import HttpLongPollApprovalClient, is_kill_phrase
from .broker import Broker
from .config import get_paths, get_settings
from .db import Database
from .events import EventBus
from .lifecycle import Lifecycle
from .operator import GatewayLLMClient, Operator
from .telegram_service import TelegramService, make_telethon_client


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


class BrokerDecideBody(BaseModel):
    session_id: str
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


class MessengerOpBody(BaseModel):
    op: Literal["list", "read", "mark_read", "style", "send"]
    chat_id: str | None = None
    message_id: str | None = None
    inbox_id: str | None = None
    text: str | None = None
    unread_only: bool = True
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
        broker = Broker(db, approval_client, events.publish)
        lifecycle = Lifecycle(
            db=db, broker=broker, approval_client=approval_client,
            events=events, settings=settings, paths=paths,
        )
        # Operator — only set up if a gateway key is configured. If not, /chat
        # returns a clear 503 explaining how to set AI_GATEWAY_API_KEY.
        operator: Operator | None = None
        if settings.gateway_key:
            llm = GatewayLLMClient(
                base_url=settings.ai_gateway_base_url,
                api_key=settings.gateway_key,
            )
        # Telegram — only set up if api_id/hash + session file are all present.
        telegram: TelegramService | None = await _maybe_start_telegram(settings, db)
        if settings.gateway_key:
            operator = Operator(
                db=db, lifecycle=lifecycle, broker=broker,
                settings=settings, paths=paths, llm=llm,
                telegram=telegram,
            )
        app.state.db = db
        app.state.events = events
        app.state.approval_client = approval_client
        app.state.broker = broker
        app.state.lifecycle = lifecycle
        app.state.operator = operator
        app.state.telegram = telegram
        # Recover any tasks left running / awaiting_approval from a prior
        # orchestrator process. The CLI's session JSONL is on disk; we
        # re-spawn with --resume and the broker's (session_id, tool_use_id)
        # dedup re-attaches to existing pending approval rows.
        await lifecycle.recover()
        try:
            yield
        finally:
            await lifecycle.shutdown()
            if telegram is not None:
                await telegram.stop()
            await db.close()

    app = FastAPI(title="oncall-agent", lifespan=lifespan)
    _register_routes(app)
    return app


async def _maybe_start_telegram(settings, db: Database) -> TelegramService | None:
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
        service = TelegramService(
            db=db,
            client=client,
            important_senders=settings.important_senders,
            important_keywords=settings.important_keywords,
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
            return {"messages": await tg.list_inbox(unread_only=body.unread_only, limit=body.limit)}
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
        result = await operator.chat_turn(session_id=session_id, user_text=body.text)
        return {
            "session_id": session_id,
            "text": result.text,
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
