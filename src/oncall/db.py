from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite

from .models import (
    ApprovalRequest,
    ApprovalResult,
    ClassifierVerdict,
    Task,
    TaskState,
    TerminalReason,
    utcnow,
)


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    state TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model TEXT,
    max_turns INTEGER,
    consecutive_denials INTEGER NOT NULL DEFAULT 0,
    dispatched_by_chat_session TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terminal_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);

CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_task_seq ON task_events(task_id, seq);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    session_id TEXT NOT NULL,
    tool_use_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_input_json TEXT NOT NULL,
    classifier_verdict TEXT NOT NULL,
    canonical_command TEXT NOT NULL,
    blast_radius TEXT NOT NULL,
    challenge_phrase TEXT,
    state TEXT NOT NULL,
    decision TEXT,
    challenge_supplied TEXT,
    challenge_matched INTEGER,
    response_message TEXT,
    requested_at TEXT NOT NULL,
    responded_at TEXT,
    auto INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_approvals_dedup ON approvals(session_id, tool_use_id);
CREATE INDEX IF NOT EXISTS idx_approvals_state ON approvals(state);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS credentials_issued (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    host TEXT NOT NULL,
    scope TEXT NOT NULL,
    ttl_s INTEGER NOT NULL,
    issued_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS messenger_inbox (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL DEFAULT 'telegram',
    chat_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    sender_username TEXT,
    sender_display_name TEXT,
    body TEXT NOT NULL,
    is_important INTEGER NOT NULL DEFAULT 0,
    received_at TEXT NOT NULL,
    read_at TEXT,
    replied_message_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_messenger_unread ON messenger_inbox(read_at) WHERE read_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_messenger_dedup ON messenger_inbox(platform, chat_id, message_id);
"""


def iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    # ---- tasks ----

    async def insert_task(self, task: Task) -> None:
        await self.conn.execute(
            """
            INSERT INTO tasks (id, session_id, state, prompt, model, max_turns,
                               consecutive_denials, dispatched_by_chat_session,
                               created_at, updated_at, terminal_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(task.id),
                task.session_id,
                task.state.value,
                task.prompt,
                task.model,
                task.max_turns,
                task.consecutive_denials,
                task.dispatched_by_chat_session,
                iso(task.created_at),
                iso(task.updated_at),
                task.terminal_reason.value if task.terminal_reason else None,
            ),
        )
        await self.conn.commit()

    async def update_task_state(
        self,
        task_id: UUID,
        state: TaskState,
        terminal_reason: TerminalReason | None = None,
    ) -> None:
        await self.conn.execute(
            "UPDATE tasks SET state = ?, terminal_reason = ?, updated_at = ? WHERE id = ?",
            (
                state.value,
                terminal_reason.value if terminal_reason else None,
                iso(utcnow()),
                str(task_id),
            ),
        )
        await self.conn.commit()

    async def increment_consecutive_denials(self, task_id: UUID) -> int:
        await self.conn.execute(
            "UPDATE tasks SET consecutive_denials = consecutive_denials + 1, updated_at = ? WHERE id = ?",
            (iso(utcnow()), str(task_id)),
        )
        row = await (await self.conn.execute(
            "SELECT consecutive_denials FROM tasks WHERE id = ?", (str(task_id),)
        )).fetchone()
        await self.conn.commit()
        return int(row["consecutive_denials"]) if row else 0

    async def reset_consecutive_denials(self, task_id: UUID) -> None:
        await self.conn.execute(
            "UPDATE tasks SET consecutive_denials = 0, updated_at = ? WHERE id = ?",
            (iso(utcnow()), str(task_id)),
        )
        await self.conn.commit()

    async def get_task(self, task_id: UUID) -> Task | None:
        row = await (await self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (str(task_id),)
        )).fetchone()
        return _row_to_task(row) if row else None

    async def get_task_by_session(self, session_id: str) -> Task | None:
        row = await (await self.conn.execute(
            "SELECT * FROM tasks WHERE session_id = ?", (session_id,)
        )).fetchone()
        return _row_to_task(row) if row else None

    async def list_tasks_in_states(self, *states: TaskState) -> list[Task]:
        if not states:
            return []
        placeholders = ",".join("?" * len(states))
        rows = await (await self.conn.execute(
            f"SELECT * FROM tasks WHERE state IN ({placeholders}) ORDER BY created_at",
            tuple(s.value for s in states),
        )).fetchall()
        return [_row_to_task(r) for r in rows]

    async def list_tasks(self, *, limit: int = 50) -> list[Task]:
        rows = await (await self.conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        )).fetchall()
        return [_row_to_task(r) for r in rows]

    # ---- task events ----

    async def append_event(self, task_id: UUID, type_: str, payload: dict[str, Any]) -> int:
        row = await (await self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM task_events WHERE task_id = ?",
            (str(task_id),),
        )).fetchone()
        seq = int(row["next_seq"])
        await self.conn.execute(
            "INSERT INTO task_events (task_id, seq, type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(task_id), seq, type_, json.dumps(payload), iso(utcnow())),
        )
        await self.conn.commit()
        return seq

    async def list_events(self, task_id: UUID, *, since_seq: int = 0) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT seq, type, payload, created_at FROM task_events "
            "WHERE task_id = ? AND seq > ? ORDER BY seq",
            (str(task_id), since_seq),
        )).fetchall()
        return [
            {
                "seq": int(r["seq"]),
                "type": r["type"],
                "payload": json.loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ---- approvals ----

    async def get_resolved_approval(
        self, session_id: str, tool_use_id: str
    ) -> tuple[ApprovalRequest, ApprovalResult] | None:
        row = await (await self.conn.execute(
            "SELECT * FROM approvals WHERE session_id = ? AND tool_use_id = ?",
            (session_id, tool_use_id),
        )).fetchone()
        if row is None or row["state"] != "resolved":
            return None
        return _row_to_approval_pair(row)

    async def get_pending_approval(self, approval_id: UUID) -> ApprovalRequest | None:
        row = await (await self.conn.execute(
            "SELECT * FROM approvals WHERE id = ? AND state = 'pending'",
            (str(approval_id),),
        )).fetchone()
        return _row_to_approval_request(row) if row else None

    async def get_approval(self, approval_id: UUID) -> dict[str, Any] | None:
        row = await (await self.conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (str(approval_id),)
        )).fetchone()
        return dict(row) if row else None

    async def list_pending_approvals(self) -> list[ApprovalRequest]:
        rows = await (await self.conn.execute(
            "SELECT * FROM approvals WHERE state = 'pending' ORDER BY requested_at"
        )).fetchall()
        return [_row_to_approval_request(r) for r in rows]

    async def create_pending_approval(self, req: ApprovalRequest) -> None:
        await self.conn.execute(
            """
            INSERT INTO approvals
              (id, task_id, session_id, tool_use_id, tool_name, tool_input_json,
               classifier_verdict, canonical_command, blast_radius, challenge_phrase,
               state, requested_at, auto)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, 0)
            """,
            (
                str(req.id),
                str(req.task_id),
                req.session_id,
                req.tool_use_id,
                req.tool_name,
                json.dumps(req.tool_input),
                req.classifier_verdict.value,
                req.canonical_command,
                req.blast_radius,
                req.challenge_phrase,
                iso(req.requested_at),
            ),
        )
        await self.conn.commit()

    async def record_auto_approval(
        self,
        req: ApprovalRequest,
        decision: str,
        response_message: str | None = None,
    ) -> None:
        now = utcnow()
        await self.conn.execute(
            """
            INSERT INTO approvals
              (id, task_id, session_id, tool_use_id, tool_name, tool_input_json,
               classifier_verdict, canonical_command, blast_radius, challenge_phrase,
               state, decision, response_message, requested_at, responded_at, auto)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL,
                    'resolved', ?, ?, ?, ?, 1)
            ON CONFLICT(session_id, tool_use_id) DO NOTHING
            """,
            (
                str(req.id),
                str(req.task_id),
                req.session_id,
                req.tool_use_id,
                req.tool_name,
                json.dumps(req.tool_input),
                req.classifier_verdict.value,
                req.canonical_command,
                req.blast_radius,
                decision,
                response_message,
                iso(req.requested_at),
                iso(now),
            ),
        )
        await self.conn.commit()

    # ---- chat sessions / messages ----

    async def ensure_chat_session(self, session_id: str) -> None:
        now = iso(utcnow())
        await self.conn.execute(
            """INSERT INTO chat_sessions (id, created_at, last_seen_at) VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET last_seen_at = excluded.last_seen_at""",
            (session_id, now, now),
        )
        await self.conn.commit()

    async def append_chat_message(self, session_id: str, role: str, content: str) -> None:
        await self.conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, iso(utcnow())),
        )
        await self.conn.commit()

    async def load_chat_history(
        self, session_id: str, *, limit: int = 60
    ) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT id, role, content, created_at FROM chat_messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )).fetchall()
        return [
            {"id": int(r["id"]), "role": r["role"], "content": r["content"],
             "created_at": r["created_at"]}
            for r in reversed(rows)
        ]

    # ---- approvals (continued) ----

    # ---- messenger inbox ----

    async def record_inbox(
        self,
        *,
        inbox_id: str,
        platform: str,
        chat_id: str,
        message_id: str,
        sender_username: str | None,
        sender_display_name: str | None,
        body: str,
        is_important: bool,
        received_at: datetime,
    ) -> bool:
        """Insert one inbound message. Returns True if a new row was inserted
        (False on duplicate (platform, chat_id, message_id) — telethon can fire
        twice in pathological cases, e.g. resubscribe after reconnect)."""
        try:
            await self.conn.execute(
                """
                INSERT INTO messenger_inbox
                  (id, platform, chat_id, message_id, sender_username,
                   sender_display_name, body, is_important, received_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inbox_id, platform, chat_id, message_id, sender_username,
                    sender_display_name, body, 1 if is_important else 0,
                    iso(received_at),
                ),
            )
            await self.conn.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def list_inbox(
        self, *, unread_only: bool = True, limit: int = 20
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM messenger_inbox "
            + ("WHERE read_at IS NULL " if unread_only else "")
            + "ORDER BY received_at DESC LIMIT ?"
        )
        rows = await (await self.conn.execute(sql, (limit,))).fetchall()
        return [_row_to_inbox(r) for r in rows]

    async def get_inbox_message(self, inbox_id: str) -> dict[str, Any] | None:
        row = await (await self.conn.execute(
            "SELECT * FROM messenger_inbox WHERE id = ?", (inbox_id,),
        )).fetchone()
        return _row_to_inbox(row) if row else None

    async def mark_inbox_read(self, inbox_id: str) -> bool:
        cur = await self.conn.execute(
            "UPDATE messenger_inbox SET read_at = ? WHERE id = ? AND read_at IS NULL",
            (iso(utcnow()), inbox_id),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def record_inbox_reply(self, inbox_id: str, reply_message_id: str) -> None:
        await self.conn.execute(
            "UPDATE messenger_inbox SET replied_message_id = ?, read_at = COALESCE(read_at, ?) WHERE id = ?",
            (reply_message_id, iso(utcnow()), inbox_id),
        )
        await self.conn.commit()

    # ---- approvals (continued) ----

    async def append_approval_response(
        self, approval_id: UUID, result: ApprovalResult
    ) -> None:
        await self.conn.execute(
            """
            UPDATE approvals
            SET state = 'resolved',
                decision = ?,
                challenge_supplied = ?,
                challenge_matched = ?,
                response_message = ?,
                responded_at = ?
            WHERE id = ? AND state = 'pending'
            """,
            (
                result.behavior,
                result.challenge_phrase_supplied,
                1 if result.challenge_matched else 0,
                result.message,
                iso(result.responded_at),
                str(approval_id),
            ),
        )
        await self.conn.commit()


# ---- row converters ----


def _row_to_task(row: aiosqlite.Row) -> Task:
    return Task(
        id=UUID(row["id"]),
        session_id=row["session_id"],
        state=TaskState(row["state"]),
        prompt=row["prompt"],
        model=row["model"],
        max_turns=row["max_turns"],
        consecutive_denials=row["consecutive_denials"],
        dispatched_by_chat_session=row["dispatched_by_chat_session"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        terminal_reason=TerminalReason(row["terminal_reason"]) if row["terminal_reason"] else None,
    )


def _row_to_approval_request(row: aiosqlite.Row) -> ApprovalRequest:
    return ApprovalRequest(
        id=UUID(row["id"]),
        task_id=UUID(row["task_id"]),
        session_id=row["session_id"],
        tool_use_id=row["tool_use_id"],
        tool_name=row["tool_name"],
        tool_input=json.loads(row["tool_input_json"]),
        classifier_verdict=ClassifierVerdict(row["classifier_verdict"]),
        canonical_command=row["canonical_command"],
        blast_radius=row["blast_radius"],
        challenge_phrase=row["challenge_phrase"],
        requested_at=datetime.fromisoformat(row["requested_at"]),
    )


def _row_to_inbox(row: aiosqlite.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "platform": row["platform"],
        "chat_id": row["chat_id"],
        "message_id": row["message_id"],
        "sender_username": row["sender_username"],
        "sender_display_name": row["sender_display_name"],
        "body": row["body"],
        "is_important": bool(row["is_important"]),
        "received_at": row["received_at"],
        "read_at": row["read_at"],
        "replied_message_id": row["replied_message_id"],
    }


def _row_to_approval_pair(row: aiosqlite.Row) -> tuple[ApprovalRequest, ApprovalResult]:
    req = _row_to_approval_request(row)
    result = ApprovalResult(
        request_id=req.id,
        behavior=row["decision"],  # type: ignore[arg-type]
        challenge_phrase_supplied=row["challenge_supplied"],
        challenge_matched=bool(row["challenge_matched"]),
        message=row["response_message"],
        responded_at=datetime.fromisoformat(row["responded_at"]) if row["responded_at"] else utcnow(),
    )
    return req, result
