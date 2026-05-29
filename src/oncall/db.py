from __future__ import annotations

import asyncio
import functools
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
    -- Telegram chat the task is locked to (autonomous-reply lockdown).
    -- Migration adds this column on existing installs.
    restricted_to_chat TEXT,
    -- Telegram chat the task is pre-authorized to op=send to. Broker
    -- auto-allows op=send when chat_id matches. NULL = no pre-approval.
    pre_approved_send_chat TEXT,
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
-- Compression checkpoints. When the live message tail grows past the model's
-- context budget, the operator summarizes everything up through one row id
-- and writes a single chat_summaries row. Future loads = (latest summary) +
-- (chat_messages with id > through_message_id).
CREATE TABLE IF NOT EXISTS chat_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id),
    summary TEXT NOT NULL,
    through_message_id INTEGER NOT NULL,
    estimated_token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_summaries_session
    ON chat_summaries(session_id, id DESC);

CREATE TABLE IF NOT EXISTS credentials_issued (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    host TEXT NOT NULL,
    scope TEXT NOT NULL,
    ttl_s INTEGER NOT NULL,
    issued_at TEXT NOT NULL,
    revoked_at TEXT
);

-- Operator memory. Each row is one short declarative fact extracted from a
-- user turn, with a packed-float32 embedding for semantic retrieval. LRU is
-- maintained via last_accessed_at (bumped both at retrieval and at near-
-- duplicate merge).
CREATE TABLE IF NOT EXISTS operator_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    -- Name of the embedding model that produced `embedding`. When the
    -- configured model changes, rows whose `model` doesn't match are
    -- invisible to retrieval until a background rebuild re-embeds them.
    -- See OperatorMemory.rebuild_stale_embeddings().
    model TEXT NOT NULL DEFAULT '',
    source_turn TEXT,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_operator_memories_lru
    ON operator_memories(last_accessed_at);

-- Pairs of memory ids the periodic dedup LLM looked at and decided NOT to
-- merge. The next dedup pass skips edges between recorded pairs so we don't
-- burn LLM calls re-asking the same question every 5 min. id_a < id_b.
CREATE TABLE IF NOT EXISTS memory_dedup_skip_pairs (
    id_a INTEGER NOT NULL,
    id_b INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (id_a, id_b)
);

-- Tracks which operator memories have already been surfaced as a
-- `[memory note: ...]` user-role message in a chat session. Used to
-- dedup memory injection across turns so the system prompt + chat
-- history prefix stays stable (KV cache hits) and the model doesn't
-- re-read the same memory text every turn. Reset on /clear; survives
-- /compress (compressed history may have lost the verbatim memory text,
-- but the model has been shown the fact at least once).
CREATE TABLE IF NOT EXISTS session_memory_shown (
    session_id TEXT NOT NULL,
    memory_id INTEGER NOT NULL,
    shown_at TEXT NOT NULL,
    PRIMARY KEY (session_id, memory_id)
);

-- Cursor into chat_messages.id, per operator chat session, marking the
-- highest message id whose content has already been forwarded to the
-- executor via hand_off. The next hand_off forwards only messages
-- newer than this cursor (capped at ~1024 chars of recent operator-
-- user dialogue), so the executor never re-sees the same chunk twice.
CREATE TABLE IF NOT EXISTS executor_handoff_cursor (
    chat_session_id TEXT PRIMARY KEY,
    last_forwarded_message_id INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

-- Inbox rows the auto-drain has shown to the operator. Decoupled from
-- `messenger_inbox.read_at` because a silent triage outcome (operator
-- chose STAY SILENT) does NOT mark the row read — the user still needs to
-- see those messages themselves. Without this table the next restart's
-- recovery would re-queue silently-triaged rows and re-burn the LLM.
CREATE TABLE IF NOT EXISTS messenger_inbox_triaged (
    inbox_id TEXT PRIMARY KEY,
    triaged_at TEXT NOT NULL
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

-- Operator-initiated `dispatch_task` calls made during an autonomous-reply
-- turn are not auto-spawned: they sit here until the user resolves via
-- `/yes <id>` or `/no <id>` in the agent chat. On allow, the task is
-- submitted with `restricted_to_chat` inherited (so the executor also runs
-- locked to that chat). On deny, nothing further happens. Empty by default.
-- See [operator.py] `_execute_tool` and [telegram_agent.py]
-- `_dispatch_approval_subscriber`.
CREATE TABLE IF NOT EXISTS pending_dispatches (
    id TEXT PRIMARY KEY,
    chat_session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model TEXT,
    restricted_to_chat TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution TEXT
);

-- Hard guardrail for `reply_to_dm`: a chat_id must be present here for the
-- operator's autonomous-reply tool to succeed. Empty by default — the user
-- has to explicitly `/allowdm <chat_id>` per chat. Even if the model is
-- prompt-injected into citing a real `authority_memory_id`, this table is
-- the final stop before bytes leave the box on the user's behalf.
CREATE TABLE IF NOT EXISTS dm_allowlist (
    chat_id TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);

-- Executor's `ask_user` tool: long-poll question from an executor task,
-- relayed by the operator to the human. The HTTP proxy parks on an
-- in-memory Future keyed by id; this table is the durable record (also
-- used to enforce per-chat queue ordering — only one "presented" ask
-- per chat at a time).
-- Per-task allowlist of directories the user pre-approved Write into.
-- Populated when the user taps "Yes (and folder)" on a Write-tool approval
-- card; subsequent Write calls to files under any allowed dir auto-allow.
-- Scope is per-task on purpose: when the task ends the trust ends.
CREATE TABLE IF NOT EXISTS task_write_dir_allowlist (
    task_id TEXT NOT NULL,
    dir TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (task_id, dir)
);

CREATE TABLE IF NOT EXISTS ask_requests (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    chat_session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    state TEXT NOT NULL,  -- pending | presented | answered | cancelled
    created_at TEXT NOT NULL,
    presented_at TEXT,
    answered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ask_requests_chat_state
  ON ask_requests(chat_session_id, state);
"""


def iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _locked(method):
    """Run a coroutine method while holding the instance's connection lock."""
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            return await method(self, *args, **kwargs)
    return wrapper


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        # Serializes ALL access to the single shared connection. aiosqlite
        # runs one worker thread and serializes *individual* ops, but NOT the
        # multi-step sequences our methods are (execute -> fetch -> commit).
        # Two tasks sharing the connection could interleave: e.g. a concurrent
        # commit() landing between append_event's `INSERT ... RETURNING` and
        # its fetchone() hit the still-active write statement and raised
        # `cannot commit transaction - SQL statements in progress`. Holding
        # this lock for each method's whole connection sequence (see
        # `_serialize_connection_access` below) makes every method atomic on
        # the connection. NB: never hold it across non-DB awaits (LLM/embed) —
        # methods here only touch sqlite, so it's held for microseconds.
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        # Idempotent migrations for columns added after the initial schema.
        # SQLite ALTER TABLE ADD COLUMN errors if the column already exists,
        # so we swallow that specific case.
        await self._migrate_add_column("tasks", "result_summary", "TEXT")
        await self._migrate_add_column("tasks", "restricted_to_chat", "TEXT")
        # Set when the operator dispatches a task that's pre-authorized to
        # send to a specific Telegram chat (memory-authorized auto-reply or
        # user-approved draft). Broker auto-allows op=send when the input
        # chat_id matches this column. NULL means no pre-approval.
        await self._migrate_add_column("tasks", "pre_approved_send_chat", "TEXT")
        await self._migrate_add_column(
            "operator_memories", "model", "TEXT NOT NULL DEFAULT ''",
        )
        # Rename existing `tg-bot-<owner>` chat sessions to `tg-agent-<owner>`.
        # The HTTP Bot API was retired in favor of a second telethon userbot;
        # the new session-id naming is `tg-agent-*`. This migration is
        # idempotent: ON CONFLICT clauses skip rows that collide with an
        # already-renamed session.
        await self._migrate_rename_bot_sessions_to_agent()
        await self._conn.commit()

    async def _migrate_rename_bot_sessions_to_agent(self) -> None:
        """Rename `tg-bot-*` session ids to `tg-agent-*` across the schema.
        Idempotent: a second run is a no-op because the old ids are gone.

        FK constraints on chat_messages.session_id / chat_summaries.session_id
        forbid an in-place rename of the parent row. We instead insert a new
        parent, redirect children, then drop the old parent."""
        try:
            cur = await self.conn.execute(
                "SELECT id, created_at, last_seen_at "
                "FROM chat_sessions WHERE id LIKE 'tg-bot-%'",
            )
            rows = await cur.fetchall()
            await cur.close()
            if not rows:
                return
            for row in rows:
                old = row["id"]
                new = "tg-agent-" + old[len("tg-bot-"):]
                # If a new-shape row already exists (partial-run recovery),
                # discard the stale bot rows and continue.
                cur = await self.conn.execute(
                    "SELECT 1 FROM chat_sessions WHERE id = ?", (new,),
                )
                exists = await cur.fetchone()
                await cur.close()
                if exists:
                    await self.conn.execute(
                        "DELETE FROM chat_messages WHERE session_id = ?", (old,),
                    )
                    await self.conn.execute(
                        "DELETE FROM chat_summaries WHERE session_id = ?", (old,),
                    )
                    await self.conn.execute(
                        "DELETE FROM chat_sessions WHERE id = ?", (old,),
                    )
                    continue
                # Insert new parent first so child redirects don't trip FK.
                await self.conn.execute(
                    "INSERT INTO chat_sessions(id, created_at, last_seen_at) "
                    "VALUES(?, ?, ?)",
                    (new, row["created_at"], row["last_seen_at"]),
                )
                await self.conn.execute(
                    "UPDATE chat_messages SET session_id = ? WHERE session_id = ?",
                    (new, old),
                )
                await self.conn.execute(
                    "UPDATE chat_summaries SET session_id = ? WHERE session_id = ?",
                    (new, old),
                )
                await self.conn.execute(
                    "UPDATE tasks SET dispatched_by_chat_session = ? "
                    "WHERE dispatched_by_chat_session = ?",
                    (new, old),
                )
                await self.conn.execute(
                    "UPDATE pending_dispatches SET chat_session_id = ? "
                    "WHERE chat_session_id = ?",
                    (new, old),
                )
                await self.conn.execute(
                    "DELETE FROM chat_sessions WHERE id = ?", (old,),
                )
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "migrate tg-bot → tg-agent session ids failed",
            )

    async def _migrate_add_column(self, table: str, column: str, type_decl: str) -> None:
        try:
            await self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        except Exception as e:
            # aiosqlite re-raises sqlite3.OperationalError; only swallow "duplicate"
            if "duplicate column" not in str(e).lower():
                raise

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
                               restricted_to_chat, pre_approved_send_chat,
                               created_at, updated_at, terminal_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                task.restricted_to_chat,
                task.pre_approved_send_chat,
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
        async with self.conn.execute(
            "SELECT consecutive_denials FROM tasks WHERE id = ?", (str(task_id),)
        ) as cur:
            row = await cur.fetchone()
        await self.conn.commit()
        return int(row["consecutive_denials"]) if row else 0

    async def reset_consecutive_denials(self, task_id: UUID) -> None:
        await self.conn.execute(
            "UPDATE tasks SET consecutive_denials = 0, updated_at = ? WHERE id = ?",
            (iso(utcnow()), str(task_id)),
        )
        await self.conn.commit()

    async def get_task(self, task_id: UUID) -> Task | None:
        async with self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (str(task_id),)
        ) as cur:
            row = await cur.fetchone()
        return _row_to_task(row) if row else None

    async def get_task_by_session(self, session_id: str) -> Task | None:
        async with self.conn.execute(
            "SELECT * FROM tasks WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return _row_to_task(row) if row else None

    async def list_tasks_in_states(self, *states: TaskState) -> list[Task]:
        if not states:
            return []
        placeholders = ",".join("?" * len(states))
        async with self.conn.execute(
            f"SELECT * FROM tasks WHERE state IN ({placeholders}) ORDER BY created_at",
            tuple(s.value for s in states),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_task(r) for r in rows]

    async def list_tasks(self, *, limit: int = 50) -> list[Task]:
        async with self.conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_task(r) for r in rows]

    # ---- task events ----

    async def append_event(self, task_id: UUID, type_: str, payload: dict[str, Any]) -> int:
        # Single INSERT with an inline MAX(seq)+1 subquery so the read
        # and the write are one SQL statement — two concurrent publishes
        # for the same task can no longer interleave their SELECT/INSERT
        # and collide on the (task_id, seq) UNIQUE index. RETURNING
        # gives us the chosen seq back without a second round-trip.
        async with self.conn.execute(
            """INSERT INTO task_events (task_id, seq, type, payload, created_at)
               VALUES (
                   ?,
                   COALESCE((SELECT MAX(seq) FROM task_events WHERE task_id = ?), 0) + 1,
                   ?, ?, ?
               )
               RETURNING seq""",
            (str(task_id), str(task_id), type_, json.dumps(payload), iso(utcnow())),
        ) as cur:
            row = await cur.fetchone()
        await self.conn.commit()
        return int(row["seq"])

    async def list_events(self, task_id: UUID, *, since_seq: int = 0) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT seq, type, payload, created_at FROM task_events "
            "WHERE task_id = ? AND seq > ? ORDER BY seq",
            (str(task_id), since_seq),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "seq": int(r["seq"]),
                "type": r["type"],
                "payload": json.loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # Event types that signal the model actually started doing work
    # (a tool call went out, text was produced, approval was requested, or
    # the task completed). State-changed and CLI-bookkeeping events are
    # excluded — they fire even when claude never read its first turn. Used
    # by crash recovery: a RUNNING task with none of these can be safely
    # re-queued, since nothing observable has happened yet.
    _MODEL_ACTIVITY_EVENT_TYPES = (
        "tool_use.requested",
        "assistant.text",
        "approval.requested",
        "result.final",
    )

    async def has_model_activity(self, task_id: UUID) -> bool:
        placeholders = ",".join("?" * len(self._MODEL_ACTIVITY_EVENT_TYPES))
        async with self.conn.execute(
            f"SELECT 1 FROM task_events WHERE task_id = ? "
            f"AND type IN ({placeholders}) LIMIT 1",
            (str(task_id), *self._MODEL_ACTIVITY_EVENT_TYPES),
        ) as cur:
            row = await cur.fetchone()
        return row is not None

    # ---- approvals ----

    async def get_resolved_approval(
        self, session_id: str, tool_use_id: str
    ) -> tuple[ApprovalRequest, ApprovalResult] | None:
        async with self.conn.execute(
            "SELECT * FROM approvals WHERE session_id = ? AND tool_use_id = ?",
            (session_id, tool_use_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row["state"] != "resolved":
            return None
        return _row_to_approval_pair(row)

    async def get_pending_approval(self, approval_id: UUID) -> ApprovalRequest | None:
        async with self.conn.execute(
            "SELECT * FROM approvals WHERE id = ? AND state = 'pending'",
            (str(approval_id),),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_approval_request(row) if row else None

    async def get_pending_by_session_and_tool(
        self, session_id: str, tool_use_id: str,
    ) -> ApprovalRequest | None:
        """Used by broker.decide on --resume: if a pending row already exists
        for this (session_id, tool_use_id), re-attach instead of inserting a
        duplicate (which would violate the UNIQUE index)."""
        async with self.conn.execute(
            "SELECT * FROM approvals "
            "WHERE session_id = ? AND tool_use_id = ? AND state = 'pending'",
            (session_id, tool_use_id),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_approval_request(row) if row else None

    async def get_approval(self, approval_id: UUID) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (str(approval_id),)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_pending_approvals(self) -> list[ApprovalRequest]:
        async with self.conn.execute(
            "SELECT * FROM approvals WHERE state = 'pending' ORDER BY requested_at"
        ) as cur:
            rows = await cur.fetchall()
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
        self, session_id: str, *, limit: int = 60, since_id: int = 0,
    ) -> list[dict[str, Any]]:
        """Most recent `limit` chat_messages for this session, in oldest-first
        order. `since_id` (exclusive lower bound on message id) lets the caller
        load only the tail newer than a compression checkpoint."""
        async with self.conn.execute(
            "SELECT id, role, content, created_at FROM chat_messages "
            "WHERE session_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
            (session_id, since_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"id": int(r["id"]), "role": r["role"], "content": r["content"],
             "created_at": r["created_at"]}
            for r in reversed(rows)
        ]

    async def delete_chat_messages(self, session_id: str) -> int:
        """Wipe every chat_messages row for this session. Returns the
        number of rows removed. Used by the bot's /clear command. Does
        NOT touch chat_summaries, chat_sessions, or operator_memories —
        callers compose these as needed."""
        async with self.conn.execute(
            "DELETE FROM chat_messages WHERE session_id = ?", (session_id,),
        ) as cur:
            rowcount = cur.rowcount
        await self.conn.commit()
        return rowcount

    async def delete_chat_summaries(self, session_id: str) -> int:
        """Wipe every compression checkpoint for this session."""
        async with self.conn.execute(
            "DELETE FROM chat_summaries WHERE session_id = ?", (session_id,),
        ) as cur:
            rowcount = cur.rowcount
        await self.conn.commit()
        return rowcount

    # ---- per-session memory injection tracking ----

    async def get_shown_memory_ids(self, session_id: str) -> set[int]:
        """Return the set of operator-memory ids that have already been
        injected as a `[memory note]` user-message in this session. Used
        by the operator to dedup memory injection across turns (so we
        only inject a memory the FIRST time it's relevant, not every
        turn it surfaces in retrieval)."""
        async with self.conn.execute(
            "SELECT memory_id FROM session_memory_shown WHERE session_id = ?",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return {int(r["memory_id"]) for r in rows}

    async def record_memory_shown(
        self, session_id: str, memory_ids: list[int],
    ) -> None:
        """Persist that `memory_ids` were just injected into this session's
        chat history. Idempotent via PRIMARY KEY (session_id, memory_id);
        re-runs on the same ids are a no-op."""
        if not memory_ids:
            return
        now = iso(utcnow())
        await self.conn.executemany(
            "INSERT OR IGNORE INTO session_memory_shown "
            "(session_id, memory_id, shown_at) VALUES (?, ?, ?)",
            [(session_id, mid, now) for mid in memory_ids],
        )
        await self.conn.commit()

    # ---- executor hand-off cursor ----

    async def get_handoff_cursor(self, chat_session_id: str) -> int:
        """Highest chat_messages.id already forwarded to the executor
        via hand_off for this operator session. 0 if nothing forwarded yet."""
        async with self.conn.execute(
            "SELECT last_forwarded_message_id FROM executor_handoff_cursor "
            "WHERE chat_session_id = ?",
            (chat_session_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["last_forwarded_message_id"]) if row else 0

    async def set_handoff_cursor(
        self, chat_session_id: str, last_forwarded_message_id: int,
    ) -> None:
        """Upsert the hand-off cursor. Idempotent — re-setting the same
        value is fine."""
        await self.conn.execute(
            """INSERT INTO executor_handoff_cursor
               (chat_session_id, last_forwarded_message_id, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(chat_session_id) DO UPDATE SET
                   last_forwarded_message_id = excluded.last_forwarded_message_id,
                   updated_at = excluded.updated_at""",
            (chat_session_id, last_forwarded_message_id, iso(utcnow())),
        )
        await self.conn.commit()

    async def clear_session_memory_shown(self, session_id: str) -> int:
        """Wipe this session's memory-shown tracking. Called from /clear
        so a freshly-cleared session re-injects relevant memories from
        scratch. Returns rows removed."""
        async with self.conn.execute(
            "DELETE FROM session_memory_shown WHERE session_id = ?",
            (session_id,),
        ) as cur:
            rowcount = cur.rowcount
        await self.conn.commit()
        return rowcount

    # ---- chat compression checkpoints ----

    async def insert_chat_summary(
        self, *, session_id: str, summary: str,
        through_message_id: int, estimated_token_count: int,
    ) -> int:
        async with self.conn.execute(
            "INSERT INTO chat_summaries "
            "(session_id, summary, through_message_id, estimated_token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, summary, through_message_id,
             estimated_token_count, iso(utcnow())),
        ) as cur:
            lastrowid = cur.lastrowid
        await self.conn.commit()
        return lastrowid or 0

    async def get_latest_chat_summary(self, session_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT id, summary, through_message_id, estimated_token_count, created_at "
            "FROM chat_summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "summary": row["summary"],
            "through_message_id": int(row["through_message_id"]),
            "estimated_token_count": int(row["estimated_token_count"]),
            "created_at": row["created_at"],
        }

    # ---- task result summaries ----

    async def update_task_result_summary(self, task_id: UUID, summary: str) -> None:
        await self.conn.execute(
            "UPDATE tasks SET result_summary = ?, updated_at = ? WHERE id = ?",
            (summary, iso(utcnow()), str(task_id)),
        )
        await self.conn.commit()

    async def get_task_result_summary(self, task_id: UUID) -> str | None:
        async with self.conn.execute(
            "SELECT result_summary FROM tasks WHERE id = ?", (str(task_id),),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return row["result_summary"]

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
        self, *,
        unread_only: bool = True,
        exclude_triaged: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        if unread_only:
            where.append("read_at IS NULL")
        if exclude_triaged:
            where.append(
                "id NOT IN (SELECT inbox_id FROM messenger_inbox_triaged)"
            )
        where_sql = ("WHERE " + " AND ".join(where) + " ") if where else ""
        sql = (
            "SELECT * FROM messenger_inbox " + where_sql
            + "ORDER BY received_at DESC LIMIT ?"
        )
        async with self.conn.execute(sql, (limit,)) as cur:
            rows = await cur.fetchall()
        return [_row_to_inbox(r) for r in rows]

    async def mark_inbox_triaged(self, inbox_ids: list[str]) -> None:
        """Record that the auto-drain has shown these rows to the operator
        (regardless of whether the outcome was a reply or a silent decision).
        Used so the next-restart recovery doesn't re-queue them."""
        if not inbox_ids:
            return
        now = iso(utcnow())
        await self.conn.executemany(
            "INSERT OR IGNORE INTO messenger_inbox_triaged "
            "(inbox_id, triaged_at) VALUES (?, ?)",
            [(i, now) for i in inbox_ids],
        )
        await self.conn.commit()

    # Total-body-char cap on what `messages` carries per chat. Bursts can
    # be 50+ messages (typing-in-progress, voice-then-text, etc.); the
    # note we send to the operator only needs enough signal to ack and
    # hand off. Newer messages are preferred — walk newest→oldest and
    # stop once their bodies sum past the cap.
    _PENDING_MSG_TOTAL_CHARS = 500

    async def list_pending_chats(self) -> list[dict[str, Any]]:
        """One row per chat_id that has not-yet-triaged inbox messages.

        "Pending" is keyed off triaged-ness alone, not `read_at`. The
        drain marks the snapshot read BEFORE invoking the operator
        (so newly-arriving DMs during a turn remain `read_at IS NULL`
        and are detectable as mid-flight arrivals), then marks
        triaged only after the turn completes. A crashed turn leaves
        rows read-but-not-triaged — still picked up here for retry,
        either by the next dirty cycle or by boot recovery.

        Each row carries: sender (from the latest pending row),
        `unread_count`, `first_unread_at` / `last_unread_at`, `inbox_ids`
        (every db-side row id contributing to this chat — the exact set
        the drain must mark read/triaged), and a bounded `messages`
        list of `{message_id, received_at, body}` whose bodies sum to
        ≤500 chars (older messages in a burst are dropped from
        `messages` but still counted in `unread_count` / `inbox_ids`).
        Callers render notes from `messages` directly so they can
        include per-message timestamps and ids."""
        async with self.conn.execute(
            """
            SELECT id, chat_id, message_id, sender_username,
                   sender_display_name, body, received_at
            FROM messenger_inbox
            WHERE id NOT IN (SELECT inbox_id FROM messenger_inbox_triaged)
            ORDER BY received_at ASC
            """
        ) as cur:
            rows = await cur.fetchall()
        # Group by chat_id, preserving oldest-first ordering.
        grouped: dict[str, list[Any]] = {}
        for r in rows:
            grouped.setdefault(r["chat_id"], []).append(r)
        out: list[dict[str, Any]] = []
        for chat_id, msgs in grouped.items():
            latest = msgs[-1]
            # Walk newest→oldest, take up to N msgs, stop when bodies sum
            # past the char cap. Reverse back to chronological for output.
            # Take messages newest→oldest. Always include at least the
            # newest (the loop body appends BEFORE the cap check), then
            # stop once the running total has reached the char cap.
            picked: list[Any] = []
            total = 0
            for m in reversed(msgs):
                picked.append(m)
                total += len(m["body"] or "")
                if total >= self._PENDING_MSG_TOTAL_CHARS:
                    break
            picked.reverse()
            messages = [
                {
                    "message_id": m["message_id"],
                    "received_at": m["received_at"],
                    "body": m["body"] or "",
                }
                for m in picked
            ]
            out.append({
                "chat_id": chat_id,
                "sender_username": latest["sender_username"],
                "sender_display_name": latest["sender_display_name"],
                "unread_count": len(msgs),
                "first_unread_at": msgs[0]["received_at"],
                "last_unread_at": latest["received_at"],
                "inbox_ids": [m["id"] for m in msgs],
                "messages": messages,
            })
        # Most-recently-updated chats first — matches what the operator
        # would expect to see if asked "any DMs?".
        out.sort(key=lambda r: r["last_unread_at"], reverse=True)
        return out

    async def latest_pending_for_chat(
        self, chat_id: str,
    ) -> dict[str, Any] | None:
        """Most-recent not-yet-triaged inbox row in `chat_id`, or None
        if the chat is fully triaged. Audit-attribution helper for the
        `reply_to_chat` path — answers "which inbox row is the bot
        replying to?". Not gated on `read_at` because the drain
        pre-marks the snapshot read before the operator turn."""
        async with self.conn.execute(
            """
            SELECT id, platform, chat_id, message_id, sender_username,
                   sender_display_name, body, is_important, received_at,
                   read_at, replied_message_id
            FROM messenger_inbox
            WHERE chat_id = ?
              AND id NOT IN (SELECT inbox_id FROM messenger_inbox_triaged)
            ORDER BY received_at DESC
            LIMIT 1
            """,
            (chat_id,),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_inbox(row) if row else None

    async def mark_chat_triaged(self, chat_id: str) -> int:
        """Mark every unread, not-yet-triaged row for `chat_id` as triaged.
        Returns the count of rows just inserted. Called once per chat after
        an inbox-drain auto-ping completes (whether the operator replied or
        stayed silent) so a restart's recovery doesn't re-burn LLM on the
        same chat."""
        async with self.conn.execute(
            """
            SELECT id FROM messenger_inbox
            WHERE chat_id = ? AND read_at IS NULL
              AND id NOT IN (SELECT inbox_id FROM messenger_inbox_triaged)
            """,
            (chat_id,),
        ) as cur:
            ids_rows = await cur.fetchall()
        ids = [r["id"] for r in ids_rows]
        if not ids:
            return 0
        # Inlined rather than calling self.mark_inbox_triaged(): both methods
        # acquire the per-connection lock, and asyncio.Lock is not reentrant,
        # so delegating would self-deadlock.
        now = iso(utcnow())
        await self.conn.executemany(
            "INSERT OR IGNORE INTO messenger_inbox_triaged "
            "(inbox_id, triaged_at) VALUES (?, ?)",
            [(i, now) for i in ids],
        )
        await self.conn.commit()
        return len(ids)

    async def mark_chat_read(self, chat_id: str) -> int:
        """Set `read_at` on every unread inbox row for this chat. Returns
        the rowcount. Used by the operator's `mark_chat_read` tool and by
        `record_chat_reply` to clear an entire conversation's unread state
        after the operator replies to it."""
        async with self.conn.execute(
            "UPDATE messenger_inbox SET read_at = ? "
            "WHERE chat_id = ? AND read_at IS NULL",
            (iso(utcnow()), chat_id),
        ) as cur:
            rowcount = cur.rowcount
        await self.conn.commit()
        return rowcount

    async def record_chat_reply(
        self, chat_id: str, reply_message_id: str,
    ) -> None:
        """After the operator successfully replies to a chat, stamp the
        latest not-yet-triaged row with `replied_message_id` (audit
        hook for "which DM did this reply address?") and mark every
        unread row in that chat read in one go. Idempotent — runs even
        if no rows match.

        The audit-stamp SELECT keys off triaged-ness, not `read_at`:
        the inbox-drain pre-marks a turn's snapshot read before the
        operator runs, so by the time this function fires from inside
        that turn, the snapshot rows are already read. Only triaged-ness
        distinguishes "the row the bot is replying to" from "already
        handled."""
        now = iso(utcnow())
        async with self.conn.execute(
            "SELECT id FROM messenger_inbox "
            "WHERE chat_id = ? "
            "  AND id NOT IN (SELECT inbox_id FROM messenger_inbox_triaged) "
            "ORDER BY received_at DESC LIMIT 1",
            (chat_id,),
        ) as cur:
            latest = await cur.fetchone()
        if latest is not None:
            await self.conn.execute(
                "UPDATE messenger_inbox SET replied_message_id = ? "
                "WHERE id = ?",
                (reply_message_id, latest["id"]),
            )
        await self.conn.execute(
            "UPDATE messenger_inbox SET read_at = ? "
            "WHERE chat_id = ? AND read_at IS NULL",
            (now, chat_id),
        )
        await self.conn.commit()

    # ---- pending dispatches (operator-initiated approval flow) ----

    async def create_pending_dispatch(
        self, *,
        dispatch_id: str,
        chat_session_id: str,
        prompt: str,
        model: str | None,
        restricted_to_chat: str,
    ) -> None:
        await self.conn.execute(
            """
            INSERT INTO pending_dispatches
                (id, chat_session_id, prompt, model, restricted_to_chat,
                 created_at, resolved_at, resolution)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (dispatch_id, chat_session_id, prompt, model,
             restricted_to_chat, iso(utcnow())),
        )
        await self.conn.commit()

    async def get_pending_dispatch(
        self, dispatch_id: str,
    ) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM pending_dispatches WHERE id = ?", (dispatch_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "chat_session_id": row["chat_session_id"],
            "prompt": row["prompt"],
            "model": row["model"],
            "restricted_to_chat": row["restricted_to_chat"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
            "resolution": row["resolution"],
        }

    async def resolve_pending_dispatch(
        self, dispatch_id: str, resolution: str,
    ) -> bool:
        """Mark a pending dispatch resolved. Returns True if the row was
        still pending and we resolved it; False if it was already resolved
        or doesn't exist (the caller MUST treat this as a no-op so a double-
        tap on Yes/No doesn't fire the task twice)."""
        async with self.conn.execute(
            "UPDATE pending_dispatches SET resolution = ?, resolved_at = ? "
            "WHERE id = ? AND resolution IS NULL",
            (resolution, iso(utcnow()), dispatch_id),
        ) as cur:
            rowcount = cur.rowcount or 0
        await self.conn.commit()
        return rowcount > 0

    # ---- dm allowlist (reply_to_dm hard guardrail) ----

    async def allow_dm(self, chat_id: str) -> bool:
        """Add `chat_id` to the autonomous-reply allowlist. Idempotent —
        returns True if a new row was created, False if it already existed."""
        async with self.conn.execute(
            "INSERT OR IGNORE INTO dm_allowlist (chat_id, added_at) VALUES (?, ?)",
            (chat_id, iso(utcnow())),
        ) as cur:
            rowcount = cur.rowcount or 0
        await self.conn.commit()
        return rowcount > 0

    async def deny_dm(self, chat_id: str) -> bool:
        """Remove `chat_id` from the allowlist. Returns True if a row was
        deleted, False if it wasn't on the list."""
        async with self.conn.execute(
            "DELETE FROM dm_allowlist WHERE chat_id = ?", (chat_id,),
        ) as cur:
            rowcount = cur.rowcount or 0
        await self.conn.commit()
        return rowcount > 0

    async def is_dm_allowed(self, chat_id: str) -> bool:
        async with self.conn.execute(
            "SELECT 1 FROM dm_allowlist WHERE chat_id = ?", (chat_id,),
        ) as cur:
            row = await cur.fetchone()
        return row is not None

    async def list_dm_allowed(self) -> list[dict[str, str]]:
        async with self.conn.execute(
            "SELECT chat_id, added_at FROM dm_allowlist ORDER BY added_at",
        ) as cur:
            rows = await cur.fetchall()
        return [{"chat_id": r["chat_id"], "added_at": r["added_at"]} for r in rows]

    # ---- per-task Write-dir allowlist ----

    async def allow_write_dir(self, task_id: str, dir_path: str) -> None:
        await self.conn.execute(
            "INSERT OR IGNORE INTO task_write_dir_allowlist "
            "(task_id, dir, added_at) VALUES (?, ?, ?)",
            (task_id, dir_path, iso(utcnow())),
        )
        await self.conn.commit()

    async def list_write_dirs(self, task_id: str) -> list[str]:
        async with self.conn.execute(
            "SELECT dir FROM task_write_dir_allowlist WHERE task_id = ?",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [r["dir"] for r in rows]

    # ---- ask_user (executor → human via operator) ----

    async def create_ask_request(
        self, *, ask_id: str, task_id: str, chat_session_id: str, question: str,
    ) -> None:
        await self.conn.execute(
            "INSERT INTO ask_requests (id, task_id, chat_session_id, question, "
            "state, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
            (ask_id, task_id, chat_session_id, question, iso(utcnow())),
        )
        await self.conn.commit()

    async def get_ask_request(self, ask_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM ask_requests WHERE id = ?", (ask_id,),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def has_presented_ask_for_chat(self, chat_session_id: str) -> bool:
        async with self.conn.execute(
            "SELECT 1 FROM ask_requests WHERE chat_session_id = ? "
            "AND state = 'presented' LIMIT 1",
            (chat_session_id,),
        ) as cur:
            row = await cur.fetchone()
        return row is not None

    async def get_presented_ask_for_chat(self, chat_session_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM ask_requests WHERE chat_session_id = ? "
            "AND state = 'presented' ORDER BY presented_at ASC LIMIT 1",
            (chat_session_id,),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def next_pending_ask_for_chat(self, chat_session_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM ask_requests WHERE chat_session_id = ? "
            "AND state = 'pending' ORDER BY created_at ASC LIMIT 1",
            (chat_session_id,),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def mark_ask_presented(self, ask_id: str) -> None:
        await self.conn.execute(
            "UPDATE ask_requests SET state = 'presented', presented_at = ? "
            "WHERE id = ? AND state = 'pending'",
            (iso(utcnow()), ask_id),
        )
        await self.conn.commit()

    async def mark_ask_answered(self, ask_id: str, answer: str) -> bool:
        async with self.conn.execute(
            "UPDATE ask_requests SET state = 'answered', answer = ?, "
            "answered_at = ? WHERE id = ? AND state IN ('pending', 'presented')",
            (answer, iso(utcnow()), ask_id),
        ) as cur:
            rowcount = cur.rowcount or 0
        await self.conn.commit()
        return rowcount > 0

    async def cancel_stale_asks(self) -> int:
        """Mark every non-terminal ask as cancelled. Run on startup — the
        executor processes that owned the in-flight futures died with
        the previous daemon, so the rows are orphans."""
        async with self.conn.execute(
            "UPDATE ask_requests SET state = 'cancelled', answered_at = ? "
            "WHERE state IN ('pending', 'presented')",
            (iso(utcnow()),),
        ) as cur:
            rowcount = cur.rowcount or 0
        await self.conn.commit()
        return rowcount

    async def get_inbox_message(self, inbox_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM messenger_inbox WHERE id = ?", (inbox_id,),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_inbox(row) if row else None

    async def mark_inbox_read(self, inbox_ids: list[str]) -> int:
        """Stamp `read_at` on the listed rows that are still unread.
        Returns the rowcount actually flipped. Used by the inbox-drain
        to mark a flush's snapshot read BEFORE invoking the operator
        so mid-flight arrivals (`read_at IS NULL`) are distinguishable
        from the rows the operator is already seeing. Also exposed via
        `telegram_service.mark_read` for the operator's single-row
        `messenger.mark_read` tool."""
        if not inbox_ids:
            return 0
        placeholders = ",".join("?" * len(inbox_ids))
        async with self.conn.execute(
            f"UPDATE messenger_inbox SET read_at = ? "
            f"WHERE id IN ({placeholders}) AND read_at IS NULL",
            (iso(utcnow()), *inbox_ids),
        ) as cur:
            rowcount = cur.rowcount
        await self.conn.commit()
        return rowcount

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

    # ---- operator memory (semantic store) ----
    # The SQL for operator_memory lives here so every connection touch goes
    # through the per-connection lock. operator_memory.py owns the embedding /
    # scoring logic and passes embeddings as raw bytes; this layer is pure SQL.

    async def memory_get(self, memory_id: int, model: str) -> aiosqlite.Row | None:
        async with self.conn.execute(
            "SELECT id, text, last_accessed_at FROM operator_memories "
            "WHERE id = ? AND model = ?",
            (memory_id, model),
        ) as cur:
            return await cur.fetchone()

    async def memory_delete(self, memory_id: int, model: str) -> bool:
        async with self.conn.execute(
            "DELETE FROM operator_memories WHERE id = ? AND model = ?",
            (memory_id, model),
        ) as cur:
            rowcount = cur.rowcount
        await self.conn.commit()
        return rowcount > 0

    async def memory_all_rows(self, model: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT id, text, embedding, last_accessed_at "
            "FROM operator_memories WHERE model = ? ORDER BY id",
            (model,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def memory_insert(
        self, *, text: str, embedding: bytes, model: str, source_turn: str | None,
    ) -> int:
        now = iso(utcnow())
        async with self.conn.execute(
            "INSERT INTO operator_memories "
            "(text, embedding, model, source_turn, created_at, last_accessed_at, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (text, embedding, model, source_turn, now, now),
        ) as cur:
            lastrowid = cur.lastrowid
        await self.conn.commit()
        return lastrowid or 0

    async def memory_entries_count(self, model: str) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM operator_memories WHERE model = ?",
            (model,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def memory_stale_count(self, model: str) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM operator_memories WHERE model != ?",
            (model,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def memory_select_stale_batch(
        self, model: str, limit: int,
    ) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT id, text FROM operator_memories "
            "WHERE model != ? ORDER BY id LIMIT ?",
            (model, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def memory_update_embeddings(
        self, rows: list[tuple[int, bytes]], model: str,
    ) -> None:
        """Re-point a batch of rows at `model` with freshly-computed
        embeddings, in one transaction. `rows` is [(id, embedding_bytes), ...]."""
        if not rows:
            return
        await self.conn.executemany(
            "UPDATE operator_memories SET embedding = ?, model = ? WHERE id = ?",
            [(emb, model, rid) for (rid, emb) in rows],
        )
        await self.conn.commit()

    async def memory_bump_access(self, row_ids: list[int]) -> None:
        if not row_ids:
            return
        now = iso(utcnow())
        placeholders = ",".join("?" * len(row_ids))
        await self.conn.execute(
            f"UPDATE operator_memories "
            f"SET last_accessed_at = ?, access_count = access_count + 1 "
            f"WHERE id IN ({placeholders})",
            (now, *row_ids),
        )
        await self.conn.commit()

    async def memory_evict_over_capacity(self, capacity: int, model: str) -> int:
        """If the model's row count exceeds `capacity`, hard-delete the oldest
        (by last_accessed_at) overflow rows. Count + delete in one locked
        transaction. Returns how many rows were evicted."""
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM operator_memories WHERE model = ?",
            (model,),
        ) as cur:
            row = await cur.fetchone()
        n = int(row["n"]) if row else 0
        if n <= capacity:
            return 0
        overflow = n - capacity
        await self.conn.execute(
            "DELETE FROM operator_memories WHERE id IN ("
            "  SELECT id FROM operator_memories "
            "  ORDER BY last_accessed_at ASC, id ASC LIMIT ?"
            ")",
            (overflow,),
        )
        await self.conn.commit()
        return overflow

    async def memory_load_skip_pairs(self) -> set[tuple[int, int]]:
        async with self.conn.execute(
            "SELECT id_a, id_b FROM memory_dedup_skip_pairs"
        ) as cur:
            rows = await cur.fetchall()
        return {(int(r["id_a"]), int(r["id_b"])) for r in rows}

    async def memory_record_skip_pairs(self, pairs: list[tuple[int, int]]) -> None:
        if not pairs:
            return
        now = iso(utcnow())
        await self.conn.executemany(
            "INSERT OR IGNORE INTO memory_dedup_skip_pairs "
            "(id_a, id_b, recorded_at) VALUES (?, ?, ?)",
            [(a, b, now) for (a, b) in pairs],
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
        restricted_to_chat=row["restricted_to_chat"],
        pre_approved_send_chat=row["pre_approved_send_chat"],
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


def _serialize_connection_access(cls: type) -> None:
    """Wrap every public coroutine method of `cls` so it runs under
    `self._lock` (see `Database.__init__`). Done in one place so a method can
    never *forget* to take the lock — that completeness is itself the safety
    property: all shared-connection access is serialized.

    Excluded: `connect`/`close` (lifecycle; run outside the concurrent phase)
    and any `_`-prefixed helper/migration (migrations run once at startup
    before tasks exist, and internal helpers are called from already-locked
    public methods, so locking them would self-deadlock — asyncio.Lock is not
    reentrant). Public methods must therefore not call each other.
    """
    skip = {"connect", "close"}
    for name, attr in list(vars(cls).items()):
        if name.startswith("_") or name in skip:
            continue
        if asyncio.iscoroutinefunction(attr):
            setattr(cls, name, _locked(attr))


_serialize_connection_access(Database)
