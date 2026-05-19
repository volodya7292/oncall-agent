"""Per-task `claude` CLI subprocess supervisor.

Spawns one `claude` process per task in headless (--print) mode with stream-json
I/O, drives it with the task's initial prompt, and forwards model events to the
event bus while persisting them to SQLite.

Pause is implicit: when claude emits a tool_use that isn't auto-allowed, our MCP
`approve` tool (over stdio, in a child of this subprocess) blocks until the
orchestrator's broker resolves. We see no traffic on stdout during the wait.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    Paths,
    Settings,
    _reset_session_initialized_marker,
    get_global_executor_session_id,
    is_executor_session_initialized,
    mark_executor_session_initialized,
)
from .db import Database
from .events import EventBus
from .models import Task, TaskState, TerminalReason


log = logging.getLogger(__name__)


class SupervisorError(Exception):
    pass


class Supervisor:
    """Owns the lifecycle of one `claude` subprocess.

    Construction is cheap; the work happens in `.run()`.
    """

    def __init__(
        self,
        *,
        db: Database,
        events: EventBus,
        settings: Settings,
        paths: Paths,
    ) -> None:
        self._db = db
        self._events = events
        self._settings = settings
        self._paths = paths
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def proc(self) -> asyncio.subprocess.Process | None:
        return self._proc

    async def run(self, task: Task, *, resuming: bool = False) -> TerminalReason:
        # One global session is shared across every executor invocation so
        # claude --resume can accumulate context turn-to-turn. The serializer
        # in lifecycle ensures only one supervisor runs at a time.
        session_id = get_global_executor_session_id()
        already_initialized = is_executor_session_initialized()
        use_resume = resuming or already_initialized

        terminal, session_missing = await self._spawn_once(
            task, session_id=session_id, use_resume=use_resume,
            write_initial_user_turn=not resuming,
        )

        # Fallback: we asked for --resume but claude couldn't find the
        # session (typically: the marker file persists across daemon
        # restarts but claude's session store was wiped). Reset the
        # marker and re-spawn with --session-id to create it fresh.
        if session_missing and use_resume:
            log.warning(
                "session %s missing on --resume; re-spawning with "
                "--session-id to recreate", session_id,
            )
            _reset_session_initialized_marker()
            terminal, _ = await self._spawn_once(
                task, session_id=session_id, use_resume=False,
                write_initial_user_turn=True,
            )

        # On the first ever successful spawn (--session-id path), mark
        # initialized so the next call uses --resume.
        if terminal == TerminalReason.SUCCESS and not is_executor_session_initialized():
            try:
                mark_executor_session_initialized()
            except OSError as e:
                log.warning("could not mark executor session initialized: %s", e)

        # Persist final state.
        final_state = {
            TerminalReason.SUCCESS: TaskState.COMPLETED,
            TerminalReason.CLI_ERROR: TaskState.FAILED,
            TerminalReason.DENIAL_LOOP: TaskState.FAILED,
            TerminalReason.KILLED: TaskState.KILLED,
            TerminalReason.TIMEOUT: TaskState.FAILED,
        }[terminal]
        await self._db.update_task_state(task.id, final_state, terminal_reason=terminal)
        await self._events.publish(task.id, "state.changed", {
            "state": final_state.value,
            "terminal_reason": terminal.value,
        })
        return terminal

    async def _spawn_once(
        self, task: Task, *, session_id: str, use_resume: bool,
        write_initial_user_turn: bool,
    ) -> tuple[TerminalReason, bool]:
        """Spawn one claude subprocess, drive it to completion. Returns
        (terminal_reason, session_missing). `session_missing` is True iff
        stderr revealed claude couldn't find the session under --resume —
        that signals the caller to retry with --session-id."""
        argv = self._build_argv(task, session_id=session_id, use_resume=use_resume)
        log.info("spawning claude for task %s (resume=%s, session=%s): %s",
                 task.id, use_resume, session_id, argv[0])

        env = os.environ.copy()
        env["ONCALL_PORT"] = str(self._settings.oncall_port)
        env["ONCALL_TOKEN"] = self._settings.oncall_token
        # Two different ids in play:
        #   - `session_id` (global, here): the claude --session-id /
        #     --resume value so the model's conversation accumulates
        #     across hand_offs.
        #   - `task.session_id` (per-row): the broker's lookup key. The
        #     MCP server forwards this on every approve/messenger/memory
        #     call so broker.decide can pin the call back to THIS task.
        #     Without it, broker.get_task_by_session(global) returns None
        #     and every mutating tool gets denied as "unknown session".
        env["ONCALL_SESSION_ID"] = task.session_id

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=32 * 1024 * 1024,
        )

        await self._db.update_task_state(task.id, TaskState.RUNNING)
        await self._events.publish(task.id, "state.changed", {"state": "running"})

        if write_initial_user_turn:
            await self._write_user_turn(task.prompt)
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass

        stderr_seen: list[bytes] = []
        try:
            stderr_task = asyncio.create_task(
                self._drain_stderr(task, sink=stderr_seen),
            )
            terminal = await self._read_stdout(task)
            stderr_task.cancel()
            await self._proc.wait()
        except asyncio.CancelledError:
            await self._terminate()
            raise
        finally:
            self._proc = None

        stderr_text = b"".join(stderr_seen).decode("utf-8", "replace")
        session_missing = (
            use_resume
            and "No conversation found with session ID" in stderr_text
        )
        return terminal, session_missing

    # ---- argv & input ----

    def _build_argv(self, task: Task, *, session_id: str, use_resume: bool) -> list[str]:
        # Use the orchestrator's own interpreter to spawn the MCP child. This
        # works identically in editable installs, wheels, and `uv tool install`
        # without depending on `uv` being on the user's PATH or on a specific
        # project venv layout.
        # ONCALL_SESSION_ID is the broker's per-task lookup key, NOT the
        # claude --session-id (which stays global across hand_offs). See
        # _spawn_once for the full explanation.
        mcp_inline = json.dumps({
            "mcpServers": {
                "oncall": {
                    "command": sys.executable,
                    "args": ["-m", "oncall", "mcp"],
                    "env": {
                        "ONCALL_PORT": str(self._settings.oncall_port),
                        "ONCALL_TOKEN": self._settings.oncall_token,
                        "ONCALL_SESSION_ID": task.session_id,
                    },
                }
            }
        })
        argv: list[str] = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            # NOTE: NOT using --bare — that would disable keychain reads and
            # break OAuth (subscription) auth. We rely on whatever auth the
            # host has set up (keychain OAuth, ANTHROPIC_API_KEY env, etc.).
            "--strict-mcp-config",
            "--mcp-config", mcp_inline,
            "--settings", str(self._paths.settings_json),
            "--setting-sources", "project",  # only our executor/settings.json, not user-level
            "--permission-mode", "default",
            "--permission-prompt-tool", "mcp__oncall__approve",
            "--effort", "medium",
            "--append-system-prompt", self._render_executor_prompt(),
            # Persistence is REQUIRED in the single-session model: every
            # hand_off after the first uses --resume to pick up the same
            # session. --no-session-persistence would erase the session
            # the moment the subprocess exited, so the next spawn would
            # fail with "No conversation found".
        ]
        if task.model:
            argv += ["--model", task.model]
        if task.max_turns:
            # claude uses --max-turns or similar — we keep it generic; if not
            # supported in this CLI version, drop silently. For now: skip if
            # unset, otherwise the flag name from help is unknown so omit.
            pass
        if use_resume:
            argv += ["--resume", session_id]
        else:
            argv += ["--session-id", session_id]
        return argv

    def _render_executor_prompt(self) -> str:
        """Read the executor system prompt and substitute spawn-time
        placeholders. Currently: `{{current_date}}` → ISO UTC datetime
        captured at spawn — the executor's headless `--print` session
        doesn't get a reliable date in its built-in context, so we inject
        one explicitly. Long-running resumed sessions still see the
        date refresh on each spawn (one per hand_off)."""
        text = self._paths.executor_prompt.read_text(encoding="utf-8")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return text.replace("{{current_date}}", now)

    async def _write_user_turn(self, text: str) -> None:
        assert self._proc and self._proc.stdin
        line = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        }) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    # ---- stdout reader ----

    async def _read_stdout(self, task: Task) -> TerminalReason:
        assert self._proc and self._proc.stdout
        reader = self._proc.stdout
        terminal: TerminalReason = TerminalReason.CLI_ERROR
        async for raw in _iter_lines(reader):
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                log.debug("skipping unparseable stream-json line: %r", raw[:120])
                continue
            term = await self._dispatch_event(task, evt)
            if term is not None:
                terminal = term
                # Don't break — drain anything remaining; claude exits on its own.
        # If we got here without seeing a `result` event, treat as cli error.
        return terminal

    async def _dispatch_event(self, task: Task, evt: dict[str, Any]) -> TerminalReason | None:
        etype = evt.get("type") or ""
        subtype = evt.get("subtype") or ""

        if etype == "system" and subtype == "init":
            # The init event echoes the session id we assigned; nothing to do.
            await self._events.publish(task.id, "system.init", {
                "model": evt.get("model"),
                "tools": evt.get("tools"),
            })
            return None

        if etype == "assistant":
            msg = evt.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else []
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    await self._events.publish(task.id, "assistant.text", {
                        "text": block.get("text", ""),
                    })
                elif block.get("type") == "tool_use":
                    await self._events.publish(task.id, "tool_use.requested", {
                        "tool_use_id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input"),
                    })
            return None

        if etype == "user":
            # Tool results echoed back to the model show up here. Forward.
            msg = evt.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else []
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    await self._events.publish(task.id, "tool_result", {
                        "tool_use_id": block.get("tool_use_id"),
                        "is_error": block.get("is_error", False),
                        # Don't echo the entire result content; cap to a preview.
                        "preview": _preview(block.get("content")),
                    })
            return None

        if etype == "result":
            is_error = bool(evt.get("is_error"))
            await self._events.publish(task.id, "result.final", {
                "is_error": is_error,
                "duration_ms": evt.get("duration_ms"),
                "total_cost_usd": evt.get("total_cost_usd"),
                "num_turns": evt.get("num_turns"),
            })
            return TerminalReason.CLI_ERROR if is_error else TerminalReason.SUCCESS

        # Hook events, api_retry, etc.: log but don't break.
        await self._events.publish(task.id, f"cli.{etype}", {"raw": evt})
        return None

    async def _drain_stderr(
        self, task: Task, *, sink: list[bytes] | None = None,
    ) -> None:
        assert self._proc and self._proc.stderr
        try:
            async for raw in _iter_lines(self._proc.stderr):
                if sink is not None:
                    sink.append(raw)
                log.warning("claude[%s] stderr: %s", task.id, raw.decode("utf-8", "replace")[:200])
        except asyncio.CancelledError:
            return

    async def _terminate(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            self._proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                self._proc.send_signal(signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._proc.wait()


async def _iter_lines(reader: asyncio.StreamReader):
    """Yield raw line bytes (newline stripped) until EOF."""
    while True:
        raw = await reader.readline()
        if not raw:
            return
        yield raw.rstrip(b"\n")


def _preview(content: Any, *, max_chars: int = 400) -> str:
    """Stringify and truncate a tool_result.content blob for the event payload."""
    if isinstance(content, str):
        s = content
    elif isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                parts.append(str(b["text"]))
            else:
                parts.append(json.dumps(b))
        s = "\n".join(parts)
    else:
        s = json.dumps(content) if content is not None else ""
    return s[:max_chars] + ("…" if len(s) > max_chars else "")
