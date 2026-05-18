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
from pathlib import Path
from typing import Any

from .config import Paths, Settings
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
        argv = self._build_argv(task, resuming=resuming)
        log.info("spawning claude for task %s (resuming=%s): %s", task.id, resuming, argv[0])

        # Env propagated to the claude subprocess. Inherit, then override the
        # MCP server's settings (token/port/session) — the MCP server is a
        # stdio child of claude and reads these from its env.
        env = os.environ.copy()
        env["ONCALL_PORT"] = str(self._settings.oncall_port)
        env["ONCALL_TOKEN"] = self._settings.oncall_token
        env["ONCALL_SESSION_ID"] = task.session_id

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            # claude emits MCP tool_results as one JSON line per result.
            # `read_image` carries the photo as inline base64; a multi-MB
            # image blows past asyncio's 64KB default StreamReader limit
            # and crashes the supervisor with LimitOverrunError. Cap is
            # 10MB on the MCP side; base64 + JSON overhead → use 32MB.
            limit=32 * 1024 * 1024,
            # Inherit the orchestrator's cwd. When `oncall api` is launched
            # from a project dir, claude operates there. As a global tool the
            # user picks the working directory by `cd`ing before starting.
        )

        await self._db.update_task_state(task.id, TaskState.RUNNING)
        await self._events.publish(task.id, "state.changed", {"state": "running"})

        # Write the initial user turn (only on first launch — on --resume the
        # transcript already contains the original prompt and any subsequent
        # turns).
        if not resuming:
            await self._write_user_turn(task.prompt)
            # Close stdin: we're not interactive. Future turns can be added by
            # reopening the API; for milestone-1 a single user-turn is enough.
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass

        try:
            stderr_task = asyncio.create_task(self._drain_stderr(task))
            terminal = await self._read_stdout(task)
            stderr_task.cancel()
            await self._proc.wait()
        except asyncio.CancelledError:
            await self._terminate()
            terminal = TerminalReason.KILLED
            raise
        finally:
            self._proc = None

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

    # ---- argv & input ----

    def _build_argv(self, task: Task, *, resuming: bool) -> list[str]:
        # Use the orchestrator's own interpreter to spawn the MCP child. This
        # works identically in editable installs, wheels, and `uv tool install`
        # without depending on `uv` being on the user's PATH or on a specific
        # project venv layout.
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
            "--append-system-prompt", self._paths.executor_prompt.read_text(encoding="utf-8"),
            "--no-session-persistence",
        ]
        if task.model:
            argv += ["--model", task.model]
        if task.max_turns:
            # claude uses --max-turns or similar — we keep it generic; if not
            # supported in this CLI version, drop silently. For now: skip if
            # unset, otherwise the flag name from help is unknown so omit.
            pass
        if resuming:
            argv += ["--resume", task.session_id]
        else:
            argv += ["--session-id", task.session_id]
        return argv

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

    async def _drain_stderr(self, task: Task) -> None:
        assert self._proc and self._proc.stderr
        try:
            async for raw in _iter_lines(self._proc.stderr):
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
