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
from collections.abc import Callable
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
from .models import Task, TaskState, TerminalReason, format_utc_now
from .result_delivery import EXECUTOR_REPLY_BUDGET_CHARS


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
        developers_snapshot_provider: "Callable[[], str] | None" = None,
    ) -> None:
        self._db = db
        self._events = events
        self._settings = settings
        self._paths = paths
        self._developers_snapshot_provider = developers_snapshot_provider
        self._proc: asyncio.subprocess.Process | None = None
        # Live context-window fill (tokens) reported by the most recent
        # task's final `result` event. Drives the post-task /compact guard.
        self._last_context_tokens: int = 0

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

        terminal, session_missing, session_in_use = await self._spawn_once(
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
            terminal, _, _ = await self._spawn_once(
                task, session_id=session_id, use_resume=False,
                write_initial_user_turn=True,
            )
        # Inverse fallback: we asked for --session-id (marker absent) but the
        # session already exists in claude's store — a prior create spawn made
        # it then died before SUCCESS, so the marker was never written. Adopt
        # the existing session: mark initialized and re-spawn with --resume
        # instead of looping on "Session ID … is already in use".
        elif session_in_use and not use_resume:
            log.warning(
                "session %s already exists but was unmarked; adopting it "
                "via --resume", session_id,
            )
            try:
                mark_executor_session_initialized()
            except OSError as e:
                log.warning("could not mark executor session initialized: %s", e)
            terminal, _, _ = await self._spawn_once(
                task, session_id=session_id, use_resume=True,
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

        # Context guard: the task is done and its result is already
        # persisted, so a now-fat session gets compacted between tasks —
        # never mid-task. Best-effort; a compaction failure must not turn a
        # successful task into a failed one.
        if terminal == TerminalReason.SUCCESS:
            await self._maybe_compact_session(task, session_id)

        return terminal

    async def _spawn_once(
        self, task: Task, *, session_id: str, use_resume: bool,
        write_initial_user_turn: bool,
    ) -> tuple[TerminalReason, bool, bool]:
        """Spawn one claude subprocess, drive it to completion. Returns
        (terminal_reason, session_missing, session_in_use).
        `session_missing` is True iff stderr revealed claude couldn't find the
        session under --resume (retry with --session-id). `session_in_use` is
        True iff a --session-id create hit an existing session (adopt it via
        --resume)."""
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
        # Inverse trap: we asked for --session-id (create) but claude already
        # has this session — a prior create spawn made it, then failed before
        # SUCCESS so the `initialized` marker was never written. Signal the
        # caller to ADOPT it (mark initialized + --resume) rather than loop
        # forever on "Session ID … is already in use".
        session_in_use = (
            not use_resume
            and "is already in use" in stderr_text
        )
        return terminal, session_missing, session_in_use

    # ---- context compaction ----

    async def _maybe_compact_session(self, task: Task, session_id: str) -> None:
        """If the live context window crossed the configured threshold, run a
        `/compact` pass on the shared session so the next task resumes against
        a summarized, much smaller history. Best-effort: any failure is logged
        and swallowed — the just-finished task already succeeded."""
        threshold = self._settings.oncall_executor_compact_at_tokens
        if threshold <= 0 or self._last_context_tokens < threshold:
            return

        before = self._last_context_tokens
        log.info(
            "executor session %s at %d tokens (>= %d); compacting",
            session_id, before, threshold,
        )
        argv = [
            "claude", "--print",
            "--output-format", "stream-json", "--verbose",
            # /compact touches no tools or MCP — keep the pass minimal and
            # isolated from user-level config, same as the executor spawn.
            "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
            "--model", task.model or "sonnet",
            "--resume", session_id,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=32 * 1024 * 1024,
            )
        except Exception:
            log.exception("executor compaction: failed to spawn claude for %s", session_id)
            return

        self._proc = proc
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=b"/compact\n"), timeout=300,
            )
        except asyncio.TimeoutError:
            log.warning("executor compaction timed out for %s; killing", session_id)
            await self._terminate()
            return
        except asyncio.CancelledError:
            await self._terminate()
            raise
        finally:
            self._proc = None

        ok, after = _parse_compact_result(stdout_b)
        if ok:
            log.info(
                "executor session %s compacted: %d -> %s tokens",
                session_id, before, after if after is not None else "?",
            )
            await self._events.publish(task.id, "session.compacted", {
                "before_tokens": before, "after_tokens": after,
            })
        else:
            log.warning(
                "executor compaction for %s did not report success (rc=%s): %s",
                session_id, proc.returncode,
                stderr_b.decode("utf-8", "replace")[:200],
            )

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
                        # Gates whether the MCP server advertises the `laptop`
                        # proxy tool (cloud-primary mode only).
                        "ONCALL_ROLE": self._settings.oncall_role,
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
            # high, not medium: at medium the executor garbled its own summary
            # while squeezing into the reply budget (dropped a load-bearing
            # noun, mutated a quoted word, then re-read its own corruption and
            # invented a meaning for it). Hand-offs are off the hot path —
            # the operator has already acked — so the extra thinking is cheap
            # here in a way it would not be on an operator turn.
            "--effort", "high",
            "--append-system-prompt", self._render_executor_prompt(),
            # Persistence is REQUIRED in the single-session model: every
            # hand_off after the first uses --resume to pick up the same
            # session. --no-session-persistence would erase the session
            # the moment the subprocess exited, so the next spawn would
            # fail with "No conversation found".
        ]
        # The built-in tool set is an ALLOWLIST, not a denylist. Beyond the
        # obvious file/shell/web tools, the CLI ships an orchestration surface
        # for its OWN runtime — timers, background jobs, subagents, workflows,
        # plan and worktree modes. All of it is scoped to a `--print` subprocess
        # that exits when this turn ends, and all of it duplicates a mechanism
        # oncall already owns (`mcp__oncall__schedule` for timers, our task
        # table for jobs, `invoke_developer` for delegation).
        #
        # The introspection half is not merely redundant, it is wrong on its
        # face: those tools answer "nothing scheduled" / "no tasks" truthfully
        # about a runtime that has never held any of the user's work, and the
        # executor reports that as an answer about oncall. It cleared a live
        # daily check via CronList; with CronList denied it reached for TaskList
        # and said the same thing. Denying decoys one bug at a time loses to
        # whatever the CLI ships next, so name what the executor may use and let
        # everything else — present and future — default to absent.
        #
        # `--tools` gates built-ins only: every `mcp__oncall__*` tool stays
        # available through --mcp-config regardless of what is listed here.
        tools = ["Read", "Glob", "Grep", "WebFetch", "WebSearch"]
        # Cloud-primary mode: this process runs in a container on a VPS, so the
        # MUTATING tools stay out and local work goes through the
        # `mcp__oncall__laptop` proxy, which runs on the user's laptop. The
        # read-only three are in the list above for both roles: the container is
        # isolated, and Telegram attachments land on the SERVER's disk
        # (~/.oncall/inbound), so without native Read the executor has no way to
        # open them (the laptop proxy looks at the wrong machine).
        if not self._settings.is_server_role:
            tools += ["Bash", "Edit", "Write", "NotebookEdit"]
        # Comma-joined into ONE argv element: --tools is variadic, so passing
        # the names as separate elements would swallow whatever follows.
        argv += ["--tools", ",".join(tools)]
        argv += ["--model", task.model or "sonnet"]
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
        placeholders.

        `{{current_date}}` → UTC datetime captured at spawn — the executor's
        headless `--print` session doesn't get a reliable date in its built-in
        context, so we inject one explicitly. Long-running resumed sessions
        still see the date refresh on each spawn (one per hand_off).

        `{{reply_budget_chars}}` → the length the executor is asked to write
        to. Deliberately BELOW both result_delivery ceilings (voice is the
        tighter one): nothing rewrites the executor now, so the gap is slack
        that absorbs a mild overrun instead of cutting the user's message
        mid-word.

        Then exactly one "# Execution environment" section is appended, per
        role. It is the ONLY place native tools are named: which tool carries
        a shell command or reads a file is the one thing that differs between
        roles, and the base prompt used to answer it too ("use Bash freely")
        — false on the server, where Bash is denied above."""
        text = self._paths.executor_prompt.read_text(encoding="utf-8")
        now = format_utc_now()
        text = text.replace("{{current_date}}", now)
        text = text.replace("{{reply_budget_chars}}", str(EXECUTOR_REPLY_BUDGET_CHARS))
        if self._settings.is_server_role:
            text += (
                "\n\n# Execution environment (cloud)\n\n"
                "You are running in a container on a cloud server, NOT the "
                "user's machine. Your native Bash/Edit/Write tools are "
                "disabled here. Read/Glob/Grep work, but they see the "
                "SERVER's filesystem — not the user's files.\n\n"
                "- Attachments the user sends over Telegram (images, PDFs, "
                "documents) are saved on the server and announced in the "
                "message as `[file attached: <path>]`. `Read` that path "
                "directly — images and PDFs come back inline. Do NOT use the "
                "laptop tool for these; the file does not exist on the "
                "laptop.\n"
                "- For web research and reasoning, use WebSearch / WebFetch "
                "directly.\n"
                "- The laptop is the user's development machine and serves ONLY "
                "their project/development work — their files, repos and local "
                "commands. Everything else you do runs server-side and never "
                "needs it. For that work use the `mcp__oncall__laptop` tool, "
                "which executes on the laptop and ONLY works while it's online. "
                "Use it for quick reads and one-off commands.\n"
                "- For any real CODE change — implementing a feature, fixing a "
                "bug, refactoring, editing files in a project/repo directory — do "
                "NOT hand-drive it through many `laptop` bash/write_file calls. "
                "Delegate the whole job to `mcp__oncall__invoke_developer`: pass a "
                "clear and detailed `task` brief and the absolute `folder` to work in. It runs "
                "an autonomous coding agent on the laptop that edits, runs tests, "
                "and commits on its own. It returns immediately with a `developer_id`; the work runs "
                "asynchronously (can take minutes) and you are notified with a "
                "`<developer-update>` turn when it finishes — do NOT block or poll. "
                "Your in-flight developer jobs are listed in the `<developers>` "
                "block at the top of each turn; never launch a second developer "
                "for a folder+task already running there. If you don't know which "
                "folder the code lives in, check your memories or ask, rather than "
                "guessing.\n"
                "- If a laptop call returns `{\"error\":\"laptop_offline\"}` or "
                "`{\"error\":\"laptop_timeout\"}`, the laptop is unreachable. "
                "State this plainly and stop — do NOT retry in a loop or invent "
                "a result. The work can be redone when the laptop is back."
            )
        else:
            text += (
                "\n\n# Execution environment (local)\n\n"
                "You are running on the user's own machine. Your native tools "
                "act on their filesystem directly: `Bash` carries shell "
                "commands, `Read`/`Glob`/`Grep` read, `Edit`/`Write` change "
                "files. Telegram attachments announced as `[file attached: "
                "<path>]` land on this same machine."
            )
        lang = self._settings.operator_language
        if lang:
            text += (
                f"\n\n# Output language\n\n"
                f"Always respond in: {lang} (ISO-639-1). Tool calls, file "
                f"contents, command output stay as-is; this only affects "
                f"natural-language replies."
            )
        return text

    def _with_developers_block(self, text: str) -> str:
        if self._developers_snapshot_provider is None:
            return text
        try:
            block = self._developers_snapshot_provider() or ""
        except Exception:
            log.warning("developers snapshot provider raised; skipping", exc_info=True)
            return text
        return f"{block}\n\n{text}" if block else text

    async def _write_user_turn(self, text: str) -> None:
        assert self._proc and self._proc.stdin
        # Prepend the current `<developers>` snapshot (in-flight autonomous
        # developer jobs) so the executor sees them and doesn't re-delegate the
        # same work. Best-effort: a provider failure must not block the turn.
        text = self._with_developers_block(text)
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
            # Record live context fill for the compaction guard. `usage` is
            # the final model call; `iterations[-1]` (when present) is that
            # same last call broken out — either way the sum of input +
            # cache_read + cache_creation is how full the window is now.
            usage = evt.get("usage") or {}
            iters = usage.get("iterations") or []
            self._last_context_tokens = _context_tokens(iters[-1] if iters else usage)
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


def _context_tokens(usage: dict[str, Any]) -> int:
    """How full the context window is, from a stream-json usage blob: the
    prompt the model just read = fresh input + cached prefix (read) + newly
    cached prefix (creation). Output tokens don't occupy the next turn's
    window, so they're excluded."""
    if not isinstance(usage, dict):
        return 0
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )


def _parse_compact_result(stdout: bytes) -> tuple[bool, int | None]:
    """Scan a /compact run's stream-json for the success signal. Returns
    (succeeded, post_compaction_tokens). `post_tokens` comes from the
    `compact_boundary` event's metadata when present."""
    ok = False
    after: int | None = None
    for raw in stdout.split(b"\n"):
        if not raw.strip():
            continue
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if evt.get("compact_result") == "success":
            ok = True
        if evt.get("type") == "system" and evt.get("subtype") == "compact_boundary":
            meta = evt.get("compact_metadata") or {}
            after = meta.get("post_tokens")
            ok = True
    return ok, after


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
