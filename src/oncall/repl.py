"""Interactive text mode for the on-call agent.

A thin HTTP client to the running orchestrator (either the launchd daemon or a
dev-mode `oncall api` in another tab). Tasks survive REPL exit; Telegram keeps
flowing; multiple terminals can attach to the same orchestrator simultaneously.

The REPL runs two cooperating asyncio tasks:
  * input loop — prompt_toolkit's PromptSession (under patch_stdout so the
    SSE printer doesn't corrupt the prompt). Sends user lines to `POST /chat`.
  * event loop — streams the global `GET /events` SSE feed and prints each
    relevant envelope above the prompt. Reconnects with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

from .config import USER_CONFIG_DIR, Settings


log = logging.getLogger(__name__)


SESSION_FILE = USER_CONFIG_DIR / "last_session"
HISTORY_FILE = USER_CONFIG_DIR / "repl_history"

DEFAULT_EVENT_TYPES = (
    "approval.requested",
    "approval.resolved",
    "result.final",
    "messenger.received",
    "chat.reply",
)
# state.changed is added in --debug mode (very chatty otherwise).


# ---------------------------------------------------------------------------
# Pure helpers (heavily unit-tested)
# ---------------------------------------------------------------------------

@dataclass
class SlashCommand:
    name: str            # "new" | "session" | "help" | "quit"
    arg: str | None      # for /session <id>


def parse_slash(line: str) -> SlashCommand | None:
    """Return a SlashCommand iff the line is a recognized /command, else None."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    head, _, rest = stripped[1:].partition(" ")
    head = head.lower()
    arg = rest.strip() or None
    if head in {"quit", "exit", "q"}:
        return SlashCommand(name="quit", arg=None)
    if head == "new":
        return SlashCommand(name="new", arg=None)
    if head == "session":
        return SlashCommand(name="session", arg=arg)
    if head in {"help", "h", "?"}:
        return SlashCommand(name="help", arg=None)
    # Unknown /commands are surfaced as help so the user notices the typo.
    return SlashCommand(name="help", arg=head)


def format_event(
    envelope: dict[str, Any], *, debug: bool = False, session_id: str | None = None,
) -> str | None:
    """Render one global-SSE envelope as a single line for the REPL, or None
    to silently drop the event. Pure function — no I/O, no terminal codes.

    `session_id`, when provided, is the REPL's active chat session. `chat.reply`
    events are filtered to this session_id (so multiple REPLs don't echo each
    other's auto-pings)."""
    type_ = envelope.get("type", "")
    payload = envelope.get("payload") or {}
    task_id = envelope.get("task_id") or ""
    short = task_id[:8] if isinstance(task_id, str) else ""

    if type_ == "approval.requested":
        canonical = (payload.get("canonical_command") or "").strip()
        phrase = payload.get("challenge_phrase") or ""
        approval_id = (payload.get("approval_id") or payload.get("id") or "")[:8]
        return (
            f"! approval {approval_id} task {short}: {canonical}"
            f" — say \"{phrase}\""
        )
    if type_ == "approval.resolved":
        # Auto-allowed readonly tools are not user-actionable; suppress them
        # unless --debug. The user only needs to see *their own* approvals
        # being accepted/denied, not every `ls`/`grep` the executor runs.
        if payload.get("auto") and payload.get("decision") == "allow" and not debug:
            return None
        decision = payload.get("decision") or "?"
        return f"~ approval task {short}: {decision}"
    if type_ == "result.final":
        if payload.get("is_error"):
            return f"* task {short} failed"
        return f"* task {short} done"
    if type_ == "messenger.received":
        sender = (
            payload.get("sender_username")
            or payload.get("sender_display_name")
            or "?"
        )
        body = payload.get("body") or ""
        if len(body) > 120:
            body = body[:117] + "..."
        return f"# DM from @{sender}: {body}"
    if type_ == "chat.reply":
        # Auto-ping reply from the operator. Filter to this REPL's session so
        # parallel terminals don't echo one another.
        if session_id is not None and payload.get("session_id") != session_id:
            return None
        return payload.get("text") or ""
    if type_ == "state.changed" and debug:
        state = payload.get("state") or "?"
        return f". task {short} → {state}"
    return None


async def parse_sse_lines(
    line_iter: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Minimal SSE parser: groups `data:` lines per blank-line-delimited event,
    skips `:` comments. Yields decoded JSON dicts; non-JSON `data` payloads are
    silently skipped."""
    buf: list[str] = []
    async for raw in line_iter:
        line = raw.rstrip("\r")
        if line == "":
            if buf:
                data = "\n".join(buf)
                buf = []
                try:
                    yield json.loads(data)
                except (ValueError, TypeError):
                    log.debug("dropping non-JSON SSE payload: %r", data[:80])
            continue
        if line.startswith(":"):
            continue  # comment / keepalive
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" "))
        # Ignore `event:` / `id:` / `retry:` fields — we encode type in the JSON.


def read_session(path: Path = SESSION_FILE) -> str | None:
    """Return the persisted last session_id, or None if absent/malformed."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return raw or None


def write_session(session_id: str, path: Path = SESSION_FILE) -> None:
    """Persist last session_id (mode 0600). Best-effort; failures are logged."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish: write tmp + rename.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(session_id, encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        log.exception("failed to persist session id")


HELP_TEXT = (
    "Slash commands:\n"
    "  /new                start a fresh chat session\n"
    "  /session <id>       resume a specific chat session\n"
    "  /help               show this\n"
    "  /quit, /exit, Ctrl-D  exit the REPL\n"
    "Anything else is sent as a chat turn to the operator."
)


# ---------------------------------------------------------------------------
# I/O: REPL run loop
# ---------------------------------------------------------------------------

async def run(
    settings: Settings,
    *,
    new_session: bool = False,
    session_override: str | None = None,
    debug: bool = False,
) -> int:
    """Entry point. Returns a Unix exit code."""
    # Local imports so `import oncall.repl` doesn't drag prompt_toolkit in for
    # the test suite or the api process.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout

    base_url = f"http://127.0.0.1:{settings.oncall_port}"
    token = settings.oncall_token
    headers = {"X-Oncall-Token": token}

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=None) as client:
        if not await _healthz_ok(client):
            print(
                f"oncall daemon not reachable at {base_url}. Start it first:\n"
                f"  oncall service start    (macOS LaunchAgent)\n"
                f"  oncall api              (foreground, dev mode)",
                file=sys.stderr,
            )
            return 2

        # Resolve session id.
        session_id = _resolve_session(
            new=new_session, override=session_override,
        )
        print(f"oncall chat — session {session_id[:8]}. Type /help for commands.")

        # Background SSE reader.
        event_types = list(DEFAULT_EVENT_TYPES)
        if debug:
            event_types.append("state.changed")
        sse_task = asyncio.create_task(
            _sse_reader(
                base_url, headers, types=event_types,
                debug=debug, session_id=session_id,
            )
        )

        # Input loop.
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            HISTORY_FILE.touch(exist_ok=True)
            os.chmod(HISTORY_FILE, 0o600)
        except OSError:
            pass
        prompt_session: PromptSession = PromptSession(history=FileHistory(str(HISTORY_FILE)))

        try:
            with patch_stdout(raw=True):
                while True:
                    try:
                        text = await prompt_session.prompt_async("oncall> ")
                    except (EOFError, KeyboardInterrupt):
                        print()
                        return 0
                    text = text.strip()
                    if not text:
                        continue
                    cmd = parse_slash(text)
                    if cmd is not None:
                        if cmd.name == "quit":
                            return 0
                        if cmd.name == "help":
                            if cmd.arg:
                                print(f"unknown command: /{cmd.arg}")
                            print(HELP_TEXT)
                            continue
                        if cmd.name == "new":
                            session_id = str(uuid4())
                            write_session(session_id)
                            print(f"new session {session_id[:8]}")
                            continue
                        if cmd.name == "session":
                            if not cmd.arg:
                                print("usage: /session <id>")
                                continue
                            session_id = cmd.arg
                            write_session(session_id)
                            print(f"resumed session {session_id[:8]}")
                            continue
                    # Plain chat turn.
                    reply = await _send_chat(client, session_id=session_id, text=text)
                    if reply is None:
                        continue
                    write_session(session_id)
                    print(reply)
        finally:
            sse_task.cancel()
            try:
                await sse_task
            except (asyncio.CancelledError, Exception):
                pass


def _resolve_session(*, new: bool, override: str | None) -> str:
    if override:
        write_session(override)
        return override
    if new:
        sid = str(uuid4())
        write_session(sid)
        return sid
    existing = read_session()
    if existing:
        return existing
    sid = str(uuid4())
    write_session(sid)
    return sid


async def _healthz_ok(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get("/healthz", timeout=2.0)
    except (httpx.HTTPError, OSError):
        return False
    return r.status_code == 200


async def _send_chat(
    client: httpx.AsyncClient, *, session_id: str, text: str,
) -> str | None:
    try:
        r = await client.post("/chat", json={"session_id": session_id, "text": text})
    except httpx.HTTPError as e:
        print(f"chat failed: {e}", file=sys.stderr)
        return None
    if r.status_code == 401:
        print("auth failed (401). Check ONCALL_TOKEN in ~/.oncall/.env.", file=sys.stderr)
        return None
    if r.status_code == 503:
        print("operator not configured: set AI_GATEWAY_API_KEY in ~/.oncall/.env.", file=sys.stderr)
        return None
    if r.status_code >= 400:
        print(f"chat error {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    body = r.json()
    return body.get("text") or "(empty reply)"


async def _sse_reader(
    base_url: str, headers: dict[str, str], *, types: list[str], debug: bool,
    session_id: str | None = None,
) -> None:
    """Reconnecting SSE consumer for GET /events. Prints formatted lines.
    Backoff: 1s, 2s, 4s, ... capped at 30s. Resets on a clean connection."""
    backoff = 1.0
    params = {"types": ",".join(types)} if types else {}
    while True:
        try:
            async with httpx.AsyncClient(
                base_url=base_url, headers=headers, timeout=None,
            ) as client:
                async with client.stream("GET", "/events", params=params) as r:
                    if r.status_code == 401:
                        print("(events) auth failed; check ONCALL_TOKEN", file=sys.stderr)
                        return
                    if r.status_code != 200:
                        raise httpx.HTTPError(f"GET /events: {r.status_code}")
                    backoff = 1.0
                    async for envelope in parse_sse_lines(r.aiter_lines()):
                        line = format_event(envelope, debug=debug, session_id=session_id)
                        if line is not None:
                            # patch_stdout in run() routes this above the prompt.
                            print(line)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("events stream lost: %s; reconnecting in %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
