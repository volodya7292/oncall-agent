"""One-shot subprocess wrapper around the local `claude` CLI.

Used for summarization tasks — chat-context compression and per-task result
summaries — where we want a single text-in / text-out call to a strong model
without paying the Vercel AI Gateway for tokens. The user's `claude` is
already authenticated (keychain OAuth for subscription users, or
ANTHROPIC_API_KEY in env for workspace API users), so this is "free" from
this code's perspective.

Tools and MCP are explicitly disabled — these calls must never run shell
commands, edit files, or escalate to the permission broker. They are pure
summarizers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol


log = logging.getLogger(__name__)


# Every Claude-built-in tool we know about. Block them all so a summarization
# prompt can't be coaxed into doing work. Unknown future tools will still be
# auto-disallowed because we leave `permissions.allow` empty and there's no
# --permission-prompt-tool wired in — the CLI's default-deny is the backstop.
_DISALLOWED_TOOLS = ",".join([
    "Bash", "Read", "Edit", "Write", "Grep", "Glob",
    "NotebookEdit", "WebFetch", "WebSearch", "Task", "TodoWrite",
])


class OneShotRunner(Protocol):
    """Anything that can turn a prompt into a string completion. Tests can
    inject a fake; production uses ClaudeCliRunner."""

    async def one_shot(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str = "sonnet",
        timeout_s: float = 60.0,
    ) -> str | None: ...


class ClaudeCliRunner:
    """Spawns the local `claude` binary in --print mode."""

    def __init__(self, *, binary: str = "claude") -> None:
        self._binary = binary

    async def one_shot(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str = "sonnet",
        timeout_s: float = 60.0,
    ) -> str | None:
        argv = [
            self._binary,
            "--print",
            "--model", model,
            "--max-turns", "1",
            # We intentionally do NOT pass --bare. For OAuth subscription
            # users the auth token lives under ~/.claude; --bare skips that
            # directory and claude exits "Not logged in · Please run /login"
            # with no stderr. Plugin/skill loading is harmless here because
            # --disallowedTools below blocks every tool anyway.
            "--strict-mcp-config",
            # No MCP servers. Claude's --mcp-config schema requires a
            # `mcpServers` key, so an empty `{}` is rejected — must pass an
            # object containing an empty mcpServers map.
            "--mcp-config", '{"mcpServers": {}}',
            "--permission-mode", "default",
            "--disallowedTools", _DISALLOWED_TOOLS,
        ]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.warning(
                "claude CLI not on PATH; skipping one-shot summarization."
                " Install it from https://docs.claude.com/en/docs/claude-code"
            )
            return None
        except Exception:
            log.exception("failed to spawn claude CLI for one-shot")
            return None

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning("claude one-shot timed out after %.0fs; killing subprocess", timeout_s)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            return None

        if proc.returncode != 0:
            log.warning(
                "claude one-shot exited %s: %s",
                proc.returncode, stderr_bytes.decode("utf-8", errors="replace")[:200],
            )
            return None

        text = stdout_bytes.decode("utf-8", errors="replace").strip()
        return text or None
