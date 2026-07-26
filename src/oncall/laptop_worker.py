"""`oncall laptop-worker` — the laptop half of cloud-primary mode.

The orchestrator runs on an always-on server; this worker runs on the user's
laptop and provides local shell/file capabilities to the server's executor.
It reaches the server ONLY via outbound HTTPS long-poll, so it works behind
NAT, through IP changes, and across sleep with no inbound exposure:

    loop:
      GET  {server}/laptop/jobs        (long-poll; also the presence heartbeat)
      → run the job locally (bounded, with a catastrophic-command backstop)
      POST {server}/laptop/jobs/{id}/result

Security: every mutating job was already gated by the server-side broker
(classifier → user approval) BEFORE it was dispatched. The deny-list backstop
here is defense in depth — it refuses catastrophic commands even if the server
were compromised, mirroring the executor's own settings.json deny list.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .classifier import catastrophic_reason
from .config import Settings
from .developer_runner import DeveloperRunner


log = logging.getLogger(__name__)

# Caps so a runaway command can't return a multi-megabyte payload over the
# wire (and blow the executor's context window).
_MAX_OUTPUT_BYTES = 100_000
_MAX_FILE_BYTES = 1_000_000
_MAX_GLOB_RESULTS = 1000
# Per-job execution ceiling. Kept below the server's job timeout (default
# 300s) so the worker returns a result before the server gives up on it.
_JOB_EXEC_TIMEOUT_S = 240


def _truncate(s: str, limit: int = _MAX_OUTPUT_BYTES) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated, {len(s) - limit} more bytes]"


async def _run_bash(command: str) -> dict[str, Any]:
    # Backstop: refuse catastrophic shapes regardless of upstream gating.
    # This is the deterministic half of the classifier only — no model call,
    # so the worker keeps working with no credentials and no reachable LLM.
    reason = catastrophic_reason(command)
    if reason:
        return {"error": "blocked_catastrophic", "detail": reason}
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return {"error": f"spawn_failed: {type(e).__name__}: {e}"}
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_JOB_EXEC_TIMEOUT_S)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return {"error": "exec_timeout", "detail": f"command exceeded {_JOB_EXEC_TIMEOUT_S}s"}
    return {
        "stdout": _truncate(out.decode("utf-8", "replace")),
        "stderr": _truncate(err.decode("utf-8", "replace")),
        "exit_code": proc.returncode,
    }


def _run_read_file(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        return {"error": "not_found", "detail": f"no such file: {path}"}
    except OSError as e:
        return {"error": "read_failed", "detail": f"{type(e).__name__}: {e}"}
    if len(data) > _MAX_FILE_BYTES:
        return {
            "error": "too_large",
            "detail": f"{len(data)} bytes exceeds {_MAX_FILE_BYTES} cap",
        }
    return {"content": data.decode("utf-8", "replace")}


def _run_write_file(path: str, content: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return {"error": "write_failed", "detail": f"{type(e).__name__}: {e}"}
    return {"ok": True, "bytes_written": len(content.encode("utf-8"))}


def _run_glob(pattern: str, base: str | None) -> dict[str, Any]:
    root = Path(base).expanduser() if base else Path.cwd()
    try:
        paths = [str(p) for p in root.glob(pattern)]
    except (OSError, ValueError) as e:
        return {"error": "glob_failed", "detail": f"{type(e).__name__}: {e}"}
    truncated = len(paths) > _MAX_GLOB_RESULTS
    return {"paths": paths[:_MAX_GLOB_RESULTS], "truncated": truncated}


async def _run_grep(pattern: str, base: str | None) -> dict[str, Any]:
    """Search file contents. Prefer ripgrep when present, else grep -rn."""
    target = base or "."
    if _which("rg"):
        argv = ["rg", "--line-number", "--no-heading", "--color=never", "--", pattern, target]
    else:
        argv = ["grep", "-rn", "--", pattern, target]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_JOB_EXEC_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"error": "exec_timeout"}
    except Exception as e:
        return {"error": f"grep_failed: {type(e).__name__}: {e}"}
    # grep/rg exit 1 = "no matches" — not an error.
    return {
        "matches": _truncate(out.decode("utf-8", "replace")),
        "exit_code": proc.returncode,
        "stderr": _truncate(err.decode("utf-8", "replace"), 4000),
    }


def _which(prog: str) -> bool:
    return any(
        os.access(os.path.join(d, prog), os.X_OK)
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if d
    )


async def execute_job(
    job: dict[str, Any], runner: DeveloperRunner | None = None,
) -> dict[str, Any]:
    """Run one job descriptor `{id, kind, input}` and return its result dict.

    `runner` owns the developer-job registry; it must be the same instance
    across calls (created once in run_worker), so the developer_* control-plane
    ops share state. The fast ops ignore it."""
    kind = job.get("kind")
    inp = job.get("input") or {}
    if kind == "bash":
        return await _run_bash(str(inp.get("command", "")))
    if kind == "read_file":
        return _run_read_file(str(inp.get("path", "")))
    if kind == "write_file":
        return _run_write_file(str(inp.get("path", "")), str(inp.get("content", "")))
    if kind == "glob":
        return _run_glob(str(inp.get("pattern", "")), inp.get("path"))
    if kind == "grep":
        return await _run_grep(str(inp.get("pattern", "")), inp.get("path"))
    if kind in ("developer_start", "developer_wait", "developer_cancel"):
        if runner is None:
            return {"error": "developer_unavailable", "detail": "no developer runner"}
        if kind == "developer_start":
            return runner.start(str(inp.get("task", "")), str(inp.get("folder", "")))
        if kind == "developer_wait":
            return await runner.wait(str(inp.get("developer_id", "")))
        return runner.cancel(str(inp.get("developer_id", "")))
    return {"error": "unknown_kind", "detail": f"worker can't run '{kind}'"}


async def run_worker(settings: Settings) -> None:
    """Long-poll loop. Self-restarting with capped backoff so the worker
    survives network loss / laptop sleep and reconnects transparently."""
    server = settings.oncall_server_url.rstrip("/")
    token = settings.oncall_laptop_token
    if not server or not token:
        raise SystemExit(
            "laptop-worker needs ONCALL_SERVER_URL and ONCALL_LAPTOP_TOKEN set "
            "in ~/.oncall/.env (the server's public URL + its laptop token)."
        )
    headers = {"X-Oncall-Laptop-Token": token}
    # One runner for the whole worker lifetime — it holds the in-flight
    # developer-job registry, which must persist across execute_job calls.
    runner = DeveloperRunner(settings)
    # Read timeout must outlast the server's long-poll hold.
    read_timeout = settings.oncall_laptop_poll_timeout_seconds + 20
    timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0)
    backoff = 1.0
    # Outage tracking so recovery is as loud as failure: monotonic ts of the
    # first failed poll in the current outage + how many polls failed in it.
    down_since: float | None = None
    failed_polls = 0
    log.info("laptop-worker polling %s", server)

    def _mark_reconnected() -> None:
        nonlocal down_since, failed_polls
        if down_since is not None:
            log.info(
                "laptop-worker reconnected to %s after %.0fs offline (%d failed polls)",
                server, time.monotonic() - down_since, failed_polls,
            )
            down_since = None
            failed_polls = 0

    def _mark_failed() -> None:
        nonlocal down_since, failed_polls
        if down_since is None:
            down_since = time.monotonic()
        failed_polls += 1

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            try:
                r = await client.get(f"{server}/laptop/jobs", headers=headers)
                if r.status_code == 401:
                    _mark_failed()
                    log.error("laptop-worker auth rejected (check ONCALL_LAPTOP_TOKEN); backing off")
                    await asyncio.sleep(30)
                    continue
                r.raise_for_status()
                _mark_reconnected()
                backoff = 1.0
                job = r.json().get("job")
                if job is None:
                    continue  # idle long-poll elapsed; re-poll immediately
                log.info("running job %s (kind=%s)", job.get("id"), job.get("kind"))
                result = await execute_job(job, runner)
                await client.post(
                    f"{server}/laptop/jobs/{job['id']}/result",
                    json={"result": result},
                    headers=headers,
                )
            except asyncio.CancelledError:
                # Clean shutdown: kill any live developer sessions so we don't
                # orphan a `claude` running detached in its own process group.
                runner.kill_all()
                raise
            except httpx.ReadTimeout:
                # Expected: the long-poll held past our read window with no
                # job. The connection itself succeeded, so this also ends an
                # outage.
                _mark_reconnected()
                backoff = 1.0
                continue
            except Exception as e:
                _mark_failed()
                log.warning("laptop-worker poll error (%s: %s); retrying in %.0fs",
                            type(e).__name__, e, min(backoff, 30))
                await asyncio.sleep(min(backoff, 30))
                backoff *= 2


def cli_main() -> None:
    from .config import get_settings
    asyncio.run(run_worker(get_settings()))
