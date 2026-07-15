"""Autonomous local coding agent — the laptop side of `invoke_developer`.

Spawns `claude --permission-mode auto` in a folder on the user's laptop to do
file/git work WITHOUT per-action approval, isolated from the oncall MCP /
broker / Telegram. This is the auto-mode tier the executor delegates to: the
executor keeps its human-in-the-loop broker; the developer runs autonomously
inside a sandboxed local directory, governed only by the auto-mode classifier
and the catastrophic deny list in developer/settings.json.

Runs on the laptop worker. Because a real coding session can take many minutes,
the `claude` subprocess runs as a BACKGROUND asyncio task — off the worker's
serial poll loop — and the control plane is three fast ops routed over the
existing laptop bridge:

  developer_start  → spawn the bg task, return a handle in <1s
  developer_wait   → block up to `oncall_developer_wait_seconds`, then return
                     the current status (terminal or still running)
  developer_cancel → cancel the task (kills the process group)

The 30-minute hard cap is authoritative HERE (the worker owns the process), so
it fires even if the server stops polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .config import Paths, Settings, get_paths


log = logging.getLogger(__name__)

# Cap the summary returned over the wire (protects the executor's context).
_MAX_OUTPUT_BYTES = 100_000
# Keep at most this many jobs in memory; evict oldest terminal ones past it.
_MAX_REGISTRY = 32
_TERMINAL: frozenset[str] = frozenset({"completed", "failed", "cancelled", "timeout"})


def _truncate(s: str, limit: int = _MAX_OUTPUT_BYTES) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated, {len(s) - limit} more bytes]"


@dataclass
class _DevJob:
    id: str
    folder: str
    task_text: str
    status: str = "running"          # running | completed | failed | cancelled | timeout
    started_at: float = 0.0
    finished_at: float | None = None
    output: str = ""
    exit_code: int | None = None
    proc: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return {
            "developer_id": self.id,
            "status": self.status,
            "exit_code": self.exit_code,
            "output": self.output,
            "elapsed_s": round(end - self.started_at, 1),
        }


class DeveloperRunner:
    """Owns the laptop's in-flight developer jobs. One instance per worker
    process (the registry must persist across `execute_job` calls). Single
    asyncio loop → no lock, same invariant as LaptopBridge."""

    def __init__(
        self, settings: Settings, paths: Paths | None = None, *, binary: str = "claude",
    ) -> None:
        self._settings = settings
        self._paths = paths or get_paths()
        self._binary = binary
        self._jobs: dict[str, _DevJob] = {}

    # ---- argv (pure; unit-tested) --------------------------------------

    def _build_argv(self) -> list[str]:
        # NOT --bare (breaks OAuth keychain). No --strict-mcp-config/--mcp-config
        # — the developer uses the laptop's normal `claude` environment. The
        # deny list in developer/settings.json survives --permission-mode auto.
        return [
            self._binary,
            "--print",
            "--model", self._settings.oncall_developer_model,
            "--effort", self._settings.oncall_developer_effort,
            "--permission-mode", "auto",
            "--settings", str(self._paths.developer_settings_json),
            "--setting-sources", "project",
            "--append-system-prompt", self._developer_prompt(),
        ]

    def _developer_prompt(self) -> str:
        try:
            return self._paths.developer_prompt.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("developer prompt unreadable (%s); using fallback", e)
            return (
                "You are an autonomous software engineer. Complete the requested "
                "task in your working directory, verify it, and print a concise "
                "summary of what you changed. Do not ask questions; act decisively."
            )

    # ---- control plane (called from execute_job) -----------------------

    def start(self, task_text: str, folder: str) -> dict:
        """Validate the folder, spawn a background `claude`, return a handle
        immediately. Idempotent for an identical (folder, task) already
        running — returns the existing handle instead of a second session."""
        p = Path(folder).expanduser()
        if not p.is_dir():
            return {"error": "folder_not_found", "detail": f"no such directory: {folder}"}
        folder_abs = str(p)
        for job in self._jobs.values():
            if job.status == "running" and job.folder == folder_abs and job.task_text == task_text:
                log.info("developer dedup: reusing running job %s for %s", job.id, folder_abs)
                return {"developer_id": job.id, "status": "already_running"}
        job = _DevJob(
            id=str(uuid4()), folder=folder_abs, task_text=task_text,
            started_at=time.monotonic(),
        )
        job.task = asyncio.create_task(self._run(job))
        job.task.add_done_callback(lambda t, j=job: self._on_done(t, j))
        self._register(job)
        log.info("developer %s started in %s", job.id, folder_abs)
        return {"developer_id": job.id, "status": "running"}

    async def wait(self, developer_id: str) -> dict:
        """Block up to `oncall_developer_wait_seconds` for the job to finish,
        then return its snapshot. Unknown id (e.g. after a worker restart) →
        status 'unknown'."""
        job = self._jobs.get(developer_id)
        if job is None:
            return {
                "developer_id": developer_id, "status": "unknown",
                "detail": "no such developer id (the worker may have restarted)",
            }
        deadline = time.monotonic() + self._settings.oncall_developer_wait_seconds
        while job.status == "running" and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
        return job.snapshot()

    def cancel(self, developer_id: str) -> dict:
        job = self._jobs.get(developer_id)
        if job is None:
            return {"developer_id": developer_id, "status": "unknown"}
        if job.task is not None and not job.task.done():
            job.task.cancel()
        return {"developer_id": developer_id, "status": "cancelled"}

    def kill_all(self) -> None:
        """Kill every running developer's process group. Called on worker
        shutdown so a clean stop doesn't orphan a live `claude`."""
        for job in self._jobs.values():
            if job.status == "running":
                self._killpg(job.proc)

    # ---- internals -----------------------------------------------------

    async def _run(self, job: _DevJob) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._build_argv(),
                cwd=job.folder,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Own process group so we can kill claude AND its children
                # (git, compilers, test runners) as a unit on timeout/cancel.
                start_new_session=True,
            )
            job.proc = proc
        except FileNotFoundError:
            job.status = "failed"
            job.output = "claude CLI not found on PATH on the laptop"
            log.warning("developer %s: claude not on PATH", job.id)
            job.finished_at = time.monotonic()
            return
        except Exception as e:
            job.status = "failed"
            job.output = f"spawn error: {type(e).__name__}: {e}"
            log.exception("developer %s spawn failed", job.id)
            job.finished_at = time.monotonic()
            return

        timeout = self._settings.oncall_developer_timeout_seconds
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(job.task_text.encode("utf-8")),
                timeout=timeout,
            )
            job.exit_code = proc.returncode
            stdout = out.decode("utf-8", "replace")
            stderr = err.decode("utf-8", "replace")
            combined = stdout
            if stderr.strip():
                combined = f"{stdout}\n[stderr]\n{stderr}" if stdout.strip() else stderr
            job.output = _truncate(combined)
            job.status = "completed" if proc.returncode == 0 else "failed"
        except asyncio.TimeoutError:
            self._killpg(proc)
            job.status = "timeout"
            job.output = f"developer exceeded {timeout}s and was killed"
            log.warning("developer %s timed out after %ss; killed process group", job.id, timeout)
        except asyncio.CancelledError:
            self._killpg(proc)
            job.status = "cancelled"
            job.output = "developer was cancelled"
            log.info("developer %s cancelled", job.id)
            raise
        except Exception as e:
            self._killpg(proc)
            job.status = "failed"
            job.output = f"exec error: {type(e).__name__}: {e}"
            log.exception("developer %s crashed", job.id)
        finally:
            job.finished_at = time.monotonic()

    def _on_done(self, t: asyncio.Task, job: _DevJob) -> None:
        # Safety net: a job whose task ended without reaching a terminal status
        # must never appear 'running' forever to a poller.
        if job.status != "running":
            return
        job.finished_at = job.finished_at or time.monotonic()
        if t.cancelled():
            job.status = "cancelled"
            return
        job.status = "failed"
        job.output = job.output or "developer task ended without a result"
        log.error("developer %s task ended while still 'running'; marking failed", job.id)

    def _killpg(self, proc: asyncio.subprocess.Process | None) -> None:
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as e:
            log.warning("developer killpg failed: %s", e)

    def _register(self, job: _DevJob) -> None:
        self._jobs[job.id] = job
        if len(self._jobs) <= _MAX_REGISTRY:
            return
        terminal = sorted(
            (j for j in self._jobs.values() if j.status in _TERMINAL),
            key=lambda j: j.finished_at or 0.0,
        )
        for j in terminal[: len(self._jobs) - _MAX_REGISTRY]:
            self._jobs.pop(j.id, None)
