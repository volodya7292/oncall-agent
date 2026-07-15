"""Server-side bridge to the laptop's local capabilities (cloud-primary mode).

In `ONCALL_ROLE=server` deployments the orchestrator runs on an always-on VPS
with no useful local filesystem of its own. The executor's local shell/file
work is instead routed to the user's laptop, which runs `oncall laptop-worker`
and reaches the server only via OUTBOUND HTTPS long-poll (NAT-friendly).

This module is the in-memory rendezvous between the two halves:

    executor claude
      └─ mcp__oncall__laptop(op=bash, …)        [mcp_server.py proxy]
           └─ POST /internal/laptop/dispatch     [api.py, loopback]
                └─ LaptopBridge.dispatch(...)     ← blocks on a Future
    laptop worker
      └─ GET  /laptop/jobs   (long-poll)          → LaptopBridge.next_job(...)
      └─ POST /laptop/jobs/{id}/result            → LaptopBridge.submit_result(...)
                                                     ← resolves the Future

State is intentionally in-memory (like HttpLongPollApprovalClient and the
operator's ask_futures): a job is an ephemeral RPC. If the daemon restarts
mid-task the executor session is killed anyway (see lifecycle.recover()), so
there is nothing to durably resume — a DB table would add complexity for no
recoverable state.

Presence: the laptop is ONLINE iff it has long-polled within
`presence_window_s`. Each poll AND each result post refreshes last-seen.

Single asyncio loop only; not safe across loops/processes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


log = logging.getLogger(__name__)


# Local capability ops the worker knows how to execute. Kept here so the
# proxy tool, the dispatch route, and the worker agree on one list.
LAPTOP_OPS: frozenset[str] = frozenset({
    "bash", "read_file", "write_file", "glob", "grep",
})

# Developer control-plane ops (invoke_developer). These ride the same bridge
# but are NOT part of the `laptop` tool's op enum — they're reachable only via
# the dedicated invoke_developer / cancel_developer MCP tools and the
# server-side DeveloperManager. All three are fast (start/cancel return
# immediately; wait blocks ≤ oncall_developer_wait_seconds, well under the
# bridge job timeout).
DEVELOPER_OPS: frozenset[str] = frozenset({
    "developer_start", "developer_wait", "developer_cancel",
})


@dataclass
class _Job:
    id: str
    kind: str
    input: dict[str, Any]
    future: "asyncio.Future[dict[str, Any]]"
    # True once a worker poll has claimed it (moved from queue to in-flight).
    claimed: bool = field(default=False)


class LaptopBridge:
    """Queues local-tool jobs for the laptop worker and blocks the executor
    on each until the worker posts a result (or a timeout fires)."""

    def __init__(
        self,
        *,
        presence_window_s: float,
        poll_timeout_s: float,
        job_timeout_s: float,
    ) -> None:
        self._presence_window_s = presence_window_s
        self._poll_timeout_s = poll_timeout_s
        self._job_timeout_s = job_timeout_s
        self._pending: "asyncio.Queue[_Job]" = asyncio.Queue()
        self._inflight: dict[str, _Job] = {}
        # Monotonic loop time of the most recent worker contact. None = the
        # worker has never been seen since this daemon started.
        self._last_seen: float | None = None

    # ---- presence -------------------------------------------------------

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def mark_seen(self) -> None:
        self._last_seen = self._now()

    def is_online(self) -> bool:
        if self._last_seen is None:
            return False
        return (self._now() - self._last_seen) <= self._presence_window_s

    def seconds_since_seen(self) -> float | None:
        if self._last_seen is None:
            return None
        return self._now() - self._last_seen

    # ---- executor side (POST /internal/laptop/dispatch) -----------------

    async def dispatch(self, kind: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Enqueue one local job and block until the worker returns its
        result. Returns the worker's result dict, or an `{"error": ...}` dict
        on offline / timeout — never raises for those expected cases, so the
        executor sees a tool error it can reason about rather than a hang."""
        if kind not in (LAPTOP_OPS | DEVELOPER_OPS):
            return {"error": "unknown_laptop_op", "detail": f"unknown op '{kind}'"}
        if not self.is_online():
            return {
                "error": "laptop_offline",
                "detail": (
                    "The user's laptop is offline; local shell/file tools are "
                    "unavailable until it reconnects. Do not retry in a loop."
                ),
            }
        loop = asyncio.get_running_loop()
        job = _Job(id=str(uuid4()), kind=kind, input=tool_input, future=loop.create_future())
        await self._pending.put(job)
        log.info("laptop job %s queued (kind=%s)", job.id, kind)
        try:
            return await asyncio.wait_for(job.future, timeout=self._job_timeout_s)
        except asyncio.TimeoutError:
            log.warning("laptop job %s timed out after %ss", job.id, self._job_timeout_s)
            return {
                "error": "laptop_timeout",
                "detail": (
                    f"The laptop did not return a result within "
                    f"{int(self._job_timeout_s)}s (it may have gone offline "
                    f"mid-job). The action's outcome is unknown."
                ),
            }
        finally:
            # Stop tracking either way. A never-claimed job left in the queue
            # is skipped by next_job() because its future is now done; an
            # in-flight job is dropped here so a late result is ignored.
            self._inflight.pop(job.id, None)

    # ---- worker side (GET /laptop/jobs) ---------------------------------

    async def next_job(self) -> dict[str, Any] | None:
        """Long-poll for the next claimable job. Blocks up to poll_timeout_s,
        then returns None so the worker re-polls (keeping presence fresh).
        Returns a JSON-serializable descriptor `{id, kind, input}`."""
        self.mark_seen()
        deadline = self._now() + self._poll_timeout_s
        while True:
            remaining = deadline - self._now()
            if remaining <= 0:
                return None
            try:
                job = await asyncio.wait_for(self._pending.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            # The executor may have already given up on this job (dispatch
            # timed out → future cancelled). Skip stale jobs and keep waiting.
            if job.future.done():
                continue
            job.claimed = True
            self._inflight[job.id] = job
            self.mark_seen()
            return {"id": job.id, "kind": job.kind, "input": job.input}

    # ---- worker side (POST /laptop/jobs/{id}/result) --------------------

    def submit_result(self, job_id: str, result: dict[str, Any]) -> bool:
        """Resolve the executor's blocked dispatch with the worker's result.
        Returns True if a job was waiting, False if it was unknown/already
        resolved (late post after a timeout)."""
        self.mark_seen()
        job = self._inflight.pop(job_id, None)
        if job is None or job.future.done():
            return False
        job.future.set_result(result)
        return True
