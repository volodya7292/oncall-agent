"""Task lifecycle controller.

Owns the single in-flight executor subprocess, accepts new hand_off
submissions onto a FIFO queue, performs crash recovery on startup, and
routes kill requests. All executor runs are serialized through one worker
so they share the same global claude --session-id without racing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from .approval_client import HttpLongPollApprovalClient
from .broker import Broker
from .config import Paths, Settings
from .db import Database
from .events import EventBus
from .models import Task, TaskState, TerminalReason
from .supervisor import Supervisor


log = logging.getLogger(__name__)


@dataclass
class RunningTask:
    task: Task
    supervisor: Supervisor
    runner: asyncio.Task[TerminalReason]


@dataclass
class Lifecycle:
    db: Database
    broker: Broker
    approval_client: HttpLongPollApprovalClient
    events: EventBus
    settings: Settings
    paths: Paths
    running: dict[UUID, RunningTask] = field(default_factory=dict)
    _queue: "asyncio.Queue[Task] | None" = field(default=None, init=False, repr=False)
    _worker_task: "asyncio.Task[None] | None" = field(default=None, init=False, repr=False)
    _current_task_id: UUID | None = field(default=None, init=False, repr=False)
    _shutdown_flag: "asyncio.Event | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue()
        self._shutdown_flag = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="executor-worker")

    async def submit_task(
        self, *, prompt: str,
        chat_session_id: str | None = None,
        restricted_to_chat: str | None = None,
        model: str | None = None,
    ) -> Task:
        """Build, persist, queue. Returns the constructed Task so callers
        that need state (deferred dispatch approval) can read it.
        `enqueue_executor` is a thin wrapper around this for callers that
        only need the lightweight dict response."""
        task = Task(
            session_id=str(uuid4()),  # per-row id for DB tracking only
            prompt=prompt,
            dispatched_by_chat_session=chat_session_id,
            restricted_to_chat=restricted_to_chat,
            model=model,
        )
        await self.db.insert_task(task)
        await self.events.publish(task.id, "state.changed", {"state": task.state.value})
        assert self._queue is not None
        await self._queue.put(task)
        return task

    async def enqueue_executor(
        self, *, prompt: str, chat_session_id: str | None = None,
        restricted_to_chat: str | None = None,
    ) -> dict[str, object]:
        """Programmatic entry point for the operator's `hand_off()` tool.
        Creates a Task row, pushes it onto the FIFO queue, and returns
        immediately. The single executor worker drains the queue.

        `restricted_to_chat` propagates the inbox-drain's "this task is a
        reply to chat X" constraint to the broker, which uses it as the
        scoping key for dm_allowlist auto-approval.
        """
        task = await self.submit_task(
            prompt=prompt,
            chat_session_id=chat_session_id,
            restricted_to_chat=restricted_to_chat,
        )
        return {
            "task_id": str(task.id),
            "queue_depth": self._queue.qsize() if self._queue is not None else 0,
            "busy": self._current_task_id is not None,
        }

    def acting_status(self) -> dict[str, object]:
        """Snapshot used by the operator's turn-time `<acting-status>`
        injection. Reports whether the worker is mid-task and how many
        items are waiting. Cheap; no I/O."""
        return {
            "busy": self._current_task_id is not None,
            "queue_depth": self._queue.qsize() if self._queue is not None else 0,
            "current_task_id": str(self._current_task_id) if self._current_task_id else None,
        }

    async def recover(self) -> None:
        """On boot, re-enqueue any tasks left in {pending}, mark stale
        RUNNING/AWAITING_APPROVAL as failed (the prior daemon's worker
        died mid-turn; with a single shared session we can't safely
        resume two), and sweep orphan approvals."""
        # Stale in-flight: a prior daemon was running this; we can't
        # cleanly resume because the new shared-session model doesn't
        # allow concurrent re-attach against an unknown number of
        # interrupted turns. Mark them failed so the queue starts clean.
        stale = await self.db.list_tasks_in_states(
            TaskState.RUNNING, TaskState.AWAITING_APPROVAL,
        )
        for task in stale:
            log.warning(
                "recover: marking stale task %s (state=%s) as FAILED — "
                "single-session model can't safely resume mid-turn",
                task.id, task.state.value,
            )
            await self.db.update_task_state(
                task.id, TaskState.FAILED, TerminalReason.KILLED,
            )
            await self.events.publish(task.id, "state.changed", {
                "state": TaskState.FAILED.value, "terminal_reason": "killed",
            })
        # Queued before the crash: still valid to run, just push them
        # back onto the queue in submission order.
        pending = await self.db.list_tasks_in_states(TaskState.PENDING)
        assert self._queue is not None
        for task in pending:
            log.info("recover: re-queueing pending task %s", task.id)
            await self._queue.put(task)
        await self._sweep_orphan_approvals()

    async def _sweep_orphan_approvals(self) -> None:
        """Resolve any pending approval whose parent task is in a terminal
        state. Called on boot and from `_run`'s finally so transient
        terminations (supervisor crash, kill) don't leave the count stuck."""
        from .models import ApprovalResult, utcnow
        try:
            pending = await self.db.list_pending_approvals()
        except Exception:
            log.exception("orphan-approval sweep: list_pending_approvals failed")
            return
        for approval in pending:
            try:
                task = await self.db.get_task(approval.task_id)
            except Exception:
                log.warning(
                    "orphan-approval sweep: get_task failed for %s",
                    approval.task_id,
                )
                continue
            if task is None or task.state not in (
                TaskState.FAILED, TaskState.KILLED, TaskState.COMPLETED,
            ):
                continue
            log.info(
                "orphan-approval sweep: resolving %s (task %s state=%s)",
                approval.id, approval.task_id, task.state.value,
            )
            result = ApprovalResult(
                request_id=approval.id,
                behavior="deny",
                message=f"orphaned: task terminated in state={task.state.value}",
                responded_at=utcnow(),
            )
            try:
                if self.approval_client.has_pending(approval.id):
                    self.approval_client.resolve(approval.id, result)
                await self.db.append_approval_response(approval.id, result)
            except Exception:
                log.exception(
                    "orphan-approval sweep: failed to resolve %s",
                    approval.id,
                )

    async def kill(self, task_id: UUID, *, reason: str = "killed") -> bool:
        rt = self.running.get(task_id)
        if rt is None:
            return False
        # If there's a pending approval future, resolve it as deny first so
        # the broker's await returns immediately. Also write the resolved
        # row to the DB ourselves rather than relying on the broker's
        # post-await commit — during shutdown the broker may not survive
        # long enough to do it.
        for approval in await self.db.list_pending_approvals():
            if approval.task_id != task_id:
                continue
            from .models import ApprovalResult, utcnow
            result = ApprovalResult(
                request_id=approval.id,
                behavior="deny",
                message=f"task killed: {reason}",
                responded_at=utcnow(),
            )
            if self.approval_client.has_pending(approval.id):
                self.approval_client.resolve(approval.id, result)
            await self.db.append_approval_response(approval.id, result)
        rt.runner.cancel()
        try:
            await rt.runner
        except (asyncio.CancelledError, Exception):
            pass
        await self.db.update_task_state(task_id, TaskState.KILLED, TerminalReason.KILLED)
        await self.events.publish(task_id, "state.changed", {
            "state": TaskState.KILLED.value, "reason": reason,
        })
        return True

    async def shutdown(self) -> None:
        """Graceful daemon shutdown. Set the flag so the worker exits
        at its next queue.get() / iteration boundary, cancel any
        in-flight runner so the current await unblocks. Tasks
        mid-flight stay in their current DB state — on next boot
        recover() marks them FAILED."""
        if self._shutdown_flag is not None:
            self._shutdown_flag.set()
        # Cancel runners (current + any future ones the worker might
        # spawn before noticing the flag) so awaits unblock.
        for rt in list(self.running.values()):
            if not rt.runner.done():
                rt.runner.cancel()
        # Cancel worker too — between draining the queue and re-checking
        # the flag, it could be parked at queue.get() with no runner to
        # ricochet through.
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self._worker_task = None
        for task_id, rt in list(self.running.items()):
            try:
                await rt.runner
            except (asyncio.CancelledError, Exception):
                pass
            self.running.pop(task_id, None)
            log.info("shutdown: cancelled runner for task %s", task_id)

    # ---- internals ----

    async def _worker_loop(self) -> None:
        """Single FIFO worker: drains the queue one task at a time so all
        executor runs share the global session id without racing.
        Exits when the shutdown flag is set or the task is cancelled."""
        assert self._queue is not None
        assert self._shutdown_flag is not None
        get_q = asyncio.ensure_future(self._queue.get())
        flag_wait = asyncio.ensure_future(self._shutdown_flag.wait())
        try:
            while True:
                if self._shutdown_flag.is_set():
                    return
                done, _ = await asyncio.wait(
                    {get_q, flag_wait}, return_when=asyncio.FIRST_COMPLETED,
                )
                if flag_wait in done:
                    return
                task = get_q.result()
                get_q = asyncio.ensure_future(self._queue.get())
                try:
                    self._current_task_id = task.id
                    await self._run_one(task)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("worker loop: unexpected error running task %s", task.id)
                finally:
                    self._current_task_id = None
        finally:
            for fut in (get_q, flag_wait):
                if not fut.done():
                    fut.cancel()

    async def _run_one(self, task: Task) -> None:
        sup = Supervisor(db=self.db, events=self.events, settings=self.settings, paths=self.paths)
        runner = asyncio.create_task(self._supervise(task, sup))
        self.running[task.id] = RunningTask(task=task, supervisor=sup, runner=runner)
        try:
            try:
                await runner
            except asyncio.CancelledError:
                # Worker cancelled mid-task. Drag the child runner down
                # with us so it doesn't pin the loop as an orphan.
                if not runner.done():
                    runner.cancel()
                    try:
                        await runner
                    except (asyncio.CancelledError, Exception):
                        pass
                raise
        finally:
            self.running.pop(task.id, None)
            # Best-effort, swallow cancellation so the cleanup doesn't
            # propagate a second cancel into surrounding code.
            try:
                await self._sweep_orphan_approvals()
            except asyncio.CancelledError:
                pass

    async def _supervise(self, task: Task, sup: Supervisor) -> TerminalReason:
        try:
            return await sup.run(task, resuming=False)
        except asyncio.CancelledError:
            return TerminalReason.KILLED
        except Exception:
            log.exception("supervisor crashed for task %s", task.id)
            await self.db.update_task_state(task.id, TaskState.FAILED, TerminalReason.CLI_ERROR)
            await self.events.publish(task.id, "state.changed", {
                "state": TaskState.FAILED.value, "terminal_reason": "cli_error",
            })
            return TerminalReason.CLI_ERROR
