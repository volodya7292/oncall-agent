"""Task lifecycle controller.

Owns the set of in-flight task subprocesses, accepts new submissions, performs
crash recovery on startup, and routes kill requests.
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
    # Cap on concurrent claude executors. Tasks beyond the cap stay in
    # `pending` state until a slot opens. Initialized in __post_init__.
    _slot_sem: "asyncio.Semaphore | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._slot_sem = asyncio.Semaphore(self.settings.oncall_max_concurrent_tasks)

    async def submit_task(
        self,
        *,
        prompt: str,
        model: str | None = None,
        max_turns: int | None = None,
        chat_session_id: str | None = None,
        restricted_to_chat: str | None = None,
    ) -> Task:
        task = Task(
            session_id=str(uuid4()),  # claude --session-id requires UUID format
            prompt=prompt,
            model=model,
            max_turns=max_turns,
            dispatched_by_chat_session=chat_session_id,
            restricted_to_chat=restricted_to_chat,
        )
        await self.db.insert_task(task)
        await self.events.publish(task.id, "state.changed", {"state": task.state.value})
        self._spawn(task, resuming=False)
        return task

    async def recover(self) -> None:
        """On boot, re-spawn any tasks left in {running, awaiting_approval},
        and sweep orphaned approvals — pending approval rows whose parent
        task already transitioned to a terminal state (failed/killed/
        completed) and will never decide them. Without the sweep, /status
        would show 'Approvals pending: N' forever for those orphans."""
        alive = await self.db.list_tasks_in_states(TaskState.RUNNING, TaskState.AWAITING_APPROVAL)
        for task in alive:
            log.info("recovering task %s (state=%s)", task.id, task.state)
            self._spawn(task, resuming=True)
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
        # long enough to do it. The DB row matters because the next daemon
        # boot's recover() would otherwise see an orphan pending row.
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
        for task_id in list(self.running.keys()):
            await self.kill(task_id, reason="shutdown")

    # ---- internals ----

    def _spawn(self, task: Task, *, resuming: bool) -> None:
        sup = Supervisor(db=self.db, events=self.events, settings=self.settings, paths=self.paths)
        runner = asyncio.create_task(self._run(task, sup, resuming=resuming))
        self.running[task.id] = RunningTask(task=task, supervisor=sup, runner=runner)

    async def _run(self, task: Task, sup: Supervisor, *, resuming: bool) -> TerminalReason:
        assert self._slot_sem is not None  # set in __post_init__
        # Log when waiting so the audit stream shows backpressure.
        if self._slot_sem.locked():
            log.info("task %s waiting for executor slot (cap=%d)",
                     task.id, self.settings.oncall_max_concurrent_tasks)
            await self.events.publish(task.id, "queued", {
                "cap": self.settings.oncall_max_concurrent_tasks,
            })
        try:
            async with self._slot_sem:
                return await sup.run(task, resuming=resuming)
        except asyncio.CancelledError:
            return TerminalReason.KILLED
        except Exception:
            log.exception("supervisor crashed for task %s", task.id)
            await self.db.update_task_state(task.id, TaskState.FAILED, TerminalReason.CLI_ERROR)
            await self.events.publish(task.id, "state.changed", {
                "state": TaskState.FAILED.value, "terminal_reason": "cli_error",
            })
            return TerminalReason.CLI_ERROR
        finally:
            self.running.pop(task.id, None)
            # If the task ended without going through `kill()` (e.g. normal
            # supervisor completion, or the crash path above), any pending
            # approval rows it left behind are now orphans. Sweep them so
            # /status' pending-approval count reflects reality.
            await self._sweep_orphan_approvals()
