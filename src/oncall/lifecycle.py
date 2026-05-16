"""Task lifecycle controller.

Owns the set of in-flight task subprocesses, accepts new submissions, performs
crash recovery on startup, and routes kill requests.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
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
    ) -> Task:
        task = Task(
            session_id=str(uuid4()),  # claude --session-id requires UUID format
            prompt=prompt,
            model=model,
            max_turns=max_turns,
            dispatched_by_chat_session=chat_session_id,
        )
        await self.db.insert_task(task)
        await self.events.publish(task.id, "state.changed", {"state": task.state.value})
        self._spawn(task, resuming=False)
        return task

    async def recover(self) -> None:
        """On boot, re-spawn any tasks left in {running, awaiting_approval}."""
        alive = await self.db.list_tasks_in_states(TaskState.RUNNING, TaskState.AWAITING_APPROVAL)
        for task in alive:
            log.info("recovering task %s (state=%s)", task.id, task.state)
            self._spawn(task, resuming=True)

    async def kill(self, task_id: UUID, *, reason: str = "killed") -> bool:
        rt = self.running.get(task_id)
        if rt is None:
            return False
        # If there's a pending approval future, resolve it as deny first so
        # the broker's await returns immediately.
        for approval in await self.db.list_pending_approvals():
            if approval.task_id == task_id and self.approval_client.has_pending(approval.id):
                from .models import ApprovalResult, utcnow
                self.approval_client.resolve(approval.id, ApprovalResult(
                    request_id=approval.id,
                    behavior="deny",
                    message=f"task killed: {reason}",
                    responded_at=utcnow(),
                ))
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
