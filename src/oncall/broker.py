"""The permission broker.

This is the orchestrator-side half of the permission chokepoint. The MCP server
process's `approve` tool is a thin loopback proxy; the actual decision-making
lives here.

Inputs come from a `claude` CLI subprocess via the MCP `approve` tool: a tool
name, an input dict, and the CLI's per-call `tool_use_id`. The broker:

  1. Deduplicates on `(session_id, tool_use_id)` so `--resume` after an
     orchestrator crash doesn't re-prompt the user.
  2. Runs the deterministic classifier.
  3. Auto-allows read-only; auto-denies catastrophic (defense in depth).
  4. For mutating: persists a pending approval row, publishes an event, and
     awaits the `ApprovalClient` (which is either a test stub or the
     HTTP long-poll Future resolved by the FastAPI handler).
  5. Enforces a consecutive-denial backstop so a misbehaving model can't
     burn through repeated approvals.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable
from uuid import UUID

from .approval_client import (
    ApprovalClient,
    generate_challenge_phrase,
    phrases_match,
)
from .audit import broker_log, fmt
from .classifier import classify
from .db import Database
from .models import (
    ApprovalRequest,
    ApprovalResult,
    ClassifierVerdict,
    PermissionResult,
    new_uuid,
    utcnow,
)


log = logging.getLogger(__name__)

MAX_CONSECUTIVE_DENIALS = 3


EventPublisher = Callable[[UUID, str, dict[str, Any]], Awaitable[None]]


class Broker:
    def __init__(
        self,
        db: Database,
        approval_client: ApprovalClient,
        publish_event: EventPublisher,
        *,
        max_consecutive_denials: int = MAX_CONSECUTIVE_DENIALS,
        approval_timeout_seconds: int | None = None,
    ) -> None:
        self._db = db
        self._client = approval_client
        self._publish = publish_event
        self._max_denials = max_consecutive_denials
        # None → use the ApprovalRequest model's own default. In production
        # api.py passes settings.oncall_approval_timeout_seconds.
        self._approval_timeout_seconds = approval_timeout_seconds

    async def decide(
        self,
        session_id: str,
        tool_use_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionResult:
        """Single entrypoint called by the MCP `approve` tool (via loopback HTTP)."""
        # 1a. Dedup (resolved): if --resume re-issues the same call after the
        # user already responded (or the timeout deny fired) — return cached.
        cached = await self._db.get_resolved_approval(session_id, tool_use_id)
        if cached is not None:
            req, result = cached
            broker_log.info("decide " + fmt(
                event="dedup_hit", session=session_id, tool_use=tool_use_id,
                tool=tool_name, decision=result.behavior,
            ))
            return self._to_permission_result(result, tool_input)

        # 1b. Dedup (pending): if --resume re-issues the same call but the
        # user never responded before the crash, RE-ATTACH to the existing
        # pending row. A fresh INSERT would violate UNIQUE(session, tool_use_id)
        # and crash the broker. Re-publishing approval.requested makes the
        # bot's button UI resurface so the user can finish responding.
        pending = await self._db.get_pending_by_session_and_tool(session_id, tool_use_id)
        if pending is not None:
            task_for_pending = await self._db.get_task_by_session(session_id)
            if task_for_pending is not None:
                await self._publish(task_for_pending.id, "approval.requested", {
                    "approval_id": str(pending.id),
                    "tool_name": pending.tool_name,
                    "canonical_command": pending.canonical_command,
                    "blast_radius": pending.blast_radius,
                    "challenge_phrase": pending.challenge_phrase,
                    "reattach": True,
                })
            broker_log.info("decide " + fmt(
                event="reattach_pending", session=session_id, tool_use=tool_use_id,
                approval=str(pending.id),
            ))
            return await self._await_pending_resolution(pending, tool_input)

        # 2. Classify (deterministic, model-free).
        verdict = classify(tool_name, tool_input)
        task = await self._db.get_task_by_session(session_id)
        if task is None:
            broker_log.warning("decide " + fmt(
                event="unknown_session", session=session_id, tool=tool_name,
            ))
            return PermissionResult(behavior="deny", message="unknown session")

        # 3. Auto paths.
        req_id = new_uuid()
        challenge = (
            None
            if verdict.kind != ClassifierVerdict.MUTATING
            else generate_challenge_phrase()
        )
        req_kwargs: dict[str, Any] = dict(
            id=req_id,
            task_id=task.id,
            session_id=session_id,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            tool_input=tool_input,
            classifier_verdict=verdict.kind,
            canonical_command=verdict.canonical,
            blast_radius=verdict.blast_radius,
            challenge_phrase=challenge,
        )
        if self._approval_timeout_seconds is not None:
            req_kwargs["timeout_seconds"] = self._approval_timeout_seconds
        req = ApprovalRequest(**req_kwargs)

        if verdict.kind == ClassifierVerdict.READONLY:
            await self._db.record_auto_approval(req, "allow", "auto:readonly")
            await self._publish(task.id, "approval.resolved", {
                "approval_id": str(req_id),
                "auto": True,
                "decision": "allow",
                "tool_name": tool_name,
                "canonical": verdict.canonical,
            })
            broker_log.info("decide " + fmt(
                event="auto_allow", task=str(task.id), tool=tool_name,
                verdict="readonly", canonical=verdict.canonical,
            ))
            return PermissionResult(behavior="allow", updatedInput=tool_input)

        # Pre-approved Telegram send: chat_id is on the user's per-chat
        # allowlist (populated via `/allowdm <chat_id>`). Mutating `op=send`
        # to that chat auto-allows without a challenge-phrase round-trip.
        # The executor's system prompt carries the no-cross-chat-leak rule;
        # this table is the final byte-level gate before a message leaves
        # the box on the user's behalf.
        if (
            verdict.kind == ClassifierVerdict.MUTATING
            and tool_name == "mcp__oncall__messenger_inbox"
            and tool_input.get("op") == "send"
        ):
            send_chat = str(tool_input.get("chat_id") or "")
            if send_chat and await self._db.is_dm_allowed(send_chat):
                await self._db.record_auto_approval(req, "allow", "auto:dm_allowlist")
                await self._publish(task.id, "approval.resolved", {
                    "approval_id": str(req_id),
                    "auto": True,
                    "decision": "allow",
                    "tool_name": tool_name,
                    "canonical": verdict.canonical,
                })
                broker_log.info("decide " + fmt(
                    event="auto_allow_dm_allowlist", task=str(task.id), tool=tool_name,
                    canonical=verdict.canonical, chat=send_chat,
                ))
                return PermissionResult(behavior="allow", updatedInput=tool_input)

        # Pre-approved Write directory: if the user previously tapped
        # "Yes (and folder)" on a Write approval, any subsequent Write to
        # a file under that dir auto-allows for this task only.
        if (
            verdict.kind == ClassifierVerdict.MUTATING
            and tool_name == "Write"
        ):
            fp = str(tool_input.get("file_path") or "")
            if fp:
                import os.path as _osp
                parent = _osp.dirname(_osp.abspath(fp))
                allowed_dirs = await self._db.list_write_dirs(str(task.id))
                for d in allowed_dirs:
                    da = _osp.abspath(d)
                    if parent == da or parent.startswith(da + _osp.sep):
                        await self._db.record_auto_approval(req, "allow", "auto:pre_approved_write_dir")
                        await self._publish(task.id, "approval.resolved", {
                            "approval_id": str(req_id),
                            "auto": True,
                            "decision": "allow",
                            "tool_name": tool_name,
                            "canonical": verdict.canonical,
                        })
                        broker_log.info("decide " + fmt(
                            event="auto_allow_write_dir", task=str(task.id),
                            tool=tool_name, file=fp, allowed_dir=d,
                        ))
                        return PermissionResult(behavior="allow", updatedInput=tool_input)

        if verdict.kind == ClassifierVerdict.CATASTROPHIC:
            await self._db.record_auto_approval(req, "deny", verdict.reason or "catastrophic")
            await self._publish(task.id, "approval.resolved", {
                "approval_id": str(req_id),
                "auto": True,
                "decision": "deny",
                "reason": verdict.reason,
                "canonical": verdict.canonical,
            })
            broker_log.warning("decide " + fmt(
                event="auto_deny_catastrophic", task=str(task.id), tool=tool_name,
                reason=verdict.reason, canonical=verdict.canonical,
            ))
            return PermissionResult(
                behavior="deny",
                message=f"BLOCKED (catastrophic): {verdict.reason or 'irreversible'}",
            )

        # 4. Mutating — check the denial backstop, then escalate.
        if task.consecutive_denials >= self._max_denials:
            broker_log.warning("decide " + fmt(
                event="halt_denial_loop", task=str(task.id),
                denials=task.consecutive_denials,
            ))
            return PermissionResult(
                behavior="deny",
                message=f"Agent halted — {task.consecutive_denials} consecutive denials.",
            )

        await self._db.create_pending_approval(req)
        await self._publish(task.id, "approval.requested", {
            "approval_id": str(req_id),
            "tool_name": tool_name,
            "canonical_command": verdict.canonical,
            "blast_radius": verdict.blast_radius,
            "challenge_phrase": challenge,
        })
        broker_log.info("decide " + fmt(
            event="escalate", task=str(task.id), tool=tool_name,
            verdict="mutating", approval=str(req_id),
            canonical=verdict.canonical, phrase=challenge,
        ))
        return await self._await_pending_resolution(req, tool_input)

    async def _await_pending_resolution(
        self, req: ApprovalRequest, tool_input: dict[str, Any],
    ) -> PermissionResult:
        """Block on the approval_client's Future until the user (or timeout)
        resolves the pending row, persist the response, fire the resolved
        event, update the denial counter. Shared between the first-time path
        and the resume re-attach path."""
        result = await self._client.request_approval(req)
        await self._db.append_approval_response(req.id, result)
        await self._publish(req.task_id, "approval.resolved", {
            "approval_id": str(req.id),
            "auto": False,
            "decision": result.behavior,
            "challenge_matched": result.challenge_matched,
            "canonical": req.canonical_command,
        })
        broker_log.info("decide " + fmt(
            event="resolved", task=str(req.task_id), approval=str(req.id),
            decision=result.behavior, matched=result.challenge_matched,
        ))
        if result.behavior == "deny":
            await self._db.increment_consecutive_denials(req.task_id)
            return PermissionResult(
                behavior="deny",
                message=result.message or "User denied.",
            )
        await self._db.reset_consecutive_denials(req.task_id)
        return PermissionResult(behavior="allow", updatedInput=tool_input)

    # ---- helpers for the HTTP `/approvals/{id}/respond` endpoint ----

    async def submit_response(
        self,
        approval_id: UUID,
        decision: str,
        challenge_phrase_supplied: str,
        message: str | None = None,
    ) -> tuple[bool, bool]:
        """Validate the challenge phrase and resolve the awaiting Future.

        Returns (approved, challenge_matched). If the supplied phrase doesn't match,
        the decision is coerced to 'deny' regardless of what the caller said.
        """
        req = await self._db.get_pending_approval(approval_id)
        if req is None or req.challenge_phrase is None:
            return False, False
        matched = phrases_match(req.challenge_phrase, challenge_phrase_supplied)
        behavior = "allow" if (matched and decision == "allow") else "deny"
        result_message = (
            message
            if matched
            else "Challenge phrase mismatch — coerced to deny."
        )
        result = ApprovalResult(
            request_id=approval_id,
            behavior=behavior,  # type: ignore[arg-type]
            challenge_phrase_supplied=challenge_phrase_supplied,
            challenge_matched=matched,
            message=result_message,
            responded_at=utcnow(),
        )
        # Resolve the in-memory future; the broker's own decide() will handle
        # the DB write + denial counters when the await returns.
        # We need the HttpLongPollApprovalClient instance to call .resolve().
        # That's wired in via the public attribute exposed by the orchestrator.
        from .approval_client import HttpLongPollApprovalClient

        if isinstance(self._client, HttpLongPollApprovalClient):
            self._client.resolve(approval_id, result)
        return behavior == "allow", matched

    # ---- internals ----

    @staticmethod
    def _to_permission_result(
        result: ApprovalResult,
        tool_input: dict[str, Any],
    ) -> PermissionResult:
        if result.behavior == "allow":
            return PermissionResult(behavior="allow", updatedInput=tool_input)
        return PermissionResult(behavior="deny", message=result.message or "denied")
