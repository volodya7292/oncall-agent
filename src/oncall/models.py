from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> UUID:
    return uuid4()


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class TerminalReason(StrEnum):
    SUCCESS = "success"
    DENIAL_LOOP = "denial_loop"
    KILLED = "killed"
    CLI_ERROR = "cli_error"
    TIMEOUT = "timeout"


class Task(BaseModel):
    id: UUID = Field(default_factory=new_uuid)
    session_id: str
    state: TaskState = TaskState.PENDING
    prompt: str
    model: str | None = None
    max_turns: int | None = None
    consecutive_denials: int = 0
    dispatched_by_chat_session: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    terminal_reason: TerminalReason | None = None


class ClassifierVerdict(StrEnum):
    READONLY = "readonly"
    MUTATING = "mutating"
    CATASTROPHIC = "catastrophic"


class Verdict(BaseModel):
    """Output of the deterministic classifier."""

    kind: ClassifierVerdict
    canonical: str
    blast_radius: str
    reason: str | None = None
    # If the command embeds a script (e.g. `python -c '...'` or `python << EOF`),
    # we extract it here so the operator can summarize the script before reading
    # the approval to the user. {"language": "python", "source": "..."}.
    embedded_code: dict[str, str] | None = None


class ApprovalRequest(BaseModel):
    id: UUID = Field(default_factory=new_uuid)
    task_id: UUID
    session_id: str
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    classifier_verdict: ClassifierVerdict
    canonical_command: str
    blast_radius: str
    challenge_phrase: str | None  # None for auto-decided rows
    requested_at: datetime = Field(default_factory=utcnow)
    # Default 24h. Pending approvals consume no compute — the executor is
    # paused at the broker — so a long backstop fits the on-call use case
    # where the user might be asleep / in a flight / etc. The broker overrides
    # this with `settings.oncall_approval_timeout_seconds` in production.
    timeout_seconds: int = 86400


class ApprovalResult(BaseModel):
    request_id: UUID
    behavior: Literal["allow", "deny"]
    challenge_phrase_supplied: str | None = None
    challenge_matched: bool = False
    message: str | None = None
    responded_at: datetime = Field(default_factory=utcnow)


class TaskEvent(BaseModel):
    task_id: UUID
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=utcnow)


class PermissionResult(BaseModel):
    """The shape that the MCP `approve` tool returns to the claude CLI."""

    behavior: Literal["allow", "deny"]
    updatedInput: dict[str, Any] | None = None  # noqa: N815 — must match CLI schema
    message: str | None = None

    def to_cli_payload(self) -> dict[str, Any]:
        if self.behavior == "allow":
            return {"behavior": "allow", "updatedInput": self.updatedInput or {}}
        return {"behavior": "deny", "message": self.message or "denied"}
