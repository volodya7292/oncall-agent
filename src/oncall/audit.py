"""Structured audit logging.

We already persist everything important to SQLite (the `approvals`,
`task_events`, `chat_messages`, `messenger_inbox` tables). This module adds a
thin INFO-level log stream so you can tail the process and see what's
happening in real time without poking the DB.

Conventions:
  * Logger names start with `oncall.audit.*` so you can filter the stream
    cheaply (`uv run oncall api 2>&1 | grep oncall.audit`).
  * One event per line, `key=value` for cheap reading + grep, JSON-escaped
    text values via shlex-like quoting.
  * Long strings are truncated to 200 chars (configurable).
  * No PII processing; we log whatever the user/executor already authored.
    The DB has full payloads if you need them.
"""

from __future__ import annotations

import logging
from typing import Any


broker_log = logging.getLogger("oncall.audit.broker")
operator_log = logging.getLogger("oncall.audit.operator")
telegram_log = logging.getLogger("oncall.audit.telegram")


_DEFAULT_MAX = 200


def trunc(text: Any, *, maxlen: int = _DEFAULT_MAX) -> str:
    """Truncate to maxlen with an ellipsis marker. Returns str regardless of input."""
    s = str(text)
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 3] + "..."


def kv(value: Any) -> str:
    """Render a value safely for a key=value line — wrap strings with spaces or
    special characters in double quotes; truncate."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    s = trunc(value)
    if any(c in s for c in (" ", "\t", "\n", "=", '"')):
        s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{s}"'
    return s


def fmt(**fields: Any) -> str:
    """Render kwargs as `k=v k=v ...` in insertion order."""
    return " ".join(f"{k}={kv(v)}" for k, v in fields.items())
