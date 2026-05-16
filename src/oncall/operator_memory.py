"""Operator memory — a single markdown file the operator reads and writes.

Lives at `~/.oncall/memory.md` by default. Loaded into the system prompt at
every turn so the operator always sees its prior notes, and exposed as two
operator tools (`remember`, `forget`).

Scope: short, declarative facts the user explicitly asked to keep, or context
the operator observed about how the user wants to be treated. Not a journal —
keep it terse and prunable.

Concurrency: single user, single orchestrator process, single writer. No
locking needed.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path


log = logging.getLogger(__name__)


_HEADER = "# Operator memory\n\n"
_INTRO = (
    "Items the user asked the operator to remember, or notable context worth\n"
    "carrying across chat sessions. Auto-managed.\n\n"
    "## Entries\n\n"
)
_ENTRY_RE = re.compile(r"^- \[(\d{4}-\d{2}-\d{2})\] (.+)$")

# Hard cap so the file can't grow unbounded and bloat the system prompt.
MAX_ENTRIES = 200
MAX_ENTRY_CHARS = 400


class OperatorMemory:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ---- reads ----

    def raw(self) -> str:
        """Full file content, or empty string if missing. Used for the system prompt."""
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            log.exception("failed to read operator memory at %s", self.path)
            return ""

    def for_prompt(self) -> str:
        """Render the memory for inclusion in the system prompt. Returns either
        the full file content (so the operator sees its own notes verbatim) or
        an "(empty)" marker on first run."""
        text = self.raw().strip()
        if not text:
            return "(no entries yet)"
        return text

    def entries(self) -> list[tuple[str, str]]:
        """Parse current entries as a list of (iso_date, text)."""
        out: list[tuple[str, str]] = []
        for line in self.raw().splitlines():
            m = _ENTRY_RE.match(line.strip())
            if m:
                out.append((m.group(1), m.group(2)))
        return out

    # ---- writes ----

    def remember(self, text: str, *, today: str | None = None) -> dict[str, object]:
        """Append a new dated entry. Returns a summary dict for the tool result.
        Quietly drops empty/whitespace input and over-long entries; operator is
        told to keep entries short."""
        text = (text or "").strip().replace("\n", " ")
        if not text:
            return {"added": False, "reason": "empty"}
        if len(text) > MAX_ENTRY_CHARS:
            return {"added": False, "reason": f"too_long (max {MAX_ENTRY_CHARS} chars)"}
        entries = self.entries()
        if any(t == text for _d, t in entries):
            return {"added": False, "reason": "duplicate"}
        if len(entries) >= MAX_ENTRIES:
            return {"added": False, "reason": f"full (max {MAX_ENTRIES})"}
        entries.append((today or date.today().isoformat(), text))
        self._write(entries)
        return {"added": True, "entries": len(entries)}

    def forget(self, substring: str) -> dict[str, object]:
        """Remove every entry whose text contains the (case-insensitive) substring."""
        needle = (substring or "").strip().lower()
        if not needle:
            return {"removed": 0, "reason": "empty"}
        entries = self.entries()
        kept = [(d, t) for d, t in entries if needle not in t.lower()]
        removed = len(entries) - len(kept)
        if removed:
            self._write(kept)
        return {"removed": removed, "entries": len(kept)}

    # ---- internals ----

    def _write(self, entries: list[tuple[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = _HEADER + _INTRO + "".join(
            f"- [{d}] {t}\n" for d, t in entries
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
