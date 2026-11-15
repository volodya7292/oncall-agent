"""One-time import of the pre-Mem0 SQLite memory store into Mem0.

Reads `operator_memories` read-only (safe while the daemon is live, WAL) and
writes each row verbatim, keeping its standing flag and creation time. Each
imported row carries `legacy_id` in metadata, so a rerun after a partial
failure skips what already landed.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .operator_memory import Mem0Memory


log = logging.getLogger(__name__)


def read_legacy_rows(db_path: Path) -> list[dict[str, Any]]:
    """Rows of `operator_memories`, oldest first; empty if the table is gone."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'operator_memories'"
        ).fetchone()
        if exists is None:
            return []
        return [dict(r) for r in conn.execute(
            "SELECT id, text, standing, created_at FROM operator_memories ORDER BY id ASC"
        )]
    finally:
        conn.close()


def _epoch(created_at: str | None) -> int | None:
    if not created_at:
        return None
    try:
        return int(datetime.fromisoformat(created_at).timestamp())
    except ValueError:
        log.warning("legacy memory has unparseable created_at %r; importing without it", created_at)
        return None


async def import_legacy(
    rows: list[dict[str, Any]], memory: Mem0Memory, *, dry_run: bool = False,
) -> dict[str, int]:
    done = {
        str((m.get("metadata") or {}).get("legacy_id"))
        for m in await memory.list_all()
        if isinstance(m.get("metadata"), dict) and "legacy_id" in m["metadata"]
    }
    imported = skipped = failed = 0
    for row in rows:
        if str(row["id"]) in done:
            skipped += 1
            continue
        standing = bool(row["standing"])
        if dry_run:
            imported += 1
            continue
        metadata: dict[str, Any] = {"legacy_id": row["id"]}
        if standing:
            metadata["standing"] = True
        try:
            await memory.add_verbatim(
                row["text"], metadata=metadata, immutable=standing,
                timestamp=_epoch(row.get("created_at")),
            )
        except Exception as e:
            failed += 1
            log.warning("legacy memory %s failed to import: %s", row["id"], e)
            continue
        imported += 1
    return {"total": len(rows), "imported": imported, "skipped": skipped, "failed": failed}
