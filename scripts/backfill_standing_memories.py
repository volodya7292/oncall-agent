"""One-off: classify existing operator memories as standing (behavioral) or not.

The `standing` column was added after the store already held rows, and the
migration backfills every one of them to 0 — correct as a default, wrong for
the instructions already in there ("reply in one line", "don't ping at
night"), which would keep being surfaced only when they happened to score
against the topic.

This walks the store in batches, asks the configured operator LLM which rows
DIRECT THE AGENT'S BEHAVIOR, and flips those to standing=1. Behavioral is the
whole test: how the agent should act, not what is true about the user.

Prints its decisions and changes nothing unless `--apply` is passed. Rows
already standing are left alone.

Run it where the database lives (the daemon's own host/container), against
the daemon's DB path:

    uv run scripts/backfill_standing_memories.py --db ~/.oncall/state.db
    uv run scripts/backfill_standing_memories.py --db ~/.oncall/state.db --apply

Requires whatever API key the configured ONCALL_OPERATOR_BACKEND needs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncall.config import get_settings  # noqa: E402

BATCH = 20

SYSTEM_PROMPT = """\
You are auditing an on-call agent's long-term memory store.

Each entry is either an INSTRUCTION about how the agent itself should behave
— its manner, timing, defaults, what it may do unasked — or it is not.
Everything else is not behavioral: facts about the user, their systems, their
people, their history, however durable or often-relevant those are. An entry
is behavioral because of what it governs, not because it is phrased as an
imperative or reads like a preference.

Behavioral entries are loaded into every single conversation, so a wrong
inclusion costs context permanently. When an entry is ambiguous, or states a
fact that merely implies how to act, leave it out.

Reply with JSON only: {"behavioral_ids": [<id>, ...]}
Include only ids from the input. An empty list is a valid answer.
"""

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_loose(text: str) -> dict[str, Any]:
    s = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        m = _JSON_OBJ_RE.search(s)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
        except (ValueError, TypeError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def build_llm(settings: Any) -> tuple[Any, str]:
    """Same client selection the daemon makes, so the backfill runs on
    whatever backend is already configured rather than a second opinion."""
    from oncall.operator import (
        AnthropicLLMClient, GatewayLLMClient, GenAILLMClient, OpenRouterLLMClient,
    )

    backend = settings.oncall_operator_backend
    model = settings.oncall_operator_model
    if backend == "openrouter" and settings.openrouter_api_key:
        order = [
            p.strip() for p in settings.oncall_operator_provider_order.split(",")
            if p.strip()
        ]
        return OpenRouterLLMClient(
            settings.openrouter_base_url, settings.openrouter_api_key,
            provider_order=order,
        ), model
    if backend == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLMClient(settings.anthropic_api_key), model
    if backend == "gemini" and settings.gemini_api_key:
        return GenAILLMClient(settings.gemini_api_key), model
    if backend == "vercel" and settings.gateway_key:
        return GatewayLLMClient(settings.ai_gateway_base_url, settings.gateway_key), model
    raise SystemExit(
        f"no usable client for ONCALL_OPERATOR_BACKEND={backend!r} — is its API key set?"
    )


async def classify(llm: Any, model: str, rows: list[dict[str, Any]]) -> set[int]:
    listing = "\n".join(f'{r["id"]}: {r["text"]}' for r in rows)
    resp = await llm.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Entries:\n{listing}"},
        ],
        tools=[],
        max_tokens=1024,
        reasoning_effort="low",
    )
    data = _parse_json_loose(resp.get("content") or "")
    raw = data.get("behavioral_ids")
    if not isinstance(raw, list):
        return set()
    valid = {int(r["id"]) for r in rows}
    out: set[int] = set()
    for i in raw:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        # A hallucinated id would flip an unrelated row into every context
        # window, so ids outside the batch are dropped rather than trusted.
        if i in valid:
            out.add(i)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path, help="path to state.db")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(operator_memories)")}
    if "standing" not in cols:
        raise SystemExit(
            "operator_memories has no `standing` column — start the upgraded "
            "daemon once so the migration runs, then re-run this."
        )

    rows = [
        dict(r) for r in conn.execute(
            "SELECT id, text FROM operator_memories WHERE standing = 0 ORDER BY id"
        )
    ]
    already = conn.execute(
        "SELECT COUNT(*) FROM operator_memories WHERE standing = 1"
    ).fetchone()[0]
    print(f"{len(rows)} contextual rows to audit; {already} already standing")
    if not rows:
        return

    settings = get_settings()
    llm, model = build_llm(settings)
    print(f"classifier: {model} via {settings.oncall_operator_backend}\n")

    promote: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        try:
            hits = await classify(llm, model, batch)
        except Exception as e:
            print(f"  batch at {start} FAILED ({type(e).__name__}: {e}) — skipped")
            continue
        for r in batch:
            if int(r["id"]) in hits:
                promote.append(r)
                print(f"  [standing] {r['id']}: {r['text']}")
    print(f"\n{len(promote)} of {len(rows)} rows classified behavioral")

    if not promote:
        return
    if not args.apply:
        print("dry run — re-run with --apply to write")
        return
    conn.executemany(
        "UPDATE operator_memories SET standing = 1 WHERE id = ?",
        [(int(r["id"]),) for r in promote],
    )
    conn.commit()
    print(f"updated {len(promote)} rows")


if __name__ == "__main__":
    asyncio.run(main())
