"""One-off live test for OperatorMemory.dedup_pass.

Stores the actual memory texts the user flagged from production logs, runs
the periodic LLM dedup pass against them, and prints the decisions. The
correctness bar:

  * Cluster of same-template-different-person memories (Person A vs
    Person B): must be KEPT separate — silently merging them is the
    bug we're fixing.
  * Cluster of paraphrases of one fact about ONE person (two
    phrasings): must be MERGED into one consolidated entry.
  * Truly unrelated memories: not even clustered (cos < 0.80).

Reads gateway / gemini keys from `~/.oncall/.env`. Uses local Ollama for
embeddings (same default as production). Runs against a temp SQLite db
so live state isn't touched.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Load ~/.oncall/.env into env without depending on python-dotenv.
ENV_PATH = Path.home() / ".oncall" / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

# Project src on sys.path so this script can be run from a clean shell.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from oncall.db import Database  # noqa: E402
from oncall.embeddings import OllamaEmbeddingClient  # noqa: E402
from oncall.operator import GenAILLMClient  # noqa: E402
from oncall.operator_memory import OperatorMemory  # noqa: E402


# Two clusters that should resolve differently. Names are synthetic.
DIFFERENT_PERSON = [
    "The user authorizes the assistant to search chat history with John "
    "Doe to answer his messages.",
    "The user authorizes the assistant to search chat history with Jane "
    "Roe to answer her messages.",
]

SAME_PERSON_PARAPHRASES = [
    "The user authorizes the assistant to chat with Sam Carter on any "
    "topic using only web search/fetch tools.",
    "The user authorizes the assistant to talk with Sam Carter on any "
    "topic via web search and web fetch tools.",
]

UNRELATED = [
    "The user's projects directory is ~/SoftwareProjects.",
    "The user prefers terse, no-fluff replies.",
]


async def main() -> None:
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set (looked in env + ~/.oncall/.env)")
        sys.exit(2)
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embed_model = os.environ.get(
        "ONCALL_MEMORY_EMBED_MODEL", "nomic-embed-text:137m-v1.5-fp16",
    )
    operator_model = os.environ.get(
        "ONCALL_OPERATOR_MODEL", "gemini-3.1-flash-lite",
    )

    embedder = OllamaEmbeddingClient(host=ollama_host, model=embed_model)
    llm = GenAILLMClient(api_key=gemini_key)

    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "dedup-test.db")
        await db.connect()
        try:
            mem = OperatorMemory(
                db, embedder,
                embed_model=embed_model,
                capacity=100,
                max_inject=10,
                relevance_floor=0.20,
                hybrid_alpha=0.7,
                hybrid_beta=0.3,
            )
            all_facts = DIFFERENT_PERSON + SAME_PERSON_PARAPHRASES + UNRELATED
            print(f"Seeding {len(all_facts)} memories…")
            await mem.store(all_facts)
            count_before = await mem.entries_count()
            print(f"  stored: {count_before} rows\n")

            print(f"Running dedup_pass with operator model={operator_model} "
                  f"reasoning=medium …")
            stats = await mem.dedup_pass(
                llm, model=operator_model, reasoning_effort="medium",
                cluster_threshold=0.80, max_cluster_size=8,
            )
            print(f"  stats: {stats}\n")

            # Print resulting rows.
            print("Resulting memory rows:")
            rows = await mem._all_rows()
            for r in rows:
                print(f"  id={r['id']} text={r['text']!r}")
            print(f"\n  total: {len(rows)} rows (was {count_before})\n")

            # Pass/fail checks.
            print("Verdict:")
            texts = [r["text"] for r in rows]
            john_kept = any("John" in t for t in texts)
            jane_kept = any("Jane" in t for t in texts)
            print(f"  ✅ John row preserved: {john_kept}"
                  if john_kept
                  else f"  ❌ John row LOST (this is the bug)")
            print(f"  ✅ Jane row preserved: {jane_kept}"
                  if jane_kept
                  else f"  ❌ Jane row LOST (this is the bug)")
            sam_rows = [t for t in texts if "Sam" in t]
            if len(sam_rows) == 1:
                print(f"  ✅ Sam paraphrases merged into one row: {sam_rows[0]!r}")
            else:
                print(f"  ⚠️  Sam: expected 1 row, got {len(sam_rows)}: "
                      f"{sam_rows}")
            unrelated_kept = sum(
                1 for orig in UNRELATED if any(orig == t for t in texts)
            )
            print(f"  ✅ Unrelated memories untouched: {unrelated_kept}/{len(UNRELATED)}"
                  if unrelated_kept == len(UNRELATED)
                  else f"  ⚠️  Unrelated memories changed: "
                       f"{unrelated_kept}/{len(UNRELATED)} preserved verbatim")
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())
