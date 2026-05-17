"""Reconstruction of the under-fire inbox-drain triage from 2026-05-17 19:24.

Scenario in prod:
  * Memories 3 + 4 loaded into the system prompt:
      id=3 "user authorizes the assistant to search the 'Piworld' project
            codebase to answer questions from Rostislav"
      id=4 "C# and general code-related questions from Rostislav are
            related to the 'Piworld' project"
  * Drain batch of 5 DMs from Rostislav; first one is "что такое swingTime?"
    (a code question).
  * Operator returned EMPTY text and made ZERO tool calls — i.e. silent
    when the correct behavior is to dispatch a search of Piworld for
    `swingTime`, then call `reply_to_dm` citing id=3 as authority.

This script replays the same prompt + same memory snapshot + same drain
note + the same `reply_to_dm` / `dispatch_task` / `query_memory` tool
schemas against gemini-3.1-flash-lite at all four thinking levels, prints
which level produces the right routing decision (tool_call vs silence),
and how much it costs in latency + reasoning tokens. The intent is to
calibrate whether `ONCALL_OPERATOR_REASONING_EFFORT=low` is enough or we
need MEDIUM/HIGH for memory-chained authority calls.

Run:
    set -a; source ~/.oncall/.env; set +a
    uv run --with google-genai scripts/reconstruct_inbox_drain.py [--iters 3]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


MODEL = "gemini-3.1-flash-lite"
LEVELS: list[str] = ["MINIMAL", "LOW", "MEDIUM"]  # HIGH = ~140s/iter, unusable

# The exact memory rows from prod at the moment of the under-fire turn
# (`sqlite3 ~/.oncall/state.db "SELECT id, text FROM operator_memories
# WHERE model = 'nomic-embed-text:137m-v1.5-fp16'"`).
MEMORIES: list[tuple[int, str]] = [
    (2, "The server `hp_smith` is used by the user."),
    (3, "The user authorizes the assistant to search the 'Piworld' project "
        "codebase to answer questions from Rostislav."),
    (4, "The user considers C# and general code-related questions from "
        "Rostislav to be related to the 'Piworld' project."),
    (6, "The user's projects are located in ~/SoftwareProjects."),
]


def _memory_block() -> str:
    """Mirror the format `OperatorMemory.for_prompt` emits, but include the
    `id=N` prefix the model needs to cite `authority_memory_id`."""
    return "\n".join(f"- (id={i}) {t}" for i, t in MEMORIES)


def _system_prompt(prompt_path: Path) -> str:
    base = prompt_path.read_text(encoding="utf-8")
    return (
        f"{base}\n\n"
        "# Your memory (auto-managed, relevant entries only)\n\n"
        "These are entries from your persistent memory that scored as "
        "relevant to this turn. Memory is auto-extracted from prior user "
        "messages; you do not manage it manually. Treat the entries below "
        "as authoritative context — if something conflicts with what the "
        "user says now, the user wins. Use `query_memory` only when you "
        "want to look up something OUTSIDE this turn's topic.\n\n"
        f"{_memory_block()}"
    )


# The drain note produced by _flush_inbox_batch for the prod batch.
DRAIN_NOTE = (
    "[system note: 5 inbound DM(s) since the last triage:\n"
    "  1. inbox_id=44f85429-70b3-4960-babb-1fd3e3d35fb0 from=@^Ростислав "
    "heuristic_important=no body='что такое swingTime?'\n"
    "  2. inbox_id=a039ca46-3268-4755-bfb4-5431e74aea14 from=@^Ростислав "
    "heuristic_important=no body='ты тут?'\n"
    "  3. inbox_id=ced23090-20cb-40f3-8b92-f2b7eb88787c from=@^Ростислав "
    "heuristic_important=no body='жду'\n"
    "  4. inbox_id=79604ce3-88aa-4de8-ba47-24f2af2b1dc2 from=@^Ростислав "
    "heuristic_important=no body='1'\n"
    "  5. inbox_id=8d7428ff-f71b-401f-bc4a-3a14f8f93e31 from=@^Ростислав "
    "heuristic_important=no body='2'\n\n"
    "For each DM you have exactly TWO options: AUTO-REPLY or STAY SILENT. "
    "No heads-up to the user — the user reads their own Telegram.\n"
    "AUTO-REPLY: if the memory entries loaded into your system prompt "
    "(possibly via JOINT inference across multiple entries) authorize "
    "you to reply on the user's behalf for THIS sender on THIS topic, "
    "execute the instruction (dispatch tasks if needed for gathering) "
    "then call `reply_to_dm` with the controlling memory's id.\n"
    "STAY SILENT: otherwise. Emit ZERO assistant content for that DM. "
    "Do not narrate non-events; the user already sees their inbox.\n"
    "If NONE of the DMs is auto-replyable, emit EMPTY text overall — "
    "no 'no instructions found' line, no 'nothing important' line, no "
    "status. heuristic_important is a hint, not a gate.]"
)


# Cut-down tool schemas mirroring OPERATOR_TOOLS — enough for the model to
# choose the right action; we capture the function_call but don't execute.
TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="query_memory",
        description="Search persistent memory for entries relevant to a query.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING"),
                "limit": types.Schema(type="INTEGER"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="dispatch_task",
        description=(
            "Hand work to the Claude executor. Use for searching directories, "
            "running shell commands, gathering info."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "prompt": types.Schema(type="STRING"),
                "model": types.Schema(
                    type="STRING", enum=["haiku", "sonnet", "opus"],
                ),
            },
            required=["prompt"],
        ),
    ),
    types.FunctionDeclaration(
        name="reply_to_dm",
        description=(
            "Send a memory-authorized auto-reply to a Telegram DM. Requires "
            "the id of the memory entry that authorizes the reply for this "
            "specific sender."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "inbox_id": types.Schema(type="STRING"),
                "text": types.Schema(type="STRING"),
                "authority_memory_id": types.Schema(type="INTEGER"),
            },
            required=["inbox_id", "text", "authority_memory_id"],
        ),
    ),
])


async def _one_run(
    client: genai.Client, *, level: str, system_prompt: str,
) -> dict[str, Any]:
    """Single replay at one thinking level. Captures TTFA, tool calls,
    final text, and reasoning-token cost."""
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        system_instruction=system_prompt,
        thinking_config=types.ThinkingConfig(thinking_level=level),
        tools=[TOOLS],
    )
    contents = [types.Content(
        role="user", parts=[types.Part.from_text(text=DRAIN_NOTE)],
    )]
    started = time.monotonic()
    first_at: float | None = None
    text_parts: list[str] = []
    fn_calls: list[dict[str, Any]] = []
    usage: Any = None
    try:
        stream = await client.aio.models.generate_content_stream(
            model=MODEL, contents=contents, config=cfg,
        )
        async for chunk in stream:
            for c in (getattr(chunk, "candidates", None) or []):
                if not (c.content and c.content.parts):
                    continue
                for part in c.content.parts:
                    if getattr(part, "thought", False):
                        continue
                    now = time.monotonic()
                    if part.text:
                        if first_at is None:
                            first_at = now
                        text_parts.append(part.text)
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        if first_at is None:
                            first_at = now
                        fn_calls.append({
                            "name": fc.name,
                            "args": dict(fc.args or {}),
                        })
            if getattr(chunk, "usage_metadata", None) is not None:
                usage = chunk.usage_metadata
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "total_s": time.monotonic() - started}
    text = "".join(text_parts)
    return {
        "ttfa_s": (first_at - started) if first_at else None,
        "total_s": time.monotonic() - started,
        "text": text,
        "fn_calls": fn_calls,
        "comp_tok": getattr(usage, "candidates_token_count", None) if usage else None,
        "reas_tok": getattr(usage, "thoughts_token_count", None) if usage else None,
    }


def _verdict(result: dict[str, Any]) -> str:
    """Classify what the model did. We want `reply_to_dm` (full action) or
    `dispatch_task` (gather first, then would auto-reply on round 2). Bare
    silence = under-fire — the bug we're reconstructing. Heads-up text =
    violates the simplified contract."""
    if "error" in result:
        return "error"
    names = [c["name"] for c in result["fn_calls"]]
    if "reply_to_dm" in names:
        return "✅ reply_to_dm"
    if "dispatch_task" in names:
        return "✅ dispatch_task (gather)"
    if "query_memory" in names:
        return "↻ query_memory (still deciding)"
    if result["text"].strip():
        return "⚠️ heads-up text (no tool)"
    return "❌ silent (under-fire)"


def _fmt(x: float | None, *, suffix: str = "") -> str:
    return "—" if x is None else f"{x:.2f}{suffix}"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in env")
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "oncall" / "prompts" / "operator_system.md"
    )
    if not prompt_path.exists():
        raise SystemExit(f"prompt not found at {prompt_path}")
    system_prompt = _system_prompt(prompt_path)
    print(f"prompt size: {len(system_prompt):,} chars "
          f"(~{len(system_prompt) // 4:,} tokens estimated)",
          file=sys.stderr)
    print(f"model: {MODEL}\n", file=sys.stderr)
    print(f"{'level':<8} {'iter':>4} {'ttfa':>6} {'total':>6} "
          f"{'comp':>5} {'reas':>5}  verdict")
    print("-" * 80)

    client = genai.Client(api_key=api_key)
    summaries: dict[str, list[dict[str, Any]]] = {}
    for level in LEVELS:
        summaries[level] = []
        for i in range(args.iters):
            if (level != LEVELS[0] or i > 0) and args.sleep > 0:
                await asyncio.sleep(args.sleep)
            r = await _one_run(client, level=level, system_prompt=system_prompt)
            verdict = _verdict(r)
            summaries[level].append({**r, "verdict": verdict})
            print(
                f"{level:<8} {i + 1:>4} "
                f"{_fmt(r.get('ttfa_s'), suffix='s'):>6} "
                f"{_fmt(r.get('total_s'), suffix='s'):>6} "
                f"{_fmt(r.get('comp_tok')):>5} "
                f"{_fmt(r.get('reas_tok')):>5}  {verdict}"
            )

    print("\n=== summary ===")
    print(f"{'level':<8} {'n':>3} {'ttfa_med':>9} {'reas_med':>9} {'verdicts'}")
    for level in LEVELS:
        rows = summaries[level]
        ttfas = [r["ttfa_s"] for r in rows if r.get("ttfa_s") is not None]
        reas = [r["reas_tok"] for r in rows if r.get("reas_tok")]
        verdicts = [r["verdict"] for r in rows]
        ttfa_med = statistics.median(ttfas) if ttfas else None
        reas_med = statistics.median(reas) if reas else None
        print(
            f"{level:<8} {len(rows):>3} "
            f"{_fmt(ttfa_med, suffix='s'):>9} "
            f"{_fmt(reas_med):>9} "
            f"{', '.join(verdicts)}"
        )

    # Show one sample of each level's text/fn output so we can sanity-check.
    print("\n=== sample of each level (first iter) ===")
    for level in LEVELS:
        r = summaries[level][0]
        print(f"\n[{level}] verdict={r['verdict']}")
        if "error" in r:
            print(f"  error: {r['error']}")
            continue
        if r["fn_calls"]:
            for c in r["fn_calls"]:
                args_str = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                print(f"  fn: {c['name']}({args_str[:200]}"
                      f"{'…' if len(args_str) > 200 else ''})")
        text = r["text"].strip()
        if text:
            print(f"  text: {text[:300]!r}{'…' if len(text) > 300 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
