"""Latency bench for gemma4:e4b on a local Ollama, mirroring
bench_thinking_genai.py so the three (Vercel, Gemini-native, Ollama-local)
columns are comparable.

Run with:
    uv run --with httpx scripts/bench_thinking_ollama.py [--iters 5 --sleep 1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any

import httpx


MODEL = "gemma4:e4b"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

PROMPTS: dict[str, str] = {
    "short_ack": (
        "You are a terse on-call operator. Reply in <= 12 words. No preamble, no apologies.\n"
        "User: I just dispatched task T1 to investigate API errors. Acknowledge."
    ),
    "tool_decision": (
        "You are an operator that routes work to a Claude executor. "
        "When the user asks for an infra check, call dispatch_task. "
        "Be terse in any user-facing text.\n"
        "User: ssh myserver and list running docker services. try again."
    ),
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "dispatch_task",
        "description": "Dispatch a task to the Claude executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string", "enum": ["haiku", "sonnet", "opus"]},
            },
            "required": ["prompt"],
        },
    },
}]

# Ollama's native /api/chat accepts `think: true|false` for thinking models.
# False == minimal/no thinking; True == let the model think.
THINK_OPTIONS: list[bool | None] = [None, False, True]


async def _stream_round(
    http: httpx.AsyncClient,
    *,
    messages: list[dict[str, Any]],
    think: bool | None,
    use_tools: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2},
    }
    if think is not None:
        body["think"] = think
    if use_tools:
        body["tools"] = TOOLS

    started = time.monotonic()
    first_at: float | None = None
    text_buf: list[str] = []
    tool_call: dict[str, Any] | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    try:
        async with http.stream("POST", f"{OLLAMA}/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                content = msg.get("content") or ""
                tc = msg.get("tool_calls") or []
                if content or tc:
                    if first_at is None:
                        first_at = time.monotonic()
                    if content:
                        text_buf.append(content)
                    if tc and tool_call is None:
                        tool_call = tc[0]
                if obj.get("done"):
                    eval_count = obj.get("eval_count")
                    eval_duration_ns = obj.get("eval_duration")
                    prompt_eval_count = obj.get("prompt_eval_count")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "total_s": time.monotonic() - started}
    return {
        "total_s": time.monotonic() - started,
        "ttft_s": (first_at - started) if first_at else None,
        "text": "".join(text_buf),
        "tool_call": tool_call,
        "comp_tok": eval_count,
        "prompt_tok": prompt_eval_count,
        "gen_tok_per_s": (eval_count / (eval_duration_ns / 1e9))
            if eval_count and eval_duration_ns else None,
    }


async def _one_call(
    http: httpx.AsyncClient,
    *,
    prompt_name: str,
    think: bool | None,
    use_tools: bool,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": PROMPTS[prompt_name]},
    ]
    wall_start = time.monotonic()
    r1 = await _stream_round(http, messages=messages, think=think, use_tools=use_tools)
    if "error" in r1:
        return {"error": r1["error"]}
    if not use_tools or r1["tool_call"] is None:
        return {
            "time_to_first_answer_s": r1["ttft_s"],
            "total_s": r1["total_s"],
            "rounds": 1,
            "comp_tok": r1["comp_tok"],
            "tok_per_s": r1["gen_tok_per_s"],
        }
    # Round 2: append tool call + synthetic result.
    tc = r1["tool_call"]
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [tc],
    })
    messages.append({
        "role": "tool",
        "content": json.dumps({"task_id": "T1", "state": "pending", "model": "sonnet"}),
    })
    r2 = await _stream_round(http, messages=messages, think=think, use_tools=use_tools)
    if "error" in r2:
        return {"error": "round2: " + r2["error"]}
    time_to_first_answer = (
        r1["total_s"] + r2["ttft_s"] if r2["ttft_s"] is not None else None
    )
    return {
        "time_to_first_answer_s": time_to_first_answer,
        "total_s": time.monotonic() - wall_start,
        "rounds": 2,
        "comp_tok": (r1["comp_tok"] or 0) + (r2["comp_tok"] or 0),
        "tok_per_s": r2["gen_tok_per_s"],
    }


def _fmt(x: float | None, *, suffix: str = "") -> str:
    return "—" if x is None else f"{x:.2f}{suffix}"


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    errs = [r["error"] for r in results if "error" in r]
    if not ok:
        return {"n": 0, "errors": errs}
    ttfa = [r["time_to_first_answer_s"] for r in ok if r.get("time_to_first_answer_s") is not None]
    comp = [r.get("comp_tok") for r in ok if r.get("comp_tok")]
    tps = [r.get("tok_per_s") for r in ok if r.get("tok_per_s")]
    totals = [r["total_s"] for r in ok]
    return {
        "n": len(ok),
        "errors": errs,
        "ttfa_med": statistics.median(ttfa) if ttfa else None,
        "ttfa_p95": (max(ttfa) if len(ttfa) < 4 else statistics.quantiles(ttfa, n=20)[-1]) if ttfa else None,
        "total_med": statistics.median(totals),
        "comp_tok_med": statistics.median(comp) if comp else None,
        "tok_per_s_med": statistics.median(tps) if tps else None,
        "rounds_med": statistics.median([r["rounds"] for r in ok]),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    # Warm up — first call after model unload eats a multi-second weight-load.
    async with httpx.AsyncClient(timeout=300.0) as http:
        print(f"warming up {MODEL}…", flush=True)
        warm = await _one_call(http, prompt_name="short_ack", think=False, use_tools=False)
        if "error" in warm:
            raise SystemExit(f"warmup failed: {warm['error']}")
        print(f"warmup ok ({warm.get('total_s', 0):.2f}s)")

        for prompt_name, use_tools in [("short_ack", False), ("tool_decision", True)]:
            print(f"\n=== prompt={prompt_name} use_tools={use_tools} model={MODEL} ===")
            print(f"{'think':<8} {'n':>3} {'ttfa_med':>9} {'ttfa_p95':>9} {'total_med':>9} "
                  f"{'comp_tok':>8} {'tok/s':>7} {'rounds':>6}")
            for think in THINK_OPTIONS:
                results: list[dict[str, Any]] = []
                for i in range(args.iters):
                    if i > 0 and args.sleep > 0:
                        await asyncio.sleep(args.sleep)
                    r = await _one_call(http, prompt_name=prompt_name, think=think, use_tools=use_tools)
                    results.append(r)
                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)
                s = _summarize(results)
                label = "(none)" if think is None else ("True" if think else "False")
                if s["n"] == 0:
                    err = s["errors"][0] if s["errors"] else "?"
                    print(f"{label:<8} {0:>3}    ERROR: {err[:140]}")
                    continue
                print(
                    f"{label:<8} {s['n']:>3} "
                    f"{_fmt(s['ttfa_med'], suffix='s'):>9} "
                    f"{_fmt(s['ttfa_p95'], suffix='s'):>9} "
                    f"{_fmt(s['total_med'], suffix='s'):>9} "
                    f"{_fmt(s['comp_tok_med']):>8} "
                    f"{_fmt(s['tok_per_s_med']):>7} "
                    f"{_fmt(s['rounds_med']):>6}"
                )


if __name__ == "__main__":
    asyncio.run(main())
