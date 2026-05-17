"""Latency + token bench for google/gemma-4-26b-a4b-it on the Vercel AI Gateway.

Compares `reasoning_effort` levels (OpenAI Chat Completions API extension that
the Vercel gateway forwards as a unified reasoning knob). We hit the same
operator-style prompt N times per level and report:

  * total wall-clock per call
  * first-token latency (TTFT) — requires streaming
  * completion tokens
  * reasoning tokens (if the provider surfaces them in usage)

Run with:
    set -a; source ~/.oncall/.env; set +a
    uv run scripts/bench_thinking.py [--iters 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any

from openai import AsyncOpenAI


MODEL = os.environ.get("BENCH_MODEL", "zai/glm-4.7-flash")

# Three prompts modelled on real operator workloads.
PROMPTS: dict[str, list[dict[str, Any]]] = {
    "short_ack": [
        {"role": "system", "content": (
            "You are a terse on-call operator. Reply in <= 12 words. "
            "No preamble, no apologies."
        )},
        {"role": "user", "content": "I just dispatched task T1 to investigate API errors. Acknowledge."},
    ],
    "tool_decision": [
        {"role": "system", "content": (
            "You are an operator that routes work to a Claude executor. "
            "When the user asks for an infra check, call dispatch_task. "
            "Be terse in any user-facing text."
        )},
        {"role": "user", "content": "ssh myserver and list running docker services. try again."},
    ],
    # User-facing ack in the SAME response as the tool_call → user sees text
    # at round-1 first-token instead of waiting for round 2.
    "tool_decision_ack_first": [
        {"role": "system", "content": (
            "You are an operator that routes work to a Claude executor. "
            "When you need to call a tool, you MUST first emit one short line of "
            "user-facing text (e.g. 'Dispatching...') and THEN call the tool — "
            "both in the same response. Be terse."
        )},
        {"role": "user", "content": "ssh myserver and list running docker services. try again."},
    ],
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

# `None` = don't send the field (baseline). The string values are forwarded
# to the underlying provider; Vercel maps them to provider-native budgets.
EFFORT_LEVELS: list[str | None] = [None, "minimal", "low", "medium", "high"]


async def _stream_round(
    client: AsyncOpenAI,
    *,
    messages: list[dict[str, Any]],
    effort: str | None,
    use_tools: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"providerOptions": {"gateway": {"sort": "ttft"}}},
    }
    if use_tools:
        kwargs["tools"] = TOOLS
    if effort is not None:
        kwargs["reasoning_effort"] = effort

    started = time.monotonic()
    first_at: float | None = None
    first_text_at: float | None = None
    text_buf: list[str] = []
    tool_calls_assembled: dict[int, dict[str, Any]] = {}
    usage: Any = None
    try:
        stream = await client.chat.completions.create(**kwargs)
        async for event in stream:
            if event.choices:
                d = event.choices[0].delta
                now = time.monotonic()
                if first_at is None and ((d.content or "") or (d.tool_calls or [])):
                    first_at = now
                if d.content:
                    if first_text_at is None:
                        first_text_at = now
                    text_buf.append(d.content)
                for tc in (d.tool_calls or []):
                    slot = tool_calls_assembled.setdefault(tc.index, {
                        "id": "", "name": "", "arguments": "",
                    })
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            if getattr(event, "usage", None) is not None:
                usage = event.usage
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "total_s": time.monotonic() - started}
    tcs = [tool_calls_assembled[i] for i in sorted(tool_calls_assembled)]
    out: dict[str, Any] = {
        "total_s": time.monotonic() - started,
        "ttft_s": (first_at - started) if first_at else None,
        "first_text_s": (first_text_at - started) if first_text_at else None,
        "text": "".join(text_buf),
        "tool_calls": tcs,
    }
    if usage is not None:
        out["comp_tok"] = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        out["reas_tok"] = getattr(details, "reasoning_tokens", None) if details else None
    return out


async def _one_call(
    client: AsyncOpenAI,
    *,
    prompt_name: str,
    effort: str | None,
    use_tools: bool,
) -> dict[str, Any]:
    messages = list(PROMPTS[prompt_name])
    wall_start = time.monotonic()
    r1 = await _stream_round(client, messages=messages, effort=effort, use_tools=use_tools)
    if "error" in r1:
        return {"error": r1["error"]}

    if not use_tools or not r1["tool_calls"]:
        return {
            "time_to_first_answer_s": r1["first_text_s"] or r1["ttft_s"],
            "total_s": r1["total_s"],
            "rounds": 1,
            "comp_tok": r1.get("comp_tok"),
            "reas_tok": r1.get("reas_tok"),
        }

    # Tool was called AND model also emitted user-facing text in round 1.
    if r1["first_text_s"] is not None:
        messages.append({
            "role": "assistant",
            "content": r1["text"] or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in r1["tool_calls"]
            ],
        })
        for tc in r1["tool_calls"]:
            messages.append({
                "role": "tool", "tool_call_id": tc["id"],
                "content": json.dumps({"task_id": "T1", "state": "pending", "model": "sonnet"}),
            })
        r2 = await _stream_round(client, messages=messages, effort=effort, use_tools=use_tools)
        return {
            "time_to_first_answer_s": r1["first_text_s"],
            "total_s": time.monotonic() - wall_start,
            "rounds": 2,
            "comp_tok": (r1.get("comp_tok") or 0) + (r2.get("comp_tok") or 0),
            "reas_tok": (r1.get("reas_tok") or 0) + (r2.get("reas_tok") or 0)
                if (r1.get("reas_tok") is not None or r2.get("reas_tok") is not None) else None,
            "ack_first": True,
        }

    # Plain two-round: emit tool call, then synthetic result, then final text.
    messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in r1["tool_calls"]
        ],
    })
    for tc in r1["tool_calls"]:
        messages.append({
            "role": "tool", "tool_call_id": tc["id"],
            "content": json.dumps({"task_id": "T1", "state": "pending", "model": "sonnet"}),
        })
    r2 = await _stream_round(client, messages=messages, effort=effort, use_tools=use_tools)
    if "error" in r2:
        return {"error": "round2: " + r2["error"]}
    ttfa = r1["total_s"] + (r2["ttft_s"] or 0) if r2["ttft_s"] is not None else None
    return {
        "time_to_first_answer_s": ttfa,
        "total_s": time.monotonic() - wall_start,
        "rounds": 2,
        "comp_tok": (r1.get("comp_tok") or 0) + (r2.get("comp_tok") or 0),
        "reas_tok": (r1.get("reas_tok") or 0) + (r2.get("reas_tok") or 0)
            if (r1.get("reas_tok") is not None or r2.get("reas_tok") is not None) else None,
    }


def _fmt(x: float | None, *, suffix: str = "") -> str:
    return "—" if x is None else f"{x:.2f}{suffix}"


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    errs = [r["error"] for r in results if "error" in r]
    if not ok:
        return {"n": 0, "errors": errs}
    ttfa = [r["time_to_first_answer_s"] for r in ok if r.get("time_to_first_answer_s") is not None]
    totals = [r["total_s"] for r in ok]
    comp = [r.get("comp_tok") for r in ok if r.get("comp_tok")]
    reas = [r.get("reas_tok") for r in ok if r.get("reas_tok")]
    rounds = [r["rounds"] for r in ok]
    return {
        "n": len(ok),
        "errors": errs,
        "ttfa_med": statistics.median(ttfa) if ttfa else None,
        "ttfa_p95": (max(ttfa) if len(ttfa) < 4 else statistics.quantiles(ttfa, n=20)[-1]) if ttfa else None,
        "total_med": statistics.median(totals),
        "comp_tok_med": statistics.median(comp) if comp else None,
        "reas_tok_med": statistics.median(reas) if reas else None,
        "rounds_med": statistics.median(rounds),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--levels", default=None,
        help="comma-separated subset of effort levels to test "
             "(e.g. 'minimal' or 'none,minimal'). Default = all.",
    )
    args = parser.parse_args()
    if args.levels:
        wanted = [x.strip().lower() for x in args.levels.split(",")]
        effort_levels = [
            lvl for lvl in EFFORT_LEVELS
            if (lvl if lvl is not None else "none") in wanted
        ]
    else:
        effort_levels = EFFORT_LEVELS

    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    base_url = os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    if not api_key:
        raise SystemExit("AI_GATEWAY_API_KEY not set in env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    for prompt_name, use_tools in [
        ("short_ack", False),
        ("tool_decision", True),
        ("tool_decision_ack_first", True),
    ]:
        print(f"\n=== prompt={prompt_name} use_tools={use_tools} model={MODEL} ===")
        print(f"{'effort':<10} {'n':>3} {'ttfa_med':>9} {'ttfa_p95':>9} {'total_med':>9} "
              f"{'comp_tok':>8} {'reas_tok':>8} {'rounds':>6}")
        for effort in effort_levels:
            results: list[dict[str, Any]] = []
            for i in range(args.iters):
                if i > 0 and args.sleep > 0:
                    await asyncio.sleep(args.sleep)
                r = await _one_call(client, prompt_name=prompt_name, effort=effort, use_tools=use_tools)
                results.append(r)
            if args.sleep > 0:
                await asyncio.sleep(args.sleep)
            s = _summarize(results)
            label = effort if effort is not None else "(none)"
            if s["n"] == 0:
                err = s["errors"][0] if s["errors"] else "?"
                short = err[:140] + ("…" if len(err) > 140 else "")
                print(f"{label:<10} {0:>3}    ERROR: {short}")
                continue
            print(
                f"{label:<10} {s['n']:>3} "
                f"{_fmt(s['ttfa_med'], suffix='s'):>9} "
                f"{_fmt(s['ttfa_p95'], suffix='s'):>9} "
                f"{_fmt(s['total_med'], suffix='s'):>9} "
                f"{_fmt(s['comp_tok_med']):>8} "
                f"{_fmt(s['reas_tok_med']):>8} "
                f"{_fmt(s['rounds_med']):>6}"
            )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
