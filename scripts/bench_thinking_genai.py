"""Latency + token bench for gemma-4-26b-a4b-it via the **native Google GenAI** SDK
(google.genai). Mirrors scripts/bench_thinking.py (which hits the same model
through the Vercel AI Gateway) so the two outputs can be compared side-by-side.

Run with:
    set -a; source .env; set +a
    uv run --with google-genai scripts/bench_thinking_genai.py [--iters 4 --sleep 2]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from typing import Any

from google import genai
from google.genai import types


MODEL = "gemma-4-31b-it"  # overridable via --model

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
    # Same intent as tool_decision but instructs the model to emit a brief
    # user-facing line in the SAME response as the tool call. Tests whether
    # we can cut TTFA from (round_1_total + round_2_ttft) down to round_1_ttft.
    "tool_decision_ack_first": (
        "You are an operator that routes work to a Claude executor. "
        "When you need to call a tool, you MUST first emit one short line of "
        "user-facing text (e.g. 'Dispatching...') and THEN call the tool — "
        "both in the same response. Be terse.\n"
        "User: ssh myserver and list running docker services. try again."
    ),
}

# Google GenAI native tool schema. Mirrors the OpenAI tool in the sibling
# benchmark so the two are comparable.
DISPATCH_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="dispatch_task",
        description="Dispatch a task to the Claude executor.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "prompt": types.Schema(type="STRING"),
                "model": types.Schema(type="STRING", enum=["haiku", "sonnet", "opus"]),
            },
            required=["prompt"],
        ),
    ),
])

# Supported by GenerateContentConfig.ThinkingConfig per the SDK. We probe all
# levels (some may be unsupported by gemma-4 and surface as an error — we
# capture and report it rather than crashing).
LEVELS: list[str | None] = ["MINIMAL", "LOW"]


async def _stream_round(
    client: genai.Client,
    *,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> dict[str, Any]:
    """Run one round and capture first-content-time, total, tokens, function
    call (if any). Returns dict with keys: started, first_at, total_s,
    ttft_s, text, fn_call (FunctionCall | None), comp_tok, reas_tok, error?"""
    started = time.monotonic()
    first_at: float | None = None
    first_text_at: float | None = None
    text_buf: list[str] = []
    fn_call: Any = None
    usage: Any = None
    try:
        stream = await client.aio.models.generate_content_stream(
            model=MODEL, contents=contents, config=config,
        )
        async for chunk in stream:
            cands = getattr(chunk, "candidates", None) or []
            for c in cands:
                if c.content and c.content.parts:
                    for part in c.content.parts:
                        # Skip thought parts — those aren't user-visible.
                        if getattr(part, "thought", False):
                            continue
                        now = time.monotonic()
                        if first_at is None and (part.text or getattr(part, "function_call", None)):
                            first_at = now
                        if part.text:
                            if first_text_at is None:
                                first_text_at = now
                            text_buf.append(part.text)
                        if getattr(part, "function_call", None) is not None and fn_call is None:
                            fn_call = part.function_call
            if getattr(chunk, "usage_metadata", None) is not None:
                usage = chunk.usage_metadata
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "total_s": time.monotonic() - started}
    return {
        "total_s": time.monotonic() - started,
        "ttft_s": (first_at - started) if first_at else None,
        "first_text_s": (first_text_at - started) if first_text_at else None,
        "text": "".join(text_buf),
        "fn_call": fn_call,
        "comp_tok": getattr(usage, "candidates_token_count", None) if usage else None,
        "reas_tok": getattr(usage, "thoughts_token_count", None) if usage else None,
    }


async def _one_call(
    client: genai.Client,
    *,
    prompt_name: str,
    level: str | None,
    use_tools: bool,
) -> dict[str, Any]:
    """Measure latency until the user sees the *first answer*. For tool-using
    turns that's round-1 + tool-exec (synthetic, ~0) + round-2 TTFT — exactly
    what the user actually waits on in production today.
    """
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=PROMPTS[prompt_name])])
    ]
    cfg_kwargs: dict[str, Any] = {"temperature": 0.2}
    if level is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    if use_tools:
        cfg_kwargs["tools"] = [DISPATCH_TOOL]
    config = types.GenerateContentConfig(**cfg_kwargs)

    wall_start = time.monotonic()
    r1 = await _stream_round(client, contents=contents, config=config)
    if "error" in r1:
        return {"error": r1["error"]}

    # No tool requested → first user-visible answer == round-1 first text token.
    if r1["fn_call"] is None or not use_tools:
        return {
            "time_to_first_answer_s": r1["first_text_s"] or r1["ttft_s"],
            "total_s": r1["total_s"],
            "rounds": 1,
            "comp_tok": r1["comp_tok"],
            "reas_tok": r1["reas_tok"],
        }

    # Tool was called AND model also emitted text in round 1 → user already
    # got something to look at before round 2 even starts. This is the
    # "ack-first" path the operator system prompt can encourage.
    if r1["first_text_s"] is not None:
        # Still run round 2 to measure full work, but report TTFA as the
        # round-1 first-text moment.
        contents.append(types.Content(role="model", parts=[types.Part(function_call=r1["fn_call"])]))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=r1["fn_call"].name,
                response={"task_id": "T1", "state": "pending", "model": "sonnet"},
            )],
        ))
        r2 = await _stream_round(client, contents=contents, config=config)
        return {
            "time_to_first_answer_s": r1["first_text_s"],
            "total_s": time.monotonic() - wall_start,
            "rounds": 2,
            "comp_tok": (r1["comp_tok"] or 0) + (r2.get("comp_tok") or 0),
            "reas_tok": (r1["reas_tok"] or 0) + (r2.get("reas_tok") or 0)
                if (r1["reas_tok"] is not None or r2.get("reas_tok") is not None) else None,
            "ack_first": True,
        }

    # Round 2: synthesize the tool result and stream the model's user-facing reply.
    contents.append(types.Content(
        role="model",
        parts=[types.Part(function_call=r1["fn_call"])],
    ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_function_response(
            name=r1["fn_call"].name,
            response={"task_id": "T1", "state": "pending", "model": "sonnet"},
        )],
    ))
    r2 = await _stream_round(client, contents=contents, config=config)
    if "error" in r2:
        return {"error": "round2: " + r2["error"]}
    # User-visible first answer = wall time from request start until round-2 first token.
    time_to_first_answer = (
        r1["total_s"] + r2["ttft_s"] if r2["ttft_s"] is not None else None
    )
    return {
        "time_to_first_answer_s": time_to_first_answer,
        "total_s": time.monotonic() - wall_start,
        "rounds": 2,
        "comp_tok": (r1["comp_tok"] or 0) + (r2["comp_tok"] or 0),
        "reas_tok": (r1["reas_tok"] or 0) + (r2["reas_tok"] or 0)
            if (r1["reas_tok"] is not None or r2["reas_tok"] is not None) else None,
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
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--model", default=MODEL,
                        help=f"Gemini model id (default {MODEL}).")
    args = parser.parse_args()
    MODEL = args.model

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in env")
    client = genai.Client(api_key=api_key)

    for prompt_name, use_tools in [
        ("short_ack", False),
        ("tool_decision", True),
        ("tool_decision_ack_first", True),
    ]:
        print(f"\n=== prompt={prompt_name} use_tools={use_tools} model={MODEL} ===")
        print("ttfa = time-to-first-answer-the-user (round-1-total + round-2-ttft if tools).")
        print(f"{'level':<8} {'n':>3} {'ttfa_med':>9} {'ttfa_p95':>9} {'total_med':>9} "
              f"{'comp_tok':>8} {'reas_tok':>8} {'rounds':>6}")
        for level in LEVELS:
            results: list[dict[str, Any]] = []
            for i in range(args.iters):
                if i > 0 and args.sleep > 0:
                    await asyncio.sleep(args.sleep)
                r = await _one_call(client, prompt_name=prompt_name, level=level, use_tools=use_tools)
                results.append(r)
            if args.sleep > 0:
                await asyncio.sleep(args.sleep)
            s = _summarize(results)
            label = level if level is not None else "(none)"
            if s["n"] == 0:
                err = s["errors"][0] if s["errors"] else "?"
                short = err[:140] + ("…" if len(err) > 140 else "")
                print(f"{label:<8} {0:>3}    ERROR: {short}")
                continue
            print(
                f"{label:<8} {s['n']:>3} "
                f"{_fmt(s['ttfa_med'], suffix='s'):>9} "
                f"{_fmt(s['ttfa_p95'], suffix='s'):>9} "
                f"{_fmt(s['total_med'], suffix='s'):>9} "
                f"{_fmt(s['comp_tok_med']):>8} "
                f"{_fmt(s['reas_tok_med']):>8} "
                f"{_fmt(s['rounds_med']):>6}"
            )


if __name__ == "__main__":
    asyncio.run(main())
