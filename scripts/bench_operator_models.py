"""E2E latency bench: gemini-3.5-flash-lite (native AI Studio) vs any set of
OpenRouter models (grok-4.20, grok-4.5, ...), at the operator's real prompt shape.

What it measures is one operator LLM round-trip — exactly what `timed("operator")`
wraps in `Operator.chat_turn` — so the numbers are comparable to the table in
`config.py`. The request shape mirrors `GenAILLMClient.chat` /
`OpenRouterLLMClient.chat` field for field (same temperature, same max_tokens,
same thinking dial translation); the only reason this doesn't import those
clients is that they discard `usage`, and reasoning-token count is the thing
that explains grok's latency curve.

Context sizes are the ones the config table is calibrated on (11k/16k/32k),
because the operator carries a big rolling history and the models diverge with
context far more than they do on a cold prompt.

Run with:
    set -a; source .env; set +a
    uv run --with google-genai --with openai scripts/bench_operator_models.py

    # cheaper smoke run
    uv run ... scripts/bench_operator_models.py --iters 2 --ctx 11000

    # one model only
    uv run ... scripts/bench_operator_models.py \
        --gemini-levels "" --or-models "x-ai/grok-4.5:low,medium"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncall.operator import OPERATOR_TOOLS  # noqa: E402

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "oncall" / "prompts"

GEMINI_MODEL = os.environ.get("BENCH_GEMINI_MODEL", "gemini-3.5-flash-lite")

# OpenRouter endpoint tag. `xai/zdr/priority` is the zero-data-retention
# priority endpoint — 2x the base xAI price. Fallbacks are OFF: a silent reroute
# to the base endpoint would quietly measure something other than what we pinned
# (and would leave ZDR behind, which is the reason for pinning in the first
# place). The operator's real client keeps fallbacks ON — worth knowing that
# these numbers are the pinned-endpoint case, not the fallback case.
OR_PROVIDER = os.environ.get("BENCH_OR_PROVIDER", "xai/zdr/priority")

# model -> "effort,effort,..." for the OpenRouter side. Efforts are empirical:
# grok-4.20 advertises no supported_efforts at all, and grok-4.5 is
# reasoning-mandatory (no "none" row is possible for it).
OR_MODELS_DEFAULT = "x-ai/grok-4.20:none,low,medium x-ai/grok-4.5:low,medium"

# $/token in/out. Gemini is AI Studio's list price for 3.5-flash-lite; the
# OpenRouter entries are filled in at startup from the pinned endpoint's own
# pricing, so the cost column can't drift from what the tag actually bills.
# Output includes reasoning tokens on both providers.
PRICING: dict[str, tuple[float, float]] = {
    GEMINI_MODEL: (0.30e-6, 2.50e-6),
}


def load_or_pricing(models: list[str], tag: str) -> None:
    """Fill PRICING from OpenRouter's endpoint listing for the pinned tag."""
    import urllib.request

    for model in models:
        url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)["data"]
        match = [e for e in data["endpoints"] if e.get("tag") == tag]
        if not match:
            raise SystemExit(
                f"{model}: no endpoint tagged {tag!r} "
                f"(have: {[e.get('tag') for e in data['endpoints']]})"
            )
        p = match[0]["pricing"]
        PRICING[model] = (float(p["prompt"]), float(p["completion"]))

# Matches the operator's own call site.
MAX_TOKENS = 2048
CTX_SIZES = [11_000, 16_000, 32_000]

# Two turn shapes the operator actually sees. `inline` should be answered
# without a tool; `handoff` should produce a hand_off call. Both are ONE
# round-trip — the operator's user-visible wait for round 1 either way, since
# the ack rides inside hand_off's ack_msg argument.
TURNS = {
    "inline": "how many tasks are running right now?",
    "handoff": "ssh into the prod box and check why nginx is throwing 502s",
}

# Filler turn used to inflate history to a target context size. Realistic in
# shape (operator dispatch chatter), which matters: a wall of lorem ipsum
# compresses differently and would flatter whichever tokenizer is greedier.
_FILLER_USER = (
    "check the disk usage on the build host and tell me if the artifact cache "
    "is what's eating it — last time it was 40GB of stale wheels and I had to "
    "clear it by hand. if it's the cache again just clear anything older than "
    "a week, otherwise report back before touching anything."
)
_FILLER_ASSISTANT = (
    "On it — dispatched. Build host is at 87% with the artifact cache holding "
    "38GB, mostly wheels from the nightly runs that never got reaped. Cleared "
    "everything older than seven days, freed 31GB, host is at 22% now. The "
    "reaper cron is present but its unit failed to load after the last "
    "upgrade, which is why this keeps recurring; I left it stopped rather "
    "than re-enabling it without asking."
)


def _est_tokens(text: str) -> int:
    """~4 chars/token. Only used to hit the padding target; every reported
    prompt-token figure comes from the provider's own usage payload."""
    return len(text) // 4


def build_messages(target_ctx: int, turn: str) -> list[dict[str, Any]]:
    """Real system prompt + synthetic rolling history padded toward
    `target_ctx` prompt tokens + the turn under test."""
    system = (PROMPTS_DIR / "operator_system.md").read_text()
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    used = _est_tokens(system) + sum(
        _est_tokens(str(t["function"]["description"])) for t in OPERATOR_TOOLS
    )
    pair_cost = _est_tokens(_FILLER_USER) + _est_tokens(_FILLER_ASSISTANT)
    final = _est_tokens(TURNS[turn])
    n_pairs = max(0, (target_ctx - used - final) // pair_cost)
    for i in range(n_pairs):
        messages.append({"role": "user", "content": f"[{i}] {_FILLER_USER}"})
        messages.append({"role": "assistant", "content": f"[{i}] {_FILLER_ASSISTANT}"})

    messages.append({"role": "user", "content": TURNS[turn]})
    return messages


# --------------------------------------------------------------------------
# Backends. Each returns:
#   {total_s, ttft_s, prompt_tok, out_tok, reas_tok, tool: str|None} or {error}
# --------------------------------------------------------------------------


async def call_gemini(
    client: Any, *, messages: list[dict[str, Any]], effort: str | None,
) -> dict[str, Any]:
    """Mirrors GenAILLMClient.chat: streamed, temperature 0.2, system_instruction
    split out, thinking_level from the effort dial."""
    from google.genai import types

    system_chunks = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    cfg_kwargs: dict[str, Any] = {
        "temperature": 0.2,
        "max_output_tokens": MAX_TOKENS,
        "system_instruction": "\n\n".join(system_chunks),
        "tools": [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters=t["function"].get("parameters") or None,
            )
            for t in OPERATOR_TOOLS
        ])],
    }
    if effort is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=effort.upper(),
        )

    started = time.monotonic()
    first_at: float | None = None
    fn_name: str | None = None
    usage: Any = None
    try:
        stream = await client.aio.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        async for chunk in stream:
            for c in getattr(chunk, "candidates", None) or []:
                for part in (c.content.parts if c.content and c.content.parts else []):
                    if getattr(part, "thought", False):
                        continue
                    if first_at is None and (part.text or getattr(part, "function_call", None)):
                        first_at = time.monotonic()
                    fc = getattr(part, "function_call", None)
                    if fc is not None and fn_name is None:
                        fn_name = fc.name
            if getattr(chunk, "usage_metadata", None) is not None:
                usage = chunk.usage_metadata
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "total_s": time.monotonic() - started}

    return {
        "total_s": time.monotonic() - started,
        "ttft_s": (first_at - started) if first_at else None,
        "prompt_tok": getattr(usage, "prompt_token_count", None) if usage else None,
        "out_tok": ((getattr(usage, "candidates_token_count", None) or 0)
                    + (getattr(usage, "thoughts_token_count", None) or 0)) if usage else None,
        "reas_tok": getattr(usage, "thoughts_token_count", None) if usage else None,
        "tool": fn_name,
    }


async def call_openrouter(
    client: Any, *, model: str, messages: list[dict[str, Any]], effort: str | None,
) -> dict[str, Any]:
    """Mirrors OpenRouterLLMClient.chat, plus streaming + usage so TTFT and
    reasoning tokens are visible. `effort="none"` takes the same
    reasoning.enabled=false path the real client uses."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": OPERATOR_TOOLS,
        "max_completion_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    extra_body: dict[str, Any] = {
        "provider": {"order": [OR_PROVIDER], "allow_fallbacks": False},
    }
    if effort is not None:
        if effort.lower() in ("none", "off", "disabled"):
            extra_body["reasoning"] = {"enabled": False}
        else:
            kwargs["reasoning_effort"] = effort
    kwargs["extra_body"] = extra_body

    started = time.monotonic()
    first_at: float | None = None
    fn_name: str | None = None
    usage: Any = None
    try:
        stream = await client.chat.completions.create(**kwargs)
        async for event in stream:
            if event.choices:
                d = event.choices[0].delta
                if first_at is None and ((d.content or "") or (d.tool_calls or [])):
                    first_at = time.monotonic()
                for tc in (d.tool_calls or []):
                    if tc.function and tc.function.name and fn_name is None:
                        fn_name = tc.function.name
            if getattr(event, "usage", None) is not None:
                usage = event.usage
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "total_s": time.monotonic() - started}

    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "total_s": time.monotonic() - started,
        "ttft_s": (first_at - started) if first_at else None,
        "prompt_tok": getattr(usage, "prompt_tokens", None) if usage else None,
        "out_tok": getattr(usage, "completion_tokens", None) if usage else None,
        "reas_tok": getattr(details, "reasoning_tokens", None) if details else None,
        "tool": fn_name,
    }


def _cost(model: str, prompt_tok: int | None, out_tok: int | None) -> float | None:
    if prompt_tok is None or out_tok is None:
        return None
    pin, pout = PRICING[model]
    return prompt_tok * pin + out_tok * pout


def _fmt(x: float | None, fmt: str = "{:.2f}") -> str:
    return "—" if x is None else fmt.format(x)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    errs = [r["error"] for r in results if "error" in r]
    if not ok:
        return {"n": 0, "errors": errs}
    totals = [r["total_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    reas = [r["reas_tok"] for r in ok if r.get("reas_tok") is not None]
    return {
        "n": len(ok),
        "errors": errs,
        "total_med": statistics.median(totals),
        "total_max": max(totals),
        "ttft_med": statistics.median(ttfts) if ttfts else None,
        "prompt_tok": statistics.median([r["prompt_tok"] for r in ok if r.get("prompt_tok")])
                      if any(r.get("prompt_tok") for r in ok) else None,
        "out_tok": statistics.median([r["out_tok"] for r in ok if r.get("out_tok") is not None])
                   if any(r.get("out_tok") is not None for r in ok) else None,
        "reas_med": statistics.median(reas) if reas else None,
        "tool_rate": sum(1 for r in ok if r.get("tool")) / len(ok),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--ctx", type=int, nargs="*", default=CTX_SIZES)
    ap.add_argument("--turns", nargs="*", default=list(TURNS))
    ap.add_argument("--gemini-levels", default="minimal,low",
                    help="empty string skips the gemini side entirely")
    ap.add_argument(
        "--or-models", nargs="*", default=OR_MODELS_DEFAULT.split(),
        help="'<slug>:<effort,effort,...>'. 'none' means reasoning.enabled=false "
             "— only valid on slugs where reasoning is optional (grok-4.20 is, "
             "grok-4.5 is not). Efforts are empirical: grok-4.20 advertises no "
             "supported_efforts at all.",
    )
    args = ap.parse_args()

    from google import genai
    from openai import AsyncOpenAI

    gem_key = os.environ.get("GEMINI_API_KEY")
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not gem_key or not or_key:
        raise SystemExit("GEMINI_API_KEY and OPENROUTER_API_KEY must both be set")

    gem_client = genai.Client(api_key=gem_key)
    or_client = AsyncOpenAI(
        api_key=or_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=120.0,
        max_retries=1,
    )

    configs: list[tuple[str, str, str | None]] = [
        (GEMINI_MODEL, "gemini", lvl)
        for lvl in args.gemini_levels.split(",") if lvl
    ]
    or_models: list[str] = []
    for spec in args.or_models:
        slug, _, levels = spec.partition(":")
        or_models.append(slug)
        configs += [(slug, "openrouter", lvl) for lvl in levels.split(",") if lvl]
    load_or_pricing(or_models, OR_PROVIDER)

    for turn in args.turns:
        for ctx in args.ctx:
            messages = build_messages(ctx, turn)
            print(f"\n=== turn={turn} target_ctx={ctx} iters={args.iters} ===")
            print(f"{'model':<24} {'effort':<8} {'n':>2} {'tot_med':>8} {'tot_max':>8} "
                  f"{'ttft':>7} {'prompt':>7} {'out':>6} {'reas':>6} {'tool':>5} {'$/call':>9}")
            for model, backend, effort in configs:
                results: list[dict[str, Any]] = []
                for i in range(args.iters):
                    if i:
                        await asyncio.sleep(args.sleep)
                    if backend == "gemini":
                        r = await call_gemini(gem_client, messages=messages, effort=effort)
                    else:
                        r = await call_openrouter(
                            or_client, model=model, messages=messages, effort=effort,
                        )
                    results.append(r)
                s = _summarize(results)
                label = model.split("/")[-1]
                if s["n"] == 0:
                    err = (s["errors"][0] if s["errors"] else "?")[:110]
                    print(f"{label:<24} {effort or '(unset)':<8}  0    ERROR: {err}")
                    continue
                cost = _cost(model, s["prompt_tok"], s["out_tok"])
                print(
                    f"{label:<24} {effort or '(unset)':<8} {s['n']:>2} "
                    f"{_fmt(s['total_med']) + 's':>8} {_fmt(s['total_max']) + 's':>8} "
                    f"{_fmt(s['ttft_med']) + 's':>7} "
                    f"{_fmt(s['prompt_tok'], '{:.0f}'):>7} "
                    f"{_fmt(s['out_tok'], '{:.0f}'):>6} "
                    f"{_fmt(s['reas_med'], '{:.0f}'):>6} "
                    f"{s['tool_rate'] * 100:>4.0f}% "
                    f"{_fmt(cost, '${:.5f}'):>9}"
                )
                if s["errors"]:
                    print(f"{'':<24} {'':<8}    partial errors: {s['errors'][0][:90]}")

    await or_client.close()


if __name__ == "__main__":
    asyncio.run(main())
