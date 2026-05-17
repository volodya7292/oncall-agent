"""Probe which parameter actually toggles thinking on minimax-m2.5 via the
Vercel gateway. We send the same prompt with several candidate knobs and
read back `usage.completion_tokens_details.reasoning_tokens` — if it drops
to 0 (or near it), that parameter works.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI


MODEL = "minimax/minimax-m2.5"

VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("baseline (no knob)", {}),
    ("reasoning_effort=minimal", {"reasoning_effort": "minimal"}),
    ("extra_body.enable_thinking=False", {
        "extra_body": {"enable_thinking": False},
    }),
    ("providerOptions.minimax.enable_thinking=False", {
        "extra_body": {"providerOptions": {"minimax": {"enable_thinking": False}}},
    }),
    ("providerOptions.minimax.thinking=False", {
        "extra_body": {"providerOptions": {"minimax": {"thinking": False}}},
    }),
    ("providerOptions.minimax.reasoning_effort=minimal", {
        "extra_body": {"providerOptions": {"minimax": {"reasoning_effort": "minimal"}}},
    }),
    ("chat_template_kwargs.enable_thinking=False", {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }),
]


async def main() -> None:
    api_key = os.environ["AI_GATEWAY_API_KEY"]
    base_url = os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": "You are a terse on-call operator. Reply in <=12 words."},
        {"role": "user", "content": "I just dispatched task T1 to investigate API errors. Acknowledge."},
    ]
    print(f"{'variant':<60} {'comp':>5} {'reas':>5}  content")
    for label, kwargs in VARIANTS:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=64,
                **kwargs,
            )
        except Exception as e:
            print(f"{label:<60}  ERROR: {type(e).__name__}: {str(e)[:80]}")
            continue
        usage = resp.usage
        comp = usage.completion_tokens if usage else None
        reas = None
        if usage and usage.completion_tokens_details:
            reas = usage.completion_tokens_details.reasoning_tokens
        content = (resp.choices[0].message.content or "").replace("\n", " ")[:60]
        print(f"{label:<60} {comp!s:>5} {reas!s:>5}  {content!r}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
