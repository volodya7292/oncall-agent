"""Dump the full chat-completion response + usage object for zai/glm-4.7-flash.
We make two back-to-back calls with the SAME long prompt — if the gateway or
the provider caches, the second call's usage should show `cached_tokens` (or
similar) > 0, and the second call's latency should drop.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from openai import AsyncOpenAI


MODEL = "zai/glm-4.7-flash"

# Make the prompt long enough that caching would be worth the gateway's while.
# Most caches require a minimum prefix length (~1k tokens).
LONG_PREFIX = (
    "You are a careful on-call operator. Below is your operating manual:\n\n"
    + ("This is filler paragraph. " * 400)
    + "\n\nNow follow the user's request below.\n"
)


async def call_once(client: AsyncOpenAI, *, label: str) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": LONG_PREFIX},
        {"role": "user", "content": "Reply with a single word: ok."},
    ]
    print(f"\n--- {label} ---")
    started = time.monotonic()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_completion_tokens=8,
        extra_body={"providerOptions": {"gateway": {"sort": "ttft"}}},
    )
    total = time.monotonic() - started
    print(f"wall: {total:.3f}s")
    print(f"content: {resp.choices[0].message.content!r}")
    # Usage is a Pydantic model in the openai SDK — model_dump gives us
    # everything the gateway forwarded.
    print("usage:")
    print(json.dumps(resp.usage.model_dump() if resp.usage else None, indent=2))
    # The raw response often hides provider-side extras under a custom field.
    raw = resp.model_dump()
    extras = {k: v for k, v in raw.items() if k not in {
        "id", "choices", "created", "model", "object", "usage", "system_fingerprint",
        "service_tier",
    }}
    if extras:
        print("response extras:")
        print(json.dumps(extras, indent=2, default=str))


async def main() -> None:
    api_key = os.environ["AI_GATEWAY_API_KEY"]
    base_url = os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    # Three calls back-to-back. If anything caches, calls 2/3 will show it.
    for i in range(3):
        await call_once(client, label=f"call {i+1}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
