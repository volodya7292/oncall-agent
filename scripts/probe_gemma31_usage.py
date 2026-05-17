"""Dump full usage_metadata for gemma-4-31b-it via Gemini native, to see if
implicit prompt caching kicks in on repeat calls with the same long prefix.

Gemini 2.5+ supports implicit caching for prompts above a minimum token
length; usage_metadata.cached_content_token_count reports the hits. Whether
gemma-4-31b-it participates in that path is what we're verifying here.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from google import genai
from google.genai import types


MODEL = "gemma-4-31b-it"

LONG_PREFIX = (
    "You are a careful on-call operator. Below is your operating manual:\n\n"
    + ("This is filler paragraph. " * 400)
    + "\n\nNow follow the user's request below.\nUser: Reply with a single word: ok."
)


async def call_once(client: genai.Client, *, label: str) -> None:
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=LONG_PREFIX)])]
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=8)
    print(f"\n--- {label} ---")
    started = time.monotonic()
    resp = await client.aio.models.generate_content(model=MODEL, contents=contents, config=cfg)
    total = time.monotonic() - started
    print(f"wall: {total:.3f}s")
    print(f"content: {resp.text!r}")
    usage = resp.usage_metadata
    if usage is None:
        print("(no usage_metadata)")
        return
    dump = usage.model_dump() if hasattr(usage, "model_dump") else {
        k: getattr(usage, k) for k in dir(usage) if not k.startswith("_")
    }
    print("usage_metadata:")
    print(json.dumps(dump, indent=2, default=str))


async def main() -> None:
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    for i in range(3):
        await call_once(client, label=f"call {i+1}")


if __name__ == "__main__":
    asyncio.run(main())
