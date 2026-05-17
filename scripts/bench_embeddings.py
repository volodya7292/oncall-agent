"""Compare embedding latency:
  - Local: ollama `embeddinggemma:300m` (308M, 768-dim)
  - Remote: Vercel gateway `alibaba/qwen3-embedding-8b` (8B, 4096-dim — current)

Same short query as a typical user turn. Reports per-call wall time, dim,
median, and p95 over N iterations.

Run:
    set -a; source ~/.oncall/.env; set +a
    uv run --with httpx scripts/bench_embeddings.py [--iters 10]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from typing import Any

import httpx


QUERIES = [
    "try again",
    "check staging is up",
    "ssh myserver and list running docker services",
    "what's the status of T1?",
]


async def ollama_embed(http: httpx.AsyncClient, *, model: str, text: str) -> dict[str, Any]:
    started = time.monotonic()
    r = await http.post(
        "http://localhost:11434/api/embed",
        json={"model": model, "input": text},
    )
    r.raise_for_status()
    body = r.json()
    vec = body["embeddings"][0]
    return {"wall_s": time.monotonic() - started, "dim": len(vec)}


async def vercel_embed(http: httpx.AsyncClient, *, model: str, text: str, api_key: str, base_url: str) -> dict[str, Any]:
    started = time.monotonic()
    r = await http.post(
        f"{base_url}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": [text]},
    )
    r.raise_for_status()
    body = r.json()
    vec = body["data"][0]["embedding"]
    return {"wall_s": time.monotonic() - started, "dim": len(vec)}


def _stats(samples: list[float]) -> tuple[float, float]:
    med = statistics.median(samples)
    p95 = max(samples) if len(samples) < 4 else statistics.quantiles(samples, n=20)[-1]
    return med, p95


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
    base_url = os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")

    async with httpx.AsyncClient(timeout=60.0) as http:
        # Warm up each backend with one call.
        print("warming up…")
        for m in ("embeddinggemma:300m", "nomic-embed-text:137m-v1.5-fp16"):
            try:
                await ollama_embed(http, model=m, text="warm")
            except Exception as e:
                print(f"ollama warmup ({m}) failed: {e}")
        if api_key:
            try:
                await vercel_embed(
                    http, model="alibaba/qwen3-embedding-8b",
                    text="warm", api_key=api_key, base_url=base_url,
                )
            except Exception as e:
                print(f"vercel warmup failed: {e}")

        for label, embed_call in [
            ("ollama/embeddinggemma:300m (local)", lambda t: ollama_embed(
                http, model="embeddinggemma:300m", text=t,
            )),
            ("ollama/nomic-embed-text:137m-v1.5-fp16 (local)", lambda t: ollama_embed(
                http, model="nomic-embed-text:137m-v1.5-fp16", text=t,
            )),
            ("vercel/alibaba/qwen3-embedding-8b (remote)", lambda t: vercel_embed(
                http, model="alibaba/qwen3-embedding-8b",
                text=t, api_key=api_key, base_url=base_url,
            )),
        ]:
            if "vercel" in label and not api_key:
                print(f"\n{label}: skipped (no AI_GATEWAY_API_KEY)")
                continue
            print(f"\n=== {label} ===")
            print(f"{'query':<55} {'wall':>7} {'dim':>5}")
            samples: list[float] = []
            dim = None
            for _ in range(args.iters // len(QUERIES) + 1):
                for q in QUERIES:
                    try:
                        r = await embed_call(q)
                    except Exception as e:
                        print(f"{q[:50]:<55}  ERROR: {type(e).__name__}: {e}")
                        continue
                    samples.append(r["wall_s"])
                    dim = r["dim"]
                    print(f"{q[:50]:<55} {r['wall_s']*1000:>6.0f}ms {dim:>5}")
                    if len(samples) >= args.iters:
                        break
                if len(samples) >= args.iters:
                    break
            if samples:
                med, p95 = _stats(samples)
                print(f"--- summary (n={len(samples)}): median={med*1000:.0f}ms  p95={p95*1000:.0f}ms  dim={dim} ---")


if __name__ == "__main__":
    asyncio.run(main())
