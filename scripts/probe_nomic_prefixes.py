"""Compare nomic-embed-text retrieval quality with vs. without task prefixes.
nomic-embed-v1.5 is trained on tasks like:
    search_query: <user query>
    search_document: <stored fact>

If the prefixes meaningfully widen the gap between cosine(query, related)
and cosine(query, unrelated), we should adopt them in production. If they
don't, the extra plumbing isn't worth it.
"""

from __future__ import annotations

import asyncio
import statistics

import httpx
import numpy as np


MODEL = "nomic-embed-text:137m-v1.5-fp16"
OLLAMA = "http://localhost:11434"


PAIRS = [
    # (query, related_doc, unrelated_doc)
    (
        "where does staging live again?",
        "the staging API runs at api-staging.example.com on port 8443",
        "alex is the on-call lead for the payments team",
    ),
    (
        "what's the prod db named",
        "the prod database is named pg-prod-1",
        "the staging API runs at api-staging.example.com on port 8443",
    ),
    (
        "who do i page for payments stuff",
        "alex is the on-call lead for the payments team",
        "the prod database is named pg-prod-1",
    ),
    (
        "ssh into the docker host",
        "myserver is the host running the docker services",
        "the postgres readonly replica is pg-readonly-2",
    ),
    (
        "is there a merge freeze active",
        "merge freeze begins 2026-03-05 for the mobile release cut",
        "renovate auto-merges go through after 3 green builds",
    ),
]


async def embed(http: httpx.AsyncClient, text: str) -> np.ndarray:
    r = await http.post(f"{OLLAMA}/api/embed", json={"model": MODEL, "input": text})
    r.raise_for_status()
    return np.asarray(r.json()["embeddings"][0], dtype=np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def main() -> None:
    async with httpx.AsyncClient(timeout=30.0) as http:
        print(f"{'mode':<14} {'query[:35]':<37} {'rel':>5} {'unrel':>6} {'gap':>5}")
        for mode, qprefix, dprefix in [
            ("no prefixes", "", ""),
            ("with prefixes", "search_query: ", "search_document: "),
        ]:
            gaps: list[float] = []
            rels: list[float] = []
            unrels: list[float] = []
            for q, rel, unrel in PAIRS:
                qv = await embed(http, qprefix + q)
                rv = await embed(http, dprefix + rel)
                uv = await embed(http, dprefix + unrel)
                c_rel = cos(qv, rv)
                c_unrel = cos(qv, uv)
                gap = c_rel - c_unrel
                rels.append(c_rel)
                unrels.append(c_unrel)
                gaps.append(gap)
                print(f"{mode:<14} {q[:35]:<37} {c_rel:>5.3f} {c_unrel:>6.3f} {gap:>5.3f}")
            print(
                f"{mode:<14} {'(summary)':<37} "
                f"{statistics.mean(rels):>5.3f} {statistics.mean(unrels):>6.3f} "
                f"{statistics.mean(gaps):>5.3f}"
            )
            print()


if __name__ == "__main__":
    asyncio.run(main())
