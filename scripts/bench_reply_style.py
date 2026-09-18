"""Reply-STYLE bench: does a model, under the real operator prompt, answer like
a person in a messenger chat (short bubbles, several per reply, no closing
period, no markdown)?

Contexts come from a Telegram Desktop JSON export of a 1:1 chat (never commit
one). For each sampled point where the friend replied to the owner, the last
turns become history (owner -> user, friend -> assistant, a burst joined by
blank lines, which is exactly the operator's bubble convention), the model
answers, and the answer is split with the production splitter. The friend's
real reply is scored the same way as the reference row.

    set -a; source .env; set +a
    uv run scripts/bench_reply_style.py --export path/to/result.json \
        --owner "Owner Name" --models "google/gemini-3.8-flash:low deepseek/deepseek-v4.1-flash:none"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncall.operator import OPERATOR_TOOLS, OpenRouterLLMClient  # noqa: E402
from oncall.telegram_format import split_messenger_reply  # noqa: E402

PROMPT = Path(__file__).resolve().parent.parent / "src/oncall/prompts/operator_system.md"
BURST_GAP_S = 120
STANDALONE = [  # no-history turns in the owner's other chat language
    "hallo",
    "Warum bildet sich Eis in alten Kühlschränken?",
    "Verkaufen Supermärkte rohe Garnelen",
    "was hältst du von pineapple auf pizza",
]


def _text(m: dict) -> str:
    t = m.get("text", "")
    return t if isinstance(t, str) else "".join(x if isinstance(x, str) else x["text"] for x in t)


def load_bursts(path: str) -> list[tuple[str, list[str]]]:
    msgs = [
        m for m in json.load(open(path))["messages"]
        if m["type"] == "message" and "forwarded_from" not in m
        and not m.get("media_type") and "photo" not in m and "file" not in m and _text(m).strip()
    ]
    bursts: list[tuple[str, list[str], int]] = []
    for m in msgs:
        ts = int(m["date_unixtime"])
        if bursts and bursts[-1][0] == m["from"] and ts - bursts[-1][2] <= BURST_GAP_S:
            bursts[-1][1].append(_text(m)); bursts[-1] = (m["from"], bursts[-1][1], ts)
        else:
            bursts.append((m["from"], [_text(m)], ts))
    return [(who, parts) for who, parts, _ in bursts]


def metrics(replies: list[list[str]]) -> dict[str, float]:
    parts = [p for r in replies for p in r]
    if not parts:
        return {}
    return {
        "bubbles/reply": st.mean(len(r) for r in replies),
        "multi%": 100 * sum(len(r) > 1 for r in replies) / len(replies),
        "chars med": st.median(len(p) for p in parts),
        "chars p90": sorted(len(p) for p in parts)[int(len(parts) * 0.9)],
        "reply chars med": st.median(sum(len(p) for p in r) for r in replies),
        "end '.'%": 100 * sum(p.rstrip().endswith(".") for p in parts) / len(parts),
        "markdown%": 100 * sum(bool(re.search(r"\*\*|^#|^\s*[-*] ", p, re.M)) for p in parts) / len(parts),
        "emoji%": 100 * sum(any(ord(c) >= 0x2600 for c in p) for p in parts) / len(parts),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--owner", required=True, help="export display name of the owner side")
    ap.add_argument("--models", default="google/gemini-3.8-flash:low deepseek/deepseek-v4.1-flash:none")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--history", type=int, default=12, help="bursts of context")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    bursts = load_bursts(a.export)
    random.seed(7)
    idx = random.sample(
        [i for i in range(a.history, len(bursts)) if bursts[i][0] != a.owner and bursts[i - 1][0] == a.owner],
        a.n,
    )
    system = PROMPT.read_text().replace("{{agent_name}}", "Agent").replace("{{owner_name}}", "Owner")
    tail = [
        {"role": "user", "content": "<acting-status>idle</acting-status>"},
        {"role": "user", "content": "<call-status>not on a call</call-status>"},
    ]
    cases: list[list[dict]] = []
    for i in idx:
        hist = [
            {"role": "user" if who == a.owner else "assistant", "content": "\n\n".join(parts)}
            for who, parts in bursts[i - a.history:i]
        ]
        cases.append([{"role": "system", "content": system}, *hist, *tail])
    for q in STANDALONE:
        cases.append([{"role": "system", "content": system}, {"role": "user", "content": q}, *tail])

    rows = {"REFERENCE (real friend)": metrics([bursts[i][1] for i in idx])}
    dump: dict[str, list] = {}
    client = OpenRouterLLMClient("https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(6)
    for spec in a.models.split():
        model, _, effort = spec.partition(":")

        async def one(msgs):
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.chat(model=model, messages=msgs, tools=OPERATOR_TOOLS,
                                          max_tokens=2048, reasoning_effort=effort or None)
                except Exception as e:  # noqa: BLE001
                    print(f"  {model}: call failed: {e}", file=sys.stderr)
                    return None
                return r, time.perf_counter() - t

        res = [x for x in await asyncio.gather(*(one(c) for c in cases)) if x]
        direct, handoffs, lat = [], 0, []
        for r, dt in res:
            lat.append(dt)
            text = (r["content"] or "").strip()
            if r["tool_calls"]:
                handoffs += 1
                args = json.loads(r["tool_calls"][0]["arguments_json"] or "{}")
                text = text or (args.get("answer") or args.get("ack_msg") or "")
            if text:
                direct.append(split_messenger_reply(text))
        m = metrics(direct)
        m["handoff%"] = 100 * handoffs / max(len(res), 1)
        m["lat med s"] = st.median(lat) if lat else 0
        m["ok"] = len(res)
        rows[spec] = m
        dump[spec] = [{"last_user": c[-3]["content"][-200:], "reply": d} for c, d in zip(cases, direct)]

    cols = ["bubbles/reply", "multi%", "chars med", "chars p90", "reply chars med",
            "end '.'%", "markdown%", "emoji%", "handoff%", "lat med s", "ok"]
    print(f"{'':42}" + "".join(f"{c:>16}" for c in cols))
    for name, m in rows.items():
        print(f"{name:42}" + "".join(f"{m.get(c, float('nan')):>16.2f}" for c in cols))
    if a.dump:
        Path(a.dump).write_text(json.dumps(dump, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
