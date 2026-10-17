"""Owner-style learning and reply rewriting.

The rewrite sits on every chat reply to the owner, so the property that
matters most is that it can never lose or mangle a message: any failure sends
the original, and exempt text (system notices, code) never reaches the LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from oncall.db import Database
from oncall.owner_style import OwnerStyle, reply_bursts
from oncall.telegram_agent import TelegramAgentService


OWNER = 4242
STYLE = {
    "profile": "lowercase, terse, no closing period",
    "examples": [{
        "context": ["did it work?"], "reply": ["yep", "all good"],
        "thoughts": "relieved", "writing_style": "two short bubbles",
    }],
}


class _ScriptedLLM:
    """Answers each call with the next scripted item; an Exception is raised."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"role": "assistant", "content": json.dumps(item), "tool_calls": []}


class _MemoryDB:
    def __init__(self, style: dict[str, Any] | None) -> None:
        self.style = style

    async def get_owner_style(self) -> dict[str, Any] | None:
        return self.style


def _agent(style: OwnerStyle) -> tuple[TelegramAgentService, list[str]]:
    sent: list[str] = []

    async def send_message(_peer, text, **_kw):
        sent.append(text)

    svc = TelegramAgentService(
        client=SimpleNamespace(send_message=send_message),
        operator=object(), events=object(), owner_user_id=OWNER,
        broker=object(), db=object(), owner_style=style,
    )
    return svc, sent


def test_reply_bursts_group_runs_and_drop_pasted_blocks():
    thread = [
        {"outgoing": True, "text": "hey"},               # no context: chat start
        {"outgoing": False, "text": "a"},
        {"outgoing": False, "text": "b"},
        {"outgoing": False, "text": "c"},
        {"outgoing": False, "text": "d"},
        {"outgoing": True, "text": "ok"},
        {"outgoing": True, "text": "on it"},             # same reply, 2nd bubble
        {"outgoing": False, "text": "send the log"},
        {"outgoing": True, "text": "x" * 401},           # pasted, not the voice
        {"outgoing": True, "text": "there"},             # new reply, no context
    ]
    assert reply_bursts(thread) == [
        {"context": [], "reply": ["hey"]},
        {"context": ["b", "c", "d"], "reply": ["ok", "on it"]},
        {"context": [], "reply": ["there"]},
    ]


async def test_rewritten_reply_is_sent_as_the_models_bubbles():
    llm = _ScriptedLLM({"messages": ["done", " ", "3 files"]})
    svc, sent = _agent(OwnerStyle(llm=llm, db=_MemoryDB(STYLE), model="m"))
    await svc._send("I finished the task. Three files changed.", styled=True)
    assert sent == ["done", "3 files"]
    assert llm.calls[0]["response_schema"]["required"] == ["messages"]


async def test_failed_or_empty_rewrite_sends_the_original():
    for outcome in (TimeoutError(), {"messages": []}, {"wrong": 1}):
        llm = _ScriptedLLM(outcome)
        svc, sent = _agent(OwnerStyle(llm=llm, db=_MemoryDB(STYLE), model="m"))
        await svc._send("first thought\n\nsecond thought", styled=True)
        assert sent == ["first thought\n\nsecond thought"], outcome


async def test_quotes_go_out_verbatim_or_the_original_does():
    """Regression: a report of a message sent to a contact quoted only its
    intro and paraphrased the rest in German; the rewrite rendered that
    paraphrase in Russian, reading like a quote of text never sent."""
    original = "Gesendet. Beginnt mit „сори, это агент)“, Rest: Vertrag ist aus."
    quote = "„сори, это агент)“"
    cases = [
        (["gesendet", "mit ⟦1⟧ am anfang", "rest: vertrag aus"],
         ["gesendet", f"mit {quote} am anfang", "rest: vertrag aus"]),
        (["gesendet, mit ⟦1⟧", "контракт закончился"], [original]),  # fake quote
        (["gesendet", "vertrag aus"], [original]),                     # quote dropped
        (["⟦1⟧ ⟦1⟧", "vertrag aus"], [original]),                      # quote repeated
    ]
    for reply, expected in cases:
        llm = _ScriptedLLM({"messages": reply})
        svc, sent = _agent(OwnerStyle(llm=llm, db=_MemoryDB(STYLE), model="m"))
        await svc._send(original, styled=True)
        assert sent == expected, reply
        assert "сори" not in llm.calls[0]["messages"][1]["content"]


async def test_exempt_text_never_reaches_the_llm():
    llm = _ScriptedLLM()  # any call would pop from an empty script and fail
    svc, sent = _agent(OwnerStyle(llm=llm, db=_MemoryDB(STYLE), model="m"))
    await svc._send("SYSTEM: memory saved", styled=True)
    await svc._send("run this:\n```\nls -la\n```", styled=True)
    await svc._send("Usage: /infer-style")  # unstyled: slash-command output
    no_style, sent2 = _agent(OwnerStyle(llm=llm, db=_MemoryDB(None), model="m"))
    await no_style._send("hello there", styled=True)
    assert llm.calls == []
    assert len(sent) == 3 and sent2 == ["hello there"]


async def test_infer_survives_a_failed_batch_and_persists(tmp_path):
    # 20 replies in one chat → two analysis batches (15 + 5); the first fails.
    thread: list[dict[str, Any]] = []
    for i in range(20):
        thread += [{"outgoing": False, "text": f"q{i}"}, {"outgoing": True, "text": f"a{i}"}]
    telegram = SimpleNamespace()

    async def get_owner_dm_threads(**_kw):
        return [thread]

    telegram.get_owner_dm_threads = get_owner_dm_threads
    # Newest first: bursts are a19..a0, so batch 2 holds indices 15..19.
    llm = _ScriptedLLM(
        RuntimeError("upstream 500"),
        {"analyses": [
            {"index": i, "thoughts": "t", "writing_style": "s"} for i in range(15, 20)
        ] + [{"index": 3, "thoughts": "outside this batch", "writing_style": "s"}]},
        {"profile": "terse", "example_indices": [1, 1, 99]},
    )
    db = Database(tmp_path / "style.db")
    await db.connect()
    try:
        style = await OwnerStyle(llm=llm, db=db, model="m").infer(telegram)
        assert style["replies_analyzed"] == 5
        # Re-indexed over analyzed replies: index 1 is the 2nd analyzed (a3);
        # duplicates and out-of-range indices are dropped.
        assert [ex["reply"] for ex in style["examples"]] == [["a3"]]
        assert (await db.get_owner_style())["profile"] == "terse"
    finally:
        await db.close()
