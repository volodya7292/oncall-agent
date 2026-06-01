"""Operator tests for the post-redesign surface.

The operator now has exactly four tools — `hand_off()`, `query_memory`,
`save_memory`, `forget_memory` — and otherwise replies directly. These
tests cover the surviving plumbing: the hand_off path into lifecycle,
the memory tools, the acting-status auto-injection, and the tool-round
cap. Removed-surface tests (dispatch_task, dispatch_handle_dm,
present_pending_approval, kill_task, read_image, etc.) are intentionally
gone — those tools no longer exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oncall.approval_client import HttpLongPollApprovalClient
from oncall.broker import Broker
from oncall.config import Paths, Settings
from oncall.db import Database
from oncall.events import EventBus
from oncall.lifecycle import Lifecycle
from oncall.operator import OPERATOR_TOOLS, Operator
from oncall.operator_memory import Memory


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """Replays a sequence of LLM responses. Each script item is either
    a plain string (text-only assistant turn) or a list of
    (tool_name, args_dict) tuples (a tool-calling turn)."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls_made: list[dict[str, Any]] = []

    async def chat(self, *, model, messages, tools, max_tokens=None, reasoning_effort=None):
        self.calls_made.append({"messages": messages, "tools": tools})
        if not self.script:
            return {"role": "assistant", "content": "(out of script)", "tool_calls": []}
        step = self.script.pop(0)
        if isinstance(step, str):
            return {"role": "assistant", "content": step, "tool_calls": []}
        tool_calls = []
        for i, (name, args) in enumerate(step):
            tool_calls.append({
                "id": f"call_{len(self.calls_made)}_{i}",
                "name": name,
                "arguments_json": json.dumps(args),
            })
        return {"role": "assistant", "content": "", "tool_calls": tool_calls}


class StubMemory:
    """Minimal MemoryStore: retrieve returns canned per-query results;
    store records inputs and reports them via entries_count."""

    def __init__(self) -> None:
        self.stored_batches: list[list[str]] = []
        self.retrieve_calls: list[str] = []
        self._entries: list[str] = []
        self._canned: dict[str, list[Memory]] = {}
        self._by_id: dict[int, Memory] = {}

    def set_retrieval(self, query: str, memories: list[Memory]) -> None:
        self._canned[query] = list(memories)
        for m in memories:
            self._by_id[m.id] = m

    async def store(self, facts, *, source_turn=None):
        kept = [f.strip() for f in facts if f and f.strip()]
        self.stored_batches.append(kept)
        self._entries.extend(kept)
        return list(kept)

    async def retrieve(self, query, *, limit=None):
        self.retrieve_calls.append(query)
        return list(self._canned.get(query, []))

    async def get_by_id(self, memory_id):
        return self._by_id.get(memory_id)

    async def delete_by_id(self, memory_id):
        existed = self._by_id.pop(memory_id, None)
        return existed is not None

    async def for_prompt(self, query):
        return ""

    async def entries_count(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    return Settings(
        oncall_token="t",
        oncall_db_path=tmp_path / "db.sqlite",
        oncall_operator_model="openai/test",
        ai_gateway_api_key="x",
    )


def test_tilde_path_is_expanded_by_settings(tmp_path, monkeypatch):
    """Regression: pydantic-settings used to treat ~ literally and
    create a directory named '~'. The expanduser validator normalizes
    before storage."""
    monkeypatch.setenv("TELEGRAM_SESSION_PATH", "~/.oncall/telegram.session")
    monkeypatch.setenv("ONCALL_DB_PATH", "~/.oncall/state.db")
    monkeypatch.setenv("TELEGRAM_AGENT_SESSION_PATH", "~/.oncall/telegram_agent.session")
    monkeypatch.setattr("oncall.config.USER_ENV_FILE", tmp_path / "nonexistent.env")
    s = Settings(_env_file=None)
    home = Path.home()
    assert str(s.telegram_session_path).startswith(str(home))
    assert "~" not in str(s.telegram_session_path)
    assert str(s.oncall_db_path).startswith(str(home))
    assert str(s.telegram_agent_session_path).startswith(str(home))


@pytest.fixture
def paths(tmp_path):
    return Paths()


@pytest.fixture
async def stack(settings, paths):
    db = Database(settings.oncall_db_path)
    await db.connect()
    events = EventBus(db)
    approval_client = HttpLongPollApprovalClient()
    broker = Broker(db, approval_client, events.publish)
    lifecycle = Lifecycle(
        db=db, broker=broker, approval_client=approval_client,
        events=events, settings=settings, paths=paths,
    )
    memory = StubMemory()
    try:
        yield {"db": db, "events": events, "approval_client": approval_client,
               "broker": broker, "lifecycle": lifecycle,
               "settings": settings, "paths": paths,
               "memory": memory}
    finally:
        await lifecycle.shutdown()
        await db.close()


def _make_operator(stack, llm) -> Operator:
    return Operator(
        db=stack["db"], lifecycle=stack["lifecycle"], broker=stack["broker"],
        settings=stack["settings"], paths=stack["paths"], llm=llm,
        memory=stack["memory"],
    )


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

def test_operator_tool_surface_is_exactly_four():
    """The redesigned operator exposes hand_off + 3 memory tools — no more."""
    names = {t["function"]["name"] for t in OPERATOR_TOOLS}
    assert names == {"hand_off", "save_memory", "query_memory", "forget_memory"}


def test_hand_off_tool_shape():
    """hand_off forwards the user's verbatim message; takes a required
    `ack_msg` (the canonical user-facing acknowledgement) and an optional
    `hint` for deictic-reply context. The user message itself is never an
    arg — it's read from chat history."""
    hand_off = next(t for t in OPERATOR_TOOLS if t["function"]["name"] == "hand_off")
    params = hand_off["function"]["parameters"]
    assert set(params.get("properties", {}).keys()) == {"ack_msg", "hint"}
    assert params.get("required", []) == ["ack_msg"]


# ---------------------------------------------------------------------------
# hand_off plumbing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hand_off_enqueues_user_message_verbatim(stack):
    """Calling hand_off forwards the original user_text into lifecycle —
    not any paraphrase the LLM might try to slip into args (it has no args)."""
    enqueued: list[dict[str, Any]] = []
    original = stack["lifecycle"].enqueue_executor

    async def spy(**kwargs):
        enqueued.append(kwargs)
        return await original(**kwargs)

    stack["lifecycle"].enqueue_executor = spy  # type: ignore[method-assign]

    llm = ScriptedLLM(script=[
        [("hand_off", {})],
        "Looking.",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="check staging health")

    assert enqueued, "hand_off did not reach lifecycle"
    assert enqueued[0]["prompt"] == "check staging health"
    assert enqueued[0]["chat_session_id"] == "s1"


@pytest.mark.asyncio
async def test_hand_off_with_hint_prepends_operator_hint_block(stack):
    """When the user's literal message is something like 'yes', the
    operator can attach a hint that the executor sees ahead of the
    user's text."""
    enqueued: list[dict[str, Any]] = []
    original = stack["lifecycle"].enqueue_executor

    async def spy(**kwargs):
        enqueued.append(kwargs)
        return await original(**kwargs)

    stack["lifecycle"].enqueue_executor = spy  # type: ignore[method-assign]

    llm = ScriptedLLM(script=[
        [("hand_off", {"hint": "user confirms retrying the deepfake check"})],
        "On it.",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="yes")

    forwarded = enqueued[0]["prompt"]
    assert forwarded.startswith("[operator hint:")
    assert "user confirms retrying" in forwarded
    assert forwarded.endswith("yes")


@pytest.mark.asyncio
async def test_hand_off_without_hint_forwards_user_text_verbatim(stack):
    """Without a hint, the forwarded prompt IS the user's message — no
    wrapping, no preamble."""
    enqueued: list[dict[str, Any]] = []
    original = stack["lifecycle"].enqueue_executor

    async def spy(**kwargs):
        enqueued.append(kwargs)
        return await original(**kwargs)

    stack["lifecycle"].enqueue_executor = spy  # type: ignore[method-assign]

    llm = ScriptedLLM(script=[
        [("hand_off", {})],
        "Looking.",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="summarize my inbox")
    assert enqueued[0]["prompt"] == "summarize my inbox"


@pytest.mark.asyncio
async def test_hand_off_forwards_dialogue_tail_then_dedups_on_second_call(stack):
    """First hand_off includes recent operator-user dialogue. Second
    hand_off forwards only what arrived after the first (cursor dedup)."""
    enqueued: list[dict[str, Any]] = []
    original = stack["lifecycle"].enqueue_executor

    async def spy(**kwargs):
        enqueued.append(kwargs)
        return await original(**kwargs)

    stack["lifecycle"].enqueue_executor = spy  # type: ignore[method-assign]

    # Seed history: a couple of prior turns the executor hasn't seen.
    await stack["db"].ensure_chat_session("s1")
    await stack["db"].append_chat_message("s1", "user", "the staging check failed earlier")
    await stack["db"].append_chat_message("s1", "assistant", "Want me to retry it?")

    # Turn 1: user says "yes" → operator hand_offs. Should bundle the
    # recent dialogue tail.
    llm1 = ScriptedLLM(script=[[("hand_off", {})], "Looking."])
    operator = _make_operator(stack, llm1)
    await operator.chat_turn(session_id="s1", user_text="yes")

    first = enqueued[0]["prompt"]
    assert "the staging check failed earlier" in first
    assert "Want me to retry it?" in first
    assert first.endswith("yes")

    # Turn 2: user follows up. Only the NEW dialogue (since the cursor)
    # should be forwarded — not the same tail again.
    llm2 = ScriptedLLM(script=[[("hand_off", {})], "On it."])
    operator2 = _make_operator(stack, llm2)
    await operator2.chat_turn(session_id="s1", user_text="and then check prod too")

    second = enqueued[1]["prompt"]
    assert "the staging check failed earlier" not in second, \
        "old dialogue should not be re-forwarded"
    assert "Want me to retry it?" not in second
    assert second.endswith("and then check prod too")


@pytest.mark.asyncio
async def test_hand_off_with_empty_user_text_returns_error_to_model(stack):
    """If hand_off fires with no fresh user text (e.g. an auto-ping turn),
    the tool returns an error so the LLM can recover with a normal reply."""
    llm = ScriptedLLM(script=[
        [("hand_off", {})],
        "ok",
    ])
    operator = _make_operator(stack, llm)
    # Simulate a turn whose "user_text" is essentially blank.
    await operator.chat_turn(session_id="s1", user_text="   ")

    # The second LLM call should carry a tool response with error.
    assert len(llm.calls_made) == 2
    tool_msgs = [m for m in llm.calls_made[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs
    payload = json.loads(tool_msgs[0]["content"])
    assert "error" in payload


# ---------------------------------------------------------------------------
# Acting-status injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acting_status_injected_as_idle_when_no_work(stack):
    """No previous hand_off → acting-status block reads idle. With no
    CallService wired (set_on_call_provider never called) the per-turn
    call-status reads "not on a call"."""
    llm = ScriptedLLM(script=["hi"])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="hello")

    contents = [
        m["content"] for m in llm.calls_made[0]["messages"]
        if m["role"] == "user" and isinstance(m["content"], str)
    ]
    assert "<acting-status>idle</acting-status>" in contents
    assert "<call-status>not on a call</call-status>" in contents


async def test_call_status_reflects_active_call(stack):
    """When the CallService reports the session is live, the per-turn
    call-status flips to "on a voice call" — this is the signal the operator
    gates voice-only expression tags on, and it's recomputed each turn (a
    different session stays "not on a call")."""
    llm = ScriptedLLM(script=["hi"])
    operator = _make_operator(stack, llm)
    operator.set_on_call_provider(lambda sid: sid == "live")

    await operator.chat_turn(session_id="live", user_text="hey")
    on = [m["content"] for m in llm.calls_made[0]["messages"] if m["role"] == "user"]
    assert "<call-status>on a voice call — your reply is spoken aloud</call-status>" in on

    await operator.chat_turn(session_id="other", user_text="hey")
    off = [m["content"] for m in llm.calls_made[1]["messages"] if m["role"] == "user"]
    assert "<call-status>not on a call</call-status>" in off


# ---------------------------------------------------------------------------
# Memory tool plumbing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_memory_writes_through(stack):
    llm = ScriptedLLM(script=[
        [("save_memory", {"text": "the user's staging is at host42"})],
        "noted",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="staging lives at host42")
    assert stack["memory"].stored_batches[-1] == ["the user's staging is at host42"]


@pytest.mark.asyncio
async def test_query_memory_returns_canned_results(stack):
    stack["memory"].set_retrieval("staging", [
        Memory(id=7, text="staging is at host42", score=0.9, cosine=0.9, last_accessed_at="x")
    ])
    llm = ScriptedLLM(script=[
        [("query_memory", {"query": "staging"})],
        "found one",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="where is staging?")

    tool_msgs = [m for m in llm.calls_made[1]["messages"] if m["role"] == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["query"] == "staging"
    assert payload["memories"][0]["text"] == "staging is at host42"


@pytest.mark.asyncio
async def test_forget_memory_deletes_by_id(stack):
    stack["memory"]._by_id[7] = Memory(
        id=7, text="old fact", score=0.0, cosine=0.0, last_accessed_at="x",
    )
    llm = ScriptedLLM(script=[
        [("forget_memory", {"memory_id": 7})],
        "ok",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="forget that")
    assert 7 not in stack["memory"]._by_id


# ---------------------------------------------------------------------------
# Loop safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_round_cap_prevents_infinite_loops(stack):
    """If the LLM keeps emitting tool calls past the cap (and none of
    them short-circuit, like hand_off does), the operator surfaces a
    stuck-message rather than looping forever."""
    # Use query_memory — it doesn't short-circuit the way hand_off does.
    llm = ScriptedLLM(script=[[("query_memory", {"query": "x"})]] * 50)
    operator = _make_operator(stack, llm)
    result = await operator.chat_turn(session_id="s1", user_text="loop check")
    assert "stuck" in result.text.lower() or "too many" in result.text.lower()


@pytest.mark.asyncio
async def test_hand_off_short_circuits_after_ack(stack):
    """After a successful hand_off, the operator returns its ack
    immediately (no second LLM round). This keeps the ack ahead of the
    async result-delivery message from a fast-completing executor."""
    llm = ScriptedLLM(script=[
        [("hand_off", {})],
        # If the operator wrongly ran a second round, it would consume
        # this script item — we assert it wasn't touched.
        "SECOND ROUND SHOULD NOT FIRE",
    ])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="check my inbox")
    # Only one LLM call should have happened.
    assert len(llm.calls_made) == 1
    # Second script item still pending.
    assert "SECOND ROUND" in llm.script[0]


# ---------------------------------------------------------------------------
# Auto-ping plumbing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_ping_runs_even_on_empty_session(stack):
    """auto_ping must fire even when the session has no history —
    otherwise inbox-drain silently drops DMs that arrive after /clear
    because the operator never gets a chance to triage them."""
    llm = ScriptedLLM(script=["acknowledged"])
    operator = _make_operator(stack, llm)
    result = await operator.auto_ping(session_id="fresh", note="new DM in chat X")
    assert result.ran is True
    assert llm.calls_made, "auto_ping should have invoked the LLM"
    assert result.text == "acknowledged"
