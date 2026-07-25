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

import asyncio
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
from oncall.operator import (
    OPERATOR_TOOLS,
    AnthropicLLMClient,
    Operator,
    summarize_llm_error,
)
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

    async def retrieve(self, query, *, limit=None, exclude_ids=None):
        self.retrieve_calls.append(query)
        hits = self._canned.get(query, [])
        if exclude_ids:
            hits = [m for m in hits if m.id not in exclude_ids]
        return list(hits)

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


@pytest.mark.asyncio
async def test_hand_off_injects_memory_context_into_executor_prompt(stack):
    """Regression: every hand_off must prepend a `# Memory context` block of
    relevant operator memories to the executor prompt.

    The executor session is wiped by /clear (commit 92e363c) and compacted at
    200K tokens, so durable behavioural rules — e.g. a per-recipient reply
    prefix — can't live only in the session's accumulated history. Before this
    fix the hand_off path forwarded dialogue + hint + user text but NO memory
    (only the dispatch_task path injected it), so resetting the long-lived
    executor session silently dropped such rules. The executor_system prompt
    promises this block exists; this test pins that the promise holds."""
    enqueued: list[dict[str, Any]] = []
    original = stack["lifecycle"].enqueue_executor

    async def spy(**kwargs):
        enqueued.append(kwargs)
        return await original(**kwargs)

    stack["lifecycle"].enqueue_executor = spy  # type: ignore[method-assign]

    rule = "When replying to others, prefix with 'сори, это агент'."
    stack["memory"].set_retrieval("reply to Sergey", [
        Memory(id=42, text=rule, score=0.7, cosine=0.7, last_accessed_at="x"),
    ])

    llm = ScriptedLLM(script=[[("hand_off", {})], "On it."])
    operator = _make_operator(stack, llm)
    await operator.chat_turn(session_id="s1", user_text="reply to Sergey")

    forwarded = enqueued[0]["prompt"]
    assert "# Memory context" in forwarded
    assert rule in forwarded
    # The user's verbatim message still rides at the tail under `# Task`.
    assert "# Task" in forwarded
    assert forwarded.rstrip().endswith("reply to Sergey")


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


@pytest.mark.asyncio
async def test_append_system_note_persists_silently(stack):
    """append_system_note is the SILENT sibling of auto_ping: it drops the same
    '[system note: ...]' shape into history but must NOT run an operator turn —
    no LLM round-trip, no reply. Regression: owner voice-call teardown uses it
    to close the lingering call-start note. Inbound teardown used to write
    nothing, so a later text turn read as still-in-call and the model leaked
    spoken expression tags ([laughter], [confirmation-en]) into text replies."""
    llm = ScriptedLLM(script=[])
    operator = _make_operator(stack, llm)
    await operator.append_system_note("tg-agent-42", "the voice call just ended.")
    # Silent: not a single LLM turn was run.
    assert llm.calls_made == []
    rows = await stack["db"].load_chat_history("tg-agent-42", limit=10)
    assert len(rows) == 1
    # Exact wrapping must match auto_ping's, so the operator and the memory
    # extractor treat this marker identically to any other system note.
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "[system note: the voice call just ended.]"


# ---------------------------------------------------------------------------
# AnthropicLLMClient: OpenAI -> Anthropic request translation
#
# _build_request is pure (no network) — it does the load-bearing, non-obvious
# work of the native backend: pull system out, pair tool_use/tool_result, merge
# the volatile-tail run of user messages into alternating turns, and drop the
# ONE cache breakpoint at the end of the STABLE prefix. The cache breakpoint
# placement is the invariant that pays for Haiku: if it lands on (or after) the
# per-turn clock, every turn writes a fresh cache entry and reads nothing.
# ---------------------------------------------------------------------------

def _openai_history_with_tail() -> list[dict]:
    """A realistic operator call: system + a tool round-trip + the current
    owner turn, then the transient per-turn tail (status / call / clock)."""
    return [
        {"role": "system", "content": "SYS-PROMPT"},
        {"role": "user", "content": "check my disk usage"},
        {"role": "assistant", "content": "On it.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "hand_off",
                                      "arguments": json.dumps({"ack_msg": "On it."})}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": json.dumps({"status": "queued"})},
        {"role": "assistant", "content": "Queued it."},
        {"role": "user", "content": "thanks, what's my timezone?"},   # last STABLE block
        {"role": "user", "content": "<acting-status>idle</acting-status>"},
        {"role": "user", "content": "<call-status>not on a call</call-status>"},
        {"role": "user", "content": "<current-time>2026-07-10T18:03:11Z</current-time>"},
    ]


def _client() -> AnthropicLLMClient:
    # No network: __init__ only constructs AsyncAnthropic; the key is unused.
    return AnthropicLLMClient(api_key="sk-test-not-real")


def _cache_breakpoints(msgs: list[dict]) -> list[dict]:
    return [b for m in msgs for b in m["content"]
            if isinstance(b, dict) and "cache_control" in b]


def test_anthropic_cache_breakpoint_is_before_the_volatile_tail():
    """Exactly one breakpoint, and it sits on the last STABLE block — the
    owner's real turn — never on a `<...-status>`/`<current-time>` block. This
    is what keeps the per-turn clock outside the cache so it reads across turns
    instead of rewriting every time."""
    kwargs = _client()._build_request(
        model="claude-haiku-4-5", messages=_openai_history_with_tail(),
        tools=OPERATOR_TOOLS, max_tokens=2048, reasoning_effort="minimal",
    )
    bps = _cache_breakpoints(kwargs["messages"])
    assert len(bps) == 1, f"expected exactly one cache breakpoint, got {len(bps)}"
    assert bps[0].get("type") == "text"
    assert bps[0]["text"] == "thanks, what's my timezone?"
    for tag in ("<acting-status>", "<call-status>", "<current-time>"):
        assert not bps[0]["text"].startswith(tag)


def test_anthropic_translation_alternates_roles_and_pairs_tools():
    """System is hoisted out; the trailing run of user messages (owner turn +
    3 status blocks) collapses to a single alternating user turn; tool_use and
    tool_result survive with a matching id."""
    kwargs = _client()._build_request(
        model="claude-haiku-4-5", messages=_openai_history_with_tail(),
        tools=OPERATOR_TOOLS, max_tokens=2048, reasoning_effort="minimal",
    )
    assert [b["text"] for b in kwargs["system"]] == ["SYS-PROMPT"]
    msgs = kwargs["messages"]
    roles = [m["role"] for m in msgs]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), roles
    assert roles[0] == "user"  # Anthropic requires the first turn to be user
    tool_use = [b for m in msgs for b in m["content"] if b.get("type") == "tool_use"]
    tool_res = [b for m in msgs for b in m["content"] if b.get("type") == "tool_result"]
    assert len(tool_use) == 1 and len(tool_res) == 1
    assert tool_use[0]["id"] == tool_res[0]["tool_use_id"] == "c1"
    assert tool_use[0]["name"] == "hand_off"
    # the 4 status/owner user messages merged into the final single user turn
    assert msgs[-1]["role"] == "user"
    assert sum(b.get("type") == "text" for b in msgs[-1]["content"]) == 4


def test_anthropic_reasoning_effort_controls_thinking():
    """'minimal'/None -> no extended thinking (fastest TTFT, the operator
    default). A real level -> thinking enabled with budget < max_tokens (the
    API rejects budget >= max_tokens), so max_tokens is bumped to fit."""
    build = _client()._build_request
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    assert "thinking" not in build(model="claude-haiku-4-5", messages=msgs,
                                   tools=[], max_tokens=2048, reasoning_effort="minimal")
    assert "thinking" not in build(model="claude-haiku-4-5", messages=msgs,
                                   tools=[], max_tokens=2048, reasoning_effort=None)
    hi = build(model="claude-haiku-4-5", messages=msgs, tools=[],
               max_tokens=2048, reasoning_effort="high")
    assert hi["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert hi["max_tokens"] > hi["thinking"]["budget_tokens"]


# ---------------------------------------------------------------------------
# <time-since-last-message>
# ---------------------------------------------------------------------------

async def _seed_message(db, session_id: str, role: str, content: str, age_s: float) -> None:
    """Insert a chat row stamped `age_s` seconds in the past. Direct SQL
    because append_chat_message always stamps 'now'."""
    from datetime import timedelta

    from oncall.db import iso
    from oncall.models import utcnow

    await db.ensure_chat_session(session_id)
    await db.conn.execute(
        "INSERT INTO chat_messages (session_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, role, content, iso(utcnow() - timedelta(seconds=age_s))),
    )
    await db.conn.commit()


def _tail_tag(llm: ScriptedLLM, tag: str) -> str | None:
    msgs = llm.calls_made[0]["messages"]
    return next((m["content"] for m in msgs
                 if isinstance(m["content"], str) and m["content"].startswith(tag)), None)


@pytest.mark.asyncio
async def test_silence_gap_measures_last_real_user_message(stack):
    """The gap is measured from the owner's last REAL message. Synthetic
    user-role rows the daemon writes to itself ([system note: ...] auto-pings,
    including the one a voice call opens with, and [memory note: ...]
    injections) must not reset the clock — otherwise every call would greet
    the owner as if they had just been talking."""
    db = stack["db"]
    await _seed_message(db, "s1", "user", "night, talk tomorrow", age_s=5 * 3600 + 12 * 60)
    await _seed_message(db, "s1", "assistant", "Night.", age_s=5 * 3600 + 11 * 60)
    await _seed_message(db, "s1", "user", "[memory note: entries about X]", age_s=90)

    llm = ScriptedLLM(["Morning."])
    await _make_operator(stack, llm).auto_ping(
        session_id="s1", note="owner voice call started",
        include_silence_gap=True,
    )

    assert _tail_tag(llm, "<time-since-last-message>") == (
        "<time-since-last-message>5h 12m since the user last spoke to you"
        "</time-since-last-message>"
    )


@pytest.mark.asyncio
async def test_silence_gap_only_on_turns_that_ask_for_it(stack):
    """The block is opt-in: ordinary chat turns and background auto-pings must
    not carry it (a clock on every turn is one the operator starts narrating).
    Only the call-start greeting opts in — and even then it is suppressed under
    LAST_CONTACT_MIN_GAP_S, absent entirely rather than reported as "0m"."""
    from oncall.operator import LAST_CONTACT_MIN_GAP_S

    db = stack["db"]
    await _seed_message(db, "s1", "user", "night", age_s=5 * 3600)

    llm = ScriptedLLM(["Sure."])
    await _make_operator(stack, llm).chat_turn(session_id="s1", user_text="hey")
    assert _tail_tag(llm, "<time-since-last-message>") is None

    llm2 = ScriptedLLM(["Noted."])
    await _make_operator(stack, llm2).auto_ping(session_id="s1", note="task 7 finished")
    assert _tail_tag(llm2, "<time-since-last-message>") is None

    await _seed_message(db, "s2", "user", "still here", age_s=LAST_CONTACT_MIN_GAP_S - 1)
    llm3 = ScriptedLLM(["Hi."])
    await _make_operator(stack, llm3).auto_ping(
        session_id="s2", note="owner voice call started", include_silence_gap=True,
    )
    assert _tail_tag(llm3, "<time-since-last-message>") is None


def test_llm_error_summary_digs_out_the_nested_provider_message():
    """google-genai re-encodes the upstream error document as a STRING inside
    its own `details` dict, so the actionable sentence sits two JSON levels
    down while `str(exc)` shows only "429 Too Many Requests" plus an unreadable
    body. Both halves are useless alone — the summary must reach the sentence.

    Non-API exceptions have no such payload and must keep their type name,
    which is the only informative part of e.g. `KeyError('session')`."""
    body = json.dumps({"error": {
        "code": 429,
        "message": "Your project has exceeded its monthly spending cap. ",
        "status": "RESOURCE_EXHAUSTED",
    }})

    class FakeClientError(Exception):
        code = 429
        details = {"message": body, "status": "Too Many Requests"}

    assert summarize_llm_error(FakeClientError(f"429 Too Many Requests. {body}")) == (
        "429: Your project has exceeded its monthly spending cap."
    )
    assert summarize_llm_error(asyncio.TimeoutError()) == (
        "the model did not respond in time"
    )
    assert summarize_llm_error(KeyError("session")) == "KeyError: 'session'"
    # A brace in a plain error message is not a JSON body to be cut away.
    assert summarize_llm_error(
        RuntimeError('bad tool args {"path": null}')
    ) == 'RuntimeError: bad tool args {"path": null}'
