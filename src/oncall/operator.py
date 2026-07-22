"""Operator — the operator tier.

The operator talks with the user. It never executes infrastructure work; its
only side-effecting capabilities are calls back into the orchestrator
(dispatching a Claude task, fetching status, forwarding an approval response).

Critical safety property: the operator NEVER decides whether a challenge phrase
matches. It forwards the user's spoken phrase to /approvals/{id}/respond, and
the orchestrator validates it canonically. A fully prompt-injected operator
still can't bypass the gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4

from .audit import fmt, operator_log
from .broker import Broker
from .config import Paths, Settings
from . import memory_extractor
from .db import Database, iso
from .events import EventBus
from .lifecycle import Lifecycle
from .local_claude import ClaudeCliRunner, OneShotRunner
from .metrics import timed
from .models import format_utc_now, utcnow
from .operator_memory import Memory, MemoryStore
from .telegram_service import TelegramService


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client abstraction — production wraps AsyncOpenAI pointed at the Vercel
# gateway; tests inject a stub.
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Return {role: 'assistant', content: str|None, tool_calls: list|None}."""
        ...


# Values of ONCALL_OPERATOR_REASONING_EFFORT that mean "no thinking pass at
# all", as distinct from None ("don't send the dial, take the model default").
# The difference is load-bearing: an unset dial on a default_enabled=true model
# reasons at its default effort.
_REASONING_OFF = frozenset({"none", "off", "disabled"})


class GenAILLMClient:
    """LLM client backed by Google's native AI Studio / Gemini API
    (`google.genai`). Translates the operator's OpenAI-Chat-style messages
    + tools into Gemini's Contents/FunctionDeclaration shape and translates
    the response back, so it's a drop-in for `GatewayLLMClient`.

    Used by default for Google models. Two reasons over the Vercel gateway:
    (1) the gateway strips assistant text when a tool_call is in the same
    response, breaking ack-first; (2) gemini-3.1-flash-lite emits a
    `thought_signature` on every function_call Part and rejects the
    follow-up turn if we don't echo it back — we capture and re-attach it
    via the OpenAI-style tool_call dict (`gemini_thought_signature_b64`)."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        import base64
        from google.genai import types
        from uuid import uuid4

        # Drop a "google/" prefix if the user pinned the model via the
        # Vercel-style slug; AI Studio uses bare names.
        gem_model = model.split("/", 1)[1] if model.startswith("google/") else model

        # OpenAI → Gemini message translation.
        # Track function-call ids → names so a later "tool" message can pair
        # its tool_call_id back to the function it was responding to (Gemini
        # function_response needs a name, not an id).
        system_chunks: list[str] = []
        contents: list[types.Content] = []
        id_to_name: dict[str, str] = {}

        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_chunks.append(m["content"])
            elif role == "user":
                content = m.get("content")
                if isinstance(content, list):
                    # List-content shape (OpenAI vision): translate each
                    # part. Used for the read_image follow-up turn that
                    # carries the loaded attachment inline.
                    u_parts: list[types.Part] = []
                    for c in content:
                        ctype = c.get("type") if isinstance(c, dict) else None
                        if ctype == "text":
                            u_parts.append(types.Part.from_text(text=c.get("text") or ""))
                        elif ctype == "image_url":
                            url = (c.get("image_url") or {}).get("url") or ""
                            if url.startswith("data:"):
                                header, _, b64 = url.partition(",")
                                meta = header.removeprefix("data:")
                                # data:<mime>;base64
                                mime = meta.split(";", 1)[0] or "application/octet-stream"
                                try:
                                    raw = base64.b64decode(b64)
                                except Exception:
                                    continue
                                u_parts.append(
                                    types.Part.from_bytes(data=raw, mime_type=mime)
                                )
                            # Non-data URLs are skipped: Gemini's
                            # inline_data path needs the bytes; we don't
                            # follow remote URLs from the model side.
                    if u_parts:
                        contents.append(types.Content(role="user", parts=u_parts))
                else:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content or "")],
                    ))
            elif role == "assistant":
                parts: list[types.Part] = []
                if m.get("content"):
                    parts.append(types.Part.from_text(text=m["content"]))
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {}
                    # gemini-3.1-flash-lite (and other thinking-enabled models)
                    # require the model's `thought_signature` to be echoed back
                    # on the function-call Part when we send the follow-up turn,
                    # or round 2 fails with 400 INVALID_ARGUMENT. We stash the
                    # signature on the OpenAI-style tool_call as a base64 string
                    # in the response translation below; reattach it here.
                    sig_b64 = tc.get("gemini_thought_signature_b64")
                    sig = base64.b64decode(sig_b64) if sig_b64 else None
                    parts.append(types.Part(
                        function_call=types.FunctionCall(name=name, args=args),
                        thought_signature=sig,
                    ))
                    if tc.get("id"):
                        id_to_name[tc["id"]] = name
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                tc_id = m.get("tool_call_id", "")
                name = id_to_name.get(tc_id, "unknown")
                raw = m.get("content") or "{}"
                try:
                    response = json.loads(raw)
                except json.JSONDecodeError:
                    response = {"result": raw}
                # Gemini's function_response expects a dict; if the result
                # was a bare scalar/list, wrap it.
                if not isinstance(response, dict):
                    response = {"result": response}
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=name, response=response,
                    )],
                ))

        cfg_kwargs: dict[str, Any] = {"temperature": 0.2}
        if system_chunks:
            cfg_kwargs["system_instruction"] = "\n\n".join(system_chunks)
        if max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_tokens
        # OpenAI's reasoning_effort levels → Gemini thinking_level. Valid
        # values per the Gemini API are MINIMAL / LOW / MEDIUM / HIGH; some
        # models accept a subset (gemma-4-31b for instance rejects LOW and
        # MEDIUM with 400). We forward whichever level the caller picked
        # so the API error surfaces honestly if it's unsupported, rather
        # than silently downgrading. Anything unrecognized → fall back to
        # MINIMAL (the latency-conservative default).
        # "none"/"off" → MINIMAL: Gemini 3.x thinking models have no true
        # off switch, so the floor is the honest translation of "don't think".
        if reasoning_effort:
            if reasoning_effort.lower() in _REASONING_OFF:
                level = "MINIMAL"
            else:
                level = reasoning_effort.upper()
                if level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
                    level = "MINIMAL"
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        if tools:
            decls: list[types.FunctionDeclaration] = []
            for t in tools:
                fn = t.get("function", {})
                decls.append(types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters") or None,
                ))
            cfg_kwargs["tools"] = [types.Tool(function_declarations=decls)]

        cfg = types.GenerateContentConfig(**cfg_kwargs)

        # Streaming: the non-streaming `generate_content` call typically waits
        # for the FULL response before returning, which on gemma-4-31b lands
        # at ~2.5–3s end-to-end. The streamed call delivers tokens as they're
        # produced — same total time on paper, but it (a) lets callers surface
        # the first text chunk immediately if they wire up a chunk sink, and
        # (b) gives us per-stream wall-clock cancellation that doesn't depend
        # on whatever retry/backoff the SDK is doing internally. We bound the
        # whole stream at 20s — that's well above the observed p99 (~5s) and
        # short enough that a stuck call doesn't block a session lock for
        # minutes when the upstream is rate-limited or wedged.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        # Last non-null usage_metadata seen on the stream. Gemini sends a
        # cumulative snapshot on (typically) the final chunk; we keep the most
        # recent so we can log cache effectiveness after the stream drains.
        usage: Any = None

        async def _consume() -> None:
            nonlocal usage
            stream = await self._client.aio.models.generate_content_stream(
                model=gem_model, contents=contents, config=cfg,
            )
            async for chunk in stream:
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = chunk.usage_metadata
                for c in (chunk.candidates or []):
                    if not (c.content and c.content.parts):
                        continue
                    for part in c.content.parts:
                        if getattr(part, "thought", False):
                            continue
                        if part.text:
                            text_parts.append(part.text)
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            entry: dict[str, Any] = {
                                "id": f"gemini_call_{uuid4().hex[:16]}",
                                "name": fc.name,
                                "arguments_json": json.dumps(dict(fc.args or {})),
                            }
                            # Preserve the part-level thought_signature for
                            # round-2: see the assistant-turn translation above
                            # for why this is mandatory on flash-lite.
                            sig = getattr(part, "thought_signature", None)
                            if sig:
                                entry["thought_signature_b64"] = (
                                    base64.b64encode(sig).decode("ascii")
                                )
                            tool_calls.append(entry)

        await asyncio.wait_for(_consume(), timeout=20.0)

        # Cache observability. Gemini 2.5+/3.x auto-cache the stable request
        # prefix (system_instruction + history) implicitly — no config on our
        # side. There's no hit/miss flag, so we log the token split: a nonzero
        # `cached` means the implicit cache fired for this turn; a persistent
        # zero on a long-lived session means the prefix is under the model's
        # min-cache floor, or entries are evicting between turns. `cached` bills
        # at ~0.1x, so cached/prompt is the cost-savings ratio.
        if usage is not None:
            prompt = getattr(usage, "prompt_token_count", None) or 0
            cached = getattr(usage, "cached_content_token_count", None) or 0
            pct = (100 * cached // prompt) if prompt else 0
            log.info(
                "gemini usage model=%s prompt=%d cached=%d (%d%%) output=%d thoughts=%d",
                gem_model, prompt, cached, pct,
                getattr(usage, "candidates_token_count", None) or 0,
                getattr(usage, "thoughts_token_count", None) or 0,
            )

        return {
            "role": "assistant",
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
        }


class GatewayLLMClient:
    """OpenAI Chat Completions via Vercel AI Gateway. Async, non-streaming for
    MVP (streaming for the API surface is added in the /chat endpoint layer)."""

    def __init__(self, base_url: str, api_key: str) -> None:
        # Import lazily so tests don't need the openai package.
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self, *, model, messages, tools, max_tokens=None, reasoning_effort=None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools or None,
            "temperature": 0.2,
            "extra_body": {"providerOptions": {"gateway": {"sort": "ttft"}}},
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        # OpenAI client rejects tools=None — strip if so.
        if kwargs["tools"] is None:
            kwargs.pop("tools")
        resp = await self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments_json": tc.function.arguments,
                }
                for tc in (msg.tool_calls or [])
            ],
        }


class OpenRouterLLMClient:
    """OpenAI Chat Completions via OpenRouter. Same wire format as
    GatewayLLMClient, but points at OpenRouter and PINS provider routing so we
    land on the lowest-TTFT backend for the operator model — e.g. gpt-oss-120b
    is ~0.26s on Groq vs ~1s if OpenRouter load-balances freely. `provider.order`
    lists preferred providers in priority order; `allow_fallbacks=True` lets it
    drop to the next one (Cerebras/BaseTen) if the top choice rate-limits.

    Caching: OpenRouter auto-caches on caching-capable providers (Groq ~0.5x,
    DeepSeek ~0.1x) with NO cache_control needed — nothing to plumb here. Note
    caching cuts cost, not TTFT (the operator's latency is round-trip-bound).

    The operator's hand_off ack rides INSIDE the tool call (ack_msg arg), not as
    parallel assistant text, so the gateway's text+tool_call stripping — the
    reason GenAILLMClient exists for Gemini — is a non-issue here."""

    def __init__(
        self, base_url: str, api_key: str, provider_order: list[str] | None = None,
    ) -> None:
        # Import lazily so tests don't need the openai package.
        from openai import AsyncOpenAI
        # Bound requests so a wedged upstream can't block a session lock forever
        # (the Gemini/Anthropic clients wrap their own 20s wait_for; the OpenAI
        # client takes a per-client timeout instead).
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1,
        )
        self._provider_order = list(provider_order or [])

    async def chat(
        self, *, model, messages, tools, max_tokens=None, reasoning_effort=None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools or None,
        }
        extra_body: dict[str, Any] = {}
        if self._provider_order:
            extra_body["provider"] = {
                "order": self._provider_order, "allow_fallbacks": True,
            }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        # "none" means explicitly OFF, which is not the same as unset (take the
        # provider default). `reasoning_effort` cannot express "off" — only
        # `reasoning.enabled=false` can — and models differ on what unset means
        # (OpenRouter advertises glm-5.2 as default_enabled=true), so the
        # caller gets to say it outright rather than inherit a default.
        if reasoning_effort is not None:
            if reasoning_effort.lower() in _REASONING_OFF:
                extra_body["reasoning"] = {"enabled": False}
            else:
                kwargs["reasoning_effort"] = reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        # OpenAI client rejects tools=None — strip if so.
        if kwargs["tools"] is None:
            kwargs.pop("tools")
        resp = await self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments_json": tc.function.arguments,
                }
                for tc in (msg.tool_calls or [])
            ],
        }


def _openai_content_to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    """Translate an OpenAI-style message `content` (a plain string, or the
    vision list-content shape used by attachments / read_image) into Anthropic
    content blocks. Empty text blocks are dropped — the Messages API rejects
    them with 400."""
    if not isinstance(content, list):
        text = content or ""
        return [{"type": "text", "text": text}] if text else []
    blocks: list[dict[str, Any]] = []
    for c in content:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        if ctype == "text":
            text = c.get("text") or ""
            if text:
                blocks.append({"type": "text", "text": text})
        elif ctype == "image_url":
            url = (c.get("image_url") or {}).get("url") or ""
            if url.startswith("data:"):
                header, _, b64 = url.partition(",")
                media = header.removeprefix("data:").split(";", 1)[0] or "application/octet-stream"
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media, "data": b64},
                })
            # Non-data URLs are skipped: we don't fetch remote bytes model-side.
    return blocks


class AnthropicLLMClient:
    """Operator LLM on the native Anthropic Messages API (`anthropic` SDK).

    Chosen for Claude models because it is the ONLY surface that supports
    prompt caching. The OpenAI-compatible endpoint at api.anthropic.com and the
    Vercel gateway both silently drop `cache_control` — confirmed against
    Anthropic's own OpenAI-SDK-compatibility docs: "Prompt caching is not
    supported, but it is supported in the Anthropic SDKs" (and `usage.
    prompt_tokens_details` there is "always empty").

    Caching is what makes Haiku viable for the operator: the ~2k-token system
    prompt plus the rolling history is a stable prefix that repeats verbatim
    every turn. We drop ONE cache breakpoint at the end of that prefix — on the
    last content block that is NOT part of the per-turn volatile tail (the
    `<acting-status>` / `<call-status>` / `<current-time>` / `<laptop-status>`
    blocks the operator appends after history, plus inline attachment bytes).
    A block-level breakpoint caches tools + system + history through that point;
    the volatile tail stays OUTSIDE the cached region, so the clock changing
    every turn doesn't invalidate the cache. Cache reads bill at ~0.1x and skip
    prefill for the cached tokens — that's what buys the sub-0.6s TTFT.

    Streams like GenAILLMClient: a 20s wall-clock bound and per-stream
    cancellation independent of whatever retry/backoff the SDK does internally.
    """

    # User content-block prefixes that mark the per-turn volatile tail. Kept in
    # sync with the transient blocks appended in Operator._run_turn. The cache
    # breakpoint is placed BEFORE any block starting with one of these.
    _VOLATILE_TAIL_TAGS = (
        "<acting-status>", "<call-status>",
        "<current-time>", "<laptop-status>",
    )

    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def _is_volatile_user(self, content: Any) -> bool:
        """A user message is 'volatile' (kept outside the cached prefix) if it's
        one of the per-turn status blocks or carries inline attachment bytes
        (list content). Everything else — the owner's real turns, tool results,
        assistant replies — is stable history and belongs in the cache."""
        if isinstance(content, list):
            return True  # inline attachment bytes: present once, then a placeholder
        text = (content or "").lstrip()
        return text.startswith(self._VOLATILE_TAIL_TAGS)

    def _build_request(
        self, *, model, messages, tools, max_tokens=None, reasoning_effort=None,
    ) -> dict[str, Any]:
        """Translate the operator's OpenAI-style call into Anthropic Messages
        API kwargs: pull system messages out, pair tool_use/tool_result, merge
        consecutive same-role turns, and drop the single cache breakpoint at the
        end of the stable prefix. Pure (no network) so it can be unit-tested."""
        # Drop an "anthropic/" prefix if the model was pinned via a gateway-style
        # slug; the native API wants the bare id (e.g. "claude-haiku-4-5").
        ant_model = model.split("/", 1)[1] if model.startswith("anthropic/") else model

        # ---- OpenAI messages -> Anthropic system + messages ----
        system_blocks: list[dict[str, Any]] = []
        conv: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []
        # Reference to the last content block that belongs in the cached prefix;
        # we tag exactly this one with cache_control after merging.
        last_stable_block: dict[str, Any] | None = None

        def _flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                conv.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_blocks.append({"type": "text", "text": m["content"]})
                continue
            if role == "tool":
                # Anthropic requires tool_results in a user turn; buffer
                # consecutive tool messages and flush them together.
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content") or "",
                }
                pending_tool_results.append(block)
                last_stable_block = block  # tool results are stable history
                continue
            _flush_tool_results()
            if role == "user":
                blocks = _openai_content_to_anthropic_blocks(m.get("content"))
                if not blocks:
                    continue
                conv.append({"role": "user", "content": blocks})
                if not self._is_volatile_user(m.get("content")):
                    last_stable_block = blocks[-1]
            elif role == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    raw = fn.get("arguments") or "{}"
                    try:
                        inp = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        inp = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": inp,
                    })
                if not blocks:
                    continue  # degenerate empty assistant row — skip
                conv.append({"role": "assistant", "content": blocks})
                last_stable_block = blocks[-1]
        _flush_tool_results()

        # Merge consecutive same-role turns (the volatile tail is a run of user
        # messages; the API wants alternating roles). The block dicts are reused
        # by reference, so the last_stable_block tag below still lands correctly.
        merged: list[dict[str, Any]] = []
        for msg in conv:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"].extend(msg["content"])
            else:
                merged.append({"role": msg["role"], "content": list(msg["content"])})

        # Single cache breakpoint at the end of the stable prefix.
        if last_stable_block is not None:
            last_stable_block["cache_control"] = {"type": "ephemeral"}

        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters") or {
                    "type": "object", "properties": {},
                },
            }
            for t in (tools or [])
        ]

        kwargs: dict[str, Any] = {
            "model": ant_model,
            "messages": merged,
            "max_tokens": max_tokens or 2048,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        # reasoning_effort -> extended thinking. Haiku 4.5 predates the 4.6
        # `adaptive` change, so depth is `thinking:{type:enabled,budget_tokens}`.
        # "minimal"/None means no thinking (fastest TTFT — the operator default).
        # budget must be < max_tokens, so bump max_tokens when thinking is on.
        if reasoning_effort and reasoning_effort.lower() not in ("minimal", "none", ""):
            budget = {"low": 1024, "medium": 4096, "high": 8192}.get(
                reasoning_effort.lower(), 1024,
            )
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(kwargs["max_tokens"], budget + 1024)

        return kwargs

    async def chat(
        self, *, model, messages, tools, max_tokens=None, reasoning_effort=None,
    ) -> dict[str, Any]:
        kwargs = self._build_request(
            model=model, messages=messages, tools=tools,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )

        # Stream to bound the call on wall-clock (mirrors GenAILLMClient). We
        # only need the assembled final message, not per-token deltas.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        async def _consume() -> None:
            async with self._client.messages.stream(**kwargs) as stream:
                final = await stream.get_final_message()
            for block in final.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments_json": json.dumps(block.input),
                    })
                # thinking / redacted_thinking blocks are not replayed.

        await asyncio.wait_for(_consume(), timeout=20.0)

        return {
            "role": "assistant",
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
        }


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI Chat Completions tool-call format)
# ---------------------------------------------------------------------------

OPERATOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "hand_off",
            "description": (
                "Hand the user's request over to ACTING — a deeper, slower "
                "process that can take whatever time it needs and answer "
                "fully. Call this whenever the request needs work: tools, "
                "files, code, web lookups, the user's data, a decision to "
                "make, OR when you don't have "
                "enough context to answer confidently.\n"
                "\n"
                "The user's verbatim message is forwarded automatically. "
                "Optionally pass `hint` to add context the user's literal "
                "words don't carry — most commonly when the user replies "
                "with a deictic / one-word answer (\"yes\", \"do it\", "
                "\"the second one\") to a question YOU asked. Keep hints "
                "short (one sentence) and factual; do NOT restate the "
                "user's message.\n"
                "\n"
                "`ack_msg` is REQUIRED — the one-line acknowledgement the "
                "user will see immediately. Pick varied phrasing each turn "
                "(see the menu in your system prompt). Do NOT emit a text "
                "body alongside this tool call; the ack lives in this "
                "parameter and nothing else."
            ),
            "parameters": {
                "type": "object",
                "required": ["ack_msg"],
                "properties": {
                    "ack_msg": {
                        "type": "string",
                        "description": (
                            "Short one-line acknowledgement shown to the "
                            "user right now (e.g. \"Looking.\", \"On it.\", "
                            "\"Let me check.\"). Match the user's language. "
                            "Vary each turn — do not repeat the previous "
                            "ack."
                        ),
                    },
                    "hint": {
                        "type": "string",
                        "description": (
                            "Optional one-sentence context to attach to "
                            "the hand-off. Use when the user's literal "
                            "message lacks standalone meaning (e.g. a "
                            "'yes' to an offer you made). Empty / omitted "
                            "when not needed."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Persist a durable fact to your long-term memory. Use this "
                "during a turn whenever you notice something worth keeping: "
                "a person + their context, an identifier, a preference, a "
                "convention the user states. Resolve deictic references "
                "first ('same for X' → spell out the full extended fact), "
                "using only what was actually said. Record, never deduce: "
                "do not save inferences, and never equate two identities "
                "unless the user stated it outright — a reference you "
                "cannot place is not a gap to fill. If the fact needs a "
                "guess to stand alone, save the narrower stated fact or "
                "save nothing and ask. "
                "Phrase the fact as one self-contained declarative sentence "
                "≤200 chars in third person about the user. Idempotent — "
                "near-duplicate facts merge into the existing memory entry "
                "instead of creating a new row. The system writes a "
                "`_Remembered: ..._` breadcrumb to chat automatically; "
                "don't echo the fact in your own reply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The fact to remember, ≤200 chars, declarative, "
                            "self-contained."
                        ),
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": (
                "Hard-delete ONE memory entry by id. Call this ONLY when the "
                "user explicitly asks to forget / drop / remove a specific "
                "stored fact (e.g. 'forget that staging is at host X', "
                "'delete the memory about Y'). Workflow: first `query_memory` "
                "to find the candidate id(s); if multiple plausible matches, "
                "list them to the user and ask which one — do NOT pick "
                "silently. NEVER call autonomously, never as housekeeping. "
                "Memory storage is otherwise auto-managed (extraction + LRU); "
                "this is the user's escape hatch for a specific wrong entry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "id from `query_memory` of the row to delete.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": (
                "Search the operator's persistent memory for facts relevant to "
                "an explicit query. Memory is auto-extracted from user turns "
                "and the most-relevant entries are already injected into your "
                "system prompt each turn — use this tool only when you want to "
                "look up something OUTSIDE the current turn's topic (e.g. "
                "before asking the user a clarifying question, check whether "
                "you already know the answer)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


# Map task_class model alias → actual model id passed to claude CLI.
MODEL_ALIAS_MAP: dict[str, str] = {
    "sonnet": "sonnet",
    "opus": "opus",
}


@dataclass
class OperatorTurnResult:
    text: str
    tool_calls_made: list[dict[str, Any]]
    # False when the turn short-circuited without actually invoking the
    # LLM — currently only happens in `auto_ping` when the session has no
    # history. Drain callers gate mark-triaged on this so a freshly
    # `/clear`'d session doesn't silently consume the inbox.
    ran: bool = True

    def user_facing_text(self) -> str:
        """The single string a transport (Telegram text / voice TTS) should
        show the user for this turn. Prefers any non-empty assistant text
        body; otherwise pulls `ack_msg` from a successful `hand_off` tool
        call (the canonical ack channel since hand_off requires the arg).
        Empty string when there's nothing to show — caller may suppress."""
        if self.text:
            return self.text
        for tc in self.tool_calls_made:
            if (
                tc.get("name") == "hand_off"
                and isinstance(tc.get("result"), dict)
                and tc["result"].get("enqueued")
            ):
                return ((tc.get("args") or {}).get("ack_msg") or "").strip()
        return ""


AUTO_PING_PREFIX = "[system note: "

# Marker introducing an operator-only "next action" footer inside an
# auto-ping system-note body. Stripped before forwarding the note as
# `user_text` to the executor so the executor doesn't read role
# instructions addressed at the operator (a class of leak that earlier
# caused "you do not decide what to send" to confuse the executor).
_OPERATOR_ACTION_MARKER = "\n\n→ ACTION:"


def _strip_operator_only_action(text: str) -> str:
    """Drop the operator-only `→ ACTION:` footer from a system-note
    string. The auto-ping wraps the note with a trailing `]`; we
    preserve that boundary. No-op for any text that doesn't contain the
    marker, so regular user messages pass through unchanged."""
    idx = text.find(_OPERATOR_ACTION_MARKER)
    if idx == -1:
        return text
    head = text[:idx]
    # Re-attach a closing `]` if the original had one — keeps the
    # `[system note: …]` framing intact for the executor.
    if text.rstrip().endswith("]"):
        return head + "]"
    return head


def _fmt_ts(ts: str) -> str:
    """Compact form for chat_messages.created_at when prefixing hand-off
    tail lines. Input is an ISO-8601 string (e.g.
    `2026-05-23T20:23:16.823+00:00`); output is `YYYY-MM-DD HH:MM`. Empty
    or unparseable inputs return ''. We trim aggressively because the
    tail is char-capped; the spawn-date anchor in the executor prompt
    handles the timezone implicitly."""
    if not ts or len(ts) < 16:
        return ""
    return ts[:10] + " " + ts[11:16]


# Below this, "time since the user last spoke" is noise: within a single
# sitting (and every turn of a live voice call) the gap is seconds, and
# announcing it each turn would train the operator to narrate it.
LAST_CONTACT_MIN_GAP_S = 10 * 60


def _fmt_elapsed(seconds: float) -> str:
    """Coarse human duration — '45m', '4h 12m', '3d 7h'. Two units is
    deliberate: the operator uses this to pick a greeting ('been a while'
    vs. picking up mid-thread), not to do arithmetic."""
    total = int(max(0.0, seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"


MEMORY_NOTE_PREFIX = "[memory note: "


COMPRESSION_SYSTEM_PROMPT = """\
You are summarizing the running history of an on-call agent's chat with its
user, so the conversation still fits a smaller context window.

You are a summarizer, never a participant. Do NOT reply to the user, continue
the conversation, or adopt the assistant's voice, language, or expression tags
(e.g. [laughter]). Describe what happened in third-person plain prose ("the
user asked...", "the operator dispatched...") — even when the transcript is
casual chit-chat, you narrate it, you do not join it.

Preserve:
- Every task ID (UUID or short form) the operator dispatched, and what the user wanted from each.
- User preferences, durable decisions, and constraints they stated.
- Open threads: things the operator owes the user, questions awaiting an answer.
- The gist of casual conversation — topics discussed, plans, opinions, and personal context the user shared. Condense it; do not discard it wholesale.

Drop:
- Verbose tool outputs — the operator can re-query the database by task ID for current state.
- Exact turn-by-turn wording; keep the substance.

Match the target length stated in the request. Output a single block of plain prose. No headers, no bullets, no markdown. End with a blank line.
"""

# Compression sanity-guard thresholds (see Operator._summarize_older). A
# summary below _MIN_SUMMARY_RATIO of its input token count, or one that shrank
# a prior summary below _PRIOR_RETENTION_FLOOR of its size, is rejected as an
# implausible (likely role-played) result rather than persisted. Set well below
# the ~10% retention target so only clearly-broken outputs trip it.
_MIN_SUMMARY_RATIO = 0.03
_PRIOR_RETENTION_FLOOR = 0.5


class Operator:
    def __init__(
        self,
        *,
        db: Database,
        lifecycle: Lifecycle,
        broker: Broker,
        settings: Settings,
        paths: Paths,
        llm: LLMClient,
        memory: MemoryStore,
        telegram: TelegramService | None = None,
        events: EventBus | None = None,
        ask_futures: dict[str, asyncio.Future[str]] | None = None,
        extract_llm: LLMClient | None = None,
        extract_model: str | None = None,
        max_history: int = 60,
        # 16 leaves headroom for multi-step drain flows (query_memory ×N +
        # dispatch_task + tool_status polls + read_chat_style + reply_to_dm +
        # save_memory) without giving runaway loops unbounded room. Bumped
        # from 10 after observing inbox-drain triage burn the cap on prep.
        max_tool_rounds: int = 16,
        runner: OneShotRunner | None = None,
    ) -> None:
        self._db = db
        self._lifecycle = lifecycle
        self._broker = broker
        self._settings = settings
        self._paths = paths
        self._llm = llm
        self._telegram = telegram
        self._memory = memory
        # Exposed so api.py can probe stale_count() / trigger rebuild
        # without breaking the existing _memory private convention.
        self.memory: MemoryStore = memory
        self._events = events
        self._ask_futures: dict[str, asyncio.Future[str]] = (
            ask_futures if ask_futures is not None else {}
        )
        # When extract_llm is None, auto-extraction is disabled (memory still
        # works for retrieval — it just never grows from conversation).
        self._extract_llm = extract_llm
        self._extract_model = (
            extract_model
            or settings.oncall_memory_extract_model
            or settings.oncall_operator_model
        )
        self._max_history = max_history
        self._max_tool_rounds = max_tool_rounds
        # One-shot Claude CLI runner for summarization (both chat compression
        # and task result summaries). Injectable for tests.
        self._runner: OneShotRunner = runner or ClaudeCliRunner()
        self._system_prompt_base = paths.operator_prompt.read_text(encoding="utf-8")
        if settings.operator_language:
            self._system_prompt_base += (
                f"\n\n# Output language\n\n"
                f"Always respond in: {settings.operator_language} "
                f"(ISO-639-1). Match the user's dialect/register where you "
                f"have evidence, but the LANGUAGE is fixed by this setting."
            )
        # One lock per chat session. Serializes user-initiated chat_turn calls
        # against auto-ping calls so chat_messages append in a consistent order
        # and the LLM never sees an interleaved state.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Strong references to in-flight extraction tasks so they aren't
        # garbage-collected while running.
        self._extraction_tasks: set[asyncio.Task[Any]] = set()
        # Same, for background history compression (see _schedule_compression),
        # plus the set of session_ids with a compression already in flight —
        # the over-budget trigger persists until the summary lands, so without
        # this every turn in between would spawn a duplicate opus one-shot.
        self._compression_tasks: set[asyncio.Task[Any]] = set()
        self._compressing: set[str] = set()
        # Per-turn buffer of facts the operator saved via `save_memory`
        # during the in-flight turn. Drained at extraction time so the
        # candidate-suggester can dedup against what's already committed.
        # Keyed by session_id; the session lock guarantees one in-flight
        # turn at a time per key, so writes here don't race.
        self._turn_saves: dict[str, list[str]] = {}
        # Returns True if the given session is currently on a live voice call.
        # Injected by the CallService at startup (None when voice is disabled →
        # always "not on a call"). Drives the per-turn <call-status> line so the
        # operator knows, this turn, whether its reply will be spoken aloud.
        self._on_call_provider: Callable[[str], bool] | None = None
        # Cloud-primary mode only: returns True if the user's laptop worker is
        # currently online (reachable via the proxy). Injected by api.py when
        # ONCALL_ROLE=server; None in legacy all-local mode (the executor has
        # native local tools, so there's no "laptop offline" state to surface).
        # Drives the per-turn <laptop-status> line so the operator declines
        # project/development work up front instead of handing off a doomed
        # task.
        self._laptop_status_provider: Callable[[], bool] | None = None

    def set_on_call_provider(self, provider: Callable[[str], bool]) -> None:
        """Register the CallService's live-call check. See _on_call_provider."""
        self._on_call_provider = provider

    def set_laptop_status_provider(self, provider: Callable[[], bool]) -> None:
        """Register the laptop-presence check (cloud-primary mode). See
        _laptop_status_provider."""
        self._laptop_status_provider = provider

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def _build_system_prompt(self, query: str | None) -> str:
        """Stable system prompt: base text only. Memories are NOT in here
        — they're delivered as `[memory note: ...]` user-role messages
        appended to chat history per turn, so the system prompt + history
        prefix stays byte-stable and the gateway's implicit KV cache hits
        across turns. The {{owner_name}} placeholder is still substituted
        every turn so /setownername edits take effect without a daemon
        restart; on a name change the cache invalidates once, then warms
        back up. `query` is unused now but kept for call-site
        compatibility."""
        del query
        from .config import read_owner_name
        agent_name = self._settings.agent_name or "On-call agent"
        return (
            self._system_prompt_base
            .replace("{{owner_name}}", read_owner_name())
            .replace("{{agent_name}}", agent_name)
        )

    async def _inject_session_memory(
        self, session_id: str, query: str,
    ) -> str | None:
        """Retrieve memories relevant to `query`, drop ones already shown
        in this session, and return a `[memory note: ...]` chat-message
        body listing the new ones. Returns None when there's nothing new
        to surface. Persists the newly-shown ids on success so the next
        turn in this session won't re-inject them — keeps the history
        prefix monotonically growing (cache-friendly) and avoids
        re-spamming the same facts.

        The model is told inline that memory notes appear at most once;
        if it later wants to look something up, it should use
        `query_memory` (operator) / `mcp__oncall__memory op=query`
        (executor)."""
        try:
            hits = await self._memory.retrieve(query, limit=10)
        except Exception:
            log.exception("memory retrieve failed for session %s", session_id)
            return None
        if not hits:
            return None
        shown = await self._db.get_shown_memory_ids(session_id)
        fresh = [m for m in hits if m.id not in shown]
        if not fresh:
            return None
        await self._db.record_memory_shown(session_id, [m.id for m in fresh])
        bullets = "\n".join(
            f"- [id={m.id}] {m.text.replace(chr(10), ' ').strip()}"
            for m in fresh
        )
        return (
            f"{MEMORY_NOTE_PREFIX}auto-loaded entries from your persistent "
            f"memory relevant to the next turn."
            f"For lookups OUTSIDE what you've already been shown, use "
            f"`query_memory`.\n{bullets}]"
        )

    async def chat_turn(
        self, session_id: str, user_text: str, *,
        language: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> OperatorTurnResult:
        # Capture the previous assistant turn BEFORE acquiring the lock —
        # under the lock we'll append the user row first, after which "the
        # previous assistant turn" would no longer be the right thing to
        # show the extractor. Reading outside the lock is safe: the lock
        # only serializes writes via _run_turn / auto_ping.
        prev_assistant_text = await self._latest_assistant_text(session_id)
        # If a `presented` ask is open for this chat, the user's message is
        # the answer — relay to the waiting executor, then run the operator
        # turn on a synthetic `[system note: relayed your answer ...]`
        # instead of the raw user text. The operator never sees the raw
        # text directly; the ask_request row holds it for audit.
        ask_relay_note = await self._maybe_relay_ask_answer(
            session_id, user_text,
        )
        if ask_relay_note is not None:
            user_text = ask_relay_note
            attachments = None
        async with self._lock_for(session_id):
            result = await self._run_turn(
                session_id, user_text, language=language,
                attachments=attachments,
            )
            # Capture the operator's saves from THIS turn while still under
            # the lock — otherwise a fast follow-up turn could clobber
            # `_turn_saves[session_id]` before we read it.
            already_saved_this_turn = list(
                self._turn_saves.pop(session_id, [])
            )
        # Fire-and-forget extraction. Skipped if disabled, or if this was
        # an auto-ping (those don't reach chat_turn anyway).
        if self._extract_llm is not None:
            task = asyncio.create_task(
                self._extract_and_propose(
                    session_id=session_id,
                    user_text=user_text,
                    prev_assistant_text=prev_assistant_text,
                    already_saved=already_saved_this_turn,
                ),
                name=f"extract-{session_id}",
            )
            self._extraction_tasks.add(task)
            task.add_done_callback(self._extraction_tasks.discard)
        return result

    async def _maybe_relay_ask_answer(
        self, session_id: str, user_text: str,
    ) -> str | None:
        """If a `presented` ask is open for this chat, treat the user's
        message as its answer: resolve the executor's future, mark
        answered, surface the next pending ask (if any) by appending it
        to the system note. Returns the synthetic `[system note: ...]`
        text to feed the operator INSTEAD of the raw user text, or None
        if no ask is open and the message should pass through normally."""
        row = await self._db.get_presented_ask_for_chat(session_id)
        if row is None:
            return None
        ask_id = row["id"]
        task_id_short = (row["task_id"] or "")[:8]
        ok = await self._db.mark_ask_answered(ask_id, user_text)
        if not ok:
            log.warning("ask_user: race marking ask=%s answered", ask_id)
            return None
        fut = self._ask_futures.pop(ask_id, None)
        if fut is not None and not fut.done():
            fut.set_result(user_text)
        elif fut is None:
            log.warning(
                "ask_user: no future for ask=%s (daemon restart?)",
                ask_id,
            )
        clipped = user_text.replace("\n", " ").strip()
        if len(clipped) > 200:
            clipped = clipped[:200] + "…"
        note = (
            f"relayed the user's answer {clipped!r} to task "
            f"{task_id_short}. The user is NOT addressing you — they "
            f"answered the task's question. Acknowledge briefly (one line, "
            f"e.g. 'relayed.') or stay silent. Do not re-engage with the "
            f"answer's content unless the user separately addresses you."
        )
        # Advance the queue: surface the next pending ask in the same note.
        nxt = await self._db.next_pending_ask_for_chat(session_id)
        if nxt is not None:
            await self._db.mark_ask_presented(nxt["id"])
            note += (
                f" Next: task {nxt['task_id'][:8]} is asking "
                f"(ask_id={nxt['id']}): {nxt['question']!r}. Relay this "
                f"verbatim to the user."
            )
        return f"{AUTO_PING_PREFIX}{note}]"

    async def auto_ping(
        self, session_id: str, note: str, *,
        retrieval_query: str | None = None,
        restricted_to_chat: str | None = None,
        include_silence_gap: bool = False,
    ) -> OperatorTurnResult:
        """Inject a synthetic '[system note: ...]' turn into a chat session.
        Used by background tasks that re-engage the operator (task terminated,
        approval requested, inbound DM landed). Always runs — even when the
        session has no prior history. Callers like the inbox-drain depend on
        this firing after /clear so a freshly-wiped session can still triage
        an incoming DM instead of silently dropping it.

        `retrieval_query`: when set, this string is used as the semantic
        retrieval key for the memory section instead of skipping retrieval.
        Pass the substance the operator should react to (e.g. an inbound DM
        body) so memory entries about the sender / topic / preferences are
        loaded; leave None for purely procedural pings (a task terminating)
        where no user-meaningful content needs surfacing.

        `include_silence_gap`: add a `<time-since-last-message>` block telling
        the operator how long the user has been silent. For pings that open a
        conversation with the user (a voice call greeting), not for background
        ones that merely report machine state."""
        async with self._lock_for(session_id):
            return await self._run_turn(
                session_id, f"{AUTO_PING_PREFIX}{note}]",
                retrieval_query=retrieval_query,
                restricted_to_chat=restricted_to_chat,
                include_silence_gap=include_silence_gap,
            )

    async def append_system_note(self, session_id: str, note: str) -> None:
        """Persist a synthetic '[system note: ...]' turn into a session WITHOUT
        running an operator turn — the silent sibling of `auto_ping`. Use when
        history needs a procedural marker the model should see on its NEXT turn,
        but which must not itself generate a reply or a user-facing ping (e.g.
        closing out a just-ended voice call so later text turns don't read as
        still-in-call). Same `[system note: ...]` wrapping as auto_ping, so the
        operator and the memory extractor treat it identically — the note text
        MUST therefore be procedural, never a user-attributed preference, or the
        extractor will paraphrase it into memory (see `_call_start_note`)."""
        await self._db.ensure_chat_session(session_id)
        await self._db.append_chat_message(
            session_id, "user", f"{AUTO_PING_PREFIX}{note}]",
        )

    async def _notify_dispatch_denied(
        self, session_id: str, prompt_preview: str,
        restricted_to_chat: str | None,
    ) -> None:
        """Inject a system-note after the user denies a deferred dispatch,
        and surface the operator's reply to the chat. Without this the
        operator dangles silently: it called dispatch_task, got
        pending_approval back, then never learned the tap was Deny."""
        preview = (prompt_preview or "").replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:160] + "…"
        note = (
            f"the user DENIED your deferred dispatch_task. No task was "
            f"spawned. The dispatch you tried: {preview!r}. Acknowledge "
            f"briefly to the user and pivot."
        )
        try:
            result = await self.auto_ping(
                session_id=session_id, note=note,
                restricted_to_chat=restricted_to_chat,
            )
        except Exception:
            log.exception(
                "dispatch_denied auto_ping failed session=%s", session_id,
            )
            return
        if not result.text or self._events is None:
            return
        await self._events.publish_global("chat.reply", {
            "session_id": session_id,
            "text": result.text,
            "voice_text": result.text,
            "trigger": "dispatch.denied",
            "task_id": None,
        })

    async def _run_turn(
        self, session_id: str, user_text: str, *,
        language: str | None = None,
        retrieval_query: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        restricted_to_chat: str | None = None,
        include_silence_gap: bool = False,
    ) -> OperatorTurnResult:
        await self._db.ensure_chat_session(session_id)
        # Read BEFORE this turn's own rows land below, otherwise the answer
        # is always "0m". Drives the <time-since-last-message> block.
        last_contact_at = (
            await self._db.last_user_message_at(session_id)
            if include_silence_gap else None
        )
        # Pick the semantic retrieval key UP FRONT so we can inject a
        # `[memory note: ...]` ahead of the actual user message.
        # `retrieval_query` is caller-supplied for inbox-drain (where
        # `user_text` is the synthetic auto-ping note and the real signal
        # is the DM body); otherwise it's the user's own message, except
        # for pure system-note auto-pings (task terminated etc.) which
        # aren't useful retrieval queries.
        effective_query: str | None
        if retrieval_query is not None:
            effective_query = retrieval_query
        else:
            effective_query = (
                None if user_text.startswith(AUTO_PING_PREFIX) else user_text
            )
        if effective_query:
            memory_note = await self._inject_session_memory(
                session_id, effective_query,
            )
            if memory_note:
                await self._db.append_chat_message(
                    session_id, "user", memory_note,
                )
        await self._db.append_chat_message(session_id, "user", user_text)
        # Attachments (e.g. a photo the user sent to the Telegram bot) are
        # persisted to history as short TEXT placeholders so reloads stay
        # bounded — the actual bytes live only in the in-memory `messages`
        # list for THIS turn. Same model `read_image` uses for its loaded
        # bytes; cross-turn the operator remembers what it described in
        # its assistant reply rather than the original pixels.
        #
        # Hard cap at 3: Gemini flash-lite (the default operator model)
        # tops out around 3 inline-image parts per request before
        # latency / reliability degrades. Any extras are dropped from
        # this turn — the model still sees the placeholders so it can
        # ask the user to resend if needed.
        attachments = list(attachments or [])[:3]
        for att in attachments:
            placeholder = (
                f"[attachment: {att.get('mime_type', '?')}, "
                f"{att.get('size_bytes', 0)} bytes — "
                f"{att.get('source', '?')}; content not persisted]"
            )
            await self._db.append_chat_message(session_id, "user", placeholder)

        # Load + possibly compress the rolling history. Compression is
        # idempotent: if no compression is needed, the summary returned is
        # whatever the previous summary was (possibly None). At this point
        # the rolling history already contains the memory note we may have
        # injected above (it was appended to chat_messages before this
        # load), so the model will see it in its proper position before
        # the new user turn.
        summary, history = await self._load_and_maybe_compress(session_id)
        # System prompt is now stable (no memory block); cache prefix
        # survives across turns.
        system_prompt = await self._build_system_prompt(None)
        if language:
            # Language hint goes at the END of the system prompt so it overrides
            # any natural-language drift from the prior history window.
            system_prompt = (
                f"{system_prompt}\n\n# Output language\n\n"
                f"Respond in: {language}. Match the user's dialect/register if "
                f"the conversation history already establishes one."
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if summary is not None:
            messages.append({
                "role": "system",
                "content": (
                    "# Earlier conversation (compressed)\n\n"
                    "The exchanges that preceded the messages below were summarized "
                    "to keep this prompt small. Treat the summary as authoritative "
                    "context. If you need current task state, query it (list_tasks / "
                    "get_task_status) — don't speculate.\n\n"
                    f"{summary['summary']}"
                ),
            })
        for row in history:
            messages.append(_row_to_openai_message(row))

        # Inline the attachment bytes for THIS turn only. Same list-content
        # shape that the `read_image` follow-up uses — Gemini sees an
        # `inline_data` Part, the Vercel gateway sees `image_url` with a
        # data URI. The DB has a short text placeholder per attachment
        # (appended above) so reload doesn't refeed the bytes. The 3-cap
        # was already applied when we wrote the placeholders.
        for att in attachments:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"(Attachment from {att.get('source', '?')} — "
                        f"{att.get('mime_type', '?')}, "
                        f"{att.get('size_bytes', 0)} bytes)"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": (
                            f"data:{att.get('mime_type', 'application/octet-stream')}"
                            f";base64,{att['data_b64']}"
                        ),
                    }},
                ],
            })

        # Acting-status: transient user-turn note so the operator knows
        # whether the previous hand_off is still in flight. Not persisted —
        # status changes per turn. The text framing ("acting") matches the
        # operator prompt's mental model; never mentions "executor"/"task".
        status = self._lifecycle.acting_status()
        if status.get("busy"):
            depth = int(status.get("queue_depth") or 0)
            extra = f" — {depth} also queued" if depth else ""
            status_block = f"<acting-status>still acting{extra}</acting-status>"
        else:
            status_block = "<acting-status>idle</acting-status>"
        messages.append({"role": "user", "content": status_block})

        # Call-status: same transient per-turn shape as acting-status. Tells the
        # operator whether THIS reply will be spoken aloud on a live voice call,
        # so it knows when voice-only expression tags are appropriate — and,
        # crucially, that a text turn after a call has ended is NOT on a call
        # (the owner's voice and text share one session, so a stale call-start
        # note can linger in history). Not persisted — recomputed each turn.
        on_call = bool(self._on_call_provider and self._on_call_provider(session_id))
        call_block = (
            "<call-status>on a voice call — your reply is spoken aloud</call-status>"
            if on_call
            else "<call-status>not on a call</call-status>"
        )
        messages.append({"role": "user", "content": call_block})

        # Current time: same transient per-turn shape as the statuses above,
        # recomputed every turn and never persisted. Without it the operator
        # has NO clock and will confidently FABRICATE one when asked the time
        # (it once told the owner "21:05" at 00:33 local). Always UTC (the
        # daemon runs on a UTC server); the operator converts to the owner's
        # local time using their timezone from memory — see the Time section
        # of the system prompt.
        time_block = f"<current-time>{format_utc_now()}</current-time>"
        messages.append({"role": "user", "content": time_block})

        # How long the owner has been silent — opt-in per caller, and today
        # only the voice call-start greeting asks for it. History carries no
        # per-message timestamps into model context, so opening a call the
        # operator cannot tell a thread the owner dropped ten minutes ago from
        # one they dropped three days ago, and picks the open thread back up
        # as if no time passed. It is NOT wanted on ordinary turns: mid-
        # conversation the gap is noise, and a clock the operator sees every
        # turn is a clock it starts narrating. Same transient shape as the
        # blocks above — recomputed, never persisted. Suppressed under
        # LAST_CONTACT_MIN_GAP_S so calling right back stays quiet.
        if last_contact_at:
            try:
                gap_s = (utcnow() - datetime.fromisoformat(last_contact_at)).total_seconds()
            except ValueError:
                log.warning("unparseable last-contact timestamp %r", last_contact_at)
                gap_s = 0.0
            if gap_s >= LAST_CONTACT_MIN_GAP_S:
                messages.append({"role": "user", "content": (
                    f"<time-since-last-message>{_fmt_elapsed(gap_s)} since the "
                    f"user last spoke to you</time-since-last-message>"
                )})

        # Laptop-status: cloud-primary mode only. Tells the operator, this
        # turn, whether the user's laptop is reachable — i.e. whether a
        # hand_off for their project/development work can succeed. When
        # offline, the operator should decline that work up front rather than
        # spawn a task that will only hit `{"error":"laptop_offline"}`. The
        # block says what the laptop is FOR, not just what's broken: the
        # operator once refused a Telegram lookup as "laptop offline" because
        # the offline text left its scope open. Not persisted — recomputed each
        # turn. Absent in legacy all-local mode.
        laptop_online: bool | None = None
        if self._laptop_status_provider is not None:
            laptop_online = False
            try:
                laptop_online = bool(self._laptop_status_provider())
            except Exception:
                log.warning("laptop status provider raised; treating as offline", exc_info=True)
            laptop_block = (
                "<laptop-status>online — the user's laptop is reachable, so "
                "project/development work on their machine can run via hand_off"
                "</laptop-status>"
                if laptop_online
                else "<laptop-status>offline — the user's laptop is unreachable, so "
                "project/development work on their machine is UNAVAILABLE this turn: "
                "say it's offline and to try again when it's back, and do not hand_off "
                "for it. The laptop serves ONLY that work — every other capability runs "
                "server-side and is unaffected, so hand off for those as normal."
                "</laptop-status>"
            )
            messages.append({"role": "user", "content": laptop_block})

        # Debug-only snapshot of THIS turn's transient statuses, persisted on
        # every message row written below (see chat_messages.statuses). Never
        # fed back into context — it exists purely so we can later inspect what
        # the operator saw when it produced a given reply (e.g. confirm
        # <call-status> was off when a voice-only tag leaked into text).
        turn_statuses: dict[str, Any] = {
            "on_call": on_call,
            "acting_busy": bool(status.get("busy")),
            "acting_queue_depth": int(status.get("queue_depth") or 0),
            "laptop_online": laptop_online,
        }

        tool_calls_made: list[dict[str, Any]] = []
        for _round in range(self._max_tool_rounds):
            # One operator LLM round-trip. `timed` records wall-clock into the
            # "operator" latency window (surfaced in /status); a raise
            # (timeout/error) lands as an error sample, not a bogus reading.
            # This times the operator's own model only — the Claude executor
            # on the hand_off path runs elsewhere and is not measured here.
            with timed("operator"):
                resp = await self._llm.chat(
                    model=self._settings.oncall_operator_model,
                    messages=messages,
                    tools=OPERATOR_TOOLS,
                    # Gemini thinking models count `thoughts_token_count`
                    # against `max_output_tokens` — at reasoning_effort=low
                    # that's ~500–1300 tokens before any visible output, so
                    # 512 truncated replies mid-sentence. 2048 leaves ~1.5k+
                    # for the actual reply even at MEDIUM. Operator replies
                    # are still terse by prompt — this is just headroom.
                    max_tokens=2048,
                    reasoning_effort=self._settings.oncall_operator_reasoning_effort,
                )
            tc_list = resp.get("tool_calls") or []
            if not tc_list:
                final_text = _strip_breadcrumb_impersonation(
                    resp.get("content") or ""
                )
                await self._db.append_chat_message(
                    session_id, "assistant", final_text, statuses=turn_statuses,
                )
                return OperatorTurnResult(text=final_text, tool_calls_made=tool_calls_made)

            # Persist the assistant turn that holds the tool_calls (OpenAI format
            # needs them present so the next API call validates).
            assistant_dict = {
                "role": "assistant",
                "content": resp.get("content") or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments_json"]},
                        # Gemini-only metadata: preserved across DB round-trip
                        # via assistant_tool_calls history rows. Backends that
                        # don't need it (Vercel gateway) just ignore the key.
                        **({"gemini_thought_signature_b64": tc["thought_signature_b64"]}
                           if tc.get("thought_signature_b64") else {}),
                    }
                    for tc in tc_list
                ],
            }
            messages.append(assistant_dict)
            await self._db.append_chat_message(
                session_id, "assistant_tool_calls", json.dumps(assistant_dict),
                statuses=turn_statuses,
            )

            for tc in tc_list:
                try:
                    args = json.loads(tc["arguments_json"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                operator_log.info("tool_call " + fmt(
                    chat=session_id, tool=tc["name"], args=json.dumps(args, ensure_ascii=False),
                ))
                try:
                    result = await self._execute_tool(
                        session_id, tc["name"], args,
                        restricted_to_chat=restricted_to_chat,
                        tool_calls_made=tool_calls_made,
                        user_text=user_text,
                    )
                except Exception as e:
                    log.exception("operator tool %s failed", tc["name"])
                    result = {"error": f"{type(e).__name__}: {e}"}
                # read_image stashes the loaded bytes under `_attachment`.
                # Strip it before logging / serializing / persisting; we
                # replay the bytes as a follow-up user message below so the
                # model can actually see the image on the next round.
                attachment = None
                if isinstance(result, dict) and "_attachment" in result:
                    attachment = result.pop("_attachment")
                operator_log.info("tool_result " + fmt(
                    chat=session_id, tool=tc["name"],
                    result=json.dumps(result, ensure_ascii=False),
                ))
                tool_calls_made.append({"name": tc["name"], "args": args, "result": result})
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                }
                messages.append(tool_msg)
                await self._db.append_chat_message(
                    session_id, "tool", json.dumps({
                        "tool_call_id": tc["id"], "name": tc["name"],
                        "args": args, "result": result,
                    }),
                    statuses=turn_statuses,
                )
                if attachment is not None:
                    # In-memory: full image bytes via a list-content user
                    # message. Both GenAILLMClient (Gemini) and the OpenAI
                    # gateway accept the {type: image_url, ...} part shape.
                    inject = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                f"(Attachment loaded by read_image — "
                                f"{attachment['mime_type']}, "
                                f"{attachment['size_bytes']} bytes, "
                                f"source: {attachment['source']})"
                            )},
                            {"type": "image_url", "image_url": {
                                "url": (
                                    f"data:{attachment['mime_type']};base64,"
                                    f"{attachment['data_b64']}"
                                ),
                            }},
                        ],
                    }
                    messages.append(inject)
                    # DB: persist a TEXT placeholder only. The next turn's
                    # load_chat_history won't refeed the bytes — the model
                    # has to call read_image again if it needs to see them.
                    placeholder = (
                        f"[attachment loaded via read_image: "
                        f"{attachment['mime_type']}, "
                        f"{attachment['size_bytes']} bytes — "
                        f"{attachment['source']}; content not persisted]"
                    )
                    await self._db.append_chat_message(
                        session_id, "user", placeholder, statuses=turn_statuses,
                    )

            # Short-circuit after a successful hand_off: the operator
            # already emitted its ack in this same turn (assistant_dict's
            # `content`), and our contract is that nothing else should be
            # said before acting completes. Skipping the next LLM round
            # also matters for ordering — the executor can fail fast and
            # publish its result-delivery message; if we waited for
            # another LLM round here, the ack would land AFTER the result.
            handed_off = any(
                tc["name"] == "hand_off"
                and not (
                    isinstance(c.get("result"), dict) and c["result"].get("error")
                )
                for tc, c in zip(tc_list, tool_calls_made[-len(tc_list):])
            )
            if handed_off:
                final_text = _strip_breadcrumb_impersonation(
                    resp.get("content") or ""
                )
                return OperatorTurnResult(
                    text=final_text, tool_calls_made=tool_calls_made,
                )

        # Hit the tool-round cap.
        msg = "I'm stuck — too many tool rounds without a final answer. Try rephrasing."
        await self._db.append_chat_message(
            session_id, "assistant", msg, statuses=turn_statuses,
        )
        return OperatorTurnResult(text=msg, tool_calls_made=tool_calls_made)

    # ---- context compression ----

    async def _load_and_maybe_compress(
        self, session_id: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Returns (summary, history). `history` is rows newer than the latest
        summary's checkpoint.

        Compression does NOT run inline. If the window is over budget we reply
        from the uncompressed history and compress in the background, so the
        turn that happens to cross the threshold pays a slightly larger prefill
        instead of waiting on an opus one-shot (tens of seconds) before it can
        say anything. The summary lands before the next turn loads."""
        summary = await self._db.get_latest_chat_summary(session_id)
        since_id = summary["through_message_id"] if summary else 0
        # Big upper bound — compression is what keeps this small, not the limit.
        history = await self._db.load_chat_history(session_id, since_id=since_id, limit=2000)

        threshold = self._settings.oncall_compression_threshold_tokens
        if _estimate_tokens(summary, history) > threshold:
            self._schedule_compression(session_id, summary, history)
        return summary, history

    def _schedule_compression(
        self,
        session_id: str,
        summary: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> None:
        """Fire-and-forget background compression for `session_id`.

        At most one in flight per session: the trigger is "history is over
        budget", which stays true until the summary lands, so every turn in
        between would otherwise queue another redundant opus one-shot against
        the same rows."""
        if session_id in self._compressing:
            return
        self._compressing.add(session_id)

        async def _run() -> None:
            try:
                new_summary = await self._compress_history(session_id, summary, history)
                if new_summary is None:
                    # Already logged by _compress_history/_summarize_older. The
                    # window stays over budget, so the next turn retries — at
                    # the cost of a bigger prefill until it succeeds.
                    log.warning(
                        "background compression produced no summary (session=%s); "
                        "history stays uncompressed", session_id,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("background compression crashed (session=%s)", session_id)
            finally:
                self._compressing.discard(session_id)

        task = asyncio.create_task(_run())
        # Strong ref so the task isn't GC'd mid-flight (same reason as
        # _extraction_tasks).
        self._compression_tasks.add(task)
        task.add_done_callback(self._compression_tasks.discard)

    async def _summarize_older(
        self,
        session_id: str,
        prior_summary: dict[str, Any] | None,
        older: list[dict[str, Any]],
    ) -> str | None:
        """Fold `older` (+ any prior summary) into an updated summary string,
        or None when there's nothing to summarize OR the model returned an
        implausible result that must NOT be persisted (persisting it would
        replace the folded rows with garbage and destroy the session's live
        context). Shared by the auto (`_compress_history`) and manual
        (`compress_now`) paths so the ~10% length target and the sanity guard
        cover both."""
        if not older:
            return None
        prior_text = (prior_summary or {}).get("summary") or "(no prior summary)"
        prior_est = int((prior_summary or {}).get("estimated_token_count") or 0)
        in_tokens = sum(len(r["content"]) // 4 for r in older) + prior_est
        # ~10% retention, clamped: enough to keep the gist of even casual
        # conversation, still a large shrink. Handed to the model as a word
        # target (tokens ≈ words × 1.33).
        target_words = max(120, min(1200, int(in_tokens * 0.10 / 1.33)))
        formatted = "\n".join(
            f"[{r['role']}]: {r['content'][:2000]}" for r in older
        )
        prompt = (
            "Fold the transcript below into an updated running summary of the "
            "conversation so far.\n\n"
            f"Summary so far:\n{prior_text}\n\n"
            f"Target length: about {target_words} words (~10% of the source). "
            "Never shorter than the summary above — you are extending it, not "
            "replacing it.\n\n"
            "<transcript>\n"
            f"{formatted}\n"
            "</transcript>\n\n"
            "Output ONLY the updated summary as third-person plain prose."
        )
        text = await self._runner.one_shot(
            prompt,
            system_prompt=COMPRESSION_SYSTEM_PROMPT,
            model=self._settings.oncall_compression_model,
            effort=self._settings.oncall_compression_effort or None,
            timeout_s=self._settings.oncall_compression_timeout_seconds,
        )
        if not text:
            log.warning(
                "compression: runner returned empty (session=%s)", session_id,
            )
            return None
        # Sanity guard. A summary far below its input size — or one that shrank
        # a prior summary it was meant to EXTEND — is almost certainly the model
        # role-playing or refusing instead of summarizing. Reject it and keep
        # the prior checkpoint rather than persist context loss. Regression:
        # Opus/Sonnet fed 471 rows of in-character small talk once returned a
        # 7-token echo of the assistant's last line ("На зв'язку!"), which
        # persisted and gutted the session's context.
        est = len(text) // 4
        if est < int(in_tokens * _MIN_SUMMARY_RATIO) or (
            prior_est and est < int(prior_est * _PRIOR_RETENTION_FLOOR)
        ):
            log.warning(
                "compression: rejecting implausible summary — model likely "
                "role-played instead of summarizing (session=%s est_tokens=%s "
                "in_tokens=%s prior_est=%s older_rows=%s); keeping prior "
                "checkpoint. Head: %r",
                session_id, est, in_tokens, prior_est, len(older), text[:120],
            )
            return None
        return text

    async def _compress_history(
        self,
        session_id: str,
        prior_summary: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Walk history backwards, choose a safe split point at a user-message
        boundary near the halfway-token mark, summarize the older portion (+ any
        prior summary), and persist a new chat_summaries row. Returns the new
        summary dict or None on failure / rejection."""
        threshold = self._settings.oncall_compression_threshold_tokens
        # Walk from newest to oldest; mark the split when we've covered ~half
        # the token budget and we're sitting at a `user` row (safe boundary —
        # tool_calls + tool rows always come AFTER a user message, never before).
        acc = 0
        split_idx: int | None = None
        for i in range(len(history) - 1, -1, -1):
            acc += len(history[i]["content"]) // 4
            if acc >= threshold // 2 and history[i]["role"] == "user":
                split_idx = i
                break
        if split_idx is None or split_idx == 0:
            # Nothing safely splittable. Skip.
            return None
        older = history[:split_idx]
        text = await self._summarize_older(session_id, prior_summary, older)
        if not text:
            return None
        through_id = older[-1]["id"]
        est = len(text) // 4
        await self._db.insert_chat_summary(
            session_id=session_id, summary=text,
            through_message_id=through_id, estimated_token_count=est,
        )
        operator_log.info("compress " + fmt(
            chat=session_id, through_id=through_id, summary_tokens=est,
            older_rows=len(older),
        ))
        return {
            "summary": text,
            "through_message_id": through_id,
            "estimated_token_count": est,
        }

    # ---- introspection (for /status surfaces) ----

    async def get_status(self, session_id: str) -> dict[str, Any]:
        """Cheap snapshot of operator-side state for /status. Reads only what's
        in the DB + in-memory configuration; no LLM round-trip."""
        summary = await self._db.get_latest_chat_summary(session_id)
        since_id = summary["through_message_id"] if summary else 0
        history = await self._db.load_chat_history(
            session_id, since_id=since_id, limit=2000,
        )
        return {
            "model": self._settings.oncall_operator_model,
            "memory_entries": await self._memory.entries_count(),
            "compression_threshold_tokens": self._settings.oncall_compression_threshold_tokens,
            "session_id": session_id,
            "session_messages_since_summary": len(history),
            "estimated_context_tokens": _estimate_tokens(summary, history),
            "latest_summary": (
                {
                    "through_message_id": summary["through_message_id"],
                    "estimated_token_count": summary["estimated_token_count"],
                    "created_at": summary["created_at"],
                }
                if summary else None
            ),
        }

    # ---- deferred dispatch resolution ----

    async def resolve_dispatch_approval(
        self, dispatch_id: str, decision: str,
    ) -> dict[str, Any]:
        """Called by the bot when the user taps Yes/No on a deferred
        dispatch keyboard. Resolves the `pending_dispatches` row; on
        'allow' actually submits the task to lifecycle with the stored
        params (and `restricted_to_chat` inherited so the executor is also
        locked). Idempotent: a double-tap returns `{"status":
        "already_resolved"}` without firing the task twice.

        Returns a dict the bot can use to update the inline-keyboard
        message (e.g. "Approved — task T1 dispatched")."""
        if decision not in {"allow", "deny"}:
            return {"status": "error", "error": f"bad decision {decision!r}"}
        row = await self._db.get_pending_dispatch(dispatch_id)
        if row is None:
            return {"status": "error", "error": "unknown dispatch_id"}
        if row["resolution"] is not None:
            return {"status": "already_resolved", "resolution": row["resolution"]}
        ok = await self._db.resolve_pending_dispatch(dispatch_id, decision)
        if not ok:
            # Raced with another tap; the other won. Report idempotent.
            row = await self._db.get_pending_dispatch(dispatch_id)
            return {"status": "already_resolved",
                    "resolution": (row or {}).get("resolution")}
        if decision == "deny":
            operator_log.info("dispatch_task.denied " + fmt(
                dispatch_id=dispatch_id,
                chat=row["chat_session_id"],
                locked_to=row["restricted_to_chat"],
            ))
            # Fire-and-forget: tell the operator the tap was Deny. Without
            # this the operator dangles — it called dispatch_task, got
            # pending_approval back as the tool result, and would never
            # learn the user denied unless they manually re-prompt.
            asyncio.create_task(self._notify_dispatch_denied(
                session_id=row["chat_session_id"],
                prompt_preview=row["prompt"],
                restricted_to_chat=row["restricted_to_chat"],
            ))
            return {"status": "denied"}
        # Approved → actually spawn. The new task inherits restricted_to_chat
        # so anything the executor calls via /internal/messenger is gated
        # to the same chat as the parent operator turn. Inject memory
        # context fresh at spawn time (not at the dispatch_task call) so
        # the executor sees what the operator knows RIGHT NOW.
        try:
            hits = await self._memory.retrieve(row["prompt"], limit=10)
        except Exception:
            log.exception("memory retrieve failed for approved dispatch %s", dispatch_id)
            hits = []
        recent = _format_recent_context(
            await self._db.load_chat_history(row["chat_session_id"], limit=40)
        )
        augmented_prompt = _inject_memory_context(row["prompt"], hits)
        if recent:
            augmented_prompt = recent + "\n\n" + augmented_prompt
        task = await self._lifecycle.submit_task(
            prompt=augmented_prompt,
            model=row["model"],
            chat_session_id=row["chat_session_id"],
            restricted_to_chat=row["restricted_to_chat"],
        )
        operator_log.info("dispatch_task.approved " + fmt(
            dispatch_id=dispatch_id, task=str(task.id),
            chat=row["chat_session_id"],
            locked_to=row["restricted_to_chat"],
        ))
        return {
            "status": "approved",
            "task_id": str(task.id),
            "session_id": task.session_id,
            "state": task.state.value,
        }

    # ---- session reset / on-demand compression ----

    async def clear_session(self, session_id: str) -> dict[str, object]:
        """Wipe a chat session's rolling history and any compression
        checkpoints, AND forget the shared executor `claude` session so the
        next hand_off starts a fresh conversation. The operator-memory store
        is NOT touched — it's cross-session and out of scope for /clear.

        The executor session is global (shared across chats), so its reset is
        best-effort and refused while a task is in-flight (see
        Lifecycle.reset_executor_session); `executor_session_reset` /
        `executor_reset_reason` report the outcome.

        Held under the session lock so an in-flight chat_turn / auto_ping
        finishes first; otherwise the user could `/clear` mid-reply and
        leak a half-deleted state to the next turn."""
        async with self._lock_for(session_id):
            messages = await self._db.delete_chat_messages(session_id)
            summaries = await self._db.delete_chat_summaries(session_id)
            # Also reset memory injection tracking. The new history is
            # empty, so memories that were shown previously have to
            # be re-injected when they next become relevant.
            shown = await self._db.clear_session_memory_shown(session_id)
        exec_reset = self._lifecycle.reset_executor_session()
        operator_log.info("session_clear " + fmt(
            chat=session_id, messages=messages, summaries=summaries,
            memory_shown_cleared=shown, executor_reset=exec_reset.get("reset"),
            executor_reset_reason=exec_reset.get("reason"),
        ))
        return {
            "messages_deleted": messages,
            "summaries_deleted": summaries,
            "memory_shown_cleared": shown,
            "executor_session_reset": bool(exec_reset.get("reset")),
            "executor_reset_reason": exec_reset.get("reason"),
        }

    async def export_context(self, session_id: str) -> str:
        """Render the operator's CURRENT context for this session as a plain
        markdown document. Includes the latest compression summary (if any)
        and every live `chat_messages` row newer than that checkpoint —
        which is exactly the window that would be fed to the LLM on the
        next turn. Memory snippets are NOT included; they vary per-query
        and aren't part of stable session state.

        Returns the document as a single UTF-8 string for the caller to
        ship (e.g. the Telegram bot uploads it via sendDocument)."""
        summary = await self._db.get_latest_chat_summary(session_id)
        since_id = summary["through_message_id"] if summary else 0
        history = await self._db.load_chat_history(
            session_id, since_id=since_id, limit=2000,
        )
        memory_count = await self._memory.entries_count()

        lines: list[str] = [
            f"# Operator context — session {session_id}",
            "",
            f"- Exported: {iso(utcnow())}",
            f"- Model: {self._settings.oncall_operator_model}",
            f"- Live messages: {len(history)}",
            f"- Memory entries (cross-session, not exported): {memory_count}",
        ]
        if summary is not None:
            lines.append(
                f"- Latest compression checkpoint: through_message_id="
                f"{summary['through_message_id']}, "
                f"~{summary['estimated_token_count']} summary tokens, "
                f"created {summary['created_at']}"
            )
        lines.append("")

        if summary is not None:
            lines += [
                "## Compression summary",
                "",
                summary["summary"].rstrip(),
                "",
            ]

        lines += [f"## Live history ({len(history)} messages)", ""]
        if not history:
            lines.append("_(empty)_")
        for row in history:
            role = row["role"]
            ts = row["created_at"]
            lines.append(f"### [{ts}] {role} (id={row['id']})")
            lines.append("")
            lines.append("```")
            lines.append(row["content"].rstrip())
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    async def compress_now(self, session_id: str) -> dict[str, Any]:
        """Force-compress this session's older history into a fresh
        chat_summaries checkpoint, bypassing the auto-compression token
        threshold. Strategy: split immediately before the most-recent
        user-role row so the active turn pair stays live; summarize
        everything older, persist as a new summary, return a metadata
        dict for the caller to render to the user.

        Returns `{"compressed": False, "reason": ...}` when there isn't
        enough material to safely split (e.g. <2 messages, no prior user
        turn, runner returned empty)."""
        async with self._lock_for(session_id):
            prior = await self._db.get_latest_chat_summary(session_id)
            since_id = prior["through_message_id"] if prior else 0
            history = await self._db.load_chat_history(
                session_id, since_id=since_id, limit=2000,
            )
            if len(history) < 2:
                return {"compressed": False, "reason": "not enough history"}
            # Walk backwards for the most-recent user-role row; everything
            # before it becomes the older bucket.
            split_idx: int | None = None
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "user":
                    split_idx = i
                    break
            if split_idx is None or split_idx == 0:
                return {"compressed": False, "reason": "no older user turn to anchor split"}
            older = history[:split_idx]
            text = await self._summarize_older(session_id, prior, older)
            if not text:
                return {
                    "compressed": False,
                    "reason": "runner returned empty or summary rejected as implausible",
                }
            through_id = older[-1]["id"]
            est = len(text) // 4
            await self._db.insert_chat_summary(
                session_id=session_id, summary=text,
                through_message_id=through_id, estimated_token_count=est,
            )
            operator_log.info("compress_now " + fmt(
                chat=session_id, through_id=through_id,
                summary_tokens=est, older_rows=len(older),
            ))
            return {
                "compressed": True,
                "through_message_id": through_id,
                "summary_tokens": est,
                "older_rows": len(older),
            }

    # ---- memory extraction (fire-and-forget background) ----

    async def _latest_assistant_text(self, session_id: str) -> str | None:
        """Most recent plain-assistant row in this session (skips
        `assistant_tool_calls` framing). Returns None if no prior assistant
        turn exists yet."""
        rows = await self._db.load_chat_history(session_id, limit=20)
        for r in reversed(rows):
            if r["role"] == "assistant":
                text = (r["content"] or "").strip()
                if text:
                    return text
        return None

    # Note used to wrap citation suggestions for the silent auto-ping that
    # follows a user turn. The operator's prompt has a matching rule —
    # derive a clean memory from each citation worth keeping, then call
    # `save_memory` with the DERIVED text; never save the citation verbatim,
    # never narrate.
    _CITATIONS_NOTE_PREFIX = "extractor flagged citations from the user"

    async def _extract_and_propose(
        self,
        *,
        session_id: str,
        user_text: str,
        prev_assistant_text: str | None,
        already_saved: list[str],
    ) -> None:
        """Run the candidate-suggester off the hot path. The operator (not
        this function) owns memory writes — we just route suggestions back
        to it via a SILENT auto-ping. The operator may then call
        `save_memory` for any candidates it judges worth keeping; if not,
        nothing happens.

        `already_saved` is the list of facts the operator committed via
        `save_memory` during THIS user turn — captured by chat_turn before
        the scheduling so a fast follow-up turn can't clobber it. The
        suggester uses it to skip near-duplicates.

        Failures stay visible — we emit a `_Memory extraction failed: ..._`
        breadcrumb so the user knows when the suggester is broken (LRU
        otherwise hides the absence)."""
        assert self._extract_llm is not None  # checked by caller
        try:
            candidates = await memory_extractor.extract_candidates(
                self._extract_llm,
                model=self._extract_model,
                user_text=user_text,
                prev_assistant_text=prev_assistant_text,
                already_saved=already_saved,
            )
        except Exception as e:
            # Don't paste the upstream's full error body into the user's chat —
            # SDK errors like google-genai's ClientError stringify to
            # `429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': ...}}`
            # and the dict is huge. Keep just the human-readable prefix.
            msg = str(e).splitlines()[0]
            brace = msg.find("{")
            if brace > 0:
                msg = msg[:brace].rstrip(" .:,-")
            if len(msg) > 160:
                msg = msg[:157] + "…"
            text = f"SYSTEM: Memory extraction failed: {type(e).__name__}: {msg}"
            operator_log.exception("memory extraction failed for session %s", session_id)
            await self._emit_breadcrumb(session_id, text)
            return

        if not candidates:
            return
        # SILENT suggestion auto-ping: the operator gets the candidates as
        # a system note and may call `save_memory` for any it wants kept.
        # We do NOT publish chat.reply for the operator's reply text on
        # this turn — only `save_memory` calls (which emit their own
        # breadcrumbs via the tool handler) reach the user. The operator
        # prompt has a matching rule telling it to emit empty content.
        bullets = "\n".join(f"  • {c}" for c in candidates)
        note = (
            f"{self._CITATIONS_NOTE_PREFIX}. These are RAW QUOTES, not "
            f"memory text. For any worth keeping, derive a clean, specific, "
            f"self-contained memory from the citation (resolve names, "
            f"identifiers, roles into a declarative sentence) and call "
            f"`save_memory` with the DERIVED text — not the citation "
            f"verbatim. Ignore citations not worth keeping. Emit empty "
            f"assistant content — no text reply.\n"
            f"{bullets}"
        )
        try:
            await self.auto_ping(
                session_id=session_id, note=note, retrieval_query=None,
            )
        except Exception:
            operator_log.exception(
                "candidate-suggestion auto-ping failed for session %s",
                session_id,
            )

    async def _emit_breadcrumb(self, session_id: str, text: str) -> None:
        """Append the breadcrumb to chat history (so future turns and
        /status reflect it) and publish a chat.reply event for live UIs."""
        async with self._lock_for(session_id):
            await self._db.append_chat_message(session_id, "assistant", text)
        if self._events is not None:
            try:
                await self._events.publish_global("chat.reply", {
                    "session_id": session_id,
                    "text": text,
                    "trigger": "memory.breadcrumb",
                })
            except Exception:
                log.exception("failed to publish memory breadcrumb event")

    # ---- tool execution ----

    # Max chars of recent operator-user dialogue prepended to a hand_off
    # prompt as context for the executor. Cap is per hand_off, not
    # cumulative — the cursor dedups so each chunk is forwarded once.
    _HANDOFF_CONTEXT_MAX_CHARS = 1024

    async def _compose_handoff_prompt(
        self, *, chat_session_id: str, user_text: str, hint: str,
    ) -> tuple[str, int]:
        """Build the prompt forwarded to the executor for one hand_off.

        Includes a tail of recent operator-user dialogue (up to
        `_HANDOFF_CONTEXT_MAX_CHARS` chars) that the executor hasn't
        seen yet, plus the user's verbatim message, plus an optional
        operator hint. Returns (prompt, new_cursor) — the cursor is the
        latest chat_messages.id covered by this hand_off; caller
        persists it after a successful enqueue.

        On any failure loading history we degrade to the user_text-only
        path (still safe — the executor session itself accumulates context
        across resumes, so missing a chunk just means slightly less ground
        truth for one turn).
        """
        try:
            cursor = await self._db.get_handoff_cursor(chat_session_id)
            history = await self._db.load_chat_history(
                chat_session_id, since_id=cursor, limit=200,
            )
        except Exception:
            log.exception("hand_off: history fetch failed; sending user_text only")
            return self._format_handoff_body(
                tail=[], user_text=user_text, hint=hint,
            ), 0

        tail: list[tuple[str, str, str]] = []  # (label, content, ts)
        new_cursor = cursor
        for row in history:
            mid = int(row["id"])
            if mid > new_cursor:
                new_cursor = mid
            role = row.get("role")
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if role not in ("user", "assistant"):
                continue
            if content.startswith("[memory note:") or content.startswith(AUTO_PING_PREFIX):
                continue
            if (
                content.startswith("<acting-status>")
                or content.startswith("<call-status>")
                or content.startswith("<laptop-status>")
            ):
                continue
            if content == user_text:
                # The latest user turn — printed as "user (now)" below.
                continue
            label = "user" if role == "user" else "operator"
            tail.append((label, content, _fmt_ts(row.get("created_at") or "")))

        # Trim from the FRONT until under the char cap (keep most recent).
        # Per-line overhead is label + content + ts + brackets + ": " + "\n".
        def total(items: list[tuple[str, str, str]]) -> int:
            return sum(len(lbl) + len(c) + len(ts) + 6 for lbl, c, ts in items)
        while tail and total(tail) > self._HANDOFF_CONTEXT_MAX_CHARS:
            tail.pop(0)

        return self._format_handoff_body(
            tail=tail, user_text=user_text, hint=hint,
        ), new_cursor

    @staticmethod
    def _format_handoff_body(
        *, tail: list[tuple[str, str, str]], user_text: str, hint: str,
    ) -> str:
        parts: list[str] = []
        if tail:
            parts.append(
                "[recent operator↔user dialogue, for context — newest last]"
            )
            for label, content, ts in tail:
                prefix = f"[{ts}] " if ts else ""
                parts.append(f"{prefix}{label}: {content}")
            parts.append("")
        if hint:
            parts.append(f"[operator hint: {hint}]")
            parts.append("")
        parts.append(_strip_operator_only_action(user_text))
        return "\n".join(parts)

    async def _execute_tool(
        self, chat_session_id: str, name: str, args: dict[str, Any],
        *, restricted_to_chat: str | None = None,
        tool_calls_made: list[dict[str, Any]] | None = None,
        user_text: str = "",
    ) -> dict[str, Any]:
        del tool_calls_made  # vestigial; kept for compat
        if name == "hand_off":
            text = (user_text or "").strip()
            if not text:
                return {
                    "error": (
                        "no fresh user message to hand off — reply to the user "
                        "directly instead."
                    ),
                }
            hint = str(args.get("hint") or "").strip()
            forwarded, new_cursor = await self._compose_handoff_prompt(
                chat_session_id=chat_session_id, user_text=text, hint=hint,
            )
            # Inject memory context fresh on EVERY hand_off. The executor
            # session is reset by /clear and compacted at 200K tokens, so we
            # can't rely on it having internalised durable rules (e.g. a
            # per-recipient reply prefix) — they must ride in on each task.
            # This is also the `# Memory context` block the executor system
            # prompt promises; without it that promise is false for hand_off.
            try:
                hits = await self._memory.retrieve(text, limit=10)
            except Exception:
                log.exception(
                    "hand_off: memory retrieve failed for chat %s", chat_session_id,
                )
                hits = []
            forwarded = _inject_memory_context(forwarded, hits)
            try:
                outcome = await self._lifecycle.enqueue_executor(
                    prompt=forwarded, chat_session_id=chat_session_id,
                    restricted_to_chat=restricted_to_chat,
                )
            except Exception as e:
                log.exception("hand_off: enqueue_executor failed")
                return {"error": f"{type(e).__name__}: {e}"}
            # Advance the cursor only after the enqueue succeeds — if
            # enqueue fails we want the next attempt to re-forward this
            # chunk, not lose it.
            if new_cursor:
                try:
                    await self._db.set_handoff_cursor(
                        chat_session_id, new_cursor,
                    )
                except Exception:
                    log.exception("hand_off: set_handoff_cursor failed")
            operator_log.info("hand_off " + fmt(
                chat=chat_session_id,
                task=outcome.get("task_id"),
                queue_depth=outcome.get("queue_depth"),
                busy=outcome.get("busy"),
                text_preview=text[:120],
                hint=hint or None,
                forwarded_len=len(forwarded),
                cursor=new_cursor,
            ))
            return {"enqueued": True, **outcome}

        if name == "save_memory":
            text = str(args.get("text") or "").strip()
            if not text:
                return {"error": "text required"}
            written = await self._memory.store(
                [text], source_turn=chat_session_id,
            )
            # Track for the extractor pass so it doesn't re-suggest a
            # near-duplicate of what we just wrote.
            self._turn_saves.setdefault(chat_session_id, []).extend(written)
            operator_log.info("save_memory " + fmt(
                chat=chat_session_id, written=len(written), text=text,
            ))
            # Breadcrumb. We're inside _execute_tool, which runs under the
            # session lock — so we append + publish directly instead of
            # going through _emit_breadcrumb (which re-acquires the lock
            # and would deadlock).
            if written:
                joined = ", ".join(written)
                if len(joined) > 400:
                    joined = joined[:397] + "…"
                breadcrumb = f"SYSTEM: Remembered: {joined}"
                await self._db.append_chat_message(
                    chat_session_id, "assistant", breadcrumb,
                )
                if self._events is not None:
                    await self._events.publish_global("chat.reply", {
                        "session_id": chat_session_id,
                        "text": breadcrumb,
                        "voice_text": breadcrumb,
                        "trigger": "memory.breadcrumb",
                        "task_id": None,
                    })
            return {"saved": written}

        if name == "forget_memory":
            try:
                memory_id = int(args.get("memory_id"))
            except (TypeError, ValueError):
                return {"error": "memory_id (integer) required"}
            # Fetch the text BEFORE delete so the audit log records what was
            # forgotten — otherwise we'd just see a bare id with no context.
            existing = await self._memory.get_by_id(memory_id)
            if existing is None:
                return {"error": f"memory_id={memory_id} not found"}
            ok = await self._memory.delete_by_id(memory_id)
            operator_log.info("forget_memory " + fmt(
                chat=chat_session_id, memory_id=memory_id,
                deleted=ok, text=existing.text,
            ))
            return {"forgotten": ok, "memory_id": memory_id}

        if name == "query_memory":
            q = str(args.get("query") or "").strip()
            if not q:
                return {"error": "query required"}
            limit_raw = args.get("limit")
            try:
                limit = int(limit_raw) if limit_raw is not None else 5
            except (TypeError, ValueError):
                limit = 5
            memories = await self._memory.retrieve(q, limit=limit)
            return {
                "query": q,
                "memories": [
                    {"id": m.id, "text": m.text, "score": round(m.score, 3)}
                    for m in memories
                ],
            }

        return {"error": f"unknown tool '{name}'"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per-tool policy for the autonomous-reply lockdown. The operator has no
# direct Telegram tools — all chat work goes through dispatched tasks.
# `dispatch_dm_send` is the only chat-targeted tool here; its chat_id arg
# must equal the restricted chat. `read_image` for Telegram attachments
# is also locked to the restricted chat (handled inline below).
_TOOLS_LOCKED_TO_CHAT_ID = {
    "dispatch_handle_dm",
}
_TOOLS_REFUSED_WHEN_RESTRICTED: set[str] = set()


def _check_restricted_access(
    name: str, args: dict[str, Any], restricted_to_chat: str,
) -> dict[str, Any] | None:
    """Enforce the autonomous-reply lockdown for one tool call. Returns the
    error dict to short-circuit `_execute_tool`, or None to proceed."""
    if name in _TOOLS_REFUSED_WHEN_RESTRICTED:
        return {"error": (
            f"`{name}` is refused during an autonomous-reply turn: reads "
            f"are locked to chat_id={restricted_to_chat}. Stay silent or "
            f"work only within that chat."
        )}
    if name in _TOOLS_LOCKED_TO_CHAT_ID:
        target = str(args.get("chat_id") or "")
        if target != restricted_to_chat:
            return {"error": (
                f"`{name}` is locked to chat_id={restricted_to_chat} for "
                f"this autonomous-reply turn; got chat_id={target!r}. "
                f"Stay silent or work only within the locked chat."
            )}
        return None
    if name == "read_image":
        # Local-file reads are not the locked chat. Telegram-attachment
        # reads must target the locked chat.
        if args.get("path"):
            return {"error": (
                "`read_image(path=...)` is refused during an autonomous-"
                "reply turn: filesystem reads are out of scope. Only "
                f"attachments from chat_id={restricted_to_chat} are allowed."
            )}
        target = str(args.get("chat_id") or "")
        if target and target != restricted_to_chat:
            return {"error": (
                f"`read_image` is locked to chat_id={restricted_to_chat} "
                f"for this autonomous-reply turn; got chat_id={target!r}."
            )}
    return None


def _format_recent_context(history: list[dict[str, Any]], max_chars: int = 1024) -> str:
    """Format the tail of operator<->user chat as a `# Recent context`
    block, capped at ~`max_chars`. Skips memory-note injections (they're
    already covered by the # Memory context block) and the synthetic
    `[system note: ...]` auto-pings. Newest at the bottom; if the cap is
    hit, oldest lines are dropped first."""
    lines: list[str] = []
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role not in ("user", "assistant"):
            continue
        if content.startswith("[memory note:") or content.startswith("[system note:"):
            continue
        label = "user" if role == "user" else "operator"
        lines.append(f"{label}: {content}")
    while lines and sum(len(l) + 1 for l in lines) > max_chars:
        lines.pop(0)
    if not lines:
        return ""
    return "# Recent context (operator<->user, newest last)\n" + "\n".join(lines)


def _inject_memory_context(prompt: str, hits: list[Memory]) -> str:
    """Prepend a `# Memory context` block to a spawned task's prompt
    listing the top-N relevant operator memories. The executor sees these
    inline, no extra tool call needed for the common case. Entries may
    include irrelevant matches — the executor is told so explicitly."""
    if not hits:
        return prompt
    lines = [
        "# Memory context (auto-loaded from operator memory; may include "
        "irrelevant entries — use your judgement)",
    ]
    for m in hits:
        text = m.text.replace("\n", " ").strip()
        lines.append(f"- [id={m.id}] {text}")
    lines.append("")
    lines.append("# Task")
    lines.append(prompt)
    return "\n".join(lines)


def _build_handle_dm_prompt(
    chat_id: str, hint: str, *, user_approved: bool, send_allowed: bool,
) -> str:
    """Build the executor prompt for a `dispatch_handle_dm` task. The
    operator hands off the situation; the executor decides whether and
    what to send after reading chat context and the user's voice. The
    operator does NOT write this prompt — that's the point: the
    history / style / image reads are unskippable, and the decision is
    the executor's based on actual context, not the operator's guess."""
    authority_para = (
        "AUTHORITY: the operator's user just asked you to send something — "
        "you should almost always send. Do not refuse unless the hint is "
        "internally inconsistent or the target is clearly wrong."
        if user_approved else
        "AUTHORITY: the operator's user has a stored memory authorizing "
        "replies to this sender on a specific kind of topic. Read the "
        "history and decide whether this inbound actually matches — if "
        "not (off-topic, ambiguous, casual filler that doesn't require a "
        "response, etc.), exit without sending. Don't force a reply just "
        "because authority exists."
    )
    if not send_allowed:
        authority_para += (
            "\n\nSEND DISABLED: this chat is NOT on the DM allowlist, so "
            "you CANNOT call `op=send` — the broker will reject it. Read "
            "history + any relevant media, then end with a final message "
            "of the form `Draft (not sent — chat not allowlisted): "
            "<draft text>` so the operator can relay it to the user, who "
            "can allowlist the chat and resend."
        )
    return (
        f"You are handling ONE Telegram chat on behalf of the operator's "
        f"user. Target chat_id: {chat_id}. You are the operator itself "
        f"(same identity), but with broader tool access — read what's "
        f"there and decide.\n\n"
        f"Situation + intent (from the operator):\n"
        f"---\n{hint}\n---\n\n"
        f"{authority_para}\n\n"
        f"Mandatory reads, in order — do these BEFORE any decision:\n"
        f"  1. `mcp__oncall__messenger_inbox` `op=history`, "
        f"`chat_id={chat_id}`, `limit=10` — load recent context, both "
        f"sides. Note any `has_media=true` (attachments; body is just "
        f"`[photo]` etc.).\n"
        f"  2. For each RECENT inbound with `has_media=true` that's "
        f"plausibly part of the reply context (last 1-2 from the other "
        f"party, especially if their last text is empty or just an "
        f"emoji), inspect the placeholder body to route correctly:\n"
        f"     - `[photo]` / `[video]` / `[file: ...]` → call "
        f"`op=read_image` with `chat_id` + `message_id` to see the "
        f"actual bytes.\n"
        f"     - `[voice: <s>s]` / `[audio: <s>s]` → call "
        f"`op=transcribe` with `chat_id` + `message_id` to get the "
        f"spoken text. Skip very short voice notes (<2s) — they're "
        f"usually noise. If the transcription returns `pending=true` "
        f"with empty/partial text, work with what you have rather than "
        f"retrying.\n"
        f"     Skip older media that isn't relevant.\n"
        f"  3. `op=style`, `chat_id={chat_id}`, `limit=20` — the user's "
        f"own outgoing samples ARE the voice. If you decide to send, "
        f"mirror length, language, register, capitalization, "
        f"punctuation, emoji.\n\n"
        f"Then decide:\n"
        f"  - If sending makes sense given what you read: compose ONE "
        f"message in the user's voice (NOT necessarily the inbound "
        f"language — match the user's outgoing samples). If the hint "
        f"says \"the user asked to say literally: '<text>'\", send that "
        f"exact text without rephrasing. Then call `op=send`, "
        f"`chat_id={chat_id}`, `text=<your message>`. The broker auto-"
        f"allows this send. Final assistant message: `Sent: <first 80 "
        f"chars>…`.\n"
        f"  - If sending does NOT make sense (off-topic for the "
        f"authority, nothing actionable, ambiguous, would be hollow): "
        f"do NOT call `op=send`. Final assistant message: `Did not "
        f"send: <one-line reason>`. Empty/no-style-samples is one such "
        f"reason — never fake a voice.\n\n"
        f"No other tool calls beyond the reads + at most one `op=send`."
    )


def _uuid(value: Any) -> UUID | None:
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _estimate_tokens(
    summary: dict[str, Any] | None, history: list[dict[str, Any]],
) -> int:
    """Rough character-based heuristic: ~4 chars per token. Good enough for
    deciding when to trigger compression — we don't need precision."""
    total_chars = 0
    if summary is not None:
        total_chars += len(summary.get("summary") or "")
    for row in history:
        total_chars += len(row.get("content") or "")
    return total_chars // 4


def _was_resolved(events: list[dict[str, Any]], approval_id: str) -> bool:
    for e in events:
        if e["type"] == "approval.resolved" and e["payload"].get("approval_id") == approval_id:
            return True
    return False


_BREADCRUMB_LINE_RE = re.compile(
    # Match either the markdown-italicized form the system actually emits
    # (`_Remembered: …_`) or the bare paraphrase the model occasionally
    # produces (`Remembered: …`). Same for the failure breadcrumb. Match
    # whole lines only — we don't want to clip mid-paragraph mentions.
    r"^\s*_?\s*(Remembered|Memory extraction (failed|skipped))\s*:.*?_?\s*$",
    re.IGNORECASE,
)


def _strip_breadcrumb_impersonation(text: str) -> str:
    """Remove any line in the operator's reply that impersonates a memory
    system breadcrumb. The system emits its own `_Remembered: …_` /
    `_Memory extraction failed: …_` message out of band; when the model
    restates one in its main reply (despite the prompt rule), the user
    sees the breadcrumb twice. Strip those lines defensively so the
    duplicate never reaches Telegram. Empty result is fine — the bot's
    chat.reply subscriber drops empty text."""
    if not text:
        return text
    kept = [
        line for line in text.splitlines()
        if not _BREADCRUMB_LINE_RE.match(line)
    ]
    return "\n".join(kept).strip()


def _row_to_openai_message(row: dict[str, Any]) -> dict[str, Any]:
    role = row["role"]
    content = row["content"]
    if role == "user":
        return {"role": "user", "content": content}
    if role == "assistant":
        return {"role": "assistant", "content": content}
    if role == "assistant_tool_calls":
        # Reconstruct the assistant turn that issued tool_calls.
        return json.loads(content)
    if role == "tool":
        payload = json.loads(content)
        return {
            "role": "tool",
            "tool_call_id": payload["tool_call_id"],
            "content": json.dumps(payload["result"]),
        }
    return {"role": "user", "content": content}
