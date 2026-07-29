"""Protocol-layer tests for the stdio MCP server.

These exist because of a real outage: mcp 2.0 removed the `@app.list_tools()` /
`@app.call_tool()` decorators the server was built on, so `oncall mcp` died at
import. The `claude` CLI reported only `MCP tool mcp__oncall__approve (passed
via --permission-prompt-tool) not found. Available MCP tools: none` and every
hand_off failed for ~3h. Nothing here mocked the MCP library, so nothing caught
it. The point of these tests is to fail loudly on the next such API break.
"""

from __future__ import annotations

import httpx
import pytest
from mcp.types import CallToolRequestParams, PaginatedRequestParams

from oncall import mcp_server as m


async def _list(monkeypatch, role: str) -> list[str]:
    monkeypatch.setenv("ONCALL_ROLE", role)
    result = await m._on_list_tools(None, PaginatedRequestParams())
    return [t.name for t in result.tools]


async def test_tools_are_advertised_and_role_gated(monkeypatch):
    """The regression: a live server must advertise `approve` (the
    permission-prompt-tool the CLI resolves by name at startup) and carry a
    usable schema on every tool. Reading `.input_schema` is deliberate — mcp
    2.0 renamed the attribute and left `inputSchema` working only as a
    constructor/wire alias, so a plain `.inputSchema` read raises."""
    laptop_side = await _list(monkeypatch, "server")
    assert "approve" in laptop_side
    # The laptop proxy and its friends are cloud-primary only.
    assert {"laptop", "invoke_developer", "cancel_developer", "schedule"} <= set(laptop_side)

    local = await _list(monkeypatch, "")
    assert "approve" in local
    assert "laptop" not in local

    result = await m._on_list_tools(None, PaginatedRequestParams())
    for tool in result.tools:
        assert tool.input_schema.get("type") == "object", tool.name


async def test_broker_failure_denies_without_faulting_the_protocol(monkeypatch):
    """Safety invariant: when the orchestrator is unreachable the broker must
    fail CLOSED and stay readable to claude — a `deny` payload carried in a
    normal result. mcp 2.0 hands raw params to the handler and turns a raise
    into a JSON-RPC error, which claude would surface as a transport fault
    (killing the turn) instead of a permission decision."""
    async def boom(*a, **kw):
        raise httpx.ConnectError("all connection attempts failed")

    monkeypatch.setattr(httpx.AsyncClient, "post", boom)

    result = await m._on_call_tool(None, CallToolRequestParams(
        name="approve",
        arguments={"tool_name": "Bash", "input": {}, "tool_use_id": "t1"},
    ))

    assert result.is_error is False
    assert '"behavior": "deny"' in result.content[0].text


async def test_bad_arguments_become_a_tool_error_not_a_raise():
    """The 1.x `@app.call_tool()` decorator validated arguments against the
    advertised inputSchema and wrapped failures as isError. A raw 2.0 handler
    does neither, so both are reimplemented — this pins them."""
    result = await m._on_call_tool(None, CallToolRequestParams(name="memory", arguments={}))
    assert result.is_error is True
    assert "'op' is a required property" in result.content[0].text


async def test_unknown_tool_stays_an_ordinary_result():
    """An unknown name is reported in-band rather than as an error, so a model
    that hallucinates a tool gets a correctable message back."""
    result = await m._on_call_tool(None, CallToolRequestParams(name="nope", arguments={}))
    assert result.is_error is False
    assert "unknown tool 'nope'" in result.content[0].text


async def test_read_image_splits_bytes_out_of_the_json_block(monkeypatch):
    """messenger_inbox.read_image returns base64 inline. It must come back as a
    separate ImageContent block with `data_b64` stripped from the JSON — left
    in, the same bytes ride along twice and blow the context window."""
    async def fake_messenger(args):
        return {"mime_type": "image/png", "data_b64": "aGk=", "caption": "c"}

    monkeypatch.setattr(m, "_proxy_messenger", fake_messenger)

    result = await m._on_call_tool(None, CallToolRequestParams(
        name="messenger_inbox", arguments={"op": "read_image", "message_id": "1"},
    ))

    assert result.is_error is False
    text, image = result.content
    assert "aGk=" not in text.text
    assert image.type == "image" and image.data == "aGk=" and image.mime_type == "image/png"


@pytest.mark.parametrize("method", ["tools/list", "tools/call"])
def test_handlers_are_registered_on_the_server(method):
    """Registration moved from decorators to constructor kwargs; if a future
    version renames them again the handlers silently vanish and the CLI is back
    to `Available MCP tools: none`."""
    assert m.app.get_request_handler(method) is not None
