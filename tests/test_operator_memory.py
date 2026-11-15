"""Tests for the Mem0-backed operator memory.

Unit tests drive `Mem0Memory` against an in-process fake of the four Mem0
endpoints it uses (httpx.MockTransport), pinning the policies Mem0 itself
does not provide: standing rows and per-session exclusion.

The live tests at the bottom run against the real Mem0 Platform under a
throwaway user_id and clean up after themselves. They skip unless
`ONCALL_RUN_MEM0_TESTS=1` and `MEM0_API_KEY` are set, because they pin
behaviour of an external system the fake can only imitate.
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest

from oncall.operator_memory import Mem0Memory


class FakeMem0:
    """Just enough of Mem0's REST surface, with its observed quirks: the
    list endpoint stringifies metadata booleans, search keeps them."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.search_bodies: list[dict] = []
        self.event_polls = 0
        self.event_status_sequence: list[str] = ["PENDING", "SUCCEEDED"]
        self.extracted: list[str] = []

    def _matches(self, row: dict, clause: dict) -> bool:
        if "AND" in clause:
            return all(self._matches(row, c) for c in clause["AND"])
        if "NOT" in clause:
            return not any(self._matches(row, c) for c in clause["NOT"])
        if "user_id" in clause:
            return row["user_id"] == clause["user_id"]
        if "memory_ids" in clause:
            return row["id"] in clause["memory_ids"]
        if "metadata" in clause:
            for key, want in clause["metadata"].items():
                have = (row.get("metadata") or {}).get(key)
                if isinstance(want, dict) and "ne" in want:
                    if have == want["ne"]:
                        return False
                elif have != want:
                    return False
            return True
        raise AssertionError(f"unexpected filter clause {clause}")

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path == "/v3/memories/add/" and body.get("infer") is False:
            mid = str(uuid.uuid4())
            self.rows[mid] = {
                "id": mid, "user_id": body["user_id"],
                "memory": body["messages"][0]["content"],
                "metadata": body.get("metadata") or {},
                "created_at": f"2026-11-15T00:00:{len(self.rows):02d}",
            }
            return httpx.Response(200, json={
                "status": "SUCCEEDED",
                "results": [{"id": mid, "data": {"memory": self.rows[mid]["memory"]}, "event": "ADD"}],
            })
        if path == "/v3/memories/add/":
            return httpx.Response(200, json={"event_id": "ev1", "status": "PENDING"})
        if path == "/v1/event/ev1/":
            status = self.event_status_sequence[min(self.event_polls, len(self.event_status_sequence) - 1)]
            self.event_polls += 1
            results = [{"event": "ADD", "data": {"memory": m}} for m in self.extracted]
            return httpx.Response(200, json={"status": status, "results": results, "error": "boom"})
        if path == "/v3/memories/search/":
            self.search_bodies.append(body)
            hits = [r for r in self.rows.values() if self._matches(r, body["filters"])]
            return httpx.Response(200, json={"results": [{**r, "score": 0.5} for r in hits]})
        if path == "/v3/memories/":
            hits = [r for r in self.rows.values() if self._matches(r, body["filters"])]
            listed = [
                {**r, "metadata": {k: str(v) if isinstance(v, bool) else v
                                   for k, v in (r.get("metadata") or {}).items()}}
                for r in hits
            ]
            return httpx.Response(200, json={"count": len(listed), "next": None, "results": listed})
        if request.method == "DELETE" and path.startswith("/v1/memories/"):
            mid = path.split("/")[3]
            if self.rows.pop(mid, None) is None:
                return httpx.Response(404, json={"error": "Memory not found!"})
            return httpx.Response(200, json={"message": "Memory deleted successfully!"})
        raise AssertionError(f"unexpected request {request.method} {path}")


@pytest.fixture
def fake() -> FakeMem0:
    return FakeMem0()


def _memory(fake: FakeMem0, **overrides) -> Mem0Memory:
    kwargs = dict(
        api_key="k", user_id="owner", max_inject=20, search_threshold=0.2,
        standing_cap=2, event_poll_seconds=0.0,
        client=httpx.AsyncClient(
            base_url="https://mem0.test", transport=httpx.MockTransport(fake.handler),
        ),
    )
    kwargs.update(overrides)
    return Mem0Memory(**kwargs)


async def test_standing_cap_refuses_at_the_boundary_and_frees_on_delete(fake):
    """The cap is a refusal, not an eviction, and it counts rows as Mem0's
    list endpoint returns them — with "True" as a string. Reading the flag as
    a bool there would count zero standing rows and never refuse anything."""
    mem = _memory(fake)
    assert await mem.store(["rule one"], standing=True) == ["rule one"]
    assert await mem.store(["rule two"], standing=True) == ["rule two"]
    assert await mem.store(["rule three"], standing=True) == []
    # Contextual facts are not subject to the standing cap.
    assert await mem.store(["a plain fact"]) == ["a plain fact"]

    standing = await mem.standing_rows()
    assert [m.text for m in standing] == ["rule one", "rule two"]
    assert await mem.delete_by_id(standing[0].id)
    # The delete invalidated the cache, so the freed slot is usable at once.
    assert await mem.store(["rule three"], standing=True) == ["rule three"]
    assert [m.text for m in await mem.standing_rows()] == ["rule two", "rule three"]


async def test_retrieve_pushes_exclusion_and_standing_filter_into_the_search(fake):
    """Exclusion must happen inside the search: filtering the top-k after the
    fact spends it on rows the session already has and injects nothing (the
    old store went memory-blind exactly that way). Standing rows stay out of
    contextual retrieval unless asked for."""
    mem = _memory(fake)
    await mem.store(["rule"], standing=True)
    await mem.store(["fact a", "fact b"])
    ids = {r["memory"]: r["id"] for r in fake.rows.values()}

    hits = await mem.retrieve("anything", exclude_ids={ids["fact a"], "not-a-uuid"})
    assert [h.text for h in hits] == ["fact b"]
    sent = fake.search_bodies[-1]["filters"]["AND"]
    # Ids that are not UUIDs are dropped: Mem0 rejects the whole request.
    assert {"NOT": [{"memory_ids": [ids["fact a"]]}]} in sent

    hits = await mem.retrieve("anything", include_standing=True)
    assert sorted(h.text for h in hits) == ["fact a", "fact b", "rule"]


async def test_ingest_turn_waits_for_the_event_and_reports_failure(fake):
    """Extraction is asynchronous on Mem0's side. Returning at submit time
    would make a failed extraction indistinguishable from "nothing new"."""
    fake.extracted = ["User's staging host is host42."]
    mem = _memory(fake)
    assert await mem.ingest_turn("host42 is staging", prev_assistant_text="Which host?") == [
        "User's staging host is host42."
    ]
    assert fake.event_polls == 2

    fake.event_polls = 0
    fake.event_status_sequence = ["PENDING", "FAILED"]
    with pytest.raises(Exception, match="extraction failed"):
        await mem.ingest_turn("anything")


# ---------------------------------------------------------------------------
# Live Mem0 Platform
# ---------------------------------------------------------------------------

_LIVE = os.environ.get("ONCALL_RUN_MEM0_TESTS") == "1" and bool(os.environ.get("MEM0_API_KEY"))
live = pytest.mark.skipif(not _LIVE, reason="set ONCALL_RUN_MEM0_TESTS=1 and MEM0_API_KEY")


@pytest.fixture
async def live_mem():
    uid = f"oncall-test-{uuid.uuid4().hex[:8]}"
    mem = Mem0Memory(
        api_key=os.environ["MEM0_API_KEY"], user_id=uid, max_inject=10,
        search_threshold=0.0, standing_cap=5,
    )
    try:
        yield mem
    finally:
        await mem._request("DELETE", "/v1/memories/", params={"user_id": uid})
        await mem.aclose()


@live
async def test_live_standing_and_exclusion_filters(live_mem):
    """Pins the filter syntax Mem0 accepts today: metadata `ne` keeps rows
    without metadata, `NOT memory_ids` excludes by id, and the list endpoint's
    stringified booleans still read as standing."""
    await live_mem.store(["Always answer the owner in German."], standing=True)
    await live_mem.store(["The staging host is staging.example.test on port 8443."])
    await live_mem.store(["The build server is ci.example.test."])

    standing = await live_mem.standing_rows(refresh=True)
    assert [m.text for m in standing] == ["Always answer the owner in German."]

    hits = await live_mem.retrieve("which host runs staging?")
    assert hits and hits[0].text.startswith("The staging host")
    assert all(h.id != standing[0].id for h in hits)

    hits = await live_mem.retrieve("which host runs staging?", exclude_ids={hits[0].id})
    assert all(not h.text.startswith("The staging host") for h in hits)
