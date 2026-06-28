"""LaptopBridge — the server-side rendezvous for laptop-worker jobs.

These lock down the properties a reader can't trivially infer from the code:
the dispatch↔claim↔result race, the decline-when-offline contract, and the
mid-job timeout that must NOT hang the (serialized) executor.
"""

from __future__ import annotations

import asyncio

from oncall.classifier import classify
from oncall.laptop_bridge import LaptopBridge
from oncall.models import ClassifierVerdict


def _bridge(*, presence=60.0, poll=0.2, job=0.3) -> LaptopBridge:
    return LaptopBridge(presence_window_s=presence, poll_timeout_s=poll, job_timeout_s=job)


async def _mark_online(b: LaptopBridge) -> None:
    """A single poll establishes presence; spawn it so the queue is ready."""
    b.mark_seen()


async def test_round_trip_dispatch_claim_result():
    b = _bridge()
    await _mark_online(b)

    async def worker():
        job = await b.next_job()
        assert job is not None
        assert job["kind"] == "bash"
        assert job["input"] == {"command": "echo hi"}
        b.submit_result(job["id"], {"stdout": "hi\n", "exit_code": 0})

    worker_task = asyncio.create_task(worker())
    result = await b.dispatch("bash", {"command": "echo hi"})
    await worker_task
    assert result == {"stdout": "hi\n", "exit_code": 0}


async def test_offline_declines_without_queueing():
    b = _bridge()
    # Never polled → offline. Dispatch must fail fast, not block.
    result = await b.dispatch("bash", {"command": "echo hi"})
    assert result["error"] == "laptop_offline"


async def test_presence_window_expiry():
    b = _bridge(presence=0.05)
    b.mark_seen()
    assert b.is_online()
    await asyncio.sleep(0.08)
    assert not b.is_online()


async def test_mid_job_timeout_does_not_hang_and_late_result_ignored():
    b = _bridge(job=0.1)
    await _mark_online(b)
    # No worker claims it → dispatch must time out and RETURN (not hang).
    result = await b.dispatch("bash", {"command": "sleep 999"})
    assert result["error"] == "laptop_timeout"
    # The stale job is left in the queue; a later poll must skip it (its
    # future is done) and return None rather than hand a dead job to a worker.
    claimed = await b.next_job()
    assert claimed is None


async def test_late_result_after_timeout_is_rejected():
    b = _bridge(poll=0.5, job=0.1)
    await _mark_online(b)
    # Claim the job, then let dispatch time out before posting the result.
    claimed_holder = {}

    async def slow_worker():
        job = await b.next_job()
        claimed_holder["job"] = job
        await asyncio.sleep(0.2)  # exceed job_timeout
        return b.submit_result(job["id"], {"stdout": "too late"})

    worker_task = asyncio.create_task(slow_worker())
    result = await b.dispatch("bash", {"command": "echo hi"})
    accepted = await worker_task
    assert result["error"] == "laptop_timeout"
    assert accepted is False  # the future was already gone; result dropped


async def test_unknown_op_rejected():
    b = _bridge()
    await _mark_online(b)
    result = await b.dispatch("rm_rf", {})
    assert result["error"] == "unknown_laptop_op"


# ---- classifier gating: the broker gates laptop ops via these verdicts ----

def test_laptop_bash_inherits_full_bash_classification():
    ro = classify("mcp__oncall__laptop", {"op": "bash", "command": "ls -la"})
    assert ro.kind == ClassifierVerdict.READONLY

    mut = classify("mcp__oncall__laptop", {"op": "bash", "command": "rm foo.txt"})
    assert mut.kind == ClassifierVerdict.MUTATING

    cat = classify("mcp__oncall__laptop", {"op": "bash", "command": "rm -rf /*"})
    assert cat.kind == ClassifierVerdict.CATASTROPHIC


def test_laptop_file_ops_classification():
    assert classify("mcp__oncall__laptop", {"op": "read_file", "path": "/x"}).kind == ClassifierVerdict.READONLY
    assert classify("mcp__oncall__laptop", {"op": "glob", "pattern": "*.py"}).kind == ClassifierVerdict.READONLY
    assert classify("mcp__oncall__laptop", {"op": "grep", "pattern": "TODO"}).kind == ClassifierVerdict.READONLY
    assert classify("mcp__oncall__laptop", {"op": "write_file", "path": "/x"}).kind == ClassifierVerdict.MUTATING
