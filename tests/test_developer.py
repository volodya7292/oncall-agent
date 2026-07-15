"""invoke_developer — the sandboxed auto-mode coding agent.

Covers the non-obvious flows only (per testing discipline):
  * the developer CLI argv (guards the auto-mode / no --bare contract),
  * the worker-side async registry lifecycle (start/wait/timeout/cancel/dedup),
  * the classifier verdicts that gate the tool,
  * the server-side watcher's wake-and-route (a multi-component flow).

Device-independent: all subprocess work uses a patched argv or tmp_path.
"""

from __future__ import annotations

import asyncio

import pytest

from oncall.classifier import classify
from oncall.config import Settings
from oncall.developer_manager import DeveloperJob, DeveloperManager
from oncall.developer_runner import _MAX_REGISTRY, DeveloperRunner, _DevJob
from oncall.models import ClassifierVerdict


def _settings(**over) -> Settings:
    base = dict(_env_file=None, oncall_developer_wait_seconds=3)
    base.update(over)
    return Settings(**base)


# --------------------------------------------------------------------------
# argv contract
# --------------------------------------------------------------------------

def test_build_argv_auto_mode_no_bare():
    argv = DeveloperRunner(_settings())._build_argv()
    assert argv[0] == "claude"
    assert "--permission-mode" in argv and argv[argv.index("--permission-mode") + 1] == "auto"
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "high"
    assert "--settings" in argv  # developer deny list
    # OAuth keychain regression + "laptop's normal env" contract:
    assert "--bare" not in argv
    assert "--strict-mcp-config" not in argv
    assert "--mcp-config" not in argv


# --------------------------------------------------------------------------
# worker-side registry lifecycle
# --------------------------------------------------------------------------

def _runner_with_argv(settings, script: str) -> DeveloperRunner:
    r = DeveloperRunner(settings)
    r._build_argv = lambda: ["/bin/sh", "-c", script]  # type: ignore[method-assign]
    return r


async def test_start_wait_completed(tmp_path):
    r = _runner_with_argv(_settings(), "cat >/dev/null; printf done")
    started = r.start("noop task", str(tmp_path))
    assert started["status"] == "running"
    snap = await r.wait(started["developer_id"])
    assert snap["status"] == "completed"
    assert snap["exit_code"] == 0
    assert snap["output"] == "done"


async def test_folder_not_found_no_job():
    r = _runner_with_argv(_settings(), "true")
    out = r.start("x", "/no/such/dir/xyz")
    assert out["error"] == "folder_not_found"
    assert r._jobs == {}  # nothing spawned


async def test_timeout_kills(tmp_path):
    r = _runner_with_argv(
        _settings(oncall_developer_timeout_seconds=1, oncall_developer_wait_seconds=5),
        "sleep 30",
    )
    started = r.start("slow", str(tmp_path))
    snap = await r.wait(started["developer_id"])
    assert snap["status"] == "timeout"
    assert "exceeded" in snap["output"]


async def test_cancel(tmp_path):
    r = _runner_with_argv(_settings(), "sleep 30")
    started = r.start("slow", str(tmp_path))
    dev_id = started["developer_id"]
    await asyncio.sleep(0.2)  # let the subprocess actually spawn
    assert r.cancel(dev_id)["status"] == "cancelled"
    await asyncio.gather(r._jobs[dev_id].task, return_exceptions=True)
    assert r._jobs[dev_id].status == "cancelled"


async def test_dedup_same_folder_task(tmp_path):
    r = _runner_with_argv(_settings(), "sleep 30")
    a = r.start("same", str(tmp_path))
    b = r.start("same", str(tmp_path))
    assert a["status"] == "running"
    assert b["status"] == "already_running"
    assert b["developer_id"] == a["developer_id"]
    r.cancel(a["developer_id"])
    await asyncio.gather(r._jobs[a["developer_id"]].task, return_exceptions=True)


async def test_wait_unknown_id():
    r = _runner_with_argv(_settings(), "true")
    assert (await r.wait("nope"))["status"] == "unknown"
    assert r.cancel("nope")["status"] == "unknown"


def test_registry_eviction_keeps_running():
    r = DeveloperRunner(_settings())
    for i in range(_MAX_REGISTRY + 8):  # all terminal
        j = _DevJob(id=f"old-{i}", folder="/f", task_text="t",
                    status="completed", finished_at=float(i))
        r._jobs[j.id] = j
    running = _DevJob(id="live", folder="/f", task_text="t", status="running")
    r._register(running)
    assert len(r._jobs) == _MAX_REGISTRY
    assert "live" in r._jobs  # running is never evicted


# --------------------------------------------------------------------------
# classifier gating
# --------------------------------------------------------------------------

def test_classify_invoke_developer_mutating():
    v = classify("mcp__oncall__invoke_developer", {"folder": "/repo", "task": "fix bug"})
    assert v.kind == ClassifierVerdict.MUTATING


def test_classify_cancel_developer_readonly():
    v = classify("mcp__oncall__cancel_developer", {"developer_id": "d1"})
    assert v.kind == ClassifierVerdict.READONLY


# --------------------------------------------------------------------------
# server-side watcher: wake-and-route
# --------------------------------------------------------------------------

class _FakeBridge:
    def __init__(self, wait_result):
        self.calls: list[tuple] = []
        self._wait = wait_result

    async def dispatch(self, op, inp):
        self.calls.append((op, inp))
        if op == "developer_start":
            return {"developer_id": "dev-1", "status": "running"}
        if op == "developer_wait":
            return self._wait
        return {}


class _FakeLifecycle:
    def __init__(self):
        self.enqueued: list[dict] = []

    async def enqueue_executor(self, *, prompt, chat_session_id=None, restricted_to_chat=None):
        self.enqueued.append({
            "prompt": prompt, "chat_session_id": chat_session_id,
            "restricted_to_chat": restricted_to_chat,
        })
        return {"task_id": "t"}


class _FakeTask:
    dispatched_by_chat_session = "chat-42"
    restricted_to_chat = None


class _FakeDB:
    async def get_task_by_session(self, sid):  # noqa: ARG002
        return _FakeTask()


async def test_watcher_wakes_executor_and_routes_to_origin_chat():
    bridge = _FakeBridge({"status": "completed", "output": "edited foo.py",
                          "exit_code": 0, "elapsed_s": 2.0})
    lc = _FakeLifecycle()
    mgr = DeveloperManager(
        bridge=bridge, lifecycle=lc, db=_FakeDB(), events=None,
        settings=_settings(), notify_session_id=None,
    )
    res = await mgr.start("sess-x", "edit foo", "/repo")
    assert res["developer_id"] == "dev-1"

    for _ in range(100):  # let the watcher observe terminal + wake
        if lc.enqueued:
            break
        await asyncio.sleep(0.02)

    assert len(lc.enqueued) == 1, "developer completion must wake the executor exactly once"
    woken = lc.enqueued[0]
    # Routed back to the chat that delegated it (captured from the invoking task).
    assert woken["chat_session_id"] == "chat-42"
    assert "dev-1" in woken["prompt"] and "edited foo.py" in woken["prompt"]
    # Job pruned from the snapshot once reported.
    assert mgr.snapshot_block() == ""


def test_snapshot_block_lists_running_jobs():
    mgr = DeveloperManager(
        bridge=_FakeBridge({}), lifecycle=_FakeLifecycle(), db=_FakeDB(),
        events=None, settings=_settings(), notify_session_id=None,
    )
    mgr._jobs["dev-9"] = DeveloperJob(
        developer_id="dev-9", folder="/srv/app", task="add a route",
        chat_session="c1", restricted_to_chat=None,
    )
    block = mgr.snapshot_block()
    assert "<developers>" in block and "dev-9" in block and "/srv/app" in block


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
