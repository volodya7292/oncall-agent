"""laptop_worker.execute_job — local execution of dispatched jobs.

Device-independent: all file ops go through tmp_path, no host state read.
"""

from __future__ import annotations

from oncall.laptop_worker import execute_job


async def test_bash_roundtrip():
    r = await execute_job({"id": "1", "kind": "bash", "input": {"command": "echo hello"}})
    assert r["exit_code"] == 0
    assert r["stdout"].strip() == "hello"


async def test_bash_catastrophic_backstop():
    # The server already auto-denies this, but the worker must refuse it too.
    r = await execute_job({"id": "1", "kind": "bash", "input": {"command": "rm -rf /*"}})
    assert r["error"] == "blocked_catastrophic"


async def test_write_then_read_file(tmp_path):
    target = tmp_path / "sub" / "note.txt"  # parent dir created on write
    w = await execute_job({
        "id": "1", "kind": "write_file",
        "input": {"path": str(target), "content": "abc"},
    })
    assert w["ok"] is True
    r = await execute_job({"id": "2", "kind": "read_file", "input": {"path": str(target)}})
    assert r["content"] == "abc"


async def test_read_missing_file():
    r = await execute_job({"id": "1", "kind": "read_file", "input": {"path": "/no/such/file/xyz"}})
    assert r["error"] == "not_found"


async def test_glob(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    r = await execute_job({
        "id": "1", "kind": "glob",
        "input": {"pattern": "*.py", "path": str(tmp_path)},
    })
    assert [p.split("/")[-1] for p in r["paths"]] == ["a.py"]


async def test_unknown_kind():
    r = await execute_job({"id": "1", "kind": "telepathy", "input": {}})
    assert r["error"] == "unknown_kind"
