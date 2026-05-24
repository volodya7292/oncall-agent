# CLAUDE.md

Project-specific notes for Claude Code working in this repo.

## Reinstalling after source changes

The user runs `oncall` from a uv-tool install at `/Users/admin/.local/share/uv/tools/oncall-agent/`, NOT from the editable checkout. That copy is a snapshot from the last `uv build` — source-tree edits do not reach it. After any change that the running daemon needs to see (prompts under `src/oncall/prompts/`, packaged settings under `src/oncall/executor/`, code changes, dependency additions), rebuild and reinstall:

```sh
uv build
uv tool install --force ./dist/oncall_agent-0.1.0-py3-none-any.whl
```

Additionally, the running `oncall api` daemon loads the operator system prompt **once at startup** (see `Operator.__init__` in [src/oncall/operator.py](src/oncall/operator.py)). Prompt edits — even on the editable install path — require a daemon restart to take effect. The daemon is launchd-managed; restart with:

```sh
oncall service start            # restarts the launchd-managed daemon
```

Do NOT `kill` the PID directly — launchd will respawn it from the same wheel anyway. `oncall service start` is the right knob.

## Testing discipline

Write tests for **non-obvious flows only**. A test earns its keep when it locks down behavior that a reader of the code couldn't trivially infer, or that has surprised someone in the past.

Skip tests that:
- Re-state a one-line method's body ("does `delete_x` call `DELETE FROM x`?").
- Assert that a stub's pre-canned return value made it through — that proves the wiring of the stub, not real behavior.
- Cover a `/help` text string or other near-tautological output. One smoke test that the command runs is enough; don't pin every word.
- Pair with the change ("I added method X, so here's `test_X_returns_what_X_returns`").

Write tests when they capture:
- A correctness claim that depends on calibration against an external system (e.g. the live-gateway integration tests pinning the 0.88 dedup threshold).
- A safety invariant (e.g. the broker refusing to allow a mutating tool call without a matching challenge phrase, even with a fully prompt-injected operator).
- A multi-step interaction or race (concurrency, ordering, replay, recovery).
- A boundary or off-by-one case (threshold equality, capacity overflow, empty-input handling, the exact split point of compression).
- Past bug regressions, with a comment naming the bug.

When in doubt, lean toward fewer, denser tests. A 200-line test file that proves 5 hard properties beats a 1000-line file that re-asserts the source code.

## Memory testing

`tests/test_operator_memory.py` includes live integration tests against a local Ollama daemon running `nomic-embed-text:137m-v1.5-fp16`. They skip unless `ONCALL_RUN_EMBEDDING_TESTS=1` is set. To run them locally:

```sh
ollama pull nomic-embed-text:137m-v1.5-fp16   # one-time
ONCALL_RUN_EMBEDDING_TESTS=1 uv run pytest tests/test_operator_memory.py -v
```

The integration tests pin the dedup threshold (0.88) to the live model's behavior — if `ONCALL_MEMORY_DEDUP_SIM` is retuned or the embedding model changes, those tests will fail loudly with a hint to retune.

## Exception handling

Every `except` must log something — at minimum a one-line `log.warning(...)` / `log.info(...)` with context (what failed, which id/chat/key). Full tracebacks (`log.exception(...)`) are encouraged for unexpected paths but not required for benign/expected ones. **Never** write `except Exception: pass` or `except X: return None` with no log line — silent swallows have already cost us one stuck `inbox-drain` loop. If the failure is truly expected and uninteresting, log at debug level and say why in the message.

## Long-lived background loops

Any `asyncio.create_task`-launched loop in [src/oncall/api.py](src/oncall/api.py) (drain, auto-ping, dedup, …) must:

1. **Self-restart on exception.** Wrap the inner body in `try / except asyncio.CancelledError: raise / except Exception: log.exception(...); notify; sleep; continue`. A transient hiccup must not kill the task.
2. **Notify Telegram on every system error**, without a traceback (the err log has the full detail). Use `_notify_system_error(events, notify_session_id, where, exc)`.
3. **Trip a circuit breaker at 3 consecutive crashes.** Reset the counter on any successful iteration. After 3 strikes the loop re-raises so the task dies — that is intentional: 3 in a row means it's a real bug, not a flake, and retrying just hides it. The `_supervise_bg_task` done-callback fires one final Telegram notification on exit. Service stays degraded until a human fixes the bug and restarts the daemon.
4. **Be supervised.** Call `_supervise_bg_task(task, events, notify_sid, "<name>")` at the create_task site so a silently-dying task still leaves a loud trail.

## Inspecting SQLite state

The orchestrator runs SQLite in WAL mode at `~/.oncall/state.db`. Safe inspection while the daemon is live:

```sh
# Ad-hoc query (WAL allows concurrent reads):
sqlite3 ~/.oncall/state.db "SELECT id, text, last_accessed_at FROM operator_memories ORDER BY last_accessed_at DESC;"

# Or a snapshot if you'll be poking around for a while:
sqlite3 ~/.oncall/state.db ".backup /tmp/oncall.db"
sqlite3 /tmp/oncall.db
```
