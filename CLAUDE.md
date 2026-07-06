# CLAUDE.md

Project-specific notes for Claude Code working in this repo.

## Deployment: server-primary, laptop as worker

The orchestrator (`oncall api`, `ONCALL_ROLE=server`) runs in Docker on an always-on hosted server, NOT on this laptop. GitHub CI builds the server image on every push to `main` and publishes it to `ghcr.io/<owner>/oncall-agent` (see [.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml)). To deploy a change to the daemon: commit, push to `main`, wait for CI, then pull the new image and restart the container on the server. Local `uv build` does nothing for the server.

The laptop runs only the **capability worker** (`oncall laptop-worker`, launchd label `com.oncall.worker`), which long-polls the server and executes local shell/file jobs. It runs from the uv-tool install at `/Users/admin/.local/share/uv/tools/oncall-agent/` — a snapshot from the last `uv build`, so source-tree edits do not reach it. After a change the worker needs to see:

```sh
uv build
uv tool install --force ./dist/oncall_agent-0.1.0-py3-none-any.whl
oncall service start --worker   # restarts the launchd-managed worker
```

Do NOT `kill` the worker PID directly — launchd respawns it from the same wheel anyway; `oncall service start --worker` is the right knob. Worker logs live at `~/.oncall/logs/worker.{out,err}.log`.

Note the `oncall api` daemon loads the operator system prompt **once at startup** (see `Operator.__init__` in [src/oncall/operator.py](src/oncall/operator.py)), so prompt edits also require a container restart on the server.

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

The orchestrator runs SQLite in WAL mode at `~/.oncall/state.db` — which, in the server-primary deployment, lives **on the hosted server** inside the container's `/root/.oncall` volume (`oncall_state`), not on this laptop. Safe inspection while the daemon is live:

```sh
# On the server, inside the container (WAL allows concurrent reads):
docker exec oncall sqlite3 /root/.oncall/state.db \
  "SELECT id, text, last_accessed_at FROM operator_memories ORDER BY last_accessed_at DESC;"

# Or a snapshot if you'll be poking around for a while:
docker exec oncall sqlite3 /root/.oncall/state.db ".backup /tmp/oncall.db"
```
