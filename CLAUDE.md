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

## Editing prompts

Everything in [src/oncall/prompts/](src/oncall/prompts/) is read by a capable model. State the constraint and stop — the model works out how to satisfy it.

Do NOT add examples, few-shot demonstrations, do/don't pairs, or a fresh rule for each observed failure. When a prompt-caused bug appears, the reflex to append "and don't do <the thing it just did>" is almost always wrong: it fixes one instance, bloats a prompt that's re-read on every spawn, and buries the constraints that carry real weight. Tighten or generalize the sentence that's already there instead.

Minimal and general beats explicit and exhaustive here. If a rule only makes sense alongside an example, the rule isn't yet stated clearly enough.

## Testing discipline

Write tests for **non-obvious flows only**. A test earns its keep when it locks down behavior that a reader of the code couldn't trivially infer, or that has surprised someone in the past.

Skip tests that:
- Re-state a one-line method's body ("does `delete_x` call `DELETE FROM x`?").
- Assert that a stub's pre-canned return value made it through — that proves the wiring of the stub, not real behavior.
- Cover a `/help` text string or other near-tautological output. One smoke test that the command runs is enough; don't pin every word.
- Pair with the change ("I added method X, so here's `test_X_returns_what_X_returns`").

Write tests when they capture:
- A correctness claim that depends on calibration against an external system (e.g. the live-embedding integration tests pinning cross-language retrieval to the model's real behavior).
- A safety invariant (e.g. the broker refusing to allow a mutating tool call without a matching challenge phrase, even with a fully prompt-injected operator).
- A multi-step interaction or race (concurrency, ordering, replay, recovery).
- A boundary or off-by-one case (threshold equality, capacity overflow, empty-input handling, the exact split point of compression).
- Past bug regressions, with a comment naming the bug.

When in doubt, lean toward fewer, denser tests. A 200-line test file that proves 5 hard properties beats a 1000-line file that re-asserts the source code.

## Operator memory: hosted on Mem0

Memory lives on the Mem0 Platform ([operator_memory.py](src/oncall/operator_memory.py), a thin REST client on `httpx`). Mem0 owns extraction, deduplication and consolidation (its "Dream" merge/supersede pass), so there is no local dedup, eviction or embedder to tune. We deliberately do not use the `mem0ai` SDK: it sends PostHog telemetry on every call (aliased to the account e-mail) unless `MEM0_TELEMETRY=False` is set before import, blocks the event loop with a `requests` ping in its constructor, and pulls qdrant-client/sqlalchemy into the image.

What stays ours, because Mem0 has no notion of it:

1. **Standing instructions.** Flagged `standing` in metadata, injected into every session and every executor task, and hard-capped (`oncall_memory_standing_cap`). Only an explicit `save_memory(standing=true)` sets the flag; automatic extraction never does, so nothing the owner merely mentioned becomes a rule the agent obeys unasked.
2. **Per-session exclusion.** Rows already shown in a session are excluded *inside* the search request (`NOT memory_ids`). Filtering the top-k afterwards is how the old store went memory-blind.
3. **Who gets ingested.** Only owner turns through `Operator.chat_turn`. Auto-pings (inbound DMs, task results) never reach it, and voice calls to third parties pass `ingest_memory=False`. Mem0 stores what it extracts as the owner's facts with no review, so this gate is the safety boundary.

Extraction rules sent with every ingested turn live in [prompts/memory_extraction.md](src/oncall/prompts/memory_extraction.md) — the same editing rules as the other prompts apply.

Observed API quirks the code relies on (re-verify with the live test after any Mem0 change): `infer=True` adds are asynchronous (poll `/v1/event/{id}/`), `infer=False` adds are synchronous; the list endpoint returns metadata booleans as the string `"True"` while search returns real bools; filters reject an `id` key, exclusion goes through `memory_ids`.

`oncall_memory_search_threshold` is on Mem0's combined score scale (semantic + BM25 + entity), not the old embedder's. Re-measure it on real rows before moving it.

## Memory testing

`tests/test_operator_memory.py` drives `Mem0Memory` against an in-process fake of Mem0's endpoints, plus a live test against the real platform under a throwaway `user_id` that is deleted afterwards. The live test skips unless enabled:

```sh
ONCALL_RUN_MEM0_TESTS=1 MEM0_API_KEY=... uv run --extra dev pytest tests/test_operator_memory.py -v
```

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

The image ships **no `sqlite3` binary** — reach the DB through the stdlib `sqlite3` module via `python`. The container name is compose-generated (`root-oncall-agent-1` at time of writing); confirm with `docker ps --filter name=oncall`. Opening with `mode=ro` keeps inspection non-mutating; WAL allows these reads while the daemon is live.

```sh
# On the server, inside the container:
docker exec root-oncall-agent-1 python -c "
import sqlite3
c = sqlite3.connect('file:/root/.oncall/state.db?mode=ro', uri=True)
for r in c.execute('SELECT id, state, created_at FROM tasks ORDER BY created_at DESC LIMIT 20'):
    print(r)
"

# Or a snapshot if you'll be poking around for a while:
docker exec root-oncall-agent-1 python -c "
import sqlite3
src = sqlite3.connect('file:/root/.oncall/state.db?mode=ro', uri=True)
dst = sqlite3.connect('/tmp/oncall.db')
src.backup(dst)
"
```
