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
- A correctness claim that depends on calibration against an external system (e.g. the live-gateway integration tests pinning the 0.88 dedup threshold).
- A safety invariant (e.g. the broker refusing to allow a mutating tool call without a matching challenge phrase, even with a fully prompt-injected operator).
- A multi-step interaction or race (concurrency, ordering, replay, recovery).
- A boundary or off-by-one case (threshold equality, capacity overflow, empty-input handling, the exact split point of compression).
- Past bug regressions, with a comment naming the bug.

When in doubt, lean toward fewer, denser tests. A 200-line test file that proves 5 hard properties beats a 1000-line file that re-asserts the source code.

## Operator memory: cosine proposes, the LLM decides

The pipeline is therefore deliberately shaped:

1. **Write time always INSERTs** (`OperatorMemory.store`). Near-duplicates are
   expected to exist transiently — that is not a bug.
2. **`dedup_pass()` (periodic, `_memory_dedup_loop` in [api.py](src/oncall/api.py))**
   builds clusters from the cosine graph at `cluster_threshold=0.80`. Cosine is
   only a *candidate generator* here; it is never the verdict.
3. **An LLM arbitrates each cluster**, reading the actual texts and returning
   merge groups. Its prompt says entities that differ (person, host, version,
   identifier) MUST NOT merge, and "when in doubt, omit".
4. **Keep-separate verdicts persist to `memory_dedup_skip_pairs`** so the next
   pass doesn't re-litigate the same cluster (and doesn't re-burn LLM calls) —
   which is why a high-cosine pair still sitting in `operator_memories` is
   usually *evidence the system worked*, not evidence it failed.

Corollary for anyone auditing memory health: a pile of cosine-similar rows
proves nothing on its own. Check `memory_dedup_skip_pairs` before concluding
dedup is broken — the pair has probably already been judged.

Retrieval note: memories are injected as `[memory note: ...]` **user-role
messages**, not into the system prompt (see `Operator._build_system_prompt`),
so the system-prompt + history prefix stays byte-stable and the provider's KV
cache keeps hitting across turns.

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

The image ships **no `sqlite3` binary** — reach the DB through the stdlib `sqlite3` module via `python`. The container name is compose-generated (`root-oncall-agent-1` at time of writing); confirm with `docker ps --filter name=oncall`. Opening with `mode=ro` keeps inspection non-mutating; WAL allows these reads while the daemon is live.

```sh
# On the server, inside the container:
docker exec root-oncall-agent-1 python -c "
import sqlite3
c = sqlite3.connect('file:/root/.oncall/state.db?mode=ro', uri=True)
for r in c.execute('SELECT id, text, last_accessed_at FROM operator_memories ORDER BY last_accessed_at DESC'):
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
