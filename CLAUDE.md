# CLAUDE.md

Project-specific notes for Claude Code working in this repo.

## Reinstalling after source changes

The user runs `oncall` from a uv-tool install at `/Users/admin/.local/share/uv/tools/oncall-agent/`, NOT from the editable checkout. That copy is a snapshot from the last `uv build` — source-tree edits do not reach it. After any change that the running daemon needs to see (prompts under `src/oncall/prompts/`, packaged settings under `src/oncall/executor/`, code changes, dependency additions), rebuild and reinstall:

```sh
uv build
uv tool install --force ./dist/oncall_agent-0.1.0-py3-none-any.whl
```

Additionally, the running `oncall api` daemon loads the operator system prompt **once at startup** (see `Operator.__init__` in [src/oncall/operator.py](src/oncall/operator.py)). Prompt edits — even on the editable install path — require a daemon restart to take effect:

```sh
pgrep -af "oncall api"          # find the PID
kill <pid> && oncall api        # restart
```

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

`tests/test_operator_memory.py` includes 3 live-gateway integration tests against `alibaba/qwen3-embedding-8b`. They skip unless `AI_GATEWAY_API_KEY` is set in the environment. To run them locally:

```sh
set -a; source ~/.oncall/.env; set +a
uv run pytest tests/test_operator_memory.py -v
```

The integration tests pin the dedup threshold (0.88) to the live model's behavior — if `ONCALL_MEMORY_DEDUP_SIM` is retuned or the embedding model changes, those tests will fail loudly with a hint to retune.

## Inspecting SQLite state

The orchestrator runs SQLite in WAL mode at `~/.oncall/state.db`. Safe inspection while the daemon is live:

```sh
# Ad-hoc query (WAL allows concurrent reads):
sqlite3 ~/.oncall/state.db "SELECT id, text, last_accessed_at FROM operator_memories ORDER BY last_accessed_at DESC;"

# Or a snapshot if you'll be poking around for a while:
sqlite3 ~/.oncall/state.db ".backup /tmp/oncall.db"
sqlite3 /tmp/oncall.db
```
