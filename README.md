# oncall-agent

A personal on-call agent. Two tiers:

- **Operator** (Gemma via Vercel AI Gateway) — operator. Handles dialogue, routes work.
- **Executor** (`claude` CLI, per-task subprocess) — does the actual work, gated by a deterministic permission broker.

User talks only to the operator. Operator dispatches tasks to the executor. Executor's tool calls go through a classifier (readonly auto-allows, mutating escalates for voice approval with a challenge phrase). Telegram is the primary inbound messenger (read DMs, propose replies in the user's own voice, send on approval).

See `DESIGN.md` for the full architecture.

## Install (global, via uv tool)

Prerequisites:
- `uv` ≥ 0.4 — https://docs.astral.sh/uv/getting-started/installation/
- Claude Code CLI on PATH — https://docs.claude.com/en/docs/claude-code (the executor subprocess invokes `claude`).
- Python 3.12+.

```sh
uv tool install git+https://github.com/<you>/oncall-agent
oncall init                # writes ~/.oncall/.env with a fresh token
# edit ~/.oncall/.env — at minimum set AI_GATEWAY_API_KEY
oncall api                 # boots the orchestrator on 127.0.0.1:8765
```

Now in another terminal, submit a chat turn:

```sh
TOKEN=$(grep ^ONCALL_TOKEN ~/.oncall/.env | cut -d= -f2)
curl -sS -X POST http://127.0.0.1:8765/chat \
  -H "X-Oncall-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "what tasks are running?"}'
```

Optional Telegram (lets the operator read your DMs and draft replies in your voice):

```sh
# Get api_id + api_hash at https://my.telegram.org/apps, paste into ~/.oncall/.env
oncall telegram-login      # interactive: phone, code, optional 2FA password
oncall api                 # now also starts the Telegram listener
```

## Install (development)

```sh
git clone https://github.com/<you>/oncall-agent
cd oncall-agent
uv sync --extra dev
uv run oncall init         # writes ~/.oncall/.env if absent
# or create a project-local .env that overrides — `cp .env.example .env`
uv run oncall api
uv run pytest
```

## Config

`oncall init` writes `~/.oncall/.env`. Project-local `.env` in cwd overrides it (handy for dev sandboxes). Both files read by pydantic-settings — anything in either is accepted via the `ONCALL_*` / `AI_GATEWAY_*` / `TELEGRAM_*` env vars too.

State lives in `~/.oncall/state.db` (SQLite, WAL).

## Security notes

- Localhost-only listener; shared-secret header auth. Real deployment needs mTLS or a Tailscale-only listener.
- `src/oncall/executor/settings.json` carries a catastrophic-command deny list (first line of defense, evaluated before the classifier). This is the policy applied to the `claude` CLI subprocess we spawn as the executor — distinct from any `.claude/settings.json` a developer working on this repo might have.
- Every tool call writes an `approvals` row, including auto-decided ones — full audit trail in SQLite.
- Real-time `oncall.audit.*` log stream: `oncall api 2>&1 | grep oncall.audit` shows every broker decision, operator tool call, Telegram inbound/send.
- Telegram archived chats are filtered from the inbox automatically.
- `mark_inbox_read` is local-only — does NOT clear Telegram's unread badge and does NOT send a read receipt.

## Status

Milestone 1 (orchestrator + broker), Milestone 2 (operator), and Milestone 3 (Telegram) all complete. 190 tests passing.
