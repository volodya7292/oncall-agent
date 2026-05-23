# oncall-agent

A personal on-call agent. Two tiers:

- **Operator** (Gemini AI Studio via Vercel AI Gateway) — handles dialogue, routes work.
- **Executor** (`claude` CLI, per-task subprocess) — does the actual work, gated by a deterministic permission broker.

User talks only to the operator. Operator dispatches tasks to the executor. Executor's tool calls go through a classifier (read-only auto-allows, mutating escalates to the user with a challenge phrase — typed as a normal message in the Telegram agent chat, or paraphrased yes/no inside a voice call). Telegram is the primary inbound messenger (read DMs, propose replies in the user's own voice, send on approval).

The operator keeps a **semantic memory** in SQLite: short declarative facts extracted automatically from each user turn (hostnames, conventions, preferences, people the user names), embedded via the same gateway and retrieved by hybrid cosine + token-overlap score. Storage and forgetting are both automatic — extraction happens off the hot path after each reply, LRU evicts at capacity (default 500). The operator never manages memory by hand; it just sees the entries most relevant to the current message.

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

The Telegram agent (a telethon userbot on a dedicated second account) is the primary client — see "Optional Telegram" below. For ad-hoc testing without it, hit the HTTP API directly:

```sh
TOKEN=$(grep ^ONCALL_TOKEN ~/.oncall/.env | cut -d= -f2)
curl -sS -X POST http://127.0.0.1:8765/chat \
  -H "X-Oncall-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "what tasks are running?"}'
```

Telegram setup (two userbot sessions on one API credential, distinguished by session file):

```sh
# Get api_id + api_hash at https://my.telegram.org/apps, paste into ~/.oncall/.env.
# Also set TELEGRAM_OWNER_USER_ID to your own numeric user id (find via @userinfobot)
# — the agent userbot only accepts messages from that id.

oncall telegram-login           # primary userbot on YOUR account — reads inbound DMs for triage
oncall telegram-login --agent   # agent userbot on a SECOND dedicated account — the user-facing chat surface
oncall api                      # boots both userbots
```

The agent userbot is what you DM with. Approvals, voice calls, and slash commands all land there.

### Telegram commands

The agent userbot is the primary interface. Slash commands you can send to it:

- `/start` — greeting.
- `/status` — running tasks, queue, pending approvals, unread DMs, operator-side memory + context stats.
- `/context` — export this session's chat history + latest compression summary as a markdown file.
- `/clear` — wipe this chat session's rolling history (operator memory is preserved).
- `/compress` — force-compress older messages into a summary now (don't wait for the auto-threshold).
- `/allowdm <chat_id>` / `/denydm <chat_id>` — add or remove a chat from the per-chat send allowlist (the broker auto-approves sends to allowlisted chats during autonomous-reply tasks).
- `/yes <id>` / `/no <id>` — resolve a pending deferred dispatch (operator-initiated `dispatch_task` during an autonomous reply).
- `/help` — list commands.

Userbots can't register a slash-command menu (no `setMyCommands`), so there's no autocomplete in the Telegram UI — type commands as plain text. Anything else is a chat turn. **Approvals arrive as a single text message** with the canonical command, blast radius, and a challenge phrase. Type the phrase back as a normal message to allow, or `/no <approval_id>` to deny explicitly; a wrong phrase routes back to the operator as a normal chat turn and the approval keeps waiting until timeout.

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

Milestone 1 (orchestrator + broker), Milestone 2 (operator), Milestone 3 (Telegram), and the operator-memory rework (auto-extracted, LRU-evicted, semantic retrieval via `alibaba/qwen3-embedding-8b`) are all complete. The Telegram agent userbot is the primary client; the older Bot API front-end and the standalone `oncall chat` REPL were both retired. Live-gateway integration tests (3 of them) skip unless `AI_GATEWAY_API_KEY` is set.
