# oncall-agent

A personal on-call agent. Two tiers:

- **Operator** (Gemini AI Studio via Vercel AI Gateway) — handles dialogue, routes work.
- **Executor** (`claude` CLI, per-task subprocess) — does the actual work, gated by a deterministic permission broker.

User talks only to the operator. Operator dispatches tasks to the executor. Executor's tool calls go through a classifier (read-only auto-allows, mutating escalates to the user with a challenge phrase — typed as a normal message in the Telegram agent chat, or paraphrased yes/no inside a voice call). Telegram is the primary inbound messenger (read DMs, propose replies in the user's own voice, send on approval).

The operator keeps a **semantic memory** in SQLite: short declarative facts extracted automatically from each user turn (hostnames, conventions, preferences, people the user names), embedded via the same gateway and retrieved by hybrid cosine + token-overlap score. Storage and forgetting are both automatic — extraction happens off the hot path after each reply, LRU evicts at capacity (default 500). The operator never manages memory by hand; it just sees the entries most relevant to the current message.

**Deployment is server-primary**: the full daemon (`oncall api`, `ONCALL_ROLE=server`) runs in Docker on an always-on server; the laptop runs only `oncall laptop-worker`, a capability worker that long-polls the server outbound (no inbound exposure) and executes local shell/file jobs on demand, gated by the same broker/approval path as native tools.

## Deploy (server-primary)

GitHub CI builds the server image on every push to `main` and publishes it to `ghcr.io/<owner>/oncall-agent` (see `.github/workflows/docker-publish.yml`). On the server:

```sh
docker run -d --name oncall --restart unless-stopped \
  -e ONCALL_BIND_HOST=0.0.0.0 \
  -e ONCALL_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  -v oncall_state:/root/.oncall -v oncall_claude:/root/.claude \
  --expose 8765 \
  ghcr.io/<owner>/oncall-agent:latest

# One-time after first deploy — the executor uses subscription OAuth
# (no ANTHROPIC_API_KEY); the credential persists on the /root/.claude volume:
docker exec -it oncall claude login
```

Put a TLS reverse proxy in front that forwards ONLY `/laptop/*` publicly; do NOT publish `:8765` raw to the internet. The `/laptop/*` routes use a dedicated `ONCALL_LAPTOP_TOKEN` and fail closed if it's unset.

On the laptop, install the capability worker (uv tool install as below, then):

```sh
oncall service install --worker   # launchd LaunchAgent com.oncall.worker → `oncall laptop-worker`
```

The worker long-polls the server; if it's offline, the operator sees a per-turn laptop status and declines local-data requests up front.

## Install (global, via uv tool)

Prerequisites:
- `uv` ≥ 0.4 — https://docs.astral.sh/uv/getting-started/installation/
- Claude Code CLI on PATH — https://docs.claude.com/en/docs/claude-code (the executor subprocess invokes `claude`).
- Python 3.12+.

```sh
uv tool install git+https://github.com/<you>/oncall-agent
oncall init                # writes ~/.oncall/.env with a fresh token
# edit ~/.oncall/.env — at minimum set AI_GATEWAY_API_KEY
oncall api                 # boots the orchestrator on 127.0.0.1:8765 (local/dev; production runs this in the server container)
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
- `/clear` — wipe this chat session's rolling history and reset the executor session (operator memory is preserved).
- `/compact` — force-compact older messages into a summary now (don't wait for the auto-threshold).
- `/allowdm <chat_id>` / `/denydm <chat_id>` — add or remove a chat from the DM allowlist (empty by default; non-allowlisted DMs are dropped, never surfaced; allowlisted chats are triaged and eligible for autonomous replies).
- `/dmlist` — show allowlisted chats.
- `/setownername <name>` — set the display name used in the operator's system prompt.
- `/yes <id>` / `/no <id>` — resolve a pending deferred dispatch (operator-initiated `dispatch_task` during an autonomous reply).
- `/restart` / `/stop` — restart or stop the daemon.
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

State lives in `~/.oncall/state.db` (SQLite, WAL) — in the server deployment that's inside the container's `/root/.oncall` volume (`oncall_state`), not on the laptop.

## Security notes

- Shared-secret header auth on the API. Locally the listener is loopback-only; in the server deployment the container binds `0.0.0.0` but only `/laptop/*` should be reverse-proxied publicly (dedicated `ONCALL_LAPTOP_TOKEN`, fail-closed if unset) — never expose `:8765` raw.
- The laptop worker is outbound-only (long-poll) — no inbound surface on the laptop — and keeps its own catastrophic-command deny-list backstop.
- `src/oncall/executor/settings.json` carries a catastrophic-command deny list (first line of defense, evaluated before the classifier). This is the policy applied to the `claude` CLI subprocess we spawn as the executor — distinct from any `.claude/settings.json` a developer working on this repo might have.
- Every tool call writes an `approvals` row, including auto-decided ones — full audit trail in SQLite.
- Real-time `oncall.audit.*` log stream: `oncall api 2>&1 | grep oncall.audit` shows every broker decision, operator tool call, Telegram inbound/send.
- Telegram inbound is surfaced regardless of the chat's archive state — archiving a chat no longer hides its DMs from the inbox.
- `mark_inbox_read` is local-only — does NOT clear Telegram's unread badge and does NOT send a read receipt.

## Status

All core milestones shipped: orchestrator + broker, operator, Telegram (two-userbot topology — primary on the user's account for DM triage, agent on a dedicated second account for the user-facing chat), voice (1:1 calls with multilingual yes/no in-call approvals), and server-primary deployment (daemon in Docker on an always-on server, laptop as an outbound-only capability worker). Live-gateway integration tests skip unless `AI_GATEWAY_API_KEY` is set.
