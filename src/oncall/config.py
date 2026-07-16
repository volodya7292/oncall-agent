from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resources (prompts, executor settings) ship inside the package so they
# resolve identically in dev (editable install) and when installed globally
# via `uv tool install`.
_PACKAGE_DIR = Path(__file__).resolve().parent

# Config home for a globally-installed tool. `oncall init` scaffolds an .env
# here. The project-local .env (if present) overrides — handy for dev.
USER_CONFIG_DIR = Path("~/.oncall").expanduser()
USER_ENV_FILE = USER_CONFIG_DIR / ".env"
# Owner display name (set by the agent's /setownername command). Read at
# every operator turn so updates take effect without a daemon restart.
OWNER_NAME_FILE = USER_CONFIG_DIR / "owner_name.txt"
OWNER_NAME_UNSET = "(unknown — ask the user to set their name with /setownername in the Telegram agent.)"

# Auto-written by `oncall telegram-login --agent` on success. Holds the
# numeric user_id of the agent Telegram account so the primary userbot's
# NewMessage handler can filter the user↔agent chat out of the inbox path.
# Missing file = filter disabled (agent hasn't been provisioned yet).
TELEGRAM_AGENT_USER_ID_FILE = USER_CONFIG_DIR / "telegram_agent_user_id"


def read_telegram_agent_user_id() -> int | None:
    try:
        s = TELEGRAM_AGENT_USER_ID_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as e:
        import logging
        logging.getLogger(__name__).warning(
            "read_telegram_agent_user_id failed: %s", e,
        )
        return None
    try:
        return int(s) if s else None
    except ValueError:
        return None


def write_telegram_agent_user_id(user_id: int) -> None:
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TELEGRAM_AGENT_USER_ID_FILE.write_text(str(int(user_id)), encoding="utf-8")

# A single long-lived claude session is reused across every executor
# invocation, so `claude --resume` accumulates context turn-to-turn.
# Persisted once on first read; rotated only by reset_executor_session().
# The "initialized" marker is created after the first successful spawn —
# subsequent spawns then use `--resume` instead of `--session-id`.
EXECUTOR_SESSION_ID_FILE = USER_CONFIG_DIR / "executor_session_id"
EXECUTOR_SESSION_INITIALIZED_FILE = USER_CONFIG_DIR / "executor_session_initialized"


def get_global_executor_session_id() -> str:
    from uuid import uuid4
    try:
        sid = EXECUTOR_SESSION_ID_FILE.read_text(encoding="utf-8").strip()
        if sid:
            return sid
    except FileNotFoundError:
        pass
    sid = str(uuid4())
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTOR_SESSION_ID_FILE.write_text(sid, encoding="utf-8")
    return sid


def is_executor_session_initialized() -> bool:
    return EXECUTOR_SESSION_INITIALIZED_FILE.exists()


def mark_executor_session_initialized() -> None:
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTOR_SESSION_INITIALIZED_FILE.touch()


def _reset_session_initialized_marker() -> None:
    try:
        EXECUTOR_SESSION_INITIALIZED_FILE.unlink()
    except FileNotFoundError:
        pass


def reset_executor_session() -> bool:
    """Forget the global executor session so the next executor spawn starts
    a brand-new `claude` conversation instead of `--resume`-ing the old one.

    Deletes both the session-id file (so a fresh uuid is minted on next read)
    and the initialized marker (so the next spawn uses --session-id, not
    --resume). The id is read fresh at each spawn, so this is safe to call
    while a task is running — the in-flight process keeps its already-captured
    id; only the *next* spawn is affected.

    Returns True if either file existed (i.e. there was a session to forget),
    False if it was already pristine.
    """
    existed = EXECUTOR_SESSION_ID_FILE.exists() or EXECUTOR_SESSION_INITIALIZED_FILE.exists()
    for f in (EXECUTOR_SESSION_ID_FILE, EXECUTOR_SESSION_INITIALIZED_FILE):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    return existed


def read_owner_name() -> str:
    """Return the owner's display name, or OWNER_NAME_UNSET if not set
    or unreadable. Never raises."""
    try:
        name = OWNER_NAME_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return OWNER_NAME_UNSET
    except OSError as e:
        # Disk error, permission issue, etc. Log via print since this
        # module is import-time-clean (no logger configured yet).
        import logging
        logging.getLogger(__name__).warning("read_owner_name failed: %s", e)
        return OWNER_NAME_UNSET
    return name or OWNER_NAME_UNSET


def write_owner_name(name: str) -> None:
    """Persist the owner's display name. Trims and caps at 80 chars."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OWNER_NAME_FILE.write_text(name.strip()[:80], encoding="utf-8")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Tuple order matters: pydantic-settings loads each in turn, later
        # overrides earlier. So project .env wins over user-global, which
        # is what we want for dev (sandbox config without touching user state).
        env_file=(str(USER_ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    oncall_token: str = "dev-token-change-me"
    oncall_port: int = 8765
    # Interface uvicorn binds. Default loopback (safe everywhere). In the
    # server-role container set ONCALL_BIND_HOST=0.0.0.0 so a TLS reverse
    # proxy on the docker network can forward the PUBLIC /laptop/* routes to
    # it. The container port must NOT be published raw to the internet — put
    # the proxy in front and let it expose only /laptop/*.
    oncall_bind_host: str = "127.0.0.1"
    # Max concurrent claude executors. Excess submissions queue in `pending`
    # state until a slot opens. Higher = more throughput but more memory and
    # more pressure on the Anthropic rate limit.
    oncall_max_concurrent_tasks: int = 4
    # How long the broker waits for the user to respond to a mutating-tool
    # approval before giving up. The executor is paused at the broker, so
    # this consumes no compute. Default 24h fits the on-call case (you might
    # be asleep / on a flight); set lower if you'd rather have approvals
    # fail-deny sooner.
    oncall_approval_timeout_seconds: int = 1800
    # Context compression: when the operator's loaded chat history exceeds
    # this many tokens (estimated as chars/4), summarize older turns via the
    # local `claude` CLI and persist the summary. 32K caps the prompt prefill
    # well short of the model's window so first-token latency stays snappy —
    # the smaller the stable prefix, the less there is to prefill each turn.
    oncall_compression_threshold_tokens: int = 32000
    # Model used for summarization one-shots (both chat compression and task
    # result summary). Passed to `claude --model`. "opus" is the alias that
    # resolves to the latest Opus on the user's subscription — chosen over
    # sonnet after Sonnet role-played the conversation (returning a 7-token
    # echo of the assistant's last line) instead of summarizing a long
    # small-talk-heavy history; the stronger model follows the "summarize, do
    # not participate" instruction more reliably. Runs via the local claude
    # subscription and is infrequent, so the extra cost is negligible.
    oncall_compression_model: str = "opus"
    # Effort for the compression one-shot (`claude --effort`). Compression is
    # off the hot path (see Operator._schedule_compression — it runs in the
    # background after the turn has already replied), so it can afford to think:
    # the failure mode this guards against is a lazy summary that drops context
    # or role-plays, not a slow one. Empty string omits the flag.
    oncall_compression_effort: str = "medium"
    # Executor context guard. The single long-lived `claude` session
    # accumulates context across every hand_off (see supervisor). When a
    # task finishes with the live context window at or above this many
    # tokens, the supervisor runs a `/compact` pass on the session before
    # the next task resumes — keeping it well clear of the model's ceiling
    # (e.g. Sonnet-1M) where quality and cost degrade. Set 0 to disable.
    oncall_executor_compact_at_tokens: int = 200000
    oncall_db_path: Path = Field(default_factory=lambda: Path("~/.oncall/state.db").expanduser())
    oncall_prod_hosts: str = ""

    # Deployment role. Empty/"laptop" = legacy all-local mode: the executor
    # uses its own native Bash/Read/Edit/Write on this machine (the original
    # single-box deployment). "server" = cloud-primary mode: this orchestrator
    # runs on an always-on VPS, the executor's native local tools are denied
    # (they'd touch the useless VPS filesystem), and local shell/file work is
    # routed to the user's laptop via the `mcp__oncall__laptop` proxy tool,
    # which only functions while the laptop's `oncall laptop-worker` is polling.
    oncall_role: str = ""
    # Shared secret authenticating the laptop worker's PUBLIC long-poll routes
    # (GET /laptop/jobs, POST /laptop/jobs/{id}/result). Distinct from
    # oncall_token (which guards the loopback/admin surface). Required for
    # server role; the worker sends it as X-Oncall-Laptop-Token.
    oncall_laptop_token: str = ""
    # Worker → server base URL (e.g. https://oncall.example.com). Used by
    # `oncall laptop-worker` to reach the long-poll routes. Server role
    # ignores it.
    oncall_server_url: str = ""
    # Laptop is considered ONLINE if it has long-polled within this many
    # seconds. Tune against the poll timeout below so brief network blips
    # don't read as offline mid-conversation.
    oncall_laptop_presence_window_seconds: int = 60
    # How long GET /laptop/jobs holds open waiting for a job before returning
    # empty (the worker immediately re-polls). Keeps presence fresh and job
    # delivery latency low.
    oncall_laptop_poll_timeout_seconds: int = 25
    # How long the server-side proxy blocks on a dispatched local job before
    # giving up and returning an error to the executor (e.g. laptop slept
    # mid-job). MUST be finite — the executor is serialized, so a hung job
    # would block every later task.
    oncall_laptop_job_timeout_seconds: int = 300

    # ---- Autonomous developer (invoke_developer) ----
    # A separate `claude --permission-mode auto` session spawned ON THE LAPTOP
    # to do file/git work autonomously, isolated from the oncall MCP / broker /
    # Telegram (see developer_runner.py). The executor delegates one coding task
    # (one broker approval) and is notified of the result via a `<developers>`
    # context update. These are read on the LAPTOP (worker) side.
    oncall_developer_model: str = "opus"
    oncall_developer_effort: str = "high"
    # Hard cap: the worker kills the developer's whole process group at this
    # age, regardless of polling. 30 min covers real coding tasks.
    oncall_developer_timeout_seconds: int = 1800
    # How long a `developer_wait` control-plane call blocks before returning the
    # current status (paces the server-side watcher's polling). MUST stay below
    # oncall_laptop_presence_window_seconds (60): while the worker is inside a
    # wait it is not refreshing its GET /laptop/jobs heartbeat, so a full-window
    # wait could flip presence to offline at the boundary.
    oncall_developer_wait_seconds: int = 45

    # Operator memory — semantic, LRU-evicted. Stored in SQLite alongside the
    # rest of state. Capacity caps the number of rows; when extraction would
    # exceed it, the least-recently-retrieved rows are dropped. Hybrid score
    # at retrieval time is `alpha * cosine + beta * token_overlap`, candidates
    # below `relevance_floor` are not injected.
    oncall_memory_capacity: int = 500
    # Local Ollama embedder. Picked over hosted gateways for ~30× lower
    # latency (~12ms median vs ~1.8s) and no rate limits. With
    # OLLAMA_KEEP_ALIVE=4h on the embed calls, the model stays resident in
    # Ollama across our daemon restarts — first user message after
    # `oncall service start` skips the cold load. Pull first:
    #     ollama pull nomic-embed-text:137m-v1.5-fp16
    # The 0.88 dedup threshold in tests is calibrated to this model;
    # swapping it likely requires retuning ONCALL_MEMORY_DEDUP_SIM.
    oncall_memory_embed_model: str = "nomic-embed-text:137m-v1.5-fp16"
    # Where to find the Ollama daemon.
    oncall_ollama_host: str = "http://localhost:11434"
    # Cheap conversational model for extracting facts from the user's turn.
    # Empty string disables auto-extraction (operator memory still works for
    # retrieval, just never grows). Defaults to the operator model.
    oncall_memory_extract_model: str = ""
    oncall_memory_hybrid_alpha: float = 0.7
    oncall_memory_hybrid_beta: float = 0.3
    oncall_memory_relevance_floor: float = 0.30
    oncall_memory_max_inject: int = 10
    # Operator model. Default is glm-5.2 via OpenRouter, pinned to Fireworks.
    # Switch models by pairing ONCALL_OPERATOR_BACKEND with the right id:
    #   openrouter → an OpenRouter slug ("z-ai/glm-5.2", "x-ai/grok-4.20")
    #   gemini     → a bare AI Studio id ("gemini-3.5-flash")
    #   anthropic  → a hyphenated Claude id ("claude-haiku-4-5")
    #
    # Measured at this operator's real prompt shape (full round-trip, the thing
    # `timed("operator")` records), glm-5.2 beats the previous gemini-3.5-flash
    # default on every axis that matters here:
    #
    #                     11k ctx   16k ctx   32k ctx   non-halluc   $/M out
    #   glm-5.2 (no-reas)   1.10s     1.14s     1.81s      0.665       2.89
    #   gemini-3.5-flash    1.52s     1.89s     7.47s      0.266       9.00
    #
    # The 32k column is why: gemini's round-trip explodes with context while
    # glm stays ~flat, and the operator carries a big rolling history. The
    # non-halluc column is Artificial Analysis' omniscience non-hallucination
    # rate — with only 4 operator tools, hallucination hurts more than
    # tool-routing finesse, and gemini-3.5-flash at minimal thinking scored
    # worst of every model surveyed (0.266).
    oncall_operator_model: str = "z-ai/glm-5.2"
    # Reasoning level. "none" asks for NO thinking pass (the OpenRouter client
    # sends reasoning.enabled=false); None just omits the dial and takes the
    # model/provider default. We say "none" explicitly rather than relying on
    # that default: OpenRouter advertises glm-5.2 as default_enabled=true /
    # default_effort=high, and while fireworks/fast in practice returns 0
    # reasoning tokens with the dial unset, the fallback providers below are
    # not guaranteed to agree. Being explicit is what the latency numbers
    # above were measured with.
    #
    # Thinking is expensive here, not free: grok-4.20 with reasoning on costs
    # 11s at 16k and 20s+ at 32k, because thinking scales with context (482
    # thinking tokens at 4k -> 1682 at 16k). glm-5.2 only offers high/xhigh —
    # there is no cheap middle setting to fall back to.
    #
    # Accepted: none/off/disabled, or a level (minimal/low/medium/high) —
    # levels are forwarded as-is, so an unsupported one surfaces as an API
    # error rather than silently downgrading.
    oncall_operator_reasoning_effort: str | None = "none"
    # Which API surface to use for the operator's LLM.
    #   "gemini" → native Google AI Studio API (google-genai SDK). The default.
    #              Required for ack-first behavior on Google models (the
    #              OpenAI-compat gateways strip assistant text when a tool_call
    #              rides in the same response). Also flash-lite's
    #              thought_signature round-trip.
    #   "openrouter" → OpenAI-compatible via OpenRouter. Pins provider routing
    #              (ONCALL_OPERATOR_PROVIDER_ORDER) for lowest TTFT and gets
    #              automatic prompt caching on capable providers.
    #   "anthropic" → native Anthropic Messages API (anthropic SDK). Kept
    #              available; the only surface with explicit cache_control.
    #   "vercel" → OpenAI-compatible via Vercel AI Gateway.
    oncall_operator_backend: str = "openrouter"
    # OpenRouter provider preference for the operator model, highest priority
    # first (comma-separated). Accepts provider names ("Groq") or endpoint tags
    # ("fireworks/fast"); tags pin a specific variant. Fallbacks stay ON, so it
    # drops to the next one if the top provider rate-limits.
    #
    # Retune per model — provider choice dominates model choice here. Across
    # glm-5.2's 28 providers, weekly-median TTFT ranges 608ms..4074ms (Z.AI's
    # own endpoint is the slowest) and tool-call error rate ranges 0.13%..7.25%.
    # This list is ordered by OpenRouter's weekly stats, gated on tool-call
    # error rate and quantization:
    #   fireworks/fast — 0.77% tool err, 81% cache hit, best E2E of the fp8-ish
    #   fireworks      — same provider, unpinned variant
    #   together       — 0.89% tool err, 74% cache hit
    # Deliberately NOT in the list: BaseTen (3.63% tool err), DeepInfra (7.25%,
    # and fp4), and the other fp4 endpoints — glm-5.2's published quality
    # scores were not measured at fp4, so a fallback there would silently
    # degrade the thing we picked this model for.
    oncall_operator_provider_order: str = "fireworks/fast,fireworks,together"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""           # OpenRouter key (oncall_operator_backend=openrouter)
    anthropic_api_key: str = ""            # Claude API key (oncall_operator_backend=anthropic)
    gemini_api_key: str = ""               # AI Studio key (oncall_operator_backend=gemini)
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v1"
    ai_gateway_api_key: str = ""           # local dev
    vercel_oidc_token: str = ""            # Vercel deployments — fallback

    @field_validator(
        "oncall_db_path",
        "telegram_session_path",
        "telegram_agent_session_path",
        mode="before",
    )
    @classmethod
    def _expand_user_path(cls, value):
        """Expand ~ in path-typed env vars. pydantic-settings reads
        TELEGRAM_SESSION_PATH=~/.oncall/telegram.session as the literal
        string '~/.oncall/...'; without this validator that becomes a
        directory named '~' under the daemon's working dir."""
        if value is None or value == "":
            return value
        if isinstance(value, Path):
            return value.expanduser()
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @property
    def gateway_key(self) -> str:
        """Auth value for the OpenAI client. Local API key wins; OIDC fallback."""
        return self.ai_gateway_api_key or self.vercel_oidc_token

    # Telegram application credentials. Both the primary and agent userbot
    # sessions use the SAME api_id/api_hash — the credential identifies the
    # registered application (https://my.telegram.org/apps), not an account.
    # The two sessions are distinguished only by their session file path.
    telegram_api_id: str = ""
    telegram_api_hash: str = ""

    # Primary userbot — runs on the user's own Telegram account. Reads
    # inbound DMs from third parties for triage + reply-by-proposal, sends
    # on the user's behalf via the broker.
    telegram_session_path: Path = Field(
        default_factory=lambda: Path("~/.oncall/telegram.session").expanduser()
    )

    # Agent userbot — runs on a dedicated second Telegram account. This is
    # the user-facing surface: the owner DMs it to talk to the operator,
    # approval prompts arrive here, chat.reply auto-pings land here. Voice
    # calls (future milestone) will also bind here.
    telegram_agent_session_path: Path = Field(
        default_factory=lambda: Path("~/.oncall/telegram_agent.session").expanduser()
    )
    # Numeric Telegram user_id of the OWNER's primary account — the only
    # sender the agent userbot will accept messages from. Get it from
    # @userinfobot.
    telegram_owner_user_id: str = ""

    # 1:1 voice calls between owner and agent. Off by default; turn on by
    # setting VOICE_CALL_ENABLED=1 AND providing both STT and TTS endpoints
    # (OpenAI-compatible, Opus-only audio, bearer-token auth). The CallService
    # binds py-tgcalls to the same telethon session as the agent text chat.
    voice_call_enabled: bool = False
    voice_tts_base_url: str = ""           # e.g. https://example.com/v1
    voice_tts_voice: str = "serhii"
    voice_tts_api_key: str = ""
    voice_stt_base_url: str = ""           # e.g. https://example.com/v1
    voice_stt_api_key: str = ""
    # Continuous low-level "office" ambience mixed under the call (keyboard
    # foley + a brown-noise floor). Doubles as comfort noise: it keeps the
    # WebRTC channel from going fully silent, which otherwise makes Telegram
    # clip the edges of utterances. Kill-switch for a live call going wrong.
    voice_ambient_bed: bool = True

    # Language the operator should respond in, and the STT language hint
    # during voice calls. ISO-639-1 (e.g. "uk", "en", "ru"). Empty = no
    # forcing — operator matches the user's last message language as it
    # always has, and STT auto-detects (which is bad on short utterances).
    # Injected into both operator and executor system prompts at startup,
    # so changes require a daemon restart.
    operator_language: str = ""

    # Display name the operator uses to refer to itself. Substituted into the
    # operator system prompt as {{agent_name}}. Empty = "On-call agent".
    agent_name: str = ""

    @property
    def is_server_role(self) -> bool:
        """True in cloud-primary mode (laptop reached via the proxy). Empty
        or 'laptop' → legacy all-local mode."""
        return self.oncall_role.strip().lower() == "server"

    @property
    def prod_hosts(self) -> set[str]:
        return {h.strip() for h in self.oncall_prod_hosts.split(",") if h.strip()}



class Paths:
    """Resource locations. All packaged inside `oncall/` so they resolve the
    same way whether running from a checkout or a globally-installed wheel."""

    def __init__(self) -> None:
        # Executor-side Claude CLI settings (catastrophic deny list, permission
        # mode). Distinct from a project-level `.claude/settings.json` which
        # belongs to whoever is editing this codebase.
        self.settings_json = _PACKAGE_DIR / "executor" / "settings.json"
        self.executor_prompt = _PACKAGE_DIR / "prompts" / "executor_system.md"
        self.operator_prompt = _PACKAGE_DIR / "prompts" / "operator_system.md"
        # Autonomous-developer Claude CLI settings (catastrophic deny list) and
        # system prompt. Spawned on the laptop by `invoke_developer`.
        self.developer_settings_json = _PACKAGE_DIR / "developer" / "settings.json"
        self.developer_prompt = _PACKAGE_DIR / "prompts" / "developer_system.md"
        # Looped office-ambience bed mixed under voice calls (see Settings).
        self.ambient_bed = _PACKAGE_DIR / "assets" / "office_bed.ogg"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    return Paths()
