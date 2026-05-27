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
# Persisted once on first read; never rotated. The "initialized" marker
# is created after the first successful spawn — subsequent spawns then
# use `--resume` instead of `--session-id`.
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
    # local `claude` CLI and persist the summary. 64K leaves plenty of head
    # room inside Gemma's 128K window while keeping prompts small enough for
    # snappy first-token latency.
    oncall_compression_threshold_tokens: int = 64000
    # Model used for summarization one-shots (both chat compression and task
    # result summary). Passed to `claude --model`. "sonnet" is the alias
    # that resolves to the latest Sonnet on the user's subscription.
    oncall_compression_model: str = "sonnet"
    oncall_db_path: Path = Field(default_factory=lambda: Path("~/.oncall/state.db").expanduser())
    oncall_prod_hosts: str = ""

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
    # Operator backend: OpenAI-compatible HTTP via Vercel AI Gateway.
    # https://vercel.com/docs/ai-gateway/sdks-and-apis/python
    # Set ONCALL_OPERATOR_MODEL to a gateway model id like "openai/gpt-oss-20b".
    # Default is gemini-3.5-flash via AI Studio: ~0.6s TTFA on a bare "Hi"
    # with thinking=minimal (vs ~0.59s on flash-lite — within noise, but
    # 3.5-flash had a tighter distribution). Natively supports ack-first
    # (text + function_call in the same response).
    oncall_operator_model: str = "gemini-3.5-flash"
    # Reasoning level. "low" buys ~100-160 reasoning tokens for noticeably
    # better triage / memory / tool-routing decisions vs "minimal", at a
    # cost of a few hundred ms TTFA. Set to "minimal" to claw back the
    # latency, or None to leave the dial unset (model default — usually
    # "medium" or higher, which is slower than we want).
    oncall_operator_reasoning_effort: str | None = "minimal"
    # Which API surface to use for the operator's LLM.
    #   "gemini" → native Google AI Studio API (google-genai SDK). Required
    #              for ack-first behavior on Google models (the Vercel gateway
    #              strips assistant text when a tool_call rides in the same
    #              response). Also required for flash-lite's thought_signature
    #              round-trip on multi-round tool flows.
    #   "vercel" → OpenAI-compatible via Vercel AI Gateway. Useful for
    #              non-Google models (zai/, minimax/, etc).
    oncall_operator_backend: str = "gemini"
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

    telegram_important_senders: str = ""
    telegram_important_keywords: str = "urgent,down,production,outage,critical"
    # Senders (lowercased @handles, comma-separated) whose messages should
    # never reach the inbox. The agent userbot's own chat with the owner is
    # filtered separately by chat_id (TELEGRAM_AGENT_USER_ID_FILE). This list
    # is for OTHER senders you'd rather not see (e.g. service bots like
    # @userinfobot).
    telegram_userbot_ignore_usernames: str = ""

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
    def prod_hosts(self) -> set[str]:
        return {h.strip() for h in self.oncall_prod_hosts.split(",") if h.strip()}

    @property
    def important_senders(self) -> set[str]:
        return {s.strip().lstrip("@") for s in self.telegram_important_senders.split(",") if s.strip()}

    @property
    def important_keywords(self) -> set[str]:
        return {k.strip().lower() for k in self.telegram_important_keywords.split(",") if k.strip()}

    @property
    def userbot_ignore_usernames(self) -> set[str]:
        return {s.strip().lstrip("@").lower()
                for s in self.telegram_userbot_ignore_usernames.split(",") if s.strip()}


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    return Paths()
