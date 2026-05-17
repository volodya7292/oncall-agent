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
    oncall_approval_timeout_seconds: int = 86400
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
    # Default embedder: local Ollama with nomic-embed-text. Picked over the
    # Vercel-hosted qwen3-embedding-8b because of ~30× lower latency (12ms
    # vs 1.8s median) and no rate limits. The same dedup threshold (0.88)
    # transfers cleanly per the live integration tests. Override to e.g.
    # "alibaba/qwen3-embedding-8b" to send embeddings via the Vercel gateway
    # instead — useful if you don't want to run Ollama on this host.
    oncall_memory_embed_model: str = "nomic-embed-text:137m-v1.5-fp16"
    # Where to find the Ollama daemon when the embed model is an Ollama tag.
    oncall_ollama_host: str = "http://localhost:11434"
    # Cheap conversational model for extracting facts from the user's turn.
    # Empty string disables auto-extraction (operator memory still works for
    # retrieval, just never grows). Defaults to the operator model.
    oncall_memory_extract_model: str = ""
    oncall_memory_hybrid_alpha: float = 0.7
    oncall_memory_hybrid_beta: float = 0.3
    oncall_memory_relevance_floor: float = 0.30
    oncall_memory_max_inject: int = 10
    # Near-duplicate threshold for the at-store merge. New facts with cosine
    # ≥ this against an existing entry update that entry instead of inserting.
    oncall_memory_dedup_sim: float = 0.88

    # Operator backend: OpenAI-compatible HTTP via Vercel AI Gateway.
    # https://vercel.com/docs/ai-gateway/sdks-and-apis/python
    # Set ONCALL_OPERATOR_MODEL to a gateway model id like "openai/gpt-oss-20b".
    oncall_operator_model: str = "google/gemma-4-31b-it"
    # Reasoning level. "minimal" cuts TTFA dramatically on gemma-4-31b-it
    # (7s → 1s) by suppressing the model's default ~150-300-token internal
    # monologue, which we don't need for routing/dispatch decisions. Set to
    # None to leave the dial unset.
    oncall_operator_reasoning_effort: str | None = "minimal"
    # Which API surface to use for the operator's LLM.
    #   "gemini" → native Google AI Studio API (google-genai SDK). Required
    #              for ack-first behavior (model emits user-facing text in
    #              the same response as tool calls) — the Vercel gateway's
    #              routing strips that on gemma-4-31b-it.
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
        "telegram_bot_session_path",
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

    telegram_api_id: str = ""
    telegram_api_hash: str = ""
    telegram_session_path: Path = Field(
        default_factory=lambda: Path("~/.oncall/telegram.session").expanduser()
    )
    telegram_important_senders: str = ""
    telegram_important_keywords: str = "urgent,down,production,outage,critical"
    # Senders (lowercased @handles, comma-separated) whose messages should
    # never reach the inbox. The own-bot front-end is auto-added at startup
    # via its user_id (see telegram_bot.py + _wire_bot_into_userbot); this
    # is for OTHER bots/people whose DMs you'd rather not see (e.g. service
    # bots like @userinfobot).
    telegram_userbot_ignore_usernames: str = ""

    # Telegram BOT front-end (separate from the userbot above). Talk to the
    # operator over Telegram. This is the primary client.
    # Get the token from @BotFather. OWNER_ID is the Telegram numeric user_id
    # of the only person allowed to talk to this bot — fetch it from
    # @userinfobot, or temporarily start the bot without OWNER_ID set and
    # watch the audit log for the inbound sender_id when you DM yours.
    telegram_bot_token: str = ""
    telegram_bot_owner_id: str = ""   # numeric string; parsed to int at start
    telegram_bot_session_path: Path = Field(
        default_factory=lambda: Path("~/.oncall/telegram_bot.session").expanduser()
    )

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
