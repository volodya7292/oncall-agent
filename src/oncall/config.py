from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
    oncall_db_path: Path = Field(default_factory=lambda: Path("~/.oncall/state.db").expanduser())
    oncall_memory_path: Path = Field(
        default_factory=lambda: Path("~/.oncall/memory.md").expanduser()
    )
    oncall_prod_hosts: str = ""

    # Operator backend: OpenAI-compatible HTTP via Vercel AI Gateway.
    # https://vercel.com/docs/ai-gateway/sdks-and-apis/python
    # Set ONCALL_OPERATOR_MODEL to a gateway model id like "openai/gpt-oss-20b".
    oncall_operator_model: str = "google/gemma-4-26b-a4b-it"
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v1"
    ai_gateway_api_key: str = ""           # local dev
    vercel_oidc_token: str = ""            # Vercel deployments — fallback

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

    @property
    def prod_hosts(self) -> set[str]:
        return {h.strip() for h in self.oncall_prod_hosts.split(",") if h.strip()}

    @property
    def important_senders(self) -> set[str]:
        return {s.strip().lstrip("@") for s in self.telegram_important_senders.split(",") if s.strip()}

    @property
    def important_keywords(self) -> set[str]:
        return {k.strip().lower() for k in self.telegram_important_keywords.split(",") if k.strip()}


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
