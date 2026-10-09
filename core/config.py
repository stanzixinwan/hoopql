"""Runtime settings for the query pipeline, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = ("openai", "anthropic")
DEFAULT_DATABASE_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class ConfigError(RuntimeError):
    pass


def load_env_file(path: Path = ENV_FILE) -> None:
    """Copy KEY=value lines from .env into os.environ without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str


def database_url() -> str:
    load_env_file()
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def llm_settings(provider: str | None = None) -> LLMSettings:
    load_env_file()
    provider = (provider or os.environ.get("HOOPQL_LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"HOOPQL_LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, got {provider!r}")
    model_var = f"HOOPQL_{provider.upper()}_MODEL"
    key_var = f"{provider.upper()}_API_KEY"
    model = os.environ.get(model_var, "").strip()
    api_key = os.environ.get(key_var, "").strip()
    if not model:
        raise ConfigError(f"Set {model_var} in .env to the {provider} model to use.")
    if not api_key:
        raise ConfigError(f"Set {key_var} in .env.")
    return LLMSettings(provider=provider, model=model, api_key=api_key)
