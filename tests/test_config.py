"""Provider selection is one environment value."""

from __future__ import annotations

import os

import pytest

from core import config

VARS = (
    "HOOPQL_LLM_PROVIDER",
    "HOOPQL_OPENAI_MODEL",
    "HOOPQL_ANTHROPIC_MODEL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    for var in VARS:
        # setenv first so monkeypatch restores the original state, even after load_env_file writes it.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


def test_provider_switch_reads_matching_model_and_key(monkeypatch) -> None:
    monkeypatch.setenv("HOOPQL_OPENAI_MODEL", "openai-model")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("HOOPQL_ANTHROPIC_MODEL", "anthropic-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    monkeypatch.setenv("HOOPQL_LLM_PROVIDER", "openai")
    assert config.llm_settings().model == "openai-model"

    monkeypatch.setenv("HOOPQL_LLM_PROVIDER", "anthropic")
    settings = config.llm_settings()
    assert settings.provider == "anthropic"
    assert settings.model == "anthropic-model"
    assert settings.api_key == "sk-anthropic"


def test_unknown_provider_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("HOOPQL_LLM_PROVIDER", "gemini")
    with pytest.raises(config.ConfigError):
        config.llm_settings()


def test_missing_key_names_the_variable(monkeypatch) -> None:
    monkeypatch.setenv("HOOPQL_OPENAI_MODEL", "openai-model")
    with pytest.raises(config.ConfigError, match="OPENAI_API_KEY"):
        config.llm_settings("openai")


def test_env_file_does_not_override_real_environment(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HOOPQL_LLM_PROVIDER=anthropic\nOPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("HOOPQL_LLM_PROVIDER", "openai")
    config.load_env_file(env_file)
    assert os.environ["HOOPQL_LLM_PROVIDER"] == "openai"
    assert os.environ["OPENAI_API_KEY"] == "from-file"
