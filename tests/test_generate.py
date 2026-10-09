"""SQLGenerator prompt assembly, SQL extraction, and provider selection."""

from __future__ import annotations

import datetime as dt

import pytest

from core import config, generate
from core.config import LLMSettings
from core.fewshot import Example
from core.generate import (
    SYSTEM_PROMPT,
    ErrorFeedback,
    LLMGenerator,
    SQLGenerator,
    build_user_prompt,
    extract_sql,
    make_generator,
)

EXAMPLE = Example(1, "How many teams are in the NBA?", "SELECT COUNT(*) FROM team;", ("team",), "easy")


def test_prompt_contains_every_section_in_order() -> None:
    prompt = build_user_prompt(
        "How many points did Jokic score?",
        "TABLE player_game_log\n  pts integer",
        [EXAMPLE],
        '"Jokic" -> player.player_id = 203999',
        today=dt.date(2026, 10, 9),
    )
    order = ["Today is 2026-10-09.", "Schema:", "Entity hints:", "Examples:", "Question: How many points"]
    positions = [prompt.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "SELECT COUNT(*) FROM team;" in prompt
    assert "previous SQL failed" not in prompt


def test_prompt_omits_empty_sections_and_includes_feedback() -> None:
    prompt = build_user_prompt(
        "q", "schema", [], "", ErrorFeedback(sql="SELECT nope FROM team", error='column "nope" does not exist')
    )
    assert "Entity hints:" not in prompt
    assert "Examples:" not in prompt
    assert prompt.endswith('Error:\ncolumn "nope" does not exist\nWrite a corrected query.')
    assert "SELECT nope FROM team" in prompt


def test_system_prompt_states_dialect_and_season_conventions() -> None:
    assert "PostgreSQL" in SYSTEM_PROMPT
    assert "'2024-25'" in SYSTEM_PROMPT
    assert "'Regular Season' or 'Playoffs'" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "text",
    [
        "```sql\nSELECT 1;\n```",
        "Here you go:\n```SQL\nSELECT 1;\n```\nThis counts rows.",
        "```\nSELECT 1;\n```",
        "  SELECT 1;  ",
    ],
)
def test_extract_sql(text: str) -> None:
    assert extract_sql(text) == "SELECT 1;"


def test_llm_generator_passes_prompts_and_extracts_sql() -> None:
    calls = []

    def complete(system: str, user: str) -> str:
        calls.append((system, user))
        return "```sql\nSELECT COUNT(*) FROM team;\n```"

    generator = LLMGenerator(complete)
    assert isinstance(generator, SQLGenerator)
    assert generator.generate("How many teams?", "schema", [EXAMPLE], "") == "SELECT COUNT(*) FROM team;"
    assert calls[0][0] == SYSTEM_PROMPT
    assert "Question: How many teams?" in calls[0][1]


class _Namespace:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def test_openai_adapter_request_and_response(monkeypatch) -> None:
    import openai

    sent = {}

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            sent["api_key"] = api_key

            def create(**kwargs):
                sent.update(kwargs)
                message = _Namespace(content="```sql\nSELECT 1;\n```")
                return _Namespace(choices=[_Namespace(message=message)])

            self.chat = _Namespace(completions=_Namespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    generator = make_generator(LLMSettings(provider="openai", model="gpt-x", api_key="sk-o"))
    assert generator.generate("q", "schema", [], "") == "SELECT 1;"
    assert sent["api_key"] == "sk-o"
    assert sent["model"] == "gpt-x"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_anthropic_adapter_request_and_response(monkeypatch) -> None:
    import anthropic

    sent = {}

    class FakeAnthropic:
        def __init__(self, api_key: str) -> None:
            sent["api_key"] = api_key

            def create(**kwargs):
                sent.update(kwargs)
                return _Namespace(
                    content=[
                        _Namespace(type="thinking", text="ignored"),
                        _Namespace(type="text", text="```sql\nSELECT 2;\n```"),
                    ]
                )

            self.messages = _Namespace(create=create)

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    generator = make_generator(LLMSettings(provider="anthropic", model="claude-x", api_key="sk-a"))
    assert generator.generate("q", "schema", [], "") == "SELECT 2;"
    assert sent["api_key"] == "sk-a"
    assert sent["system"] == SYSTEM_PROMPT
    assert [m["role"] for m in sent["messages"]] == ["user"]


@pytest.mark.parametrize("provider", config.PROVIDERS)
def test_make_generator_dispatches_on_provider(monkeypatch, provider: str) -> None:
    chosen = []
    for name in config.PROVIDERS:
        monkeypatch.setitem(generate.ADAPTERS, name, lambda settings, name=name: chosen.append(name) or (lambda s, u: ""))
    generator = make_generator(LLMSettings(provider=provider, model="m", api_key="k"))
    assert chosen == [provider]
    assert generator.name == f"{provider}:m"
