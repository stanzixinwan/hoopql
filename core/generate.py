"""The SQLGenerator interface and its hosted-LLM implementation."""

from __future__ import annotations

import datetime as dt
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from core.config import LLMSettings, llm_settings
from core.fewshot import Example, examples_text

MAX_OUTPUT_TOKENS = 1024

SYSTEM_PROMPT = """You translate questions about NBA statistics into one PostgreSQL query.

Rules:
- Write a single read-only SELECT (CTEs allowed). Never modify data.
- Use only the tables and columns in the schema below.
- Seasons are text like '2024-25'. A bare year such as "the 2025 season" means the season ending that year, '2024-25'.
- season_type is 'Regular Season' or 'Playoffs'. Default to 'Regular Season' unless the question mentions the playoffs or all games.
- When an entity hint gives an id, filter on that id instead of matching names.
- Use player_game_log for per-player stats and team_game_log for per-team stats; join player or team only for names.
- Prefer ROUND(AVG(x), 1) for averages and add ORDER BY plus LIMIT for "most", "top", or "best" questions.
- Reply with the SQL only, inside one ```sql code block."""


@dataclass(frozen=True)
class ErrorFeedback:
    sql: str
    error: str


class SQLGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        schema_text: str,
        examples: list[Example],
        grounding_hints: str,
        error_feedback: ErrorFeedback | None = None,
    ) -> str:
        """Return one SQL string for the question."""


def build_user_prompt(
    question: str,
    schema_text: str,
    examples: list[Example],
    grounding_hints: str,
    error_feedback: ErrorFeedback | None = None,
    today: dt.date | None = None,
) -> str:
    today = today or dt.date.today()
    sections = [f"Today is {today.isoformat()}.", f"Schema:\n{schema_text}"]
    if grounding_hints:
        sections.append(f"Entity hints:\n{grounding_hints}")
    if examples:
        sections.append(f"Examples:\n{examples_text(examples)}")
    sections.append(f"Question: {question}")
    if error_feedback is not None:
        sections.append(
            "Your previous SQL failed.\n"
            f"SQL:\n{error_feedback.sql}\n"
            f"Error:\n{error_feedback.error}\n"
            "Write a corrected query."
        )
    return "\n\n".join(sections)


_FENCE = re.compile(r"```(?:sql|postgresql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    match = _FENCE.search(text)
    sql = match.group(1) if match else text
    return sql.strip()


Completion = Callable[[str, str], str]


class LLMGenerator(SQLGenerator):
    def __init__(self, complete: Completion, name: str = "llm") -> None:
        self.complete = complete
        self.name = name

    def generate(
        self,
        question: str,
        schema_text: str,
        examples: list[Example],
        grounding_hints: str,
        error_feedback: ErrorFeedback | None = None,
    ) -> str:
        user = build_user_prompt(question, schema_text, examples, grounding_hints, error_feedback)
        return extract_sql(self.complete(SYSTEM_PROMPT, user))


def openai_completion(settings: LLMSettings) -> Completion:
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key)

    def complete(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content or ""

    return complete


def anthropic_completion(settings: LLMSettings) -> Completion:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.api_key)

    def complete(system: str, user: str) -> str:
        response = client.messages.create(
            model=settings.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return complete


ADAPTERS: dict[str, Callable[[LLMSettings], Completion]] = {
    "openai": openai_completion,
    "anthropic": anthropic_completion,
}


def make_generator(settings: LLMSettings | None = None) -> LLMGenerator:
    """Build the generator for HOOPQL_LLM_PROVIDER, or for explicit settings."""
    settings = settings or llm_settings()
    return LLMGenerator(ADAPTERS[settings.provider](settings), name=f"{settings.provider}:{settings.model}")
