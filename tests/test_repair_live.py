"""Repair lift on ten seeded first-attempt failures. Run with `uv run pytest -m live -s`."""

from __future__ import annotations

import psycopg
import pytest

from core.config import PROVIDERS, ConfigError, llm_settings
from core.eval import execution_accuracy
from core.execute import Executor
from core.fewshot import FewShotRetriever
from core.generate import SQLGenerator, make_generator
from core.grounding import ground, hints_text
from core.repair import generate_with_repair
from core.safety import validate
from core.schema import introspect, schema_text
from tests.repair_seeds import SEEDS

pytestmark = pytest.mark.live

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


class SeededGenerator(SQLGenerator):
    """Returns the seeded broken SQL first, then defers to the real generator for repairs."""

    def __init__(self, inner: SQLGenerator, broken: str) -> None:
        self.inner = inner
        self.broken = broken

    def generate(self, question, schema_text, examples, grounding_hints, error_feedback=None) -> str:
        if error_feedback is None:
            return self.broken
        return self.inner.generate(question, schema_text, examples, grounding_hints, error_feedback)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_repair_lifts_success_on_seeded_failures(provider: str) -> None:
    try:
        generator = make_generator(llm_settings(provider))
    except ConfigError as exc:
        pytest.skip(str(exc))
    fewshot = FewShotRetriever()
    scores = {0: 0, 2: 0}
    with psycopg.connect(READONLY_URL, autocommit=True) as conn, Executor(READONLY_URL) as executor:
        schema = introspect(conn)
        text = schema_text(schema)
        for seed in SEEDS:
            question = seed["question"]
            hints = hints_text(ground(question, conn))
            examples = fewshot.retrieve(question, k=3)
            for max_retries in scores:
                outcome = generate_with_repair(
                    SeededGenerator(generator, seed["broken"]),
                    lambda sql: validate(sql, schema),
                    executor.run,
                    question,
                    text,
                    examples,
                    hints,
                    max_retries=max_retries,
                )
                if outcome.ok and execution_accuracy(outcome.validated.sql, seed["gold"], conn):
                    scores[max_retries] += 1
    print(f"\n{generator.name}: repair off {scores[0]}/10, repair on {scores[2]}/10")
    assert scores[0] == 0
    assert scores[2] >= 6
