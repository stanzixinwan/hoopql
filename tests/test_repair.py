"""Repair loop: retries are capped, every attempt is logged, and feedback reaches the generator."""

from __future__ import annotations

import logging

import psycopg
import pytest

from core.execute import ExecutionError, Executor
from core.generate import ErrorFeedback, SQLGenerator
from core.repair import MAX_RETRIES, generate_with_repair
from core.safety import UnsafeSQLError, validate
from core.schema import introspect
from tests.repair_seeds import SEEDS

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


class ScriptedGenerator(SQLGenerator):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.feedback: list[ErrorFeedback | None] = []

    def generate(self, question, schema_text, examples, grounding_hints, error_feedback=None) -> str:
        self.feedback.append(error_feedback)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


@pytest.fixture(scope="module")
def schema():
    with psycopg.connect(READONLY_URL) as conn:
        return introspect(conn)


@pytest.fixture(scope="module")
def executor():
    with Executor(READONLY_URL) as ex:
        yield ex


def run(generator, schema, executor, max_retries=MAX_RETRIES):
    return generate_with_repair(
        generator,
        lambda sql: validate(sql, schema),
        executor.run,
        "How many teams are there?",
        "schema",
        [],
        "",
        max_retries=max_retries,
    )


def test_first_attempt_success_needs_no_retry(schema, executor) -> None:
    outcome = run(ScriptedGenerator(["SELECT COUNT(*) FROM team"]), schema, executor)
    assert outcome.ok
    assert outcome.retries == 0
    assert outcome.result.rows == [(30,)]


def test_validation_and_execution_errors_are_fed_back(schema, executor) -> None:
    generator = ScriptedGenerator(
        ["SELECT COUNT(*) FROM teams", "SELECT 1 / 0 AS x FROM team", "SELECT COUNT(*) FROM team"]
    )
    outcome = run(generator, schema, executor)
    assert outcome.ok
    assert [a.status for a in outcome.attempts] == ["validation_error", "execution_error", "ok"]
    assert generator.feedback[0] is None
    assert generator.feedback[1] == ErrorFeedback(sql="SELECT COUNT(*) FROM teams", error="Unknown table teams.")
    assert "division by zero" in generator.feedback[2].error


def test_retries_are_capped_at_two(schema, executor) -> None:
    outcome = run(ScriptedGenerator(["SELECT nope FROM team"] * 5), schema, executor)
    assert not outcome.ok
    assert len(outcome.attempts) == 1 + MAX_RETRIES == 3


def test_repair_disabled_makes_one_attempt(schema, executor) -> None:
    outcome = run(ScriptedGenerator(["SELECT nope FROM team", "SELECT 1"]), schema, executor, max_retries=0)
    assert not outcome.ok
    assert len(outcome.attempts) == 1


def test_generator_exception_stops_the_loop(schema, executor) -> None:
    outcome = run(ScriptedGenerator([RuntimeError("rate limited"), "SELECT 1"]), schema, executor)
    assert not outcome.ok
    assert outcome.attempts[0].status == "generate_error"
    assert "rate limited" in outcome.attempts[0].error


def test_every_attempt_is_logged(schema, executor, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="hoopql.repair"):
        run(ScriptedGenerator(["SELECT nope FROM team", "SELECT COUNT(*) FROM team"]), schema, executor)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert messages[0].startswith("attempt 1 validation_error")
    assert messages[1].startswith("attempt 2 ok")


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: s["question"][:40])
def test_repair_seeds_fail_first_and_gold_runs(schema, executor, seed) -> None:
    with pytest.raises((UnsafeSQLError, ExecutionError)):
        executor.run(validate(seed["broken"], schema))
    assert executor.run(validate(seed["gold"], schema)).row_count >= 1
