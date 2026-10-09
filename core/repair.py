"""Execution-feedback repair: retry generation with the failed SQL and its error."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from core.execute import ExecutionError, QueryResult
from core.fewshot import Example
from core.generate import ErrorFeedback, SQLGenerator
from core.safety import UnsafeSQLError, ValidatedSQL

MAX_RETRIES = 2

log = logging.getLogger("hoopql.repair")

Validate = Callable[[str], ValidatedSQL]
Execute = Callable[[ValidatedSQL], QueryResult]


@dataclass(frozen=True)
class Attempt:
    number: int
    sql: str
    status: str  # "ok", "generate_error", "validation_error", or "execution_error"
    error: str | None
    elapsed_ms: float


@dataclass
class RepairOutcome:
    attempts: list[Attempt] = field(default_factory=list)
    validated: ValidatedSQL | None = None
    result: QueryResult | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def generate_with_repair(
    generator: SQLGenerator,
    validate: Validate,
    execute: Execute,
    question: str,
    schema_text: str,
    examples: list[Example],
    grounding_hints: str,
    max_retries: int = MAX_RETRIES,
) -> RepairOutcome:
    """Generate, validate, and execute; on failure feed the error back up to max_retries times."""
    outcome = RepairOutcome()
    feedback: ErrorFeedback | None = None
    for number in range(1, max_retries + 2):
        started = time.perf_counter()
        sql = ""
        try:
            sql = generator.generate(question, schema_text, examples, grounding_hints, feedback)
            validated = validate(sql)
            result = execute(validated)
        except UnsafeSQLError as exc:
            status, error = "validation_error", str(exc)
        except ExecutionError as exc:
            status, error = "execution_error", str(exc)
        except Exception as exc:
            _record(outcome, number, sql, "generate_error", f"{type(exc).__name__}: {exc}", started)
            return outcome
        else:
            _record(outcome, number, sql, "ok", None, started)
            outcome.validated = validated
            outcome.result = result
            return outcome
        _record(outcome, number, sql, status, error, started)
        feedback = ErrorFeedback(sql=sql, error=error)
    return outcome


def _record(outcome: RepairOutcome, number: int, sql: str, status: str, error: str | None, started: float) -> None:
    attempt = Attempt(
        number=number,
        sql=sql,
        status=status,
        error=error,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    outcome.attempts.append(attempt)
    log.info("attempt %d %s%s | %s", number, status, f": {error}" if error else "", sql)
