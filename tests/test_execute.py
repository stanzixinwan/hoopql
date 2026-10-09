"""Sandboxed execution: timeout, row cap, and clean errors."""

from __future__ import annotations

import time

import psycopg
import pytest

from core.execute import ExecutionError, Executor
from core.safety import validate
from core.schema import introspect

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


@pytest.fixture(scope="module")
def schema():
    with psycopg.connect(READONLY_URL) as conn:
        return introspect(conn)


@pytest.fixture(scope="module")
def executor():
    with Executor(READONLY_URL, timeout_ms=500, row_cap=10) as ex:
        yield ex


def test_expensive_cross_join_is_cancelled_cleanly(schema, executor) -> None:
    query = validate("SELECT COUNT(*) FROM player_game_log a CROSS JOIN player_game_log b", schema)
    started = time.perf_counter()
    with pytest.raises(ExecutionError, match="timeout") as info:
        executor.run(query)
    assert info.value.kind == "timeout"
    assert time.perf_counter() - started < 3.0
    assert executor.run(validate("SELECT COUNT(*) FROM team", schema)).rows == [(30,)]


def test_result_shape(schema, executor) -> None:
    result = executor.run(validate("SELECT abbreviation, full_name FROM team ORDER BY abbreviation LIMIT 2", schema))
    assert result.columns == ["abbreviation", "full_name"]
    assert result.rows == [("ATL", "Atlanta Hawks"), ("BKN", "Brooklyn Nets")]
    assert result.row_count == 2
    assert not result.truncated
    assert result.elapsed_ms >= 0


def test_row_cap_truncates(schema, executor) -> None:
    result = executor.run(validate("SELECT team_id FROM team", schema))
    assert result.row_count == 10
    assert result.truncated


def test_database_error_is_reported(schema, executor) -> None:
    with pytest.raises(ExecutionError, match="division by zero") as info:
        executor.run(validate("SELECT 1 / 0 AS x FROM team", schema))
    assert info.value.kind == "database"


def test_query_timeout_does_not_leak_into_the_pooled_connection(schema, executor) -> None:
    executor.run(validate("SELECT COUNT(*) FROM team", schema))
    with executor.pool.connection() as conn:
        assert conn.execute("SHOW statement_timeout").fetchone()[0] == "5s"
