"""Execution accuracy: ordering rules, float tolerance, and the Postgres path."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import psycopg
import pytest

from core.eval import execution_accuracy, gold_is_ordered, results_match

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


def test_unordered_gold_ignores_row_order() -> None:
    assert results_match([("BOS", 1), ("LAL", 2)], [("LAL", 2), ("BOS", 1)], ordered=False)


def test_ordered_gold_requires_the_same_order() -> None:
    assert results_match([("BOS", 1), ("LAL", 2)], [("BOS", 1), ("LAL", 2)], ordered=True)
    assert not results_match([("LAL", 2), ("BOS", 1)], [("BOS", 1), ("LAL", 2)], ordered=True)


def test_duplicates_count() -> None:
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)], ordered=False)
    assert not results_match([(1,)], [(1,), (1,)], ordered=False)


def test_float_tolerance() -> None:
    assert results_match([(0.1 + 0.2,)], [(0.3,)], ordered=True)
    assert results_match([(Decimal("33.08333333"),)], [(33.083333333333,)], ordered=True)
    assert not results_match([(33.1,)], [(33.0833,)], ordered=True)


def test_unordered_float_rows_pair_up_after_sorting() -> None:
    pred = [("a", 0.30000000000000004), ("b", 0.1)]
    gold = [("b", 0.1), ("a", 0.3)]
    assert results_match(pred, gold, ordered=False)


def test_types_are_normalized() -> None:
    assert results_match([(30, dt.date(2024, 1, 2))], [(Decimal(30), dt.date(2024, 1, 2))], ordered=True)
    assert not results_match([("30",)], [(30,)], ordered=True)
    assert not results_match([(1, 2)], [(1,)], ordered=True)


@pytest.mark.parametrize(
    ("sql", "ordered"),
    [
        ("SELECT full_name FROM team ORDER BY full_name", True),
        ("SELECT full_name FROM team", False),
        ("SELECT * FROM (SELECT full_name FROM team ORDER BY full_name) t", False),
        ("SELECT 1 UNION SELECT 2 ORDER BY 1", True),
        ("SELECT COUNT(*) OVER (ORDER BY team_id) FROM team", False),
    ],
)
def test_only_top_level_order_by_counts(sql: str, ordered: bool) -> None:
    assert gold_is_ordered(sql) is ordered


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(READONLY_URL, autocommit=True) as connection:
        yield connection


def test_postgres_equivalent_queries_match(conn) -> None:
    gold = "SELECT COUNT(*) FROM game WHERE season = '2022-23' AND season_type = 'Playoffs'"
    pred = "SELECT COUNT(DISTINCT game_id) FROM team_game_log WHERE season = '2022-23' AND season_type = 'Playoffs'"
    assert execution_accuracy(pred, gold, conn)


def test_postgres_order_matters_only_when_gold_orders(conn) -> None:
    gold = "SELECT abbreviation FROM team ORDER BY abbreviation LIMIT 3"
    assert not execution_accuracy("SELECT abbreviation FROM team ORDER BY abbreviation DESC LIMIT 3", gold, conn)
    unordered_gold = "SELECT abbreviation FROM team"
    assert execution_accuracy("SELECT abbreviation FROM team ORDER BY abbreviation DESC", unordered_gold, conn)


def test_failing_prediction_scores_false_and_connection_survives(conn) -> None:
    assert not execution_accuracy("SELECT nope FROM team", "SELECT 1", conn)
    assert execution_accuracy("SELECT 1", "SELECT 1", conn)
