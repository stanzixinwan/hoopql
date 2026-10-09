"""Execution accuracy: does the predicted SQL return the same result as the gold SQL?

Ported from research/prompt_baseline.py. Differences: runs on Postgres, compares
numbers with a tolerance instead of as strings, and respects row order only when
the gold query has a top-level ORDER BY.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import psycopg
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

REL_TOL = 1e-4
ABS_TOL = 1e-6
EVAL_TIMEOUT_MS = 10_000

Row = Sequence[Any]


def gold_is_ordered(gold_sql: str) -> bool:
    try:
        root = sqlglot.parse_one(gold_sql, read="postgres")
    except ParseError:
        return False
    return isinstance(root, exp.Query) and root.args.get("order") is not None


def _normalize(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    return str(value)


def _cells_match(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return a == b


def _rows_match(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(_cells_match(x, y) for x, y in zip(a, b))


def _sort_key(row: tuple) -> tuple:
    # Round floats so values within tolerance land next to each other; type name keeps mixed types sortable.
    return tuple(
        (type(v).__name__, round(v, 3) if isinstance(v, float) else ("" if v is None else v)) for v in row
    )


def results_match(pred_rows: Sequence[Row], gold_rows: Sequence[Row], ordered: bool) -> bool:
    pred = [tuple(_normalize(v) for v in row) for row in pred_rows]
    gold = [tuple(_normalize(v) for v in row) for row in gold_rows]
    if len(pred) != len(gold):
        return False
    if not ordered:
        pred.sort(key=_sort_key)
        gold.sort(key=_sort_key)
    return all(_rows_match(p, g) for p, g in zip(pred, gold))


def _fetch(conn: psycopg.Connection, sql: str) -> list[tuple] | None:
    try:
        with conn.transaction():
            conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(EVAL_TIMEOUT_MS),))
            return conn.execute(sql).fetchall()
    except psycopg.Error:
        return None


def execution_accuracy(pred_sql: str, gold_sql: str, conn: psycopg.Connection) -> bool:
    """True when both queries run and return matching results. Use a read-only connection."""
    gold_rows = _fetch(conn, gold_sql)
    pred_rows = _fetch(conn, pred_sql)
    if gold_rows is None or pred_rows is None:
        return False
    return results_match(pred_rows, gold_rows, ordered=gold_is_ordered(gold_sql))
