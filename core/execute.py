"""Run validated SQL on the read-only pool with a server-side timeout and a row cap."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

from core.config import database_url
from core.safety import ValidatedSQL

DEFAULT_TIMEOUT_MS = 5000
DEFAULT_ROW_CAP = 1000


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool
    elapsed_ms: float


class ExecutionError(RuntimeError):
    def __init__(self, message: str, kind: str, elapsed_ms: float) -> None:
        super().__init__(message)
        self.kind = kind
        self.elapsed_ms = elapsed_ms


class Executor:
    def __init__(
        self,
        conninfo: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        row_cap: int = DEFAULT_ROW_CAP,
        max_size: int = 4,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.row_cap = row_cap
        self.pool = ConnectionPool(conninfo or database_url(), min_size=1, max_size=max_size, open=True)

    def run(self, query: ValidatedSQL) -> QueryResult:
        started = time.perf_counter()
        try:
            with self.pool.connection() as conn, conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(self.timeout_ms),))
                cursor = conn.execute(query.sql)
                rows = cursor.fetchmany(self.row_cap + 1)
                columns = [column.name for column in cursor.description or []]
        except psycopg.errors.QueryCanceled as exc:
            raise ExecutionError(
                f"Query exceeded the {self.timeout_ms} ms timeout and was cancelled.",
                kind="timeout",
                elapsed_ms=_ms_since(started),
            ) from exc
        except psycopg.Error as exc:
            message = exc.diag.message_primary or str(exc)
            raise ExecutionError(message, kind="database", elapsed_ms=_ms_since(started)) from exc
        truncated = len(rows) > self.row_cap
        rows = rows[: self.row_cap]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=_ms_since(started),
        )

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> Executor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
