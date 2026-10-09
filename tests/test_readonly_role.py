"""The API role can read game logs and cannot change them."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from psycopg.errors import InsufficientPrivilege, QueryCanceled

from tests.pg import PostgresError, apply_schema, apply_sql, recreate_database

DATABASE = "hoopql_readonly_test"
OWNER_URL = f"postgresql://hoopql:hoopql@localhost:5432/{DATABASE}"
READONLY_URL = f"postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/{DATABASE}"
ROLES_SQL = Path(__file__).resolve().parents[1] / "etl" / "roles.sql"


@pytest.fixture(scope="module")
def database() -> str:
    try:
        recreate_database(DATABASE)
        apply_schema(DATABASE)
        apply_sql(ROLES_SQL.read_text(), DATABASE)
    except PostgresError as exc:
        pytest.fail(f"Could not prepare {DATABASE}.\n{exc}")
    with psycopg.connect(OWNER_URL) as conn:
        conn.execute(
            """
            INSERT INTO team (team_id, abbreviation, full_name)
            VALUES (1610612747, 'LAL', 'Los Angeles Lakers')
            """
        )
        conn.commit()
    return DATABASE


def test_delete_is_rejected(database: str) -> None:
    with psycopg.connect(READONLY_URL) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute("DELETE FROM team")


def test_select_on_game_logs_works(database: str) -> None:
    with psycopg.connect(READONLY_URL) as conn:
        count = conn.execute("SELECT count(*) FROM team").fetchone()[0]
    assert count == 1


def test_etl_tables_are_hidden(database: str) -> None:
    with psycopg.connect(READONLY_URL) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute("SELECT count(*) FROM etl_run")


def test_statement_timeout_cancels_a_long_query(database: str) -> None:
    with psycopg.connect(READONLY_URL) as conn:
        with pytest.raises(QueryCanceled):
            conn.execute("SELECT pg_sleep(6)")
