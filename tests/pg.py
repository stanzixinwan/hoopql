"""Helpers for tests that need a throwaway Postgres database."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "etl" / "schema.sql"
MIGRATIONS_DIR = ROOT / "etl" / "migrations"


class PostgresError(RuntimeError):
    pass


def psql(sql: str, database: str = "postgres") -> str:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "hoopql",
            "-d",
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-c",
            sql,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PostgresError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def apply_sql(sql: str, database: str) -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "hoopql",
            "-d",
            database,
            "-v",
            "ON_ERROR_STOP=1",
        ],
        cwd=ROOT,
        input=sql,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PostgresError(result.stderr.strip() or result.stdout.strip())


def recreate_database(name: str) -> None:
    psql(
        f"""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{name}' AND pid <> pg_backend_pid();
        """
    )
    psql(f"DROP DATABASE IF EXISTS {name}")
    psql(f"CREATE DATABASE {name}")


def apply_schema(database: str) -> None:
    apply_sql(SCHEMA_SQL.read_text(), database)
    for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
        apply_sql(migration.read_text(), database)
