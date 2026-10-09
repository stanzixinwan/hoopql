"""Apply etl/schema.sql to an empty database and check keys and indexes."""

from __future__ import annotations

import pytest

from tests.pg import PostgresError, apply_schema, psql, recreate_database

DATABASE = "hoopql_schema_test"

EXPECTED_PRIMARY_KEYS = {
    "team": "team_id",
    "player": "player_id",
    "game": "game_id",
    "player_game_log": "player_id,game_id",
    "team_game_log": "team_id,game_id",
    "play_by_play": "game_id,event_num",
    "etl_run": "id",
    "etl_checkpoint": "endpoint,season,season_type",
}

EXPECTED_INDEXES = {
    "game_season_idx",
    "player_game_log_player_id_game_date_idx",
    "player_game_log_team_id_game_date_idx",
    "player_game_log_season_idx",
    "team_game_log_team_id_game_date_idx",
    "team_game_log_season_idx",
}


@pytest.fixture(scope="module")
def schema_database() -> str:
    try:
        recreate_database(DATABASE)
        apply_schema(DATABASE)
    except PostgresError as exc:
        pytest.fail(f"Could not apply schema. Is `docker compose up -d postgres` running?\n{exc}")
    return DATABASE


def test_primary_keys(schema_database: str) -> None:
    rows = psql(
        """
        SELECT tc.table_name, string_agg(kcu.column_name, ',' ORDER BY kcu.ordinal_position)
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.constraint_type = 'PRIMARY KEY'
        GROUP BY tc.table_name
        ORDER BY tc.table_name
        """,
        schema_database,
    )
    found = {}
    for line in rows.splitlines():
        table, columns = line.split("|")
        found[table] = columns
    assert found == EXPECTED_PRIMARY_KEYS


def test_foreign_keys(schema_database: str) -> None:
    rows = psql(
        """
        SELECT tc.table_name, kcu.column_name, ccu.table_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.constraint_type = 'FOREIGN KEY'
        ORDER BY 1, 2, 3
        """,
        schema_database,
    )
    pairs = {tuple(line.split("|")) for line in rows.splitlines() if line}
    assert ("player", "team_id", "team") in pairs
    assert ("game", "home_team_id", "team") in pairs
    assert ("game", "away_team_id", "team") in pairs
    assert ("player_game_log", "player_id", "player") in pairs
    assert ("player_game_log", "game_id", "game") in pairs
    assert ("player_game_log", "team_id", "team") in pairs
    assert ("team_game_log", "team_id", "team") in pairs
    assert ("team_game_log", "game_id", "game") in pairs
    assert ("team_game_log", "opponent_team_id", "team") in pairs
    assert ("play_by_play", "game_id", "game") in pairs


def test_lookup_indexes(schema_database: str) -> None:
    rows = psql(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY indexname
        """,
        schema_database,
    )
    names = set(rows.splitlines())
    assert EXPECTED_INDEXES <= names
