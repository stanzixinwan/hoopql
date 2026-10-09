"""Schema introspection as the API's read-only role sees it."""

from __future__ import annotations

import psycopg
import pytest

from core.schema import SchemaCache, get_schema, introspect, schema_text

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


@pytest.fixture(scope="module")
def schema():
    with psycopg.connect(READONLY_URL) as conn:
        return introspect(conn)


def test_product_tables_are_visible_and_etl_tables_are_not(schema) -> None:
    names = set(schema.table_names())
    assert {"player", "team", "game", "player_game_log", "team_game_log"} <= names
    assert "etl_run" not in names
    assert "etl_checkpoint" not in names
    assert "play_by_play" not in names


def test_columns_carry_types_and_comments(schema) -> None:
    columns = {column.name: column for column in schema.tables["player_game_log"].columns}
    assert columns["pts"].data_type == "integer"
    assert columns["pts"].nullable is False
    assert columns["fg3_pct"].nullable is True
    assert "Three-point percentage" in columns["fg3_pct"].comment
    assert all(column.comment for column in columns.values())


def test_foreign_keys_are_visible_to_the_readonly_role(schema) -> None:
    keys = {(fk.column, fk.ref_table) for fk in schema.tables["player_game_log"].foreign_keys}
    assert {("player_id", "player"), ("game_id", "game"), ("team_id", "team")} <= keys


def test_schema_text_includes_comments_and_keys(schema) -> None:
    text = schema_text(schema, ["player_game_log"])
    assert text.startswith("TABLE player_game_log")
    assert "plus_minus integer -- Point differential" in text
    assert "FOREIGN KEY (player_id) REFERENCES player (player_id)" in text
    assert "TABLE game\n" not in text


def test_get_schema_uses_the_default_cache() -> None:
    assert "player_game_log" in get_schema().tables


def test_cache_refreshes_after_ttl() -> None:
    calls = {"n": 0}
    now = {"t": 0.0}

    def connect():
        calls["n"] += 1
        return psycopg.connect(READONLY_URL)

    cache = SchemaCache(connect=connect, ttl_s=60, clock=lambda: now["t"])
    cache.get()
    now["t"] = 59
    cache.get()
    assert calls["n"] == 1
    now["t"] = 61
    cache.get()
    assert calls["n"] == 2
