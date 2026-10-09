"""Loading the same game logs twice changes zero rows."""

from __future__ import annotations

import psycopg
import pytest

from etl.ingest import (
    games_and_team_logs,
    player_from_log,
    player_logs,
    team_from_log,
    upsert_games,
    upsert_player_game_logs,
    upsert_players,
    upsert_team_game_logs,
    upsert_teams,
)
from tests.pg import PostgresError, apply_schema, psql, recreate_database

DATABASE = "hoopql_ingest_test"
URL = f"postgresql://hoopql:hoopql@localhost:5432/{DATABASE}"


def _box(pts: int, plus_minus: int) -> dict:
    return {
        "MIN": "42:00",
        "FGM": 28,
        "FGA": 46,
        "FG_PCT": 0.609,
        "FG3M": 7,
        "FG3A": 13,
        "FG3_PCT": 0.538,
        "FTM": 18,
        "FTA": 20,
        "FT_PCT": 0.9,
        "OREB": 2,
        "DREB": 4,
        "REB": 6,
        "AST": 2,
        "STL": 3,
        "BLK": 1,
        "TOV": 3,
        "PF": 1,
        "PTS": pts,
        "PLUS_MINUS": plus_minus,
        "WL": "W",
    }


def _team_row(team_id: int, abbr: str, name: str, matchup: str, pts: int) -> dict:
    return {
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": abbr,
        "TEAM_NAME": name,
        "GAME_ID": "0020500591",
        "GAME_DATE": "2006-01-22",
        "MATCHUP": matchup,
        **_box(pts, pts - 98 if "vs." in matchup else 98 - pts),
    }


HOME = _team_row(1610612747, "LAL", "Lakers", "LAL vs. TOR", 122)
AWAY = _team_row(1610612761, "TOR", "Raptors", "TOR @ LAL", 104)
PLAYER = {
    "PLAYER_ID": 977,
    "PLAYER_NAME": "Kobe Bryant",
    "TEAM_ID": 1610612747,
    "TEAM_ABBREVIATION": "LAL",
    "TEAM_NAME": "Lakers",
    "GAME_ID": "0020500591",
    "GAME_DATE": "2006-01-22",
    "MATCHUP": "LAL vs. TOR",
    **_box(81, 17),
}


@pytest.fixture(scope="module")
def database() -> str:
    try:
        recreate_database(DATABASE)
        apply_schema(DATABASE)
    except PostgresError as exc:
        pytest.fail(f"Could not prepare {DATABASE}.\n{exc}")
    return DATABASE


def _load_once(conn: psycopg.Connection) -> int:
    games, team_logs, teams = games_and_team_logs([HOME, AWAY], "2005-06", "Regular Season")
    logs, more_teams, players = player_logs([PLAYER], "2005-06", "Regular Season", {game["game_id"] for game in games})
    changed = 0
    changed += upsert_teams(conn, [*teams, *more_teams], update=True)
    changed += upsert_players(conn, [player_from_log(PLAYER), *players], update=True)
    changed += upsert_games(conn, games)
    changed += upsert_team_game_logs(conn, team_logs)
    changed += upsert_player_game_logs(conn, logs)
    return changed


def test_second_load_changes_zero_rows(database: str) -> None:
    with psycopg.connect(URL) as conn:
        with conn.transaction():
            first = _load_once(conn)
        with conn.transaction():
            second = _load_once(conn)
    assert first > 0
    assert second == 0
    assert psql("SELECT count(*) FROM player_game_log", database).strip() == "1"
    assert psql("SELECT pts FROM player_game_log", database).strip() == "81"


def test_a_changed_total_is_updated(database: str) -> None:
    edited = dict(PLAYER)
    edited["PTS"] = 80
    games, _, _ = games_and_team_logs([HOME, AWAY], "2005-06", "Regular Season")
    logs, _, _ = player_logs([edited], "2005-06", "Regular Season", {game["game_id"] for game in games})
    with psycopg.connect(URL) as conn:
        with conn.transaction():
            _load_once(conn)
        with conn.transaction():
            changed = upsert_player_game_logs(conn, logs)
    assert changed == 1
    assert psql("SELECT pts FROM player_game_log", database).strip() == "80"
