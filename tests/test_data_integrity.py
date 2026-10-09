"""Five published results, checked against basketball-reference.

- Kobe Bryant scored 81 on 2006-01-22.
- The 2015-16 Warriors won 73 regular-season games.
- Cleveland beat Golden State 4-3 in the 2016 Finals.
- The 1996-97 Bulls went 69-13.
- Stephen Curry made 402 threes in 2015-16.
"""

from __future__ import annotations

import psycopg
import pytest

from etl.db import OWNER_URL


@pytest.fixture(scope="module")
def db() -> psycopg.Connection:
    try:
        conn = psycopg.connect(OWNER_URL)
    except psycopg.OperationalError as exc:
        pytest.fail(f"Could not connect to the loaded database.\n{exc}")
    yield conn
    conn.close()


def _one(db: psycopg.Connection, sql: str) -> int:
    value = db.execute(sql).fetchone()[0]
    return int(value)


def test_kobe_scored_81_on_january_22_2006(db: psycopg.Connection) -> None:
    points = _one(
        db,
        """
        SELECT pts
        FROM player_game_log
        JOIN player USING (player_id)
        WHERE full_name = 'Kobe Bryant'
          AND game_date = DATE '2006-01-22'
        """,
    )
    assert points == 81


def test_warriors_won_73_games_in_2015_16(db: psycopg.Connection) -> None:
    wins = _one(
        db,
        """
        SELECT count(*)
        FROM team_game_log
        JOIN team USING (team_id)
        WHERE full_name = 'Golden State Warriors'
          AND season = '2015-16'
          AND season_type = 'Regular Season'
          AND wl = 'W'
        """,
    )
    assert wins == 73


def test_cavaliers_won_the_2016_finals_in_seven(db: psycopg.Connection) -> None:
    cavs_wins = _one(
        db,
        """
        SELECT count(*)
        FROM team_game_log cle
        JOIN team cle_team ON cle_team.team_id = cle.team_id
        JOIN team gsw_team ON gsw_team.team_id = cle.opponent_team_id
        WHERE cle_team.full_name = 'Cleveland Cavaliers'
          AND gsw_team.full_name = 'Golden State Warriors'
          AND cle.season = '2015-16'
          AND cle.season_type = 'Playoffs'
          AND cle.wl = 'W'
        """,
    )
    assert cavs_wins == 4


def test_bulls_won_69_games_in_1996_97(db: psycopg.Connection) -> None:
    wins = _one(
        db,
        """
        SELECT count(*)
        FROM team_game_log
        JOIN team USING (team_id)
        WHERE full_name = 'Chicago Bulls'
          AND season = '1996-97'
          AND season_type = 'Regular Season'
          AND wl = 'W'
        """,
    )
    assert wins == 69


def test_curry_made_402_threes_in_2015_16(db: psycopg.Connection) -> None:
    made = _one(
        db,
        """
        SELECT sum(fg3m)
        FROM player_game_log
        JOIN player USING (player_id)
        WHERE full_name = 'Stephen Curry'
          AND season = '2015-16'
          AND season_type = 'Regular Season'
        """,
    )
    assert made == 402
