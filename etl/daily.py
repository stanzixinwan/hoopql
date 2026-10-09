"""Load games played since the last successful ingestion run."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

import psycopg

from etl.db import connect
from etl.ingest import (
    SEASON_TYPES,
    games_and_team_logs,
    player_logs,
    upsert_games,
    upsert_player_game_logs,
    upsert_players,
    upsert_team_game_logs,
    upsert_teams,
)
from etl.nba import game_log


def season_start_year(day: date) -> int:
    """NBA seasons start in October. July through September still belong to the season that just ended."""
    return day.year if day.month >= 10 else day.year - 1


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def seasons_between(start: date, end: date) -> list[str]:
    """Season labels that overlap the inclusive date range."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    first = season_start_year(start)
    last = season_start_year(end)
    return [season_label(year) for year in range(first, last + 1)]


def plan_refresh(last_success: datetime, today: date) -> tuple[list[str], str]:
    """Seasons to refresh, and the MM/DD/YYYY date_from passed to the NBA API.

    last_success is stored in UTC. If the local calendar date is still the
    previous evening, refresh that UTC date instead of treating the range as empty.
    """
    if last_success.tzinfo is None:
        start = last_success.date()
    else:
        start = last_success.astimezone(timezone.utc).date()
    if today < start:
        today = start
    return seasons_between(start, today), start.strftime("%m/%d/%Y")


def last_success(conn: psycopg.Connection) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT finished_at
            FROM etl_run
            WHERE status = 'success' AND finished_at IS NOT NULL
            ORDER BY finished_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return None
    return row[0]


def _start_run(conn: psycopg.Connection, seasons: list[str]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO etl_run (status, kind, seasons)
            VALUES ('running', 'daily', %s)
            RETURNING id
            """,
            (",".join(seasons),),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def _finish_run(
    conn: psycopg.Connection,
    run_id: int,
    status: str,
    rows_changed: int,
    error: str | None = None,
) -> None:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE etl_run
            SET finished_at = now(),
                status = %s,
                rows_changed = %s,
                rows_upserted = %s,
                error = %s
            WHERE id = %s
            """,
            (status, rows_changed, rows_changed, error, run_id),
        )
    conn.commit()


def apply_slice(
    conn: psycopg.Connection,
    season: str,
    season_type: str,
    team_rows: list[dict],
    player_rows: list[dict],
) -> int:
    games, team_logs, teams = games_and_team_logs(team_rows, season, season_type)
    known = {game["game_id"] for game in games}
    logs, more_teams, players = player_logs(player_rows, season, season_type, known)
    changed = 0
    with conn.transaction():
        changed += upsert_teams(conn, [*teams, *more_teams], update=False)
        changed += upsert_players(conn, players, update=False)
        changed += upsert_games(conn, games)
        changed += upsert_team_game_logs(conn, team_logs)
        changed += upsert_player_game_logs(conn, logs)
    return changed


def refresh(conn: psycopg.Connection, seasons: list[str], date_from: str) -> int:
    changed = 0
    for season in seasons:
        for season_type in SEASON_TYPES:
            team_rows = game_log("team_game_log", season, season_type, date_from=date_from)
            player_rows = game_log("player_game_log", season, season_type, date_from=date_from)
            slice_changed = apply_slice(conn, season, season_type, team_rows, player_rows)
            changed += slice_changed
            print(
                f"{season} {season_type} from {date_from}: changed={slice_changed}",
                flush=True,
            )
    return changed


def main() -> None:
    today = datetime.now(timezone.utc).date()
    with connect() as conn:
        finished_at = last_success(conn)
        if finished_at is None:
            print("No successful ingestion run yet. Run: uv run python -m etl.ingest", file=sys.stderr)
            raise SystemExit(1)
        seasons, date_from = plan_refresh(finished_at, today)
        run_id = _start_run(conn, seasons)
        try:
            changed = refresh(conn, seasons, date_from)
        except Exception as exc:
            _finish_run(conn, run_id, "failed", 0, str(exc))
            raise
        _finish_run(conn, run_id, "success", changed)
    print(f"daily rows changed: {changed}", flush=True)


if __name__ == "__main__":
    main()
