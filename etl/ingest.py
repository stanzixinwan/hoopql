"""Backfill NBA players, teams, games, and game logs into Postgres.

Upserts are idempotent: loading the same rows again changes nothing.
Progress is checkpointed per season so a crash resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import date, datetime
from typing import Any

import psycopg

from etl.db import connect
from etl.nba import all_players, current_teams, game_log

FIRST_SEASON_YEAR = 1996
LAST_SEASON_YEAR = 2026
SEASON_TYPES = ("Regular Season", "Playoffs")

TEAM_COLUMNS = ("team_id", "abbreviation", "full_name", "city", "nickname")
PLAYER_COLUMNS = (
    "player_id",
    "full_name",
    "first_name",
    "last_name",
    "is_active",
    "from_year",
    "to_year",
    "team_id",
)
GAME_COLUMNS = (
    "game_id",
    "game_date",
    "season",
    "season_type",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
)
TEAM_LOG_COLUMNS = (
    "team_id",
    "game_id",
    "opponent_team_id",
    "game_date",
    "season",
    "season_type",
    "is_home",
    "wl",
    "minutes",
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "fg3_pct",
    "ftm",
    "fta",
    "ft_pct",
    "oreb",
    "dreb",
    "plus_minus",
)
PLAYER_LOG_COLUMNS = (
    "player_id",
    "game_id",
    "team_id",
    "game_date",
    "season",
    "season_type",
    "minutes",
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "fg3_pct",
    "ftm",
    "fta",
    "ft_pct",
    "oreb",
    "dreb",
    "plus_minus",
    "wl",
    "is_home",
)


def season_labels(
    first_year: int = FIRST_SEASON_YEAR,
    last_year: int = LAST_SEASON_YEAR,
) -> list[str]:
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(first_year, last_year + 1)]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value == ""


def as_int(value: Any) -> int | None:
    if _missing(value):
        return None
    return int(value)


def as_float(value: Any) -> float | None:
    if _missing(value):
        return None
    return float(value)


def require_int(value: Any) -> int:
    parsed = as_int(value)
    if parsed is None:
        raise ValueError("missing integer")
    return parsed


def parse_minutes(value: Any) -> float | None:
    if _missing(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return int(minutes) + int(seconds) / 60
    return float(text)


def parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    return date.fromisoformat(text[:10])


def parse_wl(value: Any) -> str | None:
    if _missing(value):
        return None
    text = str(value).strip().upper()
    if text in {"W", "L"}:
        return text
    return None


def game_id_text(value: Any) -> str:
    if isinstance(value, str) and not value.isdigit():
        return value
    return str(require_int(value)).zfill(10)


def is_home_matchup(matchup: str) -> bool:
    if " vs. " in matchup:
        return True
    if " @ " in matchup:
        return False
    raise ValueError(f"unrecognized matchup: {matchup}")


def split_name(full_name: str) -> tuple[str | None, str | None]:
    parts = full_name.split()
    if len(parts) < 2:
        return full_name, None
    return parts[0], " ".join(parts[1:])


def team_from_static(row: dict) -> dict:
    return {
        "team_id": int(row["id"]),
        "abbreviation": row["abbreviation"],
        "full_name": row["full_name"],
        "city": row.get("city"),
        "nickname": row.get("nickname"),
    }


def team_from_log(row: dict) -> dict:
    name = row.get("TEAM_NAME") or row["TEAM_ABBREVIATION"]
    return {
        "team_id": require_int(row["TEAM_ID"]),
        "abbreviation": row["TEAM_ABBREVIATION"],
        "full_name": name,
        "city": None,
        "nickname": row.get("TEAM_NAME"),
    }


def player_from_directory(row: dict, valid_team_ids: set[int]) -> dict:
    full_name = row.get("DISPLAY_FIRST_LAST") or row.get("PLAYER_NAME")
    if not full_name:
        raise ValueError(f"player {row.get('PERSON_ID')} has no name")
    first_name, last_name = split_name(str(full_name))
    team_id = as_int(row.get("TEAM_ID"))
    if team_id in (None, 0) or team_id not in valid_team_ids:
        team_id = None
    status = row.get("ROSTERSTATUS")
    return {
        "player_id": require_int(row.get("PERSON_ID") if row.get("PERSON_ID") is not None else row.get("PLAYER_ID")),
        "full_name": str(full_name),
        "first_name": first_name,
        "last_name": last_name,
        "is_active": status in (1, "1", "Active", True),
        "from_year": as_int(row.get("FROM_YEAR")),
        "to_year": as_int(row.get("TO_YEAR")),
        "team_id": team_id,
    }


def player_from_log(row: dict) -> dict:
    full_name = str(row["PLAYER_NAME"])
    first_name, last_name = split_name(full_name)
    return {
        "player_id": require_int(row["PLAYER_ID"]),
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "is_active": False,
        "from_year": None,
        "to_year": None,
        "team_id": None,
    }


def _stat_fields(row: dict) -> dict:
    return {
        "minutes": parse_minutes(row.get("MIN")),
        "pts": require_int(row.get("PTS")),
        "reb": require_int(row.get("REB")),
        "ast": require_int(row.get("AST")),
        "stl": require_int(row.get("STL")),
        "blk": require_int(row.get("BLK")),
        "tov": require_int(row.get("TOV")),
        "pf": require_int(row.get("PF")),
        "fgm": require_int(row.get("FGM")),
        "fga": require_int(row.get("FGA")),
        "fg_pct": as_float(row.get("FG_PCT")),
        "fg3m": require_int(row.get("FG3M")),
        "fg3a": require_int(row.get("FG3A")),
        "fg3_pct": as_float(row.get("FG3_PCT")),
        "ftm": require_int(row.get("FTM")),
        "fta": require_int(row.get("FTA")),
        "ft_pct": as_float(row.get("FT_PCT")),
        "oreb": require_int(row.get("OREB")),
        "dreb": require_int(row.get("DREB")),
        "plus_minus": as_int(row.get("PLUS_MINUS")),
        "wl": parse_wl(row.get("WL")),
    }


def games_and_team_logs(
    rows: list[dict],
    season: str,
    season_type: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Pair the two team rows for each game into a game plus two team logs."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[game_id_text(row["GAME_ID"])].append(row)

    games: list[dict] = []
    logs: list[dict] = []
    teams: list[dict] = []
    for gid, pair in grouped.items():
        if len(pair) != 2:
            print(f"skip game {gid}: expected 2 team rows, found {len(pair)}", flush=True)
            continue
        try:
            homes = [row for row in pair if is_home_matchup(str(row["MATCHUP"]))]
            aways = [row for row in pair if not is_home_matchup(str(row["MATCHUP"]))]
        except ValueError as exc:
            print(f"skip game {gid}: {exc}", flush=True)
            continue
        if len(homes) != 1 or len(aways) != 1:
            print(f"skip game {gid}: could not tell home from away", flush=True)
            continue
        home, away = homes[0], aways[0]
        played_on = parse_date(home["GAME_DATE"])
        teams.extend((team_from_log(home), team_from_log(away)))
        games.append(
            {
                "game_id": gid,
                "game_date": played_on,
                "season": season,
                "season_type": season_type,
                "home_team_id": require_int(home["TEAM_ID"]),
                "away_team_id": require_int(away["TEAM_ID"]),
                "home_score": require_int(home["PTS"]),
                "away_score": require_int(away["PTS"]),
            }
        )
        for row, opponent, home_flag in ((home, away, True), (away, home, False)):
            logs.append(
                {
                    "team_id": require_int(row["TEAM_ID"]),
                    "game_id": gid,
                    "opponent_team_id": require_int(opponent["TEAM_ID"]),
                    "game_date": played_on,
                    "season": season,
                    "season_type": season_type,
                    "is_home": home_flag,
                    **_stat_fields(row),
                }
            )
    return games, logs, teams


def player_logs(
    rows: list[dict],
    season: str,
    season_type: str,
    known_game_ids: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    logs: list[dict] = []
    teams: list[dict] = []
    players: list[dict] = []
    for row in rows:
        gid = game_id_text(row["GAME_ID"])
        if gid not in known_game_ids:
            continue
        try:
            home = is_home_matchup(str(row["MATCHUP"]))
            stat_fields = _stat_fields(row)
        except (ValueError, KeyError) as exc:
            print(f"skip player row {gid} {row.get('PLAYER_ID')}: {exc}", flush=True)
            continue
        teams.append(team_from_log(row))
        players.append(player_from_log(row))
        logs.append(
            {
                "player_id": require_int(row["PLAYER_ID"]),
                "game_id": gid,
                "team_id": require_int(row["TEAM_ID"]),
                "game_date": parse_date(row["GAME_DATE"]),
                "season": season,
                "season_type": season_type,
                "is_home": home,
                **stat_fields,
            }
        )
    return logs, teams, players


def dedupe(rows: list[dict], *keys: str) -> list[dict]:
    found: dict[tuple, dict] = {}
    for row in rows:
        found[tuple(row[key] for key in keys)] = row
    return list(found.values())


def upsert(
    conn: psycopg.Connection,
    table: str,
    columns: tuple[str, ...],
    conflict: tuple[str, ...],
    rows: list[dict],
    *,
    update: bool,
) -> int:
    """Insert rows. Return how many were inserted or actually changed."""
    rows = dedupe(rows, *conflict)
    if not rows:
        return 0
    column_list = ", ".join(columns)
    conflict_list = ", ".join(conflict)
    if update:
        mutable = [column for column in columns if column not in conflict]
        assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in mutable)
        current = ", ".join(f"{table}.{column}" for column in mutable)
        excluded = ", ".join(f"EXCLUDED.{column}" for column in mutable)
        action = f"""
            DO UPDATE SET {assignments}
            WHERE ({current}) IS DISTINCT FROM ({excluded})
        """
    else:
        action = "DO NOTHING"
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS staging_upsert")
        cur.execute(f"CREATE TEMP TABLE staging_upsert (LIKE {table} INCLUDING DEFAULTS)")
        with cur.copy(f"COPY staging_upsert ({column_list}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(tuple(row[column] for column in columns))
        cur.execute(
            f"""
            INSERT INTO {table} ({column_list})
            SELECT {column_list} FROM staging_upsert
            ON CONFLICT ({conflict_list}) {action}
            """
        )
        changed = cur.rowcount
        cur.execute("DROP TABLE staging_upsert")
        return changed


def upsert_teams(conn: psycopg.Connection, rows: list[dict], *, update: bool) -> int:
    return upsert(conn, "team", TEAM_COLUMNS, ("team_id",), rows, update=update)


def upsert_players(conn: psycopg.Connection, rows: list[dict], *, update: bool) -> int:
    return upsert(conn, "player", PLAYER_COLUMNS, ("player_id",), rows, update=update)


def upsert_games(conn: psycopg.Connection, rows: list[dict]) -> int:
    return upsert(conn, "game", GAME_COLUMNS, ("game_id",), rows, update=True)


def upsert_team_game_logs(conn: psycopg.Connection, rows: list[dict]) -> int:
    return upsert(conn, "team_game_log", TEAM_LOG_COLUMNS, ("team_id", "game_id"), rows, update=True)


def upsert_player_game_logs(conn: psycopg.Connection, rows: list[dict]) -> int:
    return upsert(
        conn,
        "player_game_log",
        PLAYER_LOG_COLUMNS,
        ("player_id", "game_id"),
        rows,
        update=True,
    )


def _checkpoint_done(conn: psycopg.Connection, endpoint: str, season: str, season_type: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM etl_checkpoint
            WHERE endpoint = %s AND season = %s AND season_type = %s
            """,
            (endpoint, season, season_type),
        )
        found = cur.fetchone() is not None
    conn.commit()
    return found


def _mark_checkpoint(
    conn: psycopg.Connection,
    endpoint: str,
    season: str,
    season_type: str,
    row_count: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO etl_checkpoint (endpoint, season, season_type, row_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (endpoint, season, season_type) DO UPDATE
            SET finished_at = now(), row_count = EXCLUDED.row_count
            """,
            (endpoint, season, season_type, row_count),
        )


def _team_ids(conn: psycopg.Connection) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT team_id FROM team")
        return {row[0] for row in cur.fetchall()}


def _game_ids(conn: psycopg.Connection, season: str, season_type: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT game_id FROM game WHERE season = %s AND season_type = %s",
            (season, season_type),
        )
        ids = {row[0] for row in cur.fetchall()}
    conn.commit()
    return ids


def _start_run(conn: psycopg.Connection, seasons: list[str]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO etl_run (status, kind, seasons)
            VALUES ('running', 'backfill', %s)
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


def load(
    conn: psycopg.Connection,
    seasons: list[str],
    season_types: tuple[str, ...],
    *,
    reapply: bool,
) -> int:
    changed = 0
    directory_rows = all_players()
    with conn.transaction():
        changed += upsert_teams(conn, [team_from_static(row) for row in current_teams()], update=True)
        directory = [player_from_directory(row, _team_ids(conn)) for row in directory_rows]
        changed += upsert_players(conn, directory, update=True)
    print(f"directory changed {changed} rows", flush=True)

    for season in seasons:
        for season_type in season_types:
            changed += _load_slice(conn, season, season_type, reapply=reapply)
    return changed


def _load_slice(conn: psycopg.Connection, season: str, season_type: str, *, reapply: bool) -> int:
    changed = 0
    if reapply or not _checkpoint_done(conn, "team_game_log", season, season_type):
        raw_teams = game_log("team_game_log", season, season_type)
        games, logs, teams = games_and_team_logs(raw_teams, season, season_type)
        with conn.transaction():
            changed += upsert_teams(conn, teams, update=False)
            changed += upsert_games(conn, games)
            changed += upsert_team_game_logs(conn, logs)
            _mark_checkpoint(conn, "team_game_log", season, season_type, len(raw_teams))
        print(
            f"{season} {season_type} teams: source={len(raw_teams)} changed={changed}",
            flush=True,
        )

    if reapply or not _checkpoint_done(conn, "player_game_log", season, season_type):
        raw_players = game_log("player_game_log", season, season_type)
        known_games = _game_ids(conn, season, season_type)
        logs, teams, players = player_logs(raw_players, season, season_type, known_games)
        before = changed
        with conn.transaction():
            changed += upsert_teams(conn, teams, update=False)
            changed += upsert_players(conn, players, update=False)
            changed += upsert_player_game_logs(conn, logs)
            _mark_checkpoint(conn, "player_game_log", season, season_type, len(raw_players))
        print(
            f"{season} {season_type} players: source={len(raw_players)} changed={changed - before}",
            flush=True,
        )
    return changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill NBA game logs into Postgres.")
    parser.add_argument("--season", action="append", help="Season label such as 2024-25. Default: 1996-97 through 2026-27.")
    parser.add_argument("--season-type", action="append", choices=SEASON_TYPES)
    parser.add_argument(
        "--reapply",
        action="store_true",
        help="Upsert again from the local cache and report how many rows change.",
    )
    args = parser.parse_args(argv)
    seasons = args.season or season_labels()
    season_types = tuple(args.season_type) if args.season_type else SEASON_TYPES

    with connect() as conn:
        run_id = _start_run(conn, seasons)
        try:
            changed = load(conn, seasons, season_types, reapply=args.reapply)
        except Exception as exc:
            _finish_run(conn, run_id, "failed", 0, str(exc))
            raise
        _finish_run(conn, run_id, "success", changed)
    print(f"backfill rows changed: {changed}", flush=True)


if __name__ == "__main__":
    main()
