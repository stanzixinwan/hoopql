"""NBA stats fetches with retry, backoff, and an on-disk cache.

A crashed backfill reads the cache instead of calling stats.nba.com again.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from nba_api.stats.endpoints import commonallplayers, leaguegamelog
from nba_api.stats.static import teams as static_teams

CACHE_ROOT = Path("data/raw/nba_api")
PLAYER_OR_TEAM = {"player_game_log": "P", "team_game_log": "T"}


def cache_path(
    endpoint: str,
    season: str,
    season_type: str,
    *,
    date_from: str = "",
    cache_root: Path = CACHE_ROOT,
) -> Path:
    slug = season_type.replace(" ", "_").lower()
    suffix = f"-from-{date_from.replace('/', '-')}" if date_from else ""
    return cache_root / endpoint / f"{season}-{slug}{suffix}.json"


def load_or_fetch(
    path: Path,
    fetcher: Callable[[], list[dict]],
    *,
    force: bool = False,
    attempts: int = 5,
    sleep_s: float = 1.5,
    backoff_s: float = 1.5,
) -> list[dict]:
    """Return cached rows, or call fetcher, write the cache, and sleep."""
    if path.exists() and not force:
        return json.loads(path.read_text())

    delay = backoff_s
    for attempt in range(1, attempts + 1):
        try:
            rows = fetcher()
            break
        except Exception as exc:  # stats.nba.com times out and rate-limits often
            if attempt == attempts:
                raise
            print(f"fetch failed ({attempt}/{attempts}): {exc}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows))
    temporary.replace(path)
    if sleep_s:
        time.sleep(sleep_s)
    return rows


def _records(endpoint) -> list[dict]:
    frames = endpoint.get_data_frames()
    if not frames or frames[0].empty:
        return []
    return json.loads(frames[0].to_json(orient="records", date_format="iso"))


def current_teams() -> list[dict]:
    """Franchises shipped with nba_api. No network call."""
    return list(static_teams.get_teams())


def all_players(*, cache_root: Path = CACHE_ROOT, sleep_s: float = 1.5) -> list[dict]:
    path = cache_root / "players" / "all.json"

    def fetch() -> list[dict]:
        print("fetching players", flush=True)
        endpoint = commonallplayers.CommonAllPlayers(
            is_only_current_season=0,
            timeout=60,
        )
        return _records(endpoint)

    return load_or_fetch(path, fetch, sleep_s=sleep_s)


def game_log(
    kind: str,
    season: str,
    season_type: str,
    *,
    date_from: str = "",
    cache_root: Path = CACHE_ROOT,
    sleep_s: float = 1.5,
) -> list[dict]:
    """Player or team league game log. date_from is MM/DD/YYYY when set."""
    if kind not in PLAYER_OR_TEAM:
        raise ValueError(f"unknown game log kind: {kind}")
    path = cache_path(
        kind,
        season,
        season_type,
        date_from=date_from,
        cache_root=cache_root,
    )

    def fetch() -> list[dict]:
        label = f"{kind} {season} {season_type}"
        if date_from:
            label = f"{label} from {date_from}"
        print(f"fetching {label}", flush=True)
        endpoint = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation=PLAYER_OR_TEAM[kind],
            date_from_nullable=date_from,
            timeout=90,
        )
        return _records(endpoint)

    return load_or_fetch(path, fetch, sleep_s=sleep_s)
