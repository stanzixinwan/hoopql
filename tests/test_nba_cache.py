"""The NBA client writes a cache and does not call the network on a hit."""

from __future__ import annotations

import json

from etl.nba import cache_path, load_or_fetch


def test_second_call_uses_the_cache(tmp_path) -> None:
    calls = {"n": 0}

    def fetcher() -> list[dict]:
        calls["n"] += 1
        return [{"PLAYER_ID": 977, "PTS": 81}]

    path = cache_path(
        "player_game_log",
        "2005-06",
        "Regular Season",
        cache_root=tmp_path,
    )
    first = load_or_fetch(path, fetcher, sleep_s=0)
    second = load_or_fetch(path, fetcher, sleep_s=0)
    assert first == second == [{"PLAYER_ID": 977, "PTS": 81}]
    assert calls["n"] == 1
    assert path.exists()


def test_retry_then_cache_the_success(tmp_path) -> None:
    calls = {"n": 0}

    def fetcher() -> list[dict]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("stats.nba.com timed out")
        return []

    path = cache_path("team_game_log", "2026-27", "Playoffs", cache_root=tmp_path)
    rows = load_or_fetch(path, fetcher, sleep_s=0, backoff_s=0)
    assert rows == []
    assert calls["n"] == 3
    assert json.loads(path.read_text()) == []


def test_date_from_is_part_of_the_cache_key(tmp_path) -> None:
    full = cache_path("player_game_log", "2025-26", "Regular Season", cache_root=tmp_path)
    partial = cache_path(
        "player_game_log",
        "2025-26",
        "Regular Season",
        date_from="10/01/2026",
        cache_root=tmp_path,
    )
    assert full != partial
    assert partial.name.endswith("-from-10-01-2026.json")
