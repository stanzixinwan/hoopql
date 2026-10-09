"""End-to-end answer with a real provider. Run with `uv run pytest -m live -s`."""

from __future__ import annotations

import psycopg
import pytest

from core.config import PROVIDERS, ConfigError, llm_settings
from core.generate import make_generator
from core.pipeline import Pipeline

pytestmark = pytest.mark.live

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"
QUESTION = "Which player had the most 40-point games in 2024-25?"
GOLD = """
SELECT p.full_name, COUNT(*) AS forty_point_games
FROM player_game_log l JOIN player p USING (player_id)
WHERE l.season = '2024-25' AND l.pts >= 40 AND l.season_type = %s
GROUP BY p.full_name ORDER BY forty_point_games DESC LIMIT 1
"""


@pytest.mark.parametrize("provider", PROVIDERS)
def test_forty_point_question_end_to_end(provider: str) -> None:
    try:
        generator = make_generator(llm_settings(provider))
    except ConfigError as exc:
        pytest.skip(str(exc))
    with psycopg.connect(READONLY_URL) as conn:
        name, regular = conn.execute(GOLD, ("Regular Season",)).fetchone()
        all_games = regular + conn.execute(
            "SELECT COUNT(*) FROM player_game_log l JOIN player p USING (player_id) "
            "WHERE p.full_name = %s AND l.season = '2024-25' AND l.season_type = 'Playoffs' AND l.pts >= 40",
            (name,),
        ).fetchone()[0]

    pipeline = Pipeline(generator=generator, conninfo=READONLY_URL)
    try:
        result = pipeline.answer(QUESTION)
    finally:
        pipeline.close()

    print(f"\n{generator.name}\n{result.sql}\n{result.rows[:3]}")
    for stage in result.trace:
        print(f"  {stage.stage:22} {stage.elapsed_ms:8.1f} ms  {stage.detail}")
    assert result.ok, result.error
    assert [s.stage for s in result.trace] == ["schema", "retrieval", "grounding", "fewshot", "generate_and_execute"]
    first = result.rows[0]
    assert name in first
    counts = [v for v in first if isinstance(v, int)]
    assert not counts or counts[0] in (regular, all_games)
