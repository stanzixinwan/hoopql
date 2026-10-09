"""Live generation against each configured provider. Run with `uv run pytest -m live`."""

from __future__ import annotations

import psycopg
import pytest

from core.config import PROVIDERS, ConfigError, llm_settings
from core.fewshot import FewShotRetriever
from core.generate import make_generator
from core.grounding import ground, hints_text
from core.schema import introspect, schema_text

pytestmark = pytest.mark.live

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"
QUESTION = "How many points per game did Nikola Jokic average in the 2023-24 regular season?"
GOLD = (
    "SELECT AVG(pts) FROM player_game_log "
    "WHERE player_id = 203999 AND season = '2023-24' AND season_type = 'Regular Season'"
)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_generates_correct_sql(provider: str) -> None:
    try:
        settings = llm_settings(provider)
    except ConfigError as exc:
        pytest.skip(str(exc))
    generator = make_generator(settings)
    with psycopg.connect(READONLY_URL) as conn:
        schema = introspect(conn)
        hints = hints_text(ground(QUESTION, conn))
        examples = FewShotRetriever().retrieve(QUESTION, k=3)
        sql = generator.generate(QUESTION, schema_text(schema), examples, hints)
        got = conn.execute(sql).fetchone()[0]
        want = conn.execute(GOLD).fetchone()[0]
    assert float(got) == pytest.approx(float(want), abs=0.05), sql
