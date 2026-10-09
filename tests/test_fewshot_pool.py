"""Every few-shot exemplar runs against Postgres; every original question is kept or dropped with a reason."""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]
POOL = json.loads((ROOT / "data/fewshot/pool.json").read_text())
DROPPED = json.loads((ROOT / "data/fewshot/dropped.json").read_text())
ORIGINAL = json.loads((ROOT / "data/nba/nba_questions.json").read_text())
READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


def test_every_original_question_is_accounted_for() -> None:
    kept = {item["id"] for item in POOL}
    dropped = {item["id"] for item in DROPPED}
    assert not kept & dropped
    assert kept | dropped == {item["id"] for item in ORIGINAL}
    assert all(item["reason"] for item in DROPPED)


@pytest.mark.parametrize("item", POOL, ids=lambda item: f"q{item['id']}")
def test_exemplar_executes(item: dict) -> None:
    with psycopg.connect(READONLY_URL) as conn:
        conn.execute(item["sql"]).fetchall()
