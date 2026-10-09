"""Name grounding against the loaded database: 20 name variants, at least 18 correct."""

from __future__ import annotations

import psycopg
import pytest

from core.grounding import candidate_spans, fold, ground, hints_text

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"

# (question, expected (kind, id) pairs)
CASES = [
    ("How many points did Nikola Jokic average in 2024-25?", {("player", 203999)}),
    ("What is Jokic's career high in assists?", {("player", 203999)}),
    ("How many triple-doubles does Giannis have?", {("player", 203507)}),
    ("Best scoring game by the Greek Freak in the playoffs", {("player", 203507)}),
    ("How many threes did Steph make in 2015-16?", {("player", 201939)}),
    ("Stephen Curry games with 10 or more threes", {("player", 201939)}),
    ("Lebron playoff points per game", {("player", 2544)}),
    ("How many 40-point games does King James have?", {("player", 2544)}),
    ("Luka Doncic rebounds per game in 2023-24", {("player", 1629029)}),
    ("Did Kevin Durrant ever score 50?", {("player", 201142)}),
    ("Kawhi Leonard steals in the 2019 playoffs", {("player", 202695)}),
    ("Shai Gilgeous Alexander free throws per game", {("player", 1628983)}),
    ("How many games did the Lakers win in 2019-20?", {("team", 1610612747)}),
    ("Boston home record in the 2023-24 playoffs", {("team", 1610612738)}),
    ("How many road wins did the Dubs have in 2015-16?", {("team", 1610612744)}),
    ("Golden State points per game in 2016-17", {("team", 1610612744)}),
    ("Joel Embiid blocks in March", {("player", 203954)}),
    ("Wemby blocks per game as a rookie", {("player", 1641705)}),
    ("Anthony Edwards 3-point percentage by month", {("player", 1630162)}),
    ("Jayson Tatum points against the Knicks", {("player", 1628369), ("team", 1610612752)}),
]


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(READONLY_URL) as connection:
        yield connection


def test_fixture_resolves_at_least_18_of_20(conn) -> None:
    failures = []
    for question, expected in CASES:
        found = {(item.kind, item.id) for item in ground(question, conn)}
        if not expected <= found:
            failures.append((question, expected, found))
    passed = len(CASES) - len(failures)
    detail = "\n".join(f"{q!r}: expected {e}, got {f}" for q, e, f in failures)
    assert passed >= 18, f"{passed}/20 resolved\n{detail}"


def test_hint_names_the_id_column(conn) -> None:
    hints = hints_text(ground("Nikola Jokic assists", conn))
    assert hints == '"Nikola Jokic" -> player.player_id = 203999 (Nikola Jokić)'


def test_an_ambiguous_city_hints_both_teams(conn) -> None:
    found = {item.id for item in ground("Los Angeles home wins in 2024-25", conn)}
    assert found == {1610612746, 1610612747}


def test_a_plain_stat_question_grounds_nothing(conn) -> None:
    assert ground("Which player scored the most points in 2024-25?", conn) == []


def test_spans_split_at_stopwords_and_skip_seasons() -> None:
    spans = [text for text, _, _ in candidate_spans("Did Jayson Tatum score more than LeBron in 2024-25?", [])]
    assert spans == ["Jayson Tatum", "LeBron"]


def test_fold_strips_accents() -> None:
    assert fold("Nikola Jokić") == "nikola jokic"
