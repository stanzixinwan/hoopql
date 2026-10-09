"""Pipeline wiring: stage trace, ablation flags, and error reporting, with a scripted generator."""

from __future__ import annotations

import pytest

from core.execute import Executor
from core.fewshot import Example, FewShotRetriever
from core.generate import SQLGenerator
from core.pipeline import Pipeline, PipelineFlags
from tests.test_fewshot import LetterEmbedder

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"
STAGES = ["schema", "retrieval", "grounding", "fewshot", "generate_and_execute"]


class RecordingGenerator(SQLGenerator):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, question, schema_text, examples, grounding_hints, error_feedback=None) -> str:
        self.calls.append(
            {"schema": schema_text, "examples": examples, "hints": grounding_hints, "feedback": error_feedback}
        )
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


EXAMPLES = [Example(1, "How many teams are in the NBA?", "SELECT COUNT(*) FROM team;", ("team",), "easy")]


@pytest.fixture(scope="module")
def executor():
    with Executor(READONLY_URL, row_cap=50) as ex:
        yield ex


def make(executor, outputs: list[str]) -> tuple[Pipeline, RecordingGenerator]:
    generator = RecordingGenerator(outputs)
    fewshot = FewShotRetriever(EXAMPLES, LetterEmbedder())
    return Pipeline(generator=generator, conninfo=READONLY_URL, executor=executor, fewshot=fewshot), generator


QUESTION = "How many points did Nikola Jokic score in the 2023-24 regular season?"
SQL = (
    "SELECT SUM(pts) FROM player_game_log WHERE player_id = 203999 "
    "AND season = '2023-24' AND season_type = 'Regular Season'"
)


def test_full_pipeline_returns_rows_and_a_trace(executor) -> None:
    pipeline, generator = make(executor, [SQL])
    result = pipeline.answer(QUESTION)
    assert result.ok
    assert result.row_count == 1
    assert result.sql.endswith("LIMIT 50")
    assert [s.stage for s in result.trace] == STAGES
    assert all(s.elapsed_ms >= 0 for s in result.trace)
    assert result.total_ms >= sum(s.elapsed_ms for s in result.trace) * 0.9
    call = generator.calls[0]
    assert "player.player_id = 203999" in call["hints"]
    assert "TABLE player_game_log" in call["schema"]
    assert call["examples"] == EXAMPLES
    retrieval = result.trace[1].detail
    assert "player_game_log" in retrieval["retrieved"]
    assert "player" in retrieval["tables"]
    assert result.to_dict()["trace"][0]["stage"] == "schema"


def test_flags_disable_each_stage(executor) -> None:
    pipeline, generator = make(executor, [SQL])
    flags = PipelineFlags(retrieval=False, grounding=False, fewshot=False)
    result = pipeline.answer(QUESTION, flags)
    assert result.ok
    call = generator.calls[0]
    assert call["hints"] == ""
    assert call["examples"] == []
    for table in ("game", "player", "player_game_log", "team", "team_game_log"):
        assert f"TABLE {table}\n" in call["schema"] + "\n"
    assert [s.detail["enabled"] for s in result.trace[1:4]] == [False, False, False]


def test_repair_runs_on_failure_and_can_be_disabled(executor) -> None:
    pipeline, generator = make(executor, ["SELECT points FROM player_game_log", SQL])
    repaired = pipeline.answer(QUESTION)
    assert repaired.ok
    assert [a.status for a in repaired.attempts] == ["validation_error", "ok"]
    assert generator.calls[1]["feedback"].error == "Unknown column points."

    pipeline, _ = make(executor, ["SELECT points FROM player_game_log", SQL])
    unrepaired = pipeline.answer(QUESTION, PipelineFlags(repair=False))
    assert not unrepaired.ok
    assert unrepaired.error == "Unknown column points."
    assert unrepaired.sql == "SELECT points FROM player_game_log"
    assert unrepaired.rows == []
    assert len(unrepaired.attempts) == 1
