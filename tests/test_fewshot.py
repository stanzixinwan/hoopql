"""Few-shot retrieval returns relevant, executable exemplars."""

from __future__ import annotations

import numpy as np
import pytest

from core.embed import default_embedder, normalize
from core.fewshot import Example, FewShotRetriever, examples_text, load_pool


@pytest.fixture(scope="module")
def retriever() -> FewShotRetriever:
    return FewShotRetriever(load_pool(), default_embedder())


RELEVANCE_CASES = [
    ("How many points did Jokic average in 2024-25?", "player_game_log"),
    ("Which team won the most home games in 2023-24?", "game"),
    ("How many playoff games were played in 2021-22?", "game"),
    ("Which players shot the best from three last season?", "player_game_log"),
    ("How many rebounds did Giannis average at home vs away?", "player_game_log"),
]


@pytest.mark.parametrize(("question", "table"), RELEVANCE_CASES)
def test_nearest_exemplars_use_the_right_table(retriever, question: str, table: str) -> None:
    top = retriever.retrieve(question, k=3)
    assert len(top) == 3
    assert table in top[0].tables or table in top[1].tables


def test_exact_pool_question_ranks_first(retriever) -> None:
    assert retriever.retrieve("How many playoff games were played in the 2021-22 season?", k=1)[0].id == 131


class LetterEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 26), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text.lower():
                if "a" <= char <= "z":
                    vectors[row, ord(char) - 97] += 1
        return normalize(vectors)


EXAMPLES = [
    Example(1, "points scored", "SELECT 1;", ("player_game_log",), "easy"),
    Example(2, "rebounds grabbed", "SELECT 2;", ("player_game_log",), "easy"),
    Example(3, "home wins", "SELECT 3;", ("game",), "easy"),
]


def test_exclude_ids_skips_the_question_being_evaluated() -> None:
    retriever = FewShotRetriever(EXAMPLES, LetterEmbedder())
    assert retriever.retrieve("points scored", k=1)[0].id == 1
    picked = retriever.retrieve("points scored", k=2, exclude_ids=frozenset({1}))
    assert len(picked) == 2
    assert 1 not in [e.id for e in picked]


def test_k_zero_and_formatting() -> None:
    retriever = FewShotRetriever(EXAMPLES, LetterEmbedder())
    assert retriever.retrieve("points", k=0) == []
    assert examples_text(EXAMPLES[:1]) == "Question: points scored\nSQL: SELECT 1;"
