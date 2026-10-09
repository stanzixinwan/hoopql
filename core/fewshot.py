"""Few-shot exemplar retrieval: the k pool questions nearest to the user's question."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.embed import Embedder, default_embedder

FEWSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "fewshot"
POOL_FILES = ("pool.json", "player_pool.json")


@dataclass(frozen=True)
class Example:
    id: int
    question: str
    sql: str
    tables: tuple[str, ...]
    difficulty: str


def load_pool(directory: Path = FEWSHOT_DIR) -> list[Example]:
    examples = []
    for name in POOL_FILES:
        for item in json.loads((directory / name).read_text()):
            examples.append(
                Example(
                    id=item["id"],
                    question=item["question"],
                    sql=item["sql"],
                    tables=tuple(item["tables"]),
                    difficulty=item["difficulty"],
                )
            )
    return examples


class FewShotRetriever:
    """Dense nearest-neighbour search over exemplar questions."""

    def __init__(self, examples: list[Example] | None = None, embedder: Embedder | None = None) -> None:
        self.examples = examples if examples is not None else load_pool()
        self.embedder = embedder or default_embedder()
        self._matrix = self.embedder.embed([e.question for e in self.examples])

    def retrieve(self, question: str, k: int = 3, exclude_ids: frozenset[int] = frozenset()) -> list[Example]:
        """Return the k most similar exemplars; `exclude_ids` keeps eval questions out of their own prompt."""
        if k <= 0:
            return []
        scores = self._matrix @ self.embedder.embed([question])[0]
        picked = []
        for i in np.argsort(-scores):
            example = self.examples[i]
            if example.id in exclude_ids:
                continue
            picked.append(example)
            if len(picked) == k:
                break
        return picked


def examples_text(examples: list[Example]) -> str:
    return "\n\n".join(f"Question: {e.question}\nSQL: {e.sql}" for e in examples)
