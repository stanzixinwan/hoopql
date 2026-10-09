"""Hybrid table retrieval built in memory from the live schema."""

from __future__ import annotations

import time

import numpy as np
import psycopg
import pytest

from core.embed import default_embedder, normalize
from core.retrieval import (
    BM25SchemaRetriever,
    HybridSchemaRetriever,
    SchemaRetriever,
    build_documents,
    build_retriever,
)
from core.schema import introspect

READONLY_URL = "postgresql://hoopql_readonly:hoopql_readonly@localhost:5432/hoopql"


@pytest.fixture(scope="module")
def schema():
    with psycopg.connect(READONLY_URL) as conn:
        return introspect(conn)


@pytest.fixture(scope="module")
def embedder():
    return default_embedder()


def test_startup_builds_both_indices_quickly(schema, embedder) -> None:
    started = time.perf_counter()
    retriever = build_retriever(schema, "hybrid", embedder)
    elapsed = time.perf_counter() - started
    assert isinstance(retriever, HybridSchemaRetriever)
    assert elapsed < 2.0, f"index build took {elapsed:.2f}s"


def test_jokic_points_question_finds_player_game_log(schema, embedder) -> None:
    retriever = build_retriever(schema, "hybrid", embedder)
    top = retriever.retrieve("How many points did Jokic average in 2024-25?", top_k=3)
    assert "player_game_log" in top


def test_team_record_question_finds_team_tables(schema, embedder) -> None:
    retriever = build_retriever(schema, "hybrid", embedder)
    top = retriever.retrieve("What was the Celtics home record in the 2023-24 playoffs?", top_k=3)
    assert {"team_game_log", "game"} & set(top)


class FakeEmbedder:
    """Bag-of-letters vectors: deterministic and fast."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 26), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text.lower():
                if "a" <= char <= "z":
                    vectors[row, ord(char) - 97] += 1
        return normalize(vectors)


DOCS = [
    {"table": "alpha", "text": "alpha points scored"},
    {"table": "beta", "text": "beta rebounds"},
    {"table": "gamma", "text": "gamma assists"},
]


def test_rrf_ranks_every_table_once() -> None:
    hybrid = HybridSchemaRetriever(SchemaRetriever(DOCS, FakeEmbedder()), BM25SchemaRetriever(DOCS))
    ranked = hybrid.retrieve_ranked("points scored")
    assert [table for table, _ in ranked][0] == "alpha"
    assert sorted(table for table, _ in ranked) == ["alpha", "beta", "gamma"]


def test_documents_come_from_the_schema(schema) -> None:
    documents = build_documents(schema)
    tables = {d["table"] for d in documents}
    assert "player_game_log" in tables
    assert "etl_run" not in tables
    text = next(d["text"] for d in documents if d["table"] == "player_game_log")
    assert "Three-point percentage" in text
