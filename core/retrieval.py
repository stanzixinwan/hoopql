"""Table retrieval: dense, BM25, and their reciprocal-rank fusion.

Ported from research/rag.py. Documents now come from the live schema and the
indexes are built in memory, so nothing is pickled to disk.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from core.embed import Embedder, default_embedder
from core.schema import Schema, table_text

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_documents(schema: Schema) -> list[dict]:
    return [{"table": name, "text": table_text(schema.tables[name])} for name in schema.table_names()]


class SchemaRetrieverProtocol(Protocol):
    documents: list[dict]

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        ...

    def retrieve_ranked(self, question: str) -> list[tuple[str, float]]:
        ...


class SchemaRetriever:
    """Dense retrieval: cosine similarity over normalized embeddings."""

    def __init__(self, documents: list[dict], embedder: Embedder | None = None) -> None:
        self.documents = documents
        self.embedder = embedder or default_embedder()
        self._matrix = self.embedder.embed([d["text"] for d in documents])

    def retrieve_ranked(self, question: str) -> list[tuple[str, float]]:
        query = self.embedder.embed([question])[0]
        scores = self._matrix @ query
        order = np.argsort(-scores)
        return [(self.documents[i]["table"], float(scores[i])) for i in order]

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        return [table for table, _ in self.retrieve_ranked(question)[:top_k]]


class BM25SchemaRetriever:
    """BM25 over the same table documents as dense retrieval."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        # Repeat the table name to strengthen its signal, as in the research retriever.
        corpus = [_tokenize(f"{d['table']} {d['table']} {d['text']}") for d in documents]
        self._bm25 = BM25Okapi(corpus)

    def retrieve_ranked(self, question: str) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(question))
        order = np.argsort(-scores)
        return [(self.documents[i]["table"], float(scores[i])) for i in order]

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        return [table for table, _ in self.retrieve_ranked(question)[:top_k]]


class HybridSchemaRetriever:
    """Reciprocal-rank fusion of the BM25 and dense rankings."""

    def __init__(self, dense: SchemaRetriever, bm25: BM25SchemaRetriever, rrf_k: int = RRF_K) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.documents = dense.documents

    def retrieve_ranked(self, question: str) -> list[tuple[str, float]]:
        rank_bm25 = {t: r + 1 for r, (t, _) in enumerate(self.bm25.retrieve_ranked(question))}
        rank_dense = {t: r + 1 for r, (t, _) in enumerate(self.dense.retrieve_ranked(question))}
        scores: dict[str, float] = {}
        for doc in self.documents:
            table = doc["table"]
            score = 0.0
            if table in rank_bm25:
                score += 1.0 / (self.rrf_k + rank_bm25[table])
            if table in rank_dense:
                score += 1.0 / (self.rrf_k + rank_dense[table])
            scores[table] = score
        return sorted(scores.items(), key=lambda item: -item[1])

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        return [table for table, _ in self.retrieve_ranked(question)[:top_k]]


def build_retriever(
    schema: Schema,
    backend: str = "hybrid",
    embedder: Embedder | None = None,
) -> SchemaRetrieverProtocol:
    documents = build_documents(schema)
    backend = backend.strip().lower()
    if backend == "bm25":
        return BM25SchemaRetriever(documents)
    if backend == "dense":
        return SchemaRetriever(documents, embedder)
    if backend == "hybrid":
        return HybridSchemaRetriever(SchemaRetriever(documents, embedder), BM25SchemaRetriever(documents))
    raise ValueError(f"Unknown retrieval backend: {backend!r} (use dense, bm25, hybrid)")
