"""
rag.py — RAG-based schema retrieval for HoopQL.

Backends:
  - dense: sentence-transformers + FAISS (inner product on normalized vectors)
  - bm25:  BM25Okapi over the same per-table schema documents
  - hybrid: RRF fusion of dense + BM25 rankings (no extra re-ranker)

Usage:
    python -m src.rag --build              # dense FAISS index
    python -m src.rag --build-bm25         # BM25 index (same documents)
    python -m src.rag --eval-retrieval --top-k 3 --backend dense
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np

from src.data_utils import (
    NBA_CORE_TABLES,
    build_nba_schema_documents,
    format_input,
    format_target,
    get_sqlite_schema,
    serialize_schema,
)

INDEX_DIR = Path("models/rag_index")
BM25_PICKLE = INDEX_DIR / "bm25.pkl"
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9_]+", text)


class SchemaRetrieverProtocol(Protocol):
    """Anything that can retrieve top-k table names for a question."""

    documents: list[dict]

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        ...


class SchemaRetriever:
    """sentence-transformer + FAISS retriever for NBA table schemas."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.documents: Optional[list[dict]] = None

    def _load_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name)

    def build(
        self,
        db_path: str = "data/raw/nba.sqlite",
        tables: Optional[list[str]] = None,
        save_dir: Path = INDEX_DIR,
    ):
        """Build FAISS index from NBA schema documents."""
        import faiss

        self._load_encoder()
        self.documents = build_nba_schema_documents(db_path, tables)
        texts = [d["text"] for d in self.documents]

        print(f"Encoding {len(texts)} table schemas with {self.model_name}...")
        embeddings = self.encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "schema.faiss"))
        with open(save_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Index built: {len(self.documents)} tables, dim={embeddings.shape[1]}")
        print(f"Saved to {save_dir}/")

    def load(self, save_dir: Path = INDEX_DIR):
        """Load a previously built index."""
        import faiss

        self._load_encoder()
        self.index = faiss.read_index(str(save_dir / "schema.faiss"))
        with open(save_dir / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Call .load() or .build() first.")
        ranked = self.retrieve_ranked(question)
        return [t for t, _ in ranked[:top_k]]

    def retrieve_ranked(self, question: str) -> list[tuple[str, float]]:
        """All tables sorted by dense score (descending)."""
        if self.index is None or self.documents is None:
            raise RuntimeError("Index not loaded. Call .load() or .build() first.")
        n = len(self.documents)
        query_emb = self.encoder.encode([question], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        scores, indices = self.index.search(query_emb, n)
        return [
            (self.documents[i]["table"], float(scores[0][j]))
            for j, i in enumerate(indices[0])
        ]

    def retrieve_with_scores(self, question: str, top_k: int = 3) -> list[tuple]:
        ranked = self.retrieve_ranked(question)
        return ranked[:top_k]


class BM25SchemaRetriever:
    """BM25 over the same table documents as dense retrieval."""

    def __init__(self):
        self.documents: Optional[list[dict]] = None
        self._bm25 = None
        self._tokenized_corpus: Optional[list[list[str]]] = None

    def build(
        self,
        db_path: str = "data/raw/nba.sqlite",
        tables: Optional[list[str]] = None,
        save_path: Path = BM25_PICKLE,
    ):
        from rank_bm25 import BM25Okapi

        self.documents = build_nba_schema_documents(db_path, tables)
        # Strengthen signal: repeat table name in pseudo-document
        texts = []
        for d in self.documents:
            t = d["table"]
            texts.append(f"{t} {t} {d['text']}")

        self._tokenized_corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenized_corpus": self._tokenized_corpus,
                    "bm25": self._bm25,
                },
                f,
            )
        print(f"BM25 index built: {len(self.documents)} tables -> {save_path}")

    def load(self, save_path: Path = BM25_PICKLE):
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = data["bm25"]

    def retrieve_ranked(self, question: str) -> list[tuple[str, float]]:
        if self._bm25 is None or self.documents is None:
            raise RuntimeError("BM25 not loaded. Call .load() or .build() first.")
        q = _tokenize(question)
        scores = self._bm25.get_scores(q)
        order = np.argsort(-scores)
        return [(self.documents[i]["table"], float(scores[i])) for i in order]

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        ranked = self.retrieve_ranked(question)
        return [t for t, _ in ranked[:top_k]]


class HybridSchemaRetriever:
    """RRF over BM25 + dense full rankings."""

    def __init__(
        self,
        dense: SchemaRetriever,
        bm25: BM25SchemaRetriever,
        rrf_k: int = RRF_K,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.documents = dense.documents

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        ranked_bm25 = self.bm25.retrieve_ranked(question)
        ranked_dense = self.dense.retrieve_ranked(question)
        rank_bm25 = {t: r + 1 for r, (t, _) in enumerate(ranked_bm25)}
        rank_dense = {t: r + 1 for r, (t, _) in enumerate(ranked_dense)}
        tables = {d["table"] for d in self.dense.documents}
        rrf_scores: dict[str, float] = {}
        for t in tables:
            s = 0.0
            if t in rank_bm25:
                s += 1.0 / (self.rrf_k + rank_bm25[t])
            if t in rank_dense:
                s += 1.0 / (self.rrf_k + rank_dense[t])
            rrf_scores[t] = s
        sorted_tables = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        return sorted_tables[:top_k]


def get_schema_retriever(
    backend: str,
    *,
    dense_model: str = "all-MiniLM-L6-v2",
    index_dir: Path = INDEX_DIR,
    bm25_path: Path = BM25_PICKLE,
) -> Union[SchemaRetriever, BM25SchemaRetriever, HybridSchemaRetriever]:
    """
    Factory for retrieval backends. Dense + hybrid require FAISS index on disk;
    BM25 requires bm25.pkl; hybrid requires both.
    """
    b = backend.strip().lower()
    if b == "dense":
        r = SchemaRetriever(model_name=dense_model)
        r.load(index_dir)
        return r
    if b == "bm25":
        r = BM25SchemaRetriever()
        r.load(bm25_path)
        return r
    if b == "hybrid":
        dense = SchemaRetriever(model_name=dense_model)
        dense.load(index_dir)
        bm25 = BM25SchemaRetriever()
        bm25.load(bm25_path)
        return HybridSchemaRetriever(dense, bm25)
    raise ValueError(f"Unknown rag backend: {backend!r} (use dense, bm25, hybrid)")


# ──────────────────────────────────────────────
# RAG-aware NBA dataset loader
# ──────────────────────────────────────────────


def load_nba_dataset_with_rag(
    questions_path: str = "data/nba/nba_questions.json",
    db_path: str = "data/raw/nba.sqlite",
    retriever: Optional[SchemaRetrieverProtocol] = None,
    backend: str = "dense",
    top_k: int = 3,
    include_types: bool = True,
) -> list[dict]:
    """
    Load NBA examples with RAG-retrieved schemas.

    Pass either ``retriever`` or ``backend`` (dense | bm25 | hybrid).
    """
    if retriever is None:
        retriever = get_schema_retriever(backend)

    full_schema = get_sqlite_schema(db_path, tables=NBA_CORE_TABLES)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    data = []
    for q in questions:
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        schema = {t: full_schema[t] for t in retrieved if t in full_schema}

        gold_tables = set(q.get("tables_used", []))
        retrieved_set = set(retrieved)
        if gold_tables:
            recall = len(gold_tables & retrieved_set) / len(gold_tables)
            perfect_retrieval = gold_tables.issubset(retrieved_set)
        else:
            recall = 1.0
            perfect_retrieval = True

        schema_text = serialize_schema(schema, include_types=include_types)
        data.append(
            {
                "input_text": format_input(q["question"], schema_text),
                "target_text": format_target(q["query"]),
                "db_id": "nba",
                "difficulty": q.get("difficulty", "unknown"),
                "tables_used": q.get("tables_used", []),
                "sql_components": q.get("sql_components", []),
                "question": q["question"],
                "gold_sql": q["query"],
                "retrieved_tables": retrieved,
                "retrieval_recall": recall,
                "perfect_retrieval": perfect_retrieval,
            }
        )

    return data


# ──────────────────────────────────────────────
# Retrieval-only evaluation (no SQL generation)
# ──────────────────────────────────────────────


def evaluate_retrieval(
    questions_path: Optional[str] = None,
    top_k: int = 3,
    retriever: Optional[SchemaRetrieverProtocol] = None,
    backend: str = "dense",
    questions: Optional[list] = None,
) -> dict:
    """Mean recall@k and perfect-retrieval rate over questions with gold tables."""
    from collections import defaultdict

    if retriever is None:
        retriever = get_schema_retriever(backend)

    if questions is None:
        if not questions_path:
            raise ValueError("Provide questions_path or questions")
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

    by_diff = defaultdict(lambda: {"total": 0, "recall_sum": 0.0, "perfect": 0})

    for q in questions:
        retrieved = set(retriever.retrieve(q["question"], top_k=top_k))
        gold = set(q.get("tables_used", []))
        if not gold:
            continue
        recall = len(gold & retrieved) / len(gold)
        perfect = int(gold.issubset(retrieved))
        diff = q.get("difficulty", "unknown")
        by_diff[diff]["total"] += 1
        by_diff[diff]["recall_sum"] += recall
        by_diff[diff]["perfect"] += perfect

    total = sum(d["total"] for d in by_diff.values())
    if total == 0:
        return {"overall_recall": 0.0, "perfect_rate": 0.0, "n": 0}
    overall_recall = sum(d["recall_sum"] for d in by_diff.values()) / total
    overall_perfect = sum(d["perfect"] for d in by_diff.values()) / total

    print(f"\n=== Retrieval backend={backend!r} @ top-{top_k} ({total} examples) ===")
    print(f"  Avg recall:        {overall_recall:.1%}")
    print(
        f"  Perfect retrieval: {overall_perfect:.1%}  (all gold tables in top-{top_k})"
    )
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard"]:
        if diff in by_diff and by_diff[diff]["total"] > 0:
            d = by_diff[diff]
            print(
                f"    {diff:11s}: recall {d['recall_sum']/d['total']:.1%} | "
                f"perfect {d['perfect']}/{d['total']} ({d['perfect']/d['total']:.1%})"
            )

    return {
        "overall_recall": overall_recall,
        "perfect_rate": overall_perfect,
        "n": total,
        "backend": backend,
        "top_k": top_k,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build FAISS dense index")
    parser.add_argument(
        "--build-bm25", action="store_true", help="Build BM25 index (same documents)"
    )
    parser.add_argument(
        "--eval-retrieval", action="store_true", help="Evaluate retrieval only"
    )
    parser.add_argument("--demo", action="store_true", help="Demo dense retrieval")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--backend",
        default="dense",
        choices=["dense", "bm25", "hybrid"],
        help="Retriever for --eval-retrieval",
    )
    parser.add_argument("--db-path", default="data/raw/nba.sqlite")
    parser.add_argument("--questions", default="data/nba/nba_questions.json")
    args = parser.parse_args()

    if args.build:
        retriever = SchemaRetriever()
        retriever.build(db_path=args.db_path)

    if args.build_bm25:
        bm25r = BM25SchemaRetriever()
        bm25r.build(db_path=args.db_path)

    if args.eval_retrieval:
        evaluate_retrieval(args.questions, top_k=args.top_k, backend=args.backend)

    if args.demo:
        retriever = SchemaRetriever()
        retriever.load()
        samples = [
            "How many teams are in the NBA?",
            "Which team had the highest 3-point shooting percentage at home in playoffs?",
            "List players drafted in the first round of 2018.",
            "What arena does the team_details list for the Lakers?",
        ]
        for q in samples:
            results = retriever.retrieve_with_scores(q, top_k=args.top_k)
            print(f"\nQ: {q}")
            for table, score in results:
                print(f"   [{score:.3f}] {table}")
