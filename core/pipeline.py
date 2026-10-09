"""One entry point for the query pipeline, with a per-stage trace and ablation flags.

Stages: schema, retrieval, grounding, fewshot, then generate/validate/execute
inside the repair loop. Each optional stage can be turned off through
PipelineFlags so the ablation can measure what it contributes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg

from core.config import database_url
from core.execute import Executor
from core.fewshot import FewShotRetriever
from core.generate import SQLGenerator, make_generator
from core.grounding import ground, hints_text
from core.repair import MAX_RETRIES, Attempt, generate_with_repair
from core.retrieval import SchemaRetrieverProtocol, build_retriever
from core.safety import validate
from core.schema import Schema, SchemaCache, schema_text


@dataclass(frozen=True)
class PipelineFlags:
    retrieval: bool = True
    grounding: bool = True
    fewshot: bool = True
    repair: bool = True
    top_k_tables: int = 3
    k_examples: int = 3
    max_retries: int = MAX_RETRIES


@dataclass(frozen=True)
class StageTrace:
    stage: str
    elapsed_ms: float
    detail: dict[str, Any]


@dataclass
class Answer:
    question: str
    sql: str | None
    columns: list[str]
    rows: list[tuple]
    row_count: int
    truncated: bool
    error: str | None
    flags: PipelineFlags
    trace: list[StageTrace] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def with_join_tables(schema: Schema, tables: list[str]) -> list[str]:
    """Add tables reachable by one foreign key so retrieved tables can be joined for names."""
    selected = list(tables)
    for name in tables:
        for fk in schema.tables[name].foreign_keys:
            if fk.ref_table not in selected and fk.ref_table in schema.tables:
                selected.append(fk.ref_table)
    return selected


class _Timer:
    def __init__(self, trace: list[StageTrace], stage: str) -> None:
        self.trace = trace
        self.stage = stage
        self.detail: dict[str, Any] = {}

    def __enter__(self) -> _Timer:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = round((time.perf_counter() - self.started) * 1000, 1)
        self.trace.append(StageTrace(self.stage, elapsed, self.detail))


class Pipeline:
    def __init__(
        self,
        generator: SQLGenerator | None = None,
        conninfo: str | None = None,
        executor: Executor | None = None,
        fewshot: FewShotRetriever | None = None,
    ) -> None:
        self.conninfo = conninfo or database_url()
        self.executor = executor or Executor(self.conninfo)
        self._generator = generator
        self._fewshot = fewshot
        self._schema_cache = SchemaCache(connect=lambda: psycopg.connect(self.conninfo))
        self._retriever: SchemaRetrieverProtocol | None = None
        self._retriever_schema: Schema | None = None
        self._lock = threading.Lock()

    @property
    def generator(self) -> SQLGenerator:
        if self._generator is None:
            self._generator = make_generator()
        return self._generator

    @property
    def fewshot(self) -> FewShotRetriever:
        if self._fewshot is None:
            self._fewshot = FewShotRetriever()
        return self._fewshot

    def _table_retriever(self, schema: Schema) -> SchemaRetrieverProtocol:
        with self._lock:
            if self._retriever is None or self._retriever_schema is not schema:
                self._retriever = build_retriever(schema)
                self._retriever_schema = schema
            return self._retriever

    def answer(self, question: str, flags: PipelineFlags | None = None) -> Answer:
        flags = flags or PipelineFlags()
        started = time.perf_counter()
        trace: list[StageTrace] = []

        with _Timer(trace, "schema") as t:
            schema = self._schema_cache.get()
            t.detail["tables"] = len(schema.tables)

        with _Timer(trace, "retrieval") as t:
            t.detail["enabled"] = flags.retrieval
            if flags.retrieval:
                retrieved = self._table_retriever(schema).retrieve(question, top_k=flags.top_k_tables)
                tables = with_join_tables(schema, retrieved)
                t.detail["retrieved"] = retrieved
            else:
                tables = schema.table_names()
            t.detail["tables"] = tables
            prompt_schema = schema_text(schema, tables)

        with _Timer(trace, "grounding") as t:
            t.detail["enabled"] = flags.grounding
            groundings = []
            if flags.grounding:
                with self.executor.pool.connection() as conn:
                    groundings = ground(question, conn)
            hints = hints_text(groundings)
            t.detail["hints"] = [g.hint() for g in groundings]

        with _Timer(trace, "fewshot") as t:
            t.detail["enabled"] = flags.fewshot
            examples = self.fewshot.retrieve(question, k=flags.k_examples) if flags.fewshot else []
            t.detail["example_ids"] = [e.id for e in examples]

        with _Timer(trace, "generate_and_execute") as t:
            max_retries = flags.max_retries if flags.repair else 0
            outcome = generate_with_repair(
                self.generator,
                lambda sql: validate(sql, schema, row_limit=self.executor.row_cap),
                self.executor.run,
                question,
                prompt_schema,
                examples,
                hints,
                max_retries=max_retries,
            )
            t.detail["max_retries"] = max_retries
            t.detail["attempts"] = len(outcome.attempts)
            t.detail["attempt_ms"] = [a.elapsed_ms for a in outcome.attempts]
            if outcome.result is not None:
                t.detail["execution_ms"] = outcome.result.elapsed_ms

        result = outcome.result
        last = outcome.attempts[-1] if outcome.attempts else None
        return Answer(
            question=question,
            sql=outcome.validated.sql if outcome.validated else (last.sql if last else None),
            columns=result.columns if result else [],
            rows=result.rows if result else [],
            row_count=result.row_count if result else 0,
            truncated=result.truncated if result else False,
            error=None if result else (last.error if last else "No attempt was made."),
            flags=flags,
            trace=trace,
            attempts=outcome.attempts,
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def close(self) -> None:
        self.executor.close()


_default: Pipeline | None = None
_default_lock = threading.Lock()


def answer(question: str, flags: PipelineFlags | None = None) -> Answer:
    """Answer a question with the process-wide pipeline built from .env settings."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Pipeline()
    return _default.answer(question, flags)
