"""
data_utils.py — Unified data pipeline for HoopQL (Spider training and NBA evaluation).

Key fix in this version:
    - load_nba_dataset() now supports use_oracle_tables=True, which restricts
      each example's schema to only the tables in `tables_used`. This brings
      NBA input length close to Spider's training distribution and isolates
      "model can't handle long schemas" from "model can't handle NBA domain".
"""

import json
import sqlite3
from typing import Optional


# ──────────────────────────────────────────────
# 1. Schema serialization
# ──────────────────────────────────────────────

def get_sqlite_schema(db_path: str, tables: Optional[list[str]] = None) -> dict:
    """Extract schema from a SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if tables is None:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]

    schema = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [{"column": row[1], "type": row[2]} for row in cur.fetchall()]
        if cols:
            schema[table] = cols

    conn.close()
    return schema


def _normalize_type(sqlite_type: str) -> str:
    """Map SQLite types to Spider-style: 'number' or 'text'."""
    if not sqlite_type:
        return "text"
    t = sqlite_type.upper()
    if any(x in t for x in ["INT", "REAL", "NUMERIC", "DOUBLE", "FLOAT"]):
        return "number"
    return "text"


def serialize_schema(schema: dict, include_types: bool = True) -> str:
    """Spider-compatible: 'table1: col1 (type), col2 (type) | table2: ...'"""
    parts = []
    for table_name, columns in schema.items():
        col_strs = []
        for c in columns:
            if include_types:
                col_strs.append(f"{c['column']} ({_normalize_type(c.get('type', ''))})")
            else:
                col_strs.append(c["column"])
        parts.append(f"{table_name}: {', '.join(col_strs)}")
    return " | ".join(parts)


def serialize_schema_for_table(schema: dict, table_name: str,
                                include_types: bool = True) -> str:
    if table_name not in schema:
        return ""
    columns = schema[table_name]
    col_strs = []
    for c in columns:
        if include_types:
            col_strs.append(f"{c['column']} ({_normalize_type(c.get('type', ''))})")
        else:
            col_strs.append(c["column"])
    return f"{table_name}: {', '.join(col_strs)}"


# ──────────────────────────────────────────────
# 2. Input/output formatting
# ──────────────────────────────────────────────

def format_input(question: str, schema_text: str) -> str:
    return f"translate to SQL: {question} | {schema_text}"


def format_target(sql: str) -> str:
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def format_example(question: str, schema: dict, sql: str,
                   include_types: bool = True) -> dict:
    """Format (question, schema, sql) into model-ready strings.
    Default include_types=True to match Spider training format."""
    schema_text = serialize_schema(schema, include_types=include_types)
    return {
        "input_text": format_input(question, schema_text),
        "target_text": format_target(sql),
    }


# ──────────────────────────────────────────────
# 3. Spider loader
# ──────────────────────────────────────────────

SPIDER_DATASET_NAME = "xlangai/spider"
SPIDER_SCHEMA_DATASET_NAME = "richardr1126/spider-schema"
SPIDER_LEGACY_SCHEMA_DATASET_NAME = "SuperMax991/spider-text2sql"


def _schema_from_spider_tables(table_examples: list[dict]) -> dict:
    schemas = {}
    for ex in table_examples:
        table_names = ex["table_names_original"]
        column_types = ex.get("column_types", [])
        schema = {t: [] for t in table_names}

        for col_idx, (table_idx, col_name) in enumerate(ex["column_names_original"]):
            if table_idx < 0:
                continue
            table_name = table_names[table_idx]
            col_type = column_types[col_idx] if col_idx < len(column_types) else ""
            schema[table_name].append({"column": col_name, "type": col_type})

        schemas[ex["db_id"]] = schema
    return schemas


def _load_spider_schemas(spider_dataset) -> dict:
    first_split = next(iter(spider_dataset))
    first_example = spider_dataset[first_split][0]

    if "table_names_original" not in first_example:
        raise ValueError(
            "Spider dataset does not include schema metadata. Use a dataset with "
            "db_schema or table_names_original fields."
        )

    schemas = {}
    for split in spider_dataset:
        for ex in spider_dataset[split]:
            db_id = ex["db_id"]
            if db_id in schemas:
                continue
            schemas[db_id] = _schema_from_spider_tables([ex])[db_id]
    return schemas


def _load_spider_schema_texts() -> dict:
    """Load db_id -> serialized schema text for Spider datasets without schemas."""
    from datasets import load_dataset

    schema_dataset = load_dataset(SPIDER_SCHEMA_DATASET_NAME)
    schemas = {}
    for split in schema_dataset:
        for ex in schema_dataset[split]:
            if "Schema (values (type))" in ex:
                schemas.setdefault(ex["db_id"], ex["Schema (values (type))"])

    legacy_dataset = load_dataset(SPIDER_LEGACY_SCHEMA_DATASET_NAME)
    for split in legacy_dataset:
        for ex in legacy_dataset[split]:
            if "db_schema" in ex:
                schemas.setdefault(ex["db_id"], ex["db_schema"])

    if not schemas:
        raise ValueError(
            f"Spider dataset does not include schema metadata, and "
            f"{SPIDER_SCHEMA_DATASET_NAME} did not provide schema fields."
        )
    return schemas


def load_spider_splits(include_types: bool = True, max_examples: Optional[int] = None):
    """Load Spider train and dev splits, formatted for model input."""
    from datasets import load_dataset
    spider = load_dataset(SPIDER_DATASET_NAME)
    has_db_schema = "db_schema" in spider["train"].column_names
    has_table_schema = "table_names_original" in spider["train"].column_names
    schema_texts = {} if has_db_schema or has_table_schema else _load_spider_schema_texts()
    schemas = {} if has_db_schema or schema_texts else _load_spider_schemas(spider)

    def process_split(split_name):
        data = []
        for ex in spider[split_name]:
            db_id = ex["db_id"]
            if "db_schema" in ex:
                formatted = {
                    "input_text": format_input(ex["question"], ex["db_schema"]),
                    "target_text": format_target(ex["query"]),
                }
            elif db_id in schema_texts:
                formatted = {
                    "input_text": format_input(ex["question"], schema_texts[db_id]),
                    "target_text": format_target(ex["query"]),
                }
            else:
                schema = schemas.get(db_id, {})
                formatted = format_example(
                    question=ex["question"],
                    schema=schema,
                    sql=ex["query"],
                    include_types=include_types,
                )
            formatted["db_id"] = db_id
            formatted["difficulty"] = ex.get("difficulty", "unknown")
            formatted["question"] = ex["question"]
            formatted["gold_sql"] = ex["query"]
            data.append(formatted)
            if max_examples and len(data) >= max_examples:
                break
        return data

    train_data = process_split("train")
    dev_split = "validation" if "validation" in spider else "test"
    dev_data = process_split(dev_split)
    return train_data, dev_data


# ──────────────────────────────────────────────
# 4. NBA loader
# ──────────────────────────────────────────────

# All tables that NBA gold SQL queries can reference.
# Verified against tables_used field in nba_questions.json.
NBA_CORE_TABLES = [
    "team",
    "player",
    "common_player_info",
    "game",
    "draft_history",
    "draft_combine_stats",
    "team_details",
    "team_history",
    "game_summary",
    "game_info",
    "line_score",
    "other_stats",
    "officials",
    "inactive_players",
    "play_by_play",  # used in Q89 only; included so that query has its schema
]


def load_nba_dataset(
    questions_path: str = "data/nba/nba_questions.json",
    db_path: str = "data/raw/nba.sqlite",
    include_types: bool = True,
    tables: Optional[list[str]] = None,
    use_oracle_tables: bool = False,
) -> list[dict]:
    """
    Load NBA question/SQL pairs, formatted for model input.

    Args:
        use_oracle_tables: If True, each example only sees schemas of tables
            listed in its `tables_used` field. This mimics perfect RAG
            retrieval and brings input length close to Spider distribution.
            Use this to validate that the trained model actually learned
            text-to-SQL (not just memorized Spider patterns).

    Returns list[dict] with keys:
        input_text, target_text, db_id, difficulty, tables_used,
        sql_components, question, gold_sql
    """
    tables = tables or NBA_CORE_TABLES
    full_schema = get_sqlite_schema(db_path, tables=tables)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    data = []
    for q in questions:
        if use_oracle_tables and q.get("tables_used"):
            schema = {t: full_schema[t] for t in q["tables_used"] if t in full_schema}
            if not schema:  # safety net
                schema = full_schema
        else:
            schema = full_schema

        formatted = format_example(
            question=q["question"],
            schema=schema,
            sql=q["query"],
            include_types=include_types,
        )
        formatted["db_id"] = "nba"
        formatted["difficulty"] = q.get("difficulty", "unknown")
        formatted["tables_used"] = q.get("tables_used", [])
        formatted["sql_components"] = q.get("sql_components", [])
        formatted["question"] = q["question"]
        formatted["gold_sql"] = q["query"]
        data.append(formatted)

    return data


NBA_SPLIT_PATH = "data/nba/nba_split.json"


def load_nba_split_ids(
    split_path: str = NBA_SPLIT_PATH,
) -> tuple[set[int], set[int]]:
    """Load (train_ids, test_ids) created by train_nba.make_or_load_split().

    Both sets contain integer indices into the original `nba_questions.json`
    list (0-based). Use them to filter the output of `load_nba_dataset` /
    `load_nba_dataset_with_rag` so evaluation does not leak training samples.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_ids"]), set(split["test_ids"])


# ──────────────────────────────────────────────
# 5. NBA schema documents for RAG
# ──────────────────────────────────────────────

def build_nba_schema_documents(
    db_path: str = "data/raw/nba.sqlite",
    tables: Optional[list[str]] = None,
) -> list[dict]:
    """One document per table, ready for sentence-transformer embedding."""
    tables = tables or NBA_CORE_TABLES
    schema = get_sqlite_schema(db_path, tables=tables)
    return [
        {"table": t, "text": serialize_schema_for_table(schema, t, include_types=True)}
        for t in schema
    ]


# ──────────────────────────────────────────────
# 6. CLI: quick tests
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-nba", action="store_true")
    parser.add_argument("--test-spider", action="store_true")
    parser.add_argument("--test-oracle", action="store_true",
                        help="Compare full-schema vs oracle-tables NBA input lengths")
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    args = parser.parse_args()

    if args.test_nba:
        nba = load_nba_dataset(args.nba_questions, args.nba_db)
        print(f"Loaded {len(nba)} NBA examples")
        print(f"\nSample input (first 300 chars):\n  {nba[0]['input_text'][:300]}...")
        print(f"\nSample target:\n  {nba[0]['target_text']}")
        avg_len = sum(len(e['input_text']) for e in nba) / len(nba)
        print(f"\nAverage input length: {avg_len:.0f} chars")

    if args.test_oracle:
        full = load_nba_dataset(args.nba_questions, args.nba_db, use_oracle_tables=False)
        oracle = load_nba_dataset(args.nba_questions, args.nba_db, use_oracle_tables=True)
        full_avg = sum(len(e['input_text']) for e in full) / len(full)
        oracle_avg = sum(len(e['input_text']) for e in oracle) / len(oracle)
        print(f"=== Schema length comparison ===")
        print(f"  Full schema mode:   avg {full_avg:.0f} chars")
        print(f"  Oracle tables mode: avg {oracle_avg:.0f} chars")
        print(f"  Reduction: {(1 - oracle_avg/full_avg)*100:.0f}%")
        print(f"\nQ1 oracle input:\n  {oracle[0]['input_text'][:300]}...")

    if args.test_spider:
        train, dev = load_spider_splits(max_examples=5)
        print(f"Spider train: {len(train)} examples (showing 5)")
        print(f"\nSample input (first 300 chars):\n  {train[0]['input_text'][:300]}...")
        print(f"\nSample target:\n  {train[0]['target_text']}")
        avg_len = sum(len(e['input_text']) for e in train) / len(train)
        print(f"\nAverage input length over 5: {avg_len:.0f} chars")
