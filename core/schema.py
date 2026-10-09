"""Live Postgres schema: tables, columns, types, foreign keys, and column comments."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from core.config import database_url

DEFAULT_TTL_S = 300.0
# Ingestion bookkeeping. Never shown to the model.
HIDDEN_TABLES = frozenset({"etl_run", "etl_checkpoint"})

COLUMNS_SQL = """
SELECT c.table_name,
       c.column_name,
       c.data_type,
       c.is_nullable = 'YES' AS nullable,
       col_description(cls.oid, c.ordinal_position::int) AS comment
FROM information_schema.columns c
JOIN pg_catalog.pg_namespace ns ON ns.nspname = c.table_schema
JOIN pg_catalog.pg_class cls ON cls.relnamespace = ns.oid AND cls.relname = c.table_name
WHERE c.table_schema = 'public'
  AND cls.relkind = 'r'
ORDER BY c.table_name, c.ordinal_position
"""

TABLE_COMMENTS_SQL = """
SELECT cls.relname, obj_description(cls.oid, 'pg_class')
FROM pg_catalog.pg_class cls
JOIN pg_catalog.pg_namespace ns ON ns.oid = cls.relnamespace
WHERE ns.nspname = 'public' AND cls.relkind = 'r'
"""

# pg_catalog instead of information_schema: the latter hides constraints on tables the role does not own.
FOREIGN_KEYS_SQL = """
SELECT src.relname, src_col.attname, dst.relname, dst_col.attname
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class src ON src.oid = con.conrelid
JOIN pg_catalog.pg_class dst ON dst.oid = con.confrelid
JOIN pg_catalog.pg_namespace ns ON ns.oid = src.relnamespace
CROSS JOIN LATERAL unnest(con.conkey, con.confkey) AS k(src_attnum, dst_attnum)
JOIN pg_catalog.pg_attribute src_col ON src_col.attrelid = src.oid AND src_col.attnum = k.src_attnum
JOIN pg_catalog.pg_attribute dst_col ON dst_col.attrelid = dst.oid AND dst_col.attnum = k.dst_attnum
WHERE con.contype = 'f' AND ns.nspname = 'public'
ORDER BY src.relname, src_col.attname
"""


@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    nullable: bool
    comment: str | None


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    comment: str | None
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]

    def column_names(self) -> set[str]:
        return {column.name for column in self.columns}


@dataclass(frozen=True)
class Schema:
    tables: dict[str, Table]

    def table_names(self) -> list[str]:
        return sorted(self.tables)


def introspect(conn: psycopg.Connection) -> Schema:
    columns: dict[str, list[Column]] = {}
    for table, name, data_type, nullable, comment in conn.execute(COLUMNS_SQL).fetchall():
        if table in HIDDEN_TABLES:
            continue
        columns.setdefault(table, []).append(Column(name, data_type, nullable, comment))

    table_comments = dict(conn.execute(TABLE_COMMENTS_SQL).fetchall())

    foreign_keys: dict[str, list[ForeignKey]] = {}
    for table, column, ref_table, ref_column in conn.execute(FOREIGN_KEYS_SQL).fetchall():
        if table in columns:
            foreign_keys.setdefault(table, []).append(ForeignKey(column, ref_table, ref_column))

    tables = {
        name: Table(
            name=name,
            comment=table_comments.get(name),
            columns=tuple(cols),
            foreign_keys=tuple(foreign_keys.get(name, ())),
        )
        for name, cols in columns.items()
    }
    return Schema(tables=tables)


class SchemaCache:
    def __init__(
        self,
        connect: Callable[[], psycopg.Connection] | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._connect = connect or (lambda: psycopg.connect(database_url()))
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._schema: Schema | None = None
        self._loaded_at = 0.0

    def get(self) -> Schema:
        with self._lock:
            now = self._clock()
            if self._schema is None or now - self._loaded_at >= self._ttl_s:
                with self._connect() as conn:
                    self._schema = introspect(conn)
                self._loaded_at = now
            return self._schema

    def clear(self) -> None:
        with self._lock:
            self._schema = None


_default_cache = SchemaCache()


def get_schema() -> Schema:
    return _default_cache.get()


def table_text(table: Table) -> str:
    """One table rendered for the prompt and for retrieval documents."""
    lines = [f"TABLE {table.name}" + (f" -- {table.comment}" if table.comment else "")]
    for column in table.columns:
        line = f"  {column.name} {column.data_type}"
        if not column.nullable:
            line += " NOT NULL"
        if column.comment:
            line += f" -- {column.comment}"
        lines.append(line)
    for fk in table.foreign_keys:
        lines.append(f"  FOREIGN KEY ({fk.column}) REFERENCES {fk.ref_table} ({fk.ref_column})")
    return "\n".join(lines)


def schema_text(schema: Schema, tables: list[str] | None = None) -> str:
    names = tables if tables is not None else schema.table_names()
    return "\n\n".join(table_text(schema.tables[name]) for name in names if name in schema.tables)
