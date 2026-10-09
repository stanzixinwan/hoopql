"""Validate generated SQL before it reaches the database.

The read-only role and statement_timeout are the real boundary; this layer
rejects anything that is not a single SELECT over known tables and columns, so
bad queries fail fast with a message the repair loop can act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from core.schema import Schema

DEFAULT_ROW_LIMIT = 1000
DIALECT = "postgres"

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Command,
    exp.Into,
    exp.Lock,
    exp.Set,
    exp.Grant,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "current_setting",
        "set_config",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "cursor_to_xml",
        "txid_current",
    }
)
ALLOWED_SCHEMAS = frozenset({"", "public"})


class UnsafeSQLError(ValueError):
    """The SQL was rejected before execution; the message is safe to show the generator."""


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str
    tables: tuple[str, ...]
    limit_added: bool


def _reject_comments(sql: str) -> None:
    try:
        tokens = sqlglot.tokenize(sql, read=DIALECT)
    except TokenError as exc:
        raise UnsafeSQLError(f"Could not tokenize SQL: {exc}") from exc
    if any(token.comments for token in tokens):
        raise UnsafeSQLError("SQL comments are not allowed.")


def _parse_single(sql: str) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except ParseError as exc:
        raise UnsafeSQLError(f"Could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeSQLError(f"Expected exactly one statement, got {len(statements)}.")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed, got {root.key.upper()}.")
    return root


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def _reject_forbidden(root: exp.Expression) -> None:
    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed in a read-only query.")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
                raise UnsafeSQLError(f"Function {name}() is not allowed.")


def _check_references(root: exp.Expression, schema: Schema) -> tuple[str, ...]:
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    alias_to_table: dict[str, str] = {}
    derived_aliases: set[str] = set(cte_names)
    output_names: set[str] = set()
    tables: set[str] = set()

    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue
        name = table.name.lower()
        if table.db.lower() not in ALLOWED_SCHEMAS or table.catalog:
            raise UnsafeSQLError(f"Table {table.sql(DIALECT)} is outside the public schema.")
        if name in cte_names and not table.db:
            derived_aliases.add(table.alias_or_name.lower())
            continue
        if name not in schema.tables:
            raise UnsafeSQLError(f"Unknown table {name}.")
        tables.add(name)
        alias_to_table[table.alias_or_name.lower()] = name
        alias_to_table.setdefault(name, name)

    for node in root.find_all(exp.Subquery, exp.Lateral, exp.Values, exp.Unnest):
        if node.alias:
            derived_aliases.add(node.alias.lower())
    for node in root.find_all(exp.Alias):
        output_names.add(node.alias.lower())
    for node in root.find_all(exp.TableAlias):
        output_names.update(col.name.lower() for col in node.columns)

    known_columns = set().union(*(schema.tables[t].column_names() for t in tables)) if tables else set()

    for column in root.find_all(exp.Column):
        name = column.name.lower()
        if not name or isinstance(column.this, exp.Star):
            continue
        qualifier = column.table.lower()
        if not qualifier:
            if name not in known_columns and name not in output_names:
                raise UnsafeSQLError(f"Unknown column {name}.")
            continue
        if qualifier in derived_aliases:
            continue
        if qualifier not in alias_to_table:
            raise UnsafeSQLError(f"Unknown table or alias {qualifier} for column {name}.")
        table_name = alias_to_table[qualifier]
        if name not in schema.tables[table_name].column_names():
            raise UnsafeSQLError(f"Unknown column {table_name}.{name}.")

    return tuple(sorted(tables))


def validate(sql: str, schema: Schema, row_limit: int = DEFAULT_ROW_LIMIT) -> ValidatedSQL:
    """Return normalized SQL with a LIMIT, or raise UnsafeSQLError."""
    _reject_comments(sql)
    root = _parse_single(sql)
    _reject_forbidden(root)
    tables = _check_references(root, schema)
    limit_added = root.args.get("limit") is None
    if limit_added:
        root = root.limit(row_limit)
    return ValidatedSQL(sql=root.sql(dialect=DIALECT), tables=tables, limit_added=limit_added)
