"""Owner connection for ingestion. The API uses the read-only role instead."""

from __future__ import annotations

import os

import psycopg

OWNER_URL = "postgresql://hoopql:hoopql@localhost:5432/hoopql"


def database_url() -> str:
    return os.environ.get("ETL_DATABASE_URL") or OWNER_URL


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url())
