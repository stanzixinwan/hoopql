-- Read-only role for the API. Local password is a dev default, not a production secret.
-- Ingestion keeps using the database owner via ETL_DATABASE_URL.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'hoopql_readonly') THEN
        CREATE ROLE hoopql_readonly LOGIN PASSWORD 'hoopql_readonly';
    END IF;
END
$$;

ALTER ROLE hoopql_readonly WITH LOGIN PASSWORD 'hoopql_readonly';
ALTER ROLE hoopql_readonly SET statement_timeout = '5s';
ALTER ROLE hoopql_readonly SET idle_in_transaction_session_timeout = '5s';

DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO hoopql_readonly', current_database());
END
$$;

GRANT USAGE ON SCHEMA public TO hoopql_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO hoopql_readonly;
REVOKE ALL PRIVILEGES ON TABLE etl_run FROM hoopql_readonly;
REVOKE ALL PRIVILEGES ON TABLE etl_checkpoint FROM hoopql_readonly;
