-- Fuzzy, accent-insensitive name lookup for grounding ("Jokic" finds "Nikola Jokić").
-- Safe to re-run. Applied after etl/schema.sql on new Compose volumes and in tests.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- unaccent() is STABLE, so an IMMUTABLE wrapper with a fixed dictionary is needed for indexes.
CREATE OR REPLACE FUNCTION fold_name(text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
AS $$ SELECT lower(public.unaccent('public.unaccent'::regdictionary, $1)) $$;

CREATE INDEX IF NOT EXISTS player_full_name_trgm_idx ON player USING gin (fold_name(full_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS team_full_name_trgm_idx ON team USING gin (fold_name(full_name) gin_trgm_ops);
