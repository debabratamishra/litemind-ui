-- Runs automatically on first initialization of the pgdata volume
-- (via /docker-entrypoint-initdb.d). Creates the schema GoTrue migrates into.
CREATE SCHEMA IF NOT EXISTS auth;
ALTER ROLE postgres SET search_path TO auth, public;
