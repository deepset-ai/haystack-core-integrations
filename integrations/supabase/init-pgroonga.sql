-- Enable PGroonga extension
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- PostgreSQL role that PostgREST switches to when a service_role JWT is presented.
-- The role must exist before PostgREST connects.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'service_role') THEN
        CREATE ROLE service_role NOLOGIN;
    END IF;
END
$$;

GRANT ALL ON SCHEMA public TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO service_role;

-- exec_sql: allows the document store to create/drop tables and indexes via RPC.
CREATE OR REPLACE FUNCTION exec_sql(query TEXT)
RETURNS VOID AS $$
BEGIN
    EXECUTE query;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION exec_sql(TEXT) TO service_role;

-- groonga_search: full-text search via PGroonga, called by _groonga_retrieval().
CREATE OR REPLACE FUNCTION groonga_search(query_text TEXT, table_name TEXT, top_k INT)
RETURNS TABLE(id TEXT, content TEXT, meta JSONB, score REAL) AS $$
DECLARE
    sql TEXT;
BEGIN
    sql := format(
        'SELECT id, content, meta, pgroonga_score(tableoid, ctid)::REAL AS score
         FROM %I
         WHERE content &@~ %L
         ORDER BY score DESC
         LIMIT %s',
        table_name, query_text, top_k
    );
    RETURN QUERY EXECUTE sql;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION groonga_search(TEXT, TEXT, INT) TO service_role;

-- Pre-create the test table so PostgREST includes it in its schema cache at startup.
-- Tests use this fixed table and clear data between runs instead of recreating the table.
CREATE TABLE IF NOT EXISTS haystack_groonga_test (
    id TEXT PRIMARY KEY,
    content TEXT,
    meta JSONB,
    score REAL
);

CREATE INDEX IF NOT EXISTS pgroonga_haystack_groonga_test_index
ON haystack_groonga_test
USING pgroonga (content);

GRANT ALL ON TABLE haystack_groonga_test TO postgres;
