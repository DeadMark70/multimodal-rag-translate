-- Small compatibility functions for the existing JSON-text repository queries.
-- Data representation is deliberately unchanged during the storage migration.
CREATE FUNCTION json_extract(document text, path text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT document::jsonb #>> string_to_array(substr(path, 3), '.')
$$;

CREATE FUNCTION json_type(document text, path text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT jsonb_typeof(document::jsonb #> string_to_array(substr(path, 3), '.'))
$$;

CREATE FUNCTION min(integer, integer) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$ SELECT LEAST($1, $2) $$;
CREATE FUNCTION max(integer, integer) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$ SELECT GREATEST($1, $2) $$;

CREATE INDEX idx_eval_job_item_work_latest
ON evaluation_job_items(work_item_id, created_at DESC, id DESC);
CREATE INDEX idx_eval_attempts_running_heartbeat
ON evaluation_attempts(last_heartbeat_at, started_at) WHERE status = 'running';
