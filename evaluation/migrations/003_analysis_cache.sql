CREATE TABLE evaluation_analysis_state (
    campaign_id TEXT PRIMARY KEY REFERENCES campaigns(id) ON DELETE CASCADE ON UPDATE CASCADE,
    revision BIGINT NOT NULL DEFAULT 1
);
CREATE TABLE evaluation_analysis_cache (
    campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE ON UPDATE CASCADE,
    kind TEXT NOT NULL,
    revision BIGINT NOT NULL,
    format_version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    built_at TEXT NOT NULL,
    PRIMARY KEY(campaign_id, kind)
);

CREATE FUNCTION evaluation_analysis_changed() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE cid text; data jsonb;
BEGIN
    data := CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END;
    IF TG_TABLE_NAME = 'campaigns' THEN
        cid := data->>'id';
    ELSIF TG_TABLE_NAME = 'evaluation_accounting_scope_targets' THEN
        SELECT campaign_id INTO cid FROM evaluation_accounting_scopes WHERE scope_id = data->>'scope_id';
    ELSIF TG_TABLE_NAME = 'evaluation_graph_evidence_items' THEN
        SELECT campaign_id INTO cid FROM evaluation_graph_events WHERE graph_event_id = data->>'graph_event_id';
    ELSE
        cid := data->>'campaign_id';
    END IF;
    IF cid IS NOT NULL AND EXISTS(SELECT 1 FROM campaigns WHERE id=cid) THEN
        INSERT INTO evaluation_analysis_state(campaign_id) VALUES(cid)
        ON CONFLICT(campaign_id) DO UPDATE SET revision=evaluation_analysis_state.revision+1;
    END IF;
    RETURN NULL;
END $$;

DO $$
DECLARE table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'campaigns','campaign_results','ragas_scores','agent_traces',
        'evaluation_work_items','evaluation_accounting_scopes',
        'evaluation_accounting_scope_targets','evaluation_usage_events',
        'evaluation_trace_events','evaluation_llm_calls','evaluation_retrieval_events',
        'evaluation_retrieval_chunks','evaluation_context_packs','evaluation_tool_calls',
        'evaluation_routing_decisions','evaluation_claims','evaluation_evidence_packets',
        'evaluation_slot_resolutions','evaluation_v9_attempt_materializations',
        'evaluation_human_ratings','evaluation_graph_events','evaluation_graph_evidence_items'
    ] LOOP
        EXECUTE format('CREATE TRIGGER analysis_changed AFTER INSERT OR UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION evaluation_analysis_changed()', table_name);
    END LOOP;
END $$;

INSERT INTO evaluation_analysis_state(campaign_id) SELECT id FROM campaigns;
