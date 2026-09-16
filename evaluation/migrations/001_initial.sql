-- Preserve existing IDs, JSON text and ISO timestamps for a lossless migration.
CREATE TABLE campaigns (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'execution',
    config_json TEXT NOT NULL,
    completed_units INTEGER NOT NULL DEFAULT 0,
    total_units INTEGER NOT NULL DEFAULT 0,
    evaluation_completed_units INTEGER NOT NULL DEFAULT 0,
    evaluation_total_units INTEGER NOT NULL DEFAULT 0,
    current_question_id TEXT,
    current_mode TEXT,
    error_message TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE campaign_results (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    ground_truth_short TEXT,
    key_points_json TEXT NOT NULL DEFAULT '[]',
    ragas_focus_json TEXT NOT NULL DEFAULT '[]',
    mode TEXT NOT NULL,
    execution_profile TEXT,
    context_policy_version TEXT,
    run_number INTEGER NOT NULL,
    condition_id TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL,
    contexts_json TEXT NOT NULL,
    source_doc_ids_json TEXT NOT NULL,
    expected_sources_json TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    token_usage_json TEXT NOT NULL,
    category TEXT,
    difficulty TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    source_attempt_id TEXT,
    created_at TEXT NOT NULL, question_version TEXT, request_id TEXT, started_at TEXT, completed_at TEXT, total_latency_ms DOUBLE PRECISION, total_tokens INTEGER NOT NULL DEFAULT 0, estimated_cost_usd DOUBLE PRECISION, estimated_cost_twd DOUBLE PRECISION, test_suite_id TEXT, test_case_hash TEXT, ground_truth_hash TEXT, expected_evidence_hash TEXT, knowledge_base_id TEXT, index_version TEXT, retriever_config_hash TEXT, prompt_pack_version TEXT, price_snapshot_id TEXT, question_snapshot_json TEXT NOT NULL DEFAULT '{}', model_config_snapshot_json TEXT NOT NULL DEFAULT '{}', system_version_snapshot_json TEXT NOT NULL DEFAULT '{}', ablation_flags_json TEXT NOT NULL DEFAULT '{}', derived_metrics_json TEXT NOT NULL DEFAULT '{}', final_answer_hash TEXT,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    selection_json TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_work_items (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    work_type TEXT NOT NULL,
    input_snapshot_json TEXT NOT NULL,
    latest_success_attempt_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_job_items (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    work_item_id TEXT NOT NULL,
    status TEXT NOT NULL,
    max_attempts INTEGER NOT NULL,
    next_retry_at TEXT,
    active_attempt_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(work_item_id) REFERENCES evaluation_work_items(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_attempts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    job_item_id TEXT NOT NULL,
    work_item_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    last_heartbeat_at TEXT,
    finished_at TEXT,
    error_type TEXT,
    safe_error_message TEXT,
    output_json TEXT,
    FOREIGN KEY(job_id) REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(job_item_id) REFERENCES evaluation_job_items(id) ON DELETE CASCADE,
    FOREIGN KEY(work_item_id) REFERENCES evaluation_work_items(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_accounting_scopes (
    scope_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    run_id TEXT,
    metric_name TEXT,
    accounting_schema_version TEXT NOT NULL,
    status TEXT NOT NULL,
    observed_call_count INTEGER NOT NULL DEFAULT 0,
    measured_call_count INTEGER NOT NULL DEFAULT 0,
    missing_usage_call_count INTEGER NOT NULL DEFAULT 0,
    unclassified_phase_call_count INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_accounting_scope_targets (
    scope_id TEXT NOT NULL,
    campaign_result_id TEXT,
    job_id TEXT NOT NULL,
    work_item_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    mode TEXT,
    metric_name TEXT,
    is_official INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY(scope_id, attempt_id),
    FOREIGN KEY(scope_id) REFERENCES evaluation_accounting_scopes(scope_id) ON DELETE CASCADE
);

CREATE TABLE evaluation_usage_events (
    usage_event_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    run_id TEXT,
    provider_run_id TEXT,
    phase TEXT NOT NULL,
    purpose TEXT NOT NULL,
    metric_name TEXT,
    provider TEXT,
    model_name TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_text_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    other_tokens INTEGER NOT NULL DEFAULT 0,
    reported_total_tokens INTEGER,
    raw_usage_json TEXT NOT NULL DEFAULT '{}',
    usage_status TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL,
    estimated_cost_usd DOUBLE PRECISION,
    estimated_cost_twd DOUBLE PRECISION,
    pricing_status TEXT NOT NULL,
    price_snapshot_id TEXT,
    latency_ms DOUBLE PRECISION,
    status TEXT NOT NULL,
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(scope_id) REFERENCES evaluation_accounting_scopes(scope_id) ON DELETE CASCADE,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_trace_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_event_id TEXT,
    parent_span_id TEXT,
    event_type TEXT NOT NULL,
    event_schema_version TEXT NOT NULL DEFAULT '1.0',
    sequence INTEGER NOT NULL,
    stage_type TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms DOUBLE PRECISION,
    status TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_llm_calls (
    llm_call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    provider TEXT,
    model_name TEXT,
    phase TEXT NOT NULL DEFAULT 'unknown',
    purpose TEXT NOT NULL DEFAULT 'unknown',
    reservation_id TEXT,
    provider_attempt INTEGER,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER,
    other_tokens INTEGER,
    estimated_cost_usd DOUBLE PRECISION,
    estimated_cost_twd DOUBLE PRECISION,
    prompt_hash TEXT,
    prompt_preview TEXT,
    prompt_capture_status TEXT NOT NULL DEFAULT 'unknown',
    full_prompt_capture_status TEXT NOT NULL DEFAULT 'unknown',
    response_hash TEXT,
    latency_ms DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'success',
    error_json TEXT NOT NULL DEFAULT '{}',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_retrieval_events (
    retrieval_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    query TEXT,
    query_hash TEXT,
    retriever_name TEXT,
    top_k INTEGER,
    result_count INTEGER NOT NULL DEFAULT 0,
    latency_ms DOUBLE PRECISION,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_retrieval_chunks (
    retrieval_chunk_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    retrieval_event_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    doc_id TEXT,
    page_start INTEGER,
    page_end INTEGER,
    modality TEXT,
    rank_before_rerank INTEGER,
    rank_after_rerank INTEGER,
    dense_score DOUBLE PRECISION,
    bm25_score DOUBLE PRECISION,
    rerank_score DOUBLE PRECISION,
    used_in_context INTEGER NOT NULL DEFAULT 0,
    used_in_answer INTEGER NOT NULL DEFAULT 0,
    expected_evidence_match INTEGER NOT NULL DEFAULT 0,
    excerpt TEXT,
    content_hash TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY(retrieval_event_id) REFERENCES evaluation_retrieval_events(retrieval_event_id) ON DELETE CASCADE
);

CREATE TABLE evaluation_context_packs (
    context_pack_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    attempt_id TEXT,
    condition_id TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL DEFAULT '1',
    span_id TEXT,
    input_chunk_count INTEGER NOT NULL DEFAULT 0,
    packed_chunk_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    retrieved_but_not_packed_evidence_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY(attempt_id) REFERENCES evaluation_attempts(id) ON DELETE SET NULL
);

CREATE TABLE evaluation_tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    tool_name TEXT NOT NULL,
    action TEXT,
    latency_ms DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'success',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_routing_decisions (
    routing_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    selected_mode TEXT NOT NULL,
    analysis_type TEXT NOT NULL DEFAULT 'retrospective',
    confidence DOUBLE PRECISION,
    reason TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_claims (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    attempt_id TEXT,
    condition_id TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL DEFAULT '1',
    span_id TEXT,
    claim_text TEXT NOT NULL,
    claim_type TEXT,
    support_status TEXT NOT NULL DEFAULT 'unsupported',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    unsupported_reason TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY(attempt_id) REFERENCES evaluation_attempts(id) ON DELETE SET NULL
);

CREATE TABLE evaluation_evidence_packets (
    evidence_packet_row_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    condition_id TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL DEFAULT '1',
    evidence_id TEXT NOT NULL,
    packet_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(attempt_id) REFERENCES evaluation_attempts(id) ON DELETE CASCADE,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_slot_resolutions (
    slot_resolution_row_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    condition_id TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL DEFAULT '1',
    slot_id TEXT NOT NULL,
    resolution_stage TEXT NOT NULL DEFAULT 'sufficiency',
    resolution_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(attempt_id) REFERENCES evaluation_attempts(id) ON DELETE CASCADE,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_v9_attempt_materializations (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    condition_id TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL DEFAULT '1',
    trace_json TEXT NOT NULL DEFAULT '{}',
    materialization_status TEXT NOT NULL CHECK (materialization_status IN ('completed', 'cancelled')),
    completed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(attempt_id) REFERENCES evaluation_attempts(id) ON DELETE CASCADE,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_human_ratings (
    human_rating_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    rater_id_hash TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    correctness_score DOUBLE PRECISION NOT NULL,
    faithfulness_score DOUBLE PRECISION NOT NULL,
    completeness_score DOUBLE PRECISION NOT NULL,
    citation_quality_score DOUBLE PRECISION NOT NULL,
    usefulness_score DOUBLE PRECISION NOT NULL,
    comments TEXT,
    is_blinded INTEGER NOT NULL DEFAULT 1,
    shown_mode_label INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_graph_events (
    graph_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT,
    span_id TEXT,
    graph_query TEXT NOT NULL,
    graph_search_mode TEXT NOT NULL,
    graph_evidence_mode TEXT NOT NULL DEFAULT 'raw_current',
    graph_route TEXT NOT NULL,
    router_reason TEXT,
    graph_feature_flags_json TEXT NOT NULL DEFAULT '{}',
    graph_snapshot_version TEXT,
    graph_schema_version TEXT,
    graph_extraction_prompt_version TEXT,
    matched_entity_ids_json TEXT NOT NULL DEFAULT '[]',
    community_ids_json TEXT NOT NULL DEFAULT '[]',
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0,
    path_count INTEGER NOT NULL DEFAULT 0,
    graph_latency_ms INTEGER,
    graph_context_tokens INTEGER NOT NULL DEFAULT 0,
    graph_to_chunk_success_rate DOUBLE PRECISION,
    graph_noise_ratio DOUBLE PRECISION,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE evaluation_graph_evidence_items (
    graph_evidence_item_id TEXT PRIMARY KEY,
    graph_event_id TEXT NOT NULL,
    node_ids_json TEXT NOT NULL DEFAULT '[]',
    edge_ids_json TEXT NOT NULL DEFAULT '[]',
    relation_path_json TEXT NOT NULL DEFAULT '[]',
    source_doc_ids_json TEXT NOT NULL DEFAULT '[]',
    source_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
    pages_json TEXT NOT NULL DEFAULT '[]',
    asset_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    provenance_status TEXT NOT NULL DEFAULT 'missing',
    used_as_locator INTEGER NOT NULL DEFAULT 1,
    packed_in_context INTEGER NOT NULL DEFAULT 0,
    used_in_answer INTEGER NOT NULL DEFAULT 0,
    supported_claim_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY(graph_event_id) REFERENCES evaluation_graph_events(graph_event_id) ON DELETE CASCADE
);

CREATE TABLE agent_traces (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    campaign_result_id TEXT,
    user_id TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    summary_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE ragas_scores (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    campaign_result_id TEXT,
    user_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    details_json TEXT NOT NULL,
    source_attempt_id TEXT,
    evaluation_signature TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_campaigns_user_created
ON campaigns(user_id, created_at DESC);

CREATE INDEX idx_campaign_results_campaign_created
ON campaign_results(campaign_id, created_at ASC);

CREATE INDEX idx_campaign_results_campaign_user_order
ON campaign_results(campaign_id, user_id, created_at ASC, question_id ASC, mode ASC, run_number ASC, id ASC);

CREATE INDEX idx_eval_jobs_user_campaign_created
ON evaluation_jobs(user_id, campaign_id, created_at ASC);

CREATE UNIQUE INDEX idx_eval_work_item_logical_key
ON evaluation_work_items(campaign_id, logical_key);

CREATE UNIQUE INDEX idx_eval_job_item_pair
ON evaluation_job_items(job_id, work_item_id);

CREATE INDEX idx_eval_job_item_ready
ON evaluation_job_items(status, next_retry_at, created_at);

CREATE UNIQUE INDEX idx_eval_attempt_number
ON evaluation_attempts(work_item_id, attempt_number);

CREATE UNIQUE INDEX idx_ragas_scores_result_metric
ON ragas_scores(campaign_result_id, metric_name);

CREATE INDEX idx_ragas_scores_campaign_user_result
ON ragas_scores(campaign_id, user_id, campaign_result_id, metric_name);

CREATE UNIQUE INDEX idx_agent_traces_result
ON agent_traces(campaign_result_id);

CREATE INDEX idx_agent_traces_campaign_user_created
ON agent_traces(campaign_id, user_id, created_at DESC);

CREATE UNIQUE INDEX idx_campaign_results_unit_unique
        ON campaign_results(campaign_id, question_id, mode, run_number, condition_id)
        ;

CREATE INDEX idx_eval_trace_events_run_started
        ON evaluation_trace_events(run_id, started_at ASC)
        ;

CREATE INDEX idx_eval_trace_events_campaign_run
        ON evaluation_trace_events(campaign_id, run_id, sequence ASC, started_at ASC)
        ;

CREATE INDEX idx_eval_llm_calls_run_purpose
        ON evaluation_llm_calls(run_id, purpose)
        ;

CREATE INDEX idx_eval_llm_calls_campaign_run
        ON evaluation_llm_calls(campaign_id, run_id, created_at ASC)
        ;

CREATE UNIQUE INDEX idx_eval_llm_calls_attempt_identity
        ON evaluation_llm_calls(run_id, reservation_id, provider_attempt)
        WHERE reservation_id IS NOT NULL AND provider_attempt IS NOT NULL
        ;

CREATE INDEX idx_eval_retrieval_events_run_span
        ON evaluation_retrieval_events(run_id, span_id)
        ;

CREATE INDEX idx_eval_retrieval_chunks_event
        ON evaluation_retrieval_chunks(retrieval_event_id)
        ;

CREATE INDEX idx_eval_retrieval_chunks_run_event
        ON evaluation_retrieval_chunks(run_id, retrieval_event_id)
        ;

CREATE INDEX idx_eval_retrieval_chunks_campaign_run
        ON evaluation_retrieval_chunks(
            campaign_id,
            run_id,
            retrieval_event_id,
            rank_after_rerank,
            created_at ASC
        )
        ;

CREATE INDEX idx_eval_context_packs_run_created
        ON evaluation_context_packs(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_context_packs_attempt_created
        ON evaluation_context_packs(attempt_id, created_at ASC)
        ;

CREATE INDEX idx_eval_tool_calls_run_created
        ON evaluation_tool_calls(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_routing_decisions_run_created
        ON evaluation_routing_decisions(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_routing_decisions_campaign_run_created
        ON evaluation_routing_decisions(campaign_id, run_id, created_at ASC, routing_decision_id ASC)
        ;

CREATE INDEX idx_eval_claims_run_created
        ON evaluation_claims(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_claims_campaign_run
        ON evaluation_claims(campaign_id, run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_claims_attempt_created
        ON evaluation_claims(attempt_id, created_at ASC)
        ;

CREATE UNIQUE INDEX idx_eval_evidence_packets_attempt_evidence
        ON evaluation_evidence_packets(attempt_id, evidence_id)
        ;

CREATE INDEX idx_eval_evidence_packets_run_created
        ON evaluation_evidence_packets(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_evidence_packets_campaign_run
        ON evaluation_evidence_packets(campaign_id, run_id, created_at ASC)
        ;

CREATE UNIQUE INDEX idx_eval_slot_resolutions_attempt_slot_stage
        ON evaluation_slot_resolutions(attempt_id, slot_id, resolution_stage)
        ;

CREATE INDEX idx_eval_slot_resolutions_run_created
        ON evaluation_slot_resolutions(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_slot_resolutions_campaign_run
        ON evaluation_slot_resolutions(campaign_id, run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_v9_materializations_campaign_run
        ON evaluation_v9_attempt_materializations(campaign_id, run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_human_ratings_run_created
        ON evaluation_human_ratings(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_human_ratings_campaign_run
        ON evaluation_human_ratings(campaign_id, run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_graph_events_run_created
        ON evaluation_graph_events(run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_graph_events_campaign_run
        ON evaluation_graph_events(campaign_id, run_id, created_at ASC)
        ;

CREATE INDEX idx_eval_graph_evidence_items_event
        ON evaluation_graph_evidence_items(graph_event_id, created_at ASC)
        ;

CREATE INDEX idx_eval_accounting_scopes_campaign_type_status
        ON evaluation_accounting_scopes(campaign_id, scope_type, status)
        ;

CREATE INDEX idx_eval_usage_events_scope_created
        ON evaluation_usage_events(scope_id, created_at ASC)
        ;

CREATE INDEX idx_eval_usage_events_run_phase
        ON evaluation_usage_events(run_id, phase)
        ;

CREATE INDEX idx_eval_accounting_targets_attempt_official
        ON evaluation_accounting_scope_targets(attempt_id, is_official)
        ;

CREATE INDEX idx_eval_usage_events_campaign_metric
        ON evaluation_usage_events(campaign_id, metric_name)
        ;
