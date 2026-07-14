"""SQLite persistence for evaluation campaigns."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import aiosqlite

from core.errors import AppError, ErrorCode
from evaluation.agentic_evaluation_service import LEGACY_SHARED_PROFILE
from evaluation.campaign_schemas import (
    CampaignConfig,
    CampaignLifecycleStatus,
    CampaignResult,
    CampaignResultStatus,
    CampaignStatus,
)
from evaluation.trace_schemas import AgentTraceDetail, AgentTraceSummary, summarize_agent_trace

EVALUATION_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation.db"
_UNSET = object()
logger = logging.getLogger(__name__)
_INITIALIZED_DB_PATHS: set[str] = set()
_INIT_LOCKS: dict[str, asyncio.Lock] = {}
ROUTE_PROFILE_ALIASES = {
    "hybrid_graph": "generic_graph",
}
_OBSERVABILITY_TABLE_COLUMNS = {
    "evaluation_trace_events": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT NOT NULL DEFAULT ''",
        "parent_event_id": "TEXT",
        "parent_span_id": "TEXT",
        "event_type": "TEXT NOT NULL DEFAULT ''",
        "event_schema_version": "TEXT NOT NULL DEFAULT '1.0'",
        "sequence": "INTEGER NOT NULL DEFAULT 0",
        "stage_type": "TEXT NOT NULL DEFAULT ''",
        "stage_name": "TEXT NOT NULL DEFAULT ''",
        "started_at": "TEXT NOT NULL DEFAULT ''",
        "ended_at": "TEXT",
        "duration_ms": "REAL",
        "status": "TEXT NOT NULL DEFAULT 'running'",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "error_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_llm_calls": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "provider": "TEXT",
        "model_name": "TEXT",
        "purpose": "TEXT NOT NULL DEFAULT 'unknown'",
        "prompt_tokens": "INTEGER NOT NULL DEFAULT 0",
        "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
        "total_tokens": "INTEGER NOT NULL DEFAULT 0",
        "estimated_cost_usd": "REAL",
        "estimated_cost_twd": "REAL",
        "prompt_hash": "TEXT",
        "prompt_preview": "TEXT",
        "response_hash": "TEXT",
        "latency_ms": "REAL",
        "status": "TEXT NOT NULL DEFAULT 'success'",
        "error_json": "TEXT NOT NULL DEFAULT '{}'",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_retrieval_events": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "query": "TEXT",
        "query_hash": "TEXT",
        "retriever_name": "TEXT",
        "top_k": "INTEGER",
        "result_count": "INTEGER NOT NULL DEFAULT 0",
        "latency_ms": "REAL",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_retrieval_chunks": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "retrieval_event_id": "TEXT NOT NULL DEFAULT ''",
        "chunk_id": "TEXT NOT NULL DEFAULT ''",
        "doc_id": "TEXT",
        "page_start": "INTEGER",
        "page_end": "INTEGER",
        "modality": "TEXT",
        "rank_before_rerank": "INTEGER",
        "rank_after_rerank": "INTEGER",
        "dense_score": "REAL",
        "bm25_score": "REAL",
        "rerank_score": "REAL",
        "used_in_context": "INTEGER NOT NULL DEFAULT 0",
        "used_in_answer": "INTEGER NOT NULL DEFAULT 0",
        "expected_evidence_match": "INTEGER NOT NULL DEFAULT 0",
        "excerpt": "TEXT",
        "content_hash": "TEXT",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_context_packs": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "input_chunk_count": "INTEGER NOT NULL DEFAULT 0",
        "packed_chunk_count": "INTEGER NOT NULL DEFAULT 0",
        "token_count": "INTEGER NOT NULL DEFAULT 0",
        "retrieved_but_not_packed_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_tool_calls": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "tool_name": "TEXT NOT NULL DEFAULT ''",
        "action": "TEXT",
        "latency_ms": "REAL",
        "status": "TEXT NOT NULL DEFAULT 'success'",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_routing_decisions": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "selected_mode": "TEXT NOT NULL DEFAULT ''",
        "analysis_type": "TEXT NOT NULL DEFAULT 'retrospective'",
        "confidence": "REAL",
        "reason": "TEXT",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_claims": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "claim_text": "TEXT NOT NULL DEFAULT ''",
        "claim_type": "TEXT",
        "support_status": "TEXT NOT NULL DEFAULT 'unsupported'",
        "evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "unsupported_reason": "TEXT",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_human_ratings": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT NOT NULL DEFAULT ''",
        "span_id": "TEXT",
        "rater_id_hash": "TEXT NOT NULL DEFAULT ''",
        "rubric_version": "TEXT NOT NULL DEFAULT ''",
        "correctness_score": "REAL NOT NULL DEFAULT 0",
        "faithfulness_score": "REAL NOT NULL DEFAULT 0",
        "completeness_score": "REAL NOT NULL DEFAULT 0",
        "citation_quality_score": "REAL NOT NULL DEFAULT 0",
        "usefulness_score": "REAL NOT NULL DEFAULT 0",
        "comments": "TEXT",
        "is_blinded": "INTEGER NOT NULL DEFAULT 1",
        "shown_mode_label": "INTEGER NOT NULL DEFAULT 0",
        "payload_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_graph_events": {
        "run_id": "TEXT NOT NULL DEFAULT ''",
        "campaign_id": "TEXT",
        "span_id": "TEXT",
        "graph_query": "TEXT NOT NULL DEFAULT ''",
        "graph_search_mode": "TEXT NOT NULL DEFAULT ''",
        "graph_evidence_mode": "TEXT NOT NULL DEFAULT 'raw_current'",
        "graph_route": "TEXT NOT NULL DEFAULT ''",
        "router_reason": "TEXT",
        "graph_feature_flags_json": "TEXT NOT NULL DEFAULT '{}'",
        "graph_snapshot_version": "TEXT",
        "graph_schema_version": "TEXT",
        "graph_extraction_prompt_version": "TEXT",
        "matched_entity_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "community_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "node_count": "INTEGER NOT NULL DEFAULT 0",
        "edge_count": "INTEGER NOT NULL DEFAULT 0",
        "path_count": "INTEGER NOT NULL DEFAULT 0",
        "graph_latency_ms": "INTEGER",
        "graph_context_tokens": "INTEGER NOT NULL DEFAULT 0",
        "graph_to_chunk_success_rate": "REAL",
        "graph_noise_ratio": "REAL",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
    "evaluation_graph_evidence_items": {
        "graph_event_id": "TEXT NOT NULL DEFAULT ''",
        "node_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "edge_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "relation_path_json": "TEXT NOT NULL DEFAULT '[]'",
        "source_doc_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "source_chunk_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "pages_json": "TEXT NOT NULL DEFAULT '[]'",
        "asset_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "confidence": "REAL NOT NULL DEFAULT 0",
        "provenance_status": "TEXT NOT NULL DEFAULT 'missing'",
        "used_as_locator": "INTEGER NOT NULL DEFAULT 1",
        "packed_in_context": "INTEGER NOT NULL DEFAULT 0",
        "used_in_answer": "INTEGER NOT NULL DEFAULT 0",
        "supported_claim_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    },
}
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS campaigns (
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

CREATE INDEX IF NOT EXISTS idx_campaigns_user_created
ON campaigns(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS campaign_results (
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
    latency_ms REAL NOT NULL DEFAULT 0,
    token_usage_json TEXT NOT NULL,
    category TEXT,
    difficulty TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    source_attempt_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_campaign_results_campaign_created
ON campaign_results(campaign_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_campaign_results_campaign_user_order
ON campaign_results(campaign_id, user_id, created_at ASC, question_id ASC, mode ASC, run_number ASC, id ASC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_results_unit_unique
ON campaign_results(campaign_id, question_id, mode, run_number, condition_id);

CREATE TABLE IF NOT EXISTS evaluation_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    selection_json TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_jobs_user_campaign_created
ON evaluation_jobs(user_id, campaign_id, created_at ASC);

CREATE TABLE IF NOT EXISTS evaluation_work_items (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    work_type TEXT NOT NULL,
    input_snapshot_json TEXT NOT NULL,
    latest_success_attempt_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_work_item_logical_key
ON evaluation_work_items(campaign_id, logical_key);

CREATE TABLE IF NOT EXISTS evaluation_job_items (
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_job_item_pair
ON evaluation_job_items(job_id, work_item_id);

CREATE INDEX IF NOT EXISTS idx_eval_job_item_ready
ON evaluation_job_items(status, next_retry_at, created_at);

CREATE TABLE IF NOT EXISTS evaluation_attempts (
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_attempt_number
ON evaluation_attempts(work_item_id, attempt_number);

CREATE TABLE IF NOT EXISTS evaluation_trace_events (
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
    duration_ms REAL,
    status TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_llm_calls (
    llm_call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    provider TEXT,
    model_name TEXT,
    purpose TEXT NOT NULL DEFAULT 'unknown',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL,
    estimated_cost_twd REAL,
    prompt_hash TEXT,
    prompt_preview TEXT,
    response_hash TEXT,
    latency_ms REAL,
    status TEXT NOT NULL DEFAULT 'success',
    error_json TEXT NOT NULL DEFAULT '{}',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_retrieval_events (
    retrieval_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    query TEXT,
    query_hash TEXT,
    retriever_name TEXT,
    top_k INTEGER,
    result_count INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_retrieval_chunks (
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
    dense_score REAL,
    bm25_score REAL,
    rerank_score REAL,
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

CREATE TABLE IF NOT EXISTS evaluation_context_packs (
    context_pack_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    input_chunk_count INTEGER NOT NULL DEFAULT 0,
    packed_chunk_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    retrieved_but_not_packed_evidence_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    tool_name TEXT NOT NULL,
    action TEXT,
    latency_ms REAL,
    status TEXT NOT NULL DEFAULT 'success',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_routing_decisions (
    routing_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    selected_mode TEXT NOT NULL,
    analysis_type TEXT NOT NULL DEFAULT 'retrospective',
    confidence REAL,
    reason TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_claims (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    claim_text TEXT NOT NULL,
    claim_type TEXT,
    support_status TEXT NOT NULL DEFAULT 'unsupported',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    unsupported_reason TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_human_ratings (
    human_rating_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    span_id TEXT,
    rater_id_hash TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    correctness_score REAL NOT NULL,
    faithfulness_score REAL NOT NULL,
    completeness_score REAL NOT NULL,
    citation_quality_score REAL NOT NULL,
    usefulness_score REAL NOT NULL,
    comments TEXT,
    is_blinded INTEGER NOT NULL DEFAULT 1,
    shown_mode_label INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_graph_events (
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
    graph_to_chunk_success_rate REAL,
    graph_noise_ratio REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_graph_evidence_items (
    graph_evidence_item_id TEXT PRIMARY KEY,
    graph_event_id TEXT NOT NULL,
    node_ids_json TEXT NOT NULL DEFAULT '[]',
    edge_ids_json TEXT NOT NULL DEFAULT '[]',
    relation_path_json TEXT NOT NULL DEFAULT '[]',
    source_doc_ids_json TEXT NOT NULL DEFAULT '[]',
    source_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
    pages_json TEXT NOT NULL DEFAULT '[]',
    asset_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0,
    provenance_status TEXT NOT NULL DEFAULT 'missing',
    used_as_locator INTEGER NOT NULL DEFAULT 1,
    packed_in_context INTEGER NOT NULL DEFAULT 0,
    used_in_answer INTEGER NOT NULL DEFAULT 0,
    supported_claim_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY(graph_event_id) REFERENCES evaluation_graph_events(graph_event_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_traces (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    campaign_result_id TEXT,
    user_id TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ragas_scores (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    campaign_result_id TEXT,
    user_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    details_json TEXT NOT NULL,
    source_attempt_id TEXT,
    evaluation_signature TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ragas_scores_result_metric
ON ragas_scores(campaign_result_id, metric_name);

CREATE INDEX IF NOT EXISTS idx_ragas_scores_campaign_user_result
ON ragas_scores(campaign_id, user_id, campaign_result_id, metric_name);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_traces_result
ON agent_traces(campaign_result_id);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def connect_db():
    """Open SQLite connection with WAL-friendly settings."""
    db_path = Path(EVALUATION_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA journal_mode=WAL;")
    await connection.execute("PRAGMA synchronous=NORMAL;")
    await connection.execute("PRAGMA busy_timeout=5000;")
    await connection.execute("PRAGMA foreign_keys=ON;")
    try:
        yield connection
    finally:
        await connection.close()


async def init_db() -> None:
    """Initialize evaluation database and future-proof tables."""
    db_path = str(Path(EVALUATION_DB_PATH).resolve())
    if db_path in _INITIALIZED_DB_PATHS and Path(db_path).exists():
        return
    lock = _INIT_LOCKS.setdefault(db_path, asyncio.Lock())
    async with lock:
        if db_path in _INITIALIZED_DB_PATHS and Path(db_path).exists():
            return
        async with connect_db() as connection:
            await connection.executescript(_INIT_SQL)
            await _apply_migrations(connection)
            await connection.commit()
        _INITIALIZED_DB_PATHS.add(db_path)


async def force_init_db() -> None:
    """Run schema creation and migrations even if the current DB path is cached."""
    db_path = str(Path(EVALUATION_DB_PATH).resolve())
    async with connect_db() as connection:
        await connection.executescript(_INIT_SQL)
        await _apply_migrations(connection)
        await connection.commit()
    _INITIALIZED_DB_PATHS.add(db_path)


async def _apply_migrations(connection: aiosqlite.Connection) -> None:
    """Apply additive migrations for existing Phase 2 databases."""
    campaign_columns = await _table_columns(connection, "campaigns")
    if "phase" not in campaign_columns:
        await connection.execute(
            "ALTER TABLE campaigns ADD COLUMN phase TEXT NOT NULL DEFAULT 'execution'"
        )
    if "evaluation_completed_units" not in campaign_columns:
        await connection.execute(
            "ALTER TABLE campaigns ADD COLUMN evaluation_completed_units INTEGER NOT NULL DEFAULT 0"
        )
    if "evaluation_total_units" not in campaign_columns:
        await connection.execute(
            "ALTER TABLE campaigns ADD COLUMN evaluation_total_units INTEGER NOT NULL DEFAULT 0"
        )

    campaign_result_columns = await _table_columns(connection, "campaign_results")
    if "execution_profile" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN execution_profile TEXT"
        )
    if "context_policy_version" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN context_policy_version TEXT"
        )
    if "ground_truth_short" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN ground_truth_short TEXT"
        )
    if "key_points_json" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN key_points_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "ragas_focus_json" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN ragas_focus_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "source_attempt_id" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN source_attempt_id TEXT"
        )
    if "condition_id" not in campaign_result_columns:
        await connection.execute(
            "ALTER TABLE campaign_results ADD COLUMN condition_id TEXT NOT NULL DEFAULT ''"
        )

    work_item_columns = await _table_columns(connection, "evaluation_work_items")
    if "latest_success_attempt_id" not in work_item_columns:
        await connection.execute(
            "ALTER TABLE evaluation_work_items ADD COLUMN latest_success_attempt_id TEXT"
        )
    attempt_columns = await _table_columns(connection, "evaluation_attempts")
    if "output_json" not in attempt_columns:
        await connection.execute("ALTER TABLE evaluation_attempts ADD COLUMN output_json TEXT")

    ragas_score_columns = await _table_columns(connection, "ragas_scores")
    if "source_attempt_id" not in ragas_score_columns:
        await connection.execute(
            "ALTER TABLE ragas_scores ADD COLUMN source_attempt_id TEXT"
        )
    if "evaluation_signature" not in ragas_score_columns:
        await connection.execute(
            "ALTER TABLE ragas_scores ADD COLUMN evaluation_signature TEXT"
        )
    campaign_result_research_columns = {
        "question_version": "TEXT",
        "request_id": "TEXT",
        "started_at": "TEXT",
        "completed_at": "TEXT",
        "total_latency_ms": "REAL",
        "total_tokens": "INTEGER NOT NULL DEFAULT 0",
        "estimated_cost_usd": "REAL",
        "estimated_cost_twd": "REAL",
        "test_suite_id": "TEXT",
        "test_case_hash": "TEXT",
        "ground_truth_hash": "TEXT",
        "expected_evidence_hash": "TEXT",
        "knowledge_base_id": "TEXT",
        "index_version": "TEXT",
        "retriever_config_hash": "TEXT",
        "prompt_pack_version": "TEXT",
        "price_snapshot_id": "TEXT",
        "question_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
        "model_config_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
        "system_version_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
        "ablation_flags_json": "TEXT NOT NULL DEFAULT '{}'",
        "derived_metrics_json": "TEXT NOT NULL DEFAULT '{}'",
        "final_answer_hash": "TEXT",
    }
    for column_name, column_type in campaign_result_research_columns.items():
        if column_name not in campaign_result_columns:
            await connection.execute(
                f"ALTER TABLE campaign_results ADD COLUMN {column_name} {column_type}"
            )

    for table_name, column_definitions in _OBSERVABILITY_TABLE_COLUMNS.items():
        await _ensure_table_columns(connection, table_name, column_definitions)

    await connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ragas_scores_result_metric
        ON ragas_scores(campaign_result_id, metric_name)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ragas_scores_campaign_user_result
        ON ragas_scores(campaign_id, user_id, campaign_result_id, metric_name)
        """
    )
    await connection.execute("DROP INDEX IF EXISTS idx_campaign_results_unit_unique")
    await connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_results_unit_unique
        ON campaign_results(campaign_id, question_id, mode, run_number, condition_id)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_campaign_results_campaign_user_order
        ON campaign_results(
            campaign_id,
            user_id,
            created_at ASC,
            question_id ASC,
            mode ASC,
            run_number ASC,
            id ASC
        )
        """
    )
    await connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_traces_result
        ON agent_traces(campaign_result_id)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_trace_events_run_started
        ON evaluation_trace_events(run_id, started_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_trace_events_campaign_run
        ON evaluation_trace_events(campaign_id, run_id, sequence ASC, started_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_llm_calls_run_purpose
        ON evaluation_llm_calls(run_id, purpose)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_llm_calls_campaign_run
        ON evaluation_llm_calls(campaign_id, run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_retrieval_events_run_span
        ON evaluation_retrieval_events(run_id, span_id)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_retrieval_chunks_event
        ON evaluation_retrieval_chunks(retrieval_event_id)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_retrieval_chunks_run_event
        ON evaluation_retrieval_chunks(run_id, retrieval_event_id)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_retrieval_chunks_campaign_run
        ON evaluation_retrieval_chunks(
            campaign_id,
            run_id,
            retrieval_event_id,
            rank_after_rerank,
            created_at ASC
        )
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_context_packs_run_created
        ON evaluation_context_packs(run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_tool_calls_run_created
        ON evaluation_tool_calls(run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_routing_decisions_run_created
        ON evaluation_routing_decisions(run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_routing_decisions_campaign_run_created
        ON evaluation_routing_decisions(campaign_id, run_id, created_at ASC, routing_decision_id ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_claims_run_created
        ON evaluation_claims(run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_claims_campaign_run
        ON evaluation_claims(campaign_id, run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_human_ratings_run_created
        ON evaluation_human_ratings(run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_human_ratings_campaign_run
        ON evaluation_human_ratings(campaign_id, run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_graph_events_run_created
        ON evaluation_graph_events(run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_graph_events_campaign_run
        ON evaluation_graph_events(campaign_id, run_id, created_at ASC)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eval_graph_evidence_items_event
        ON evaluation_graph_evidence_items(graph_event_id, created_at ASC)
        """
    )


async def _table_columns(connection: aiosqlite.Connection, table_name: str) -> set[str]:
    cursor = await connection.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()
    return {str(row[1]) for row in rows}


async def _ensure_table_columns(
    connection: aiosqlite.Connection,
    table_name: str,
    column_definitions: dict[str, str],
) -> None:
    existing_columns = await _table_columns(connection, table_name)
    for column_name, column_definition in column_definitions.items():
        if column_name in existing_columns:
            continue
        await connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(payload: str | None, fallback: Any) -> Any:
    if not payload:
        return fallback
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return fallback


def _row_to_campaign_status(row: aiosqlite.Row) -> CampaignStatus:
    config_payload = _json_loads(row["config_json"], {})
    return CampaignStatus(
        id=row["id"],
        name=row["name"],
        status=CampaignLifecycleStatus(row["status"]),
        phase=row["phase"],
        config=CampaignConfig.model_validate(config_payload),
        completed_units=row["completed_units"],
        total_units=row["total_units"],
        evaluation_completed_units=row["evaluation_completed_units"],
        evaluation_total_units=row["evaluation_total_units"],
        current_question_id=row["current_question_id"],
        current_mode=row["current_mode"],
        error_message=row["error_message"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_campaign_result(row: aiosqlite.Row) -> CampaignResult:
    execution_profile = row["execution_profile"] if "execution_profile" in row.keys() else None
    context_policy_version = (
        row["context_policy_version"] if "context_policy_version" in row.keys() else None
    )
    question_snapshot = (
        _json_loads(row["question_snapshot_json"], {})
        if "question_snapshot_json" in row.keys()
        else {}
    )
    model_config_snapshot = (
        _json_loads(row["model_config_snapshot_json"], {})
        if "model_config_snapshot_json" in row.keys()
        else {}
    )
    system_version_snapshot = (
        _json_loads(row["system_version_snapshot_json"], {})
        if "system_version_snapshot_json" in row.keys()
        else {}
    )
    derived_metrics = (
        _json_loads(row["derived_metrics_json"], {})
        if "derived_metrics_json" in row.keys()
        else {}
    )
    request_id = row["request_id"] if "request_id" in row.keys() else None
    started_at = (
        datetime.fromisoformat(row["started_at"])
        if "started_at" in row.keys() and row["started_at"]
        else None
    )
    completed_at = (
        datetime.fromisoformat(row["completed_at"])
        if "completed_at" in row.keys() and row["completed_at"]
        else None
    )
    total_latency_ms = row["total_latency_ms"] if "total_latency_ms" in row.keys() else None
    total_tokens = row["total_tokens"] if "total_tokens" in row.keys() else None
    final_answer_hash = row["final_answer_hash"] if "final_answer_hash" in row.keys() else None
    snapshot_missing = (
        request_id is None
        and started_at is None
        and completed_at is None
        and total_latency_ms is None
        and not question_snapshot
        and not model_config_snapshot
        and not system_version_snapshot
        and not derived_metrics
        and final_answer_hash is None
    )
    if snapshot_missing and total_tokens == 0:
        total_tokens = None
    if not execution_profile and row["mode"] == "agentic":
        execution_profile = LEGACY_SHARED_PROFILE

    return CampaignResult(
        id=row["id"],
        campaign_id=row["campaign_id"],
        question_id=row["question_id"],
        question=row["question"],
        ground_truth=row["ground_truth"],
        ground_truth_short=row["ground_truth_short"] if "ground_truth_short" in row.keys() else None,
        key_points=_json_loads(row["key_points_json"], []) if "key_points_json" in row.keys() else [],
        ragas_focus=_json_loads(row["ragas_focus_json"], []) if "ragas_focus_json" in row.keys() else [],
        mode=row["mode"],
        execution_profile=execution_profile,
        context_policy_version=context_policy_version,
        run_number=row["run_number"],
        repeat_number=(
            derived_metrics["repeat_number"]
            if isinstance(derived_metrics.get("repeat_number"), int)
            else row["run_number"]
        ),
        condition_id=(row["condition_id"] or None) if "condition_id" in row.keys() else None,
        answer=row["answer"],
        contexts=_json_loads(row["contexts_json"], []),
        source_doc_ids=_json_loads(row["source_doc_ids_json"], []),
        expected_sources=_json_loads(row["expected_sources_json"], []),
        latency_ms=row["latency_ms"],
        token_usage=_json_loads(row["token_usage_json"], {}),
        category=row["category"],
        difficulty=row["difficulty"],
        question_version=row["question_version"] if "question_version" in row.keys() else None,
        request_id=request_id,
        started_at=started_at,
        completed_at=completed_at,
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
        question_snapshot=question_snapshot,
        model_config_snapshot=model_config_snapshot,
        system_version_snapshot=system_version_snapshot,
        derived_metrics=derived_metrics,
        final_answer_hash=final_answer_hash,
        source_attempt_id=(
            row["source_attempt_id"] if "source_attempt_id" in row.keys() else None
        ),
        status=CampaignResultStatus(row["status"]),
        error_message=row["error_message"],
        has_trace=bool(row["has_trace"]) if "has_trace" in row.keys() else False,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


@dataclass(frozen=True)
class CampaignAnalyticsResult:
    """Bounded result projection used by campaign analytics pages."""

    id: str
    campaign_id: str
    question_id: str
    question: str
    mode: str
    execution_profile: Optional[str]
    context_policy_version: Optional[str]
    run_number: int
    latency_ms: float
    total_latency_ms: Optional[float]
    total_tokens: Optional[int]
    category: Optional[str]
    difficulty: Optional[str]
    question_version: Optional[str]
    status: CampaignResultStatus
    error_message: Optional[str]
    derived_metrics: dict[str, Any]
    answer_preview: str
    created_at: datetime


def _normalize_route_profile(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return ROUTE_PROFILE_ALIASES.get(value, value)


def _normalize_trace_route_profiles(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["route_profile"] = _normalize_route_profile(normalized.get("route_profile"))
    steps = normalized.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            metadata = step.get("metadata")
            if isinstance(metadata, dict):
                metadata["route_profile"] = _normalize_route_profile(metadata.get("route_profile"))
    return normalized


def _row_to_agent_trace_detail(row: aiosqlite.Row) -> AgentTraceDetail:
    payload = _normalize_trace_route_profiles(_json_loads(row["trace_json"], {}))
    if not payload:
        raise AppError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Stored agent trace is invalid",
            status_code=500,
        )
    try:
        if not payload.get("execution_profile") and payload.get("mode") == "agentic":
            payload["execution_profile"] = LEGACY_SHARED_PROFILE
        return AgentTraceDetail.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse agent trace row %s: %s", row["id"], exc)
        raise AppError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Stored agent trace is invalid",
            status_code=500,
        ) from exc


class CampaignRepository:
    """CRUD operations for campaign lifecycle rows."""

    async def create(
        self,
        *,
        user_id: str,
        name: Optional[str],
        config: CampaignConfig,
    ) -> CampaignStatus:
        await init_db()
        campaign_id = str(uuid4())
        now = _utc_now_iso()
        total_units = len(config.test_case_ids) * len(config.modes) * config.repeat_count
        async with connect_db() as connection:
            await connection.execute(
                """
                INSERT INTO campaigns (
                    id, user_id, name, status, phase, config_json, completed_units, total_units,
                    evaluation_completed_units, evaluation_total_units, current_question_id,
                    current_mode, error_message, cancel_requested, created_at, started_at,
                    completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
                """,
                (
                    campaign_id,
                    user_id,
                    name,
                    CampaignLifecycleStatus.PENDING.value,
                    "execution",
                    _json_dumps(config.model_dump(mode="json", by_alias=True)),
                    0,
                    total_units,
                    now,
                    now,
                ),
            )
            await connection.commit()
        return await self.get(user_id=user_id, campaign_id=campaign_id)

    async def get(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM campaigns WHERE id = ? AND user_id = ?",
                (campaign_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Campaign not found",
                status_code=404,
            )
        return _row_to_campaign_status(row)

    async def list_by_user(self, *, user_id: str) -> list[CampaignStatus]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT * FROM campaigns WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [_row_to_campaign_status(row) for row in rows]

    async def list_inflight(self) -> list[tuple[str, CampaignStatus]]:
        """List non-terminal campaigns across all users for startup recovery."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM campaigns
                WHERE status IN (?, ?, ?)
                ORDER BY created_at ASC
                """,
                (
                    CampaignLifecycleStatus.PENDING.value,
                    CampaignLifecycleStatus.RUNNING.value,
                    CampaignLifecycleStatus.EVALUATING.value,
                ),
            )
            rows = await cursor.fetchall()
        return [(str(row["user_id"]), _row_to_campaign_status(row)) for row in rows]

    async def mark_running(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.RUNNING,
            phase="execution",
            started_at=_utc_now_iso(),
        )

    async def update_progress(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_completed_units: Optional[int] = None,
        evaluation_total_units: Optional[int] = None,
        current_question_id: Optional[str],
        current_mode: Optional[str],
    ) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            completed_units=completed_units,
            evaluation_completed_units=evaluation_completed_units,
            evaluation_total_units=evaluation_total_units,
            current_question_id=current_question_id,
            current_mode=current_mode,
        )

    async def mark_evaluating(
        self,
        *,
        user_id: str,
        campaign_id: str,
        evaluation_total_units: int,
    ) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.EVALUATING,
            phase="evaluation",
            evaluation_completed_units=0,
            evaluation_total_units=evaluation_total_units,
            error_message=None,
        )

    async def mark_completed(
        self,
        *,
        user_id: str,
        campaign_id: str,
        phase: Optional[str] = None,
    ) -> CampaignStatus:
        now = _utc_now_iso()
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.COMPLETED,
            phase=phase,
            completed_at=now,
            current_question_id=None,
            current_mode=None,
        )

    async def derive_execution_state(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        """Derive durable campaign execution state from its current ledger items."""
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(
                        CASE WHEN item.status = 'succeeded' AND EXISTS (
                            SELECT 1
                            FROM campaign_results AS result
                            WHERE result.campaign_id = job.campaign_id
                              AND result.user_id = job.user_id
                              AND result.source_attempt_id = work.latest_success_attempt_id
                        ) THEN 1 ELSE 0 END
                    ) AS compatible_succeeded,
                    SUM(CASE WHEN item.status = 'failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN item.status IN ('pending', 'running', 'retry_wait') THEN 1 ELSE 0 END) AS unresolved,
                    SUM(CASE WHEN item.status = 'succeeded' AND NOT EXISTS (
                        SELECT 1
                        FROM campaign_results AS result
                        WHERE result.campaign_id = job.campaign_id
                          AND result.user_id = job.user_id
                          AND result.source_attempt_id = work.latest_success_attempt_id
                    ) THEN 1 ELSE 0 END) AS incompatible
                FROM evaluation_job_items AS item
                JOIN evaluation_jobs AS job ON job.id = item.job_id
                JOIN evaluation_work_items AS work ON work.id = item.work_item_id
                WHERE job.user_id = ? AND job.campaign_id = ?
                  AND work.work_type = 'dataset_execution'
                """,
                (user_id, campaign_id),
            )
            row = await cursor.fetchone()
        total = int(row["total"] or 0)
        compatible_succeeded = int(row["compatible_succeeded"] or 0)
        failed = int(row["failed"] or 0)
        unresolved = int(row["unresolved"] or 0)
        incompatible = int(row["incompatible"] or 0)
        if total == 0 or unresolved:
            return await self._update_campaign(
                user_id=user_id,
                campaign_id=campaign_id,
                status=CampaignLifecycleStatus.PENDING if total == 0 else CampaignLifecycleStatus.RUNNING,
                phase="execution",
                completed_units=compatible_succeeded,
            )
        if (failed or incompatible) and compatible_succeeded:
            return await self._update_campaign(
                user_id=user_id,
                campaign_id=campaign_id,
                status=CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
                phase="execution",
                completed_units=compatible_succeeded,
                error_message="Some dataset execution units failed or lacked compatible results.",
                completed_at=_utc_now_iso(),
                current_question_id=None,
                current_mode=None,
            )
        if failed or incompatible:
            return await self.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message="No usable dataset execution result was produced.",
                phase="execution",
            )
        return await self.mark_completed(user_id=user_id, campaign_id=campaign_id, phase="execution")

    async def derive_ragas_state(
        self,
        *,
        user_id: str,
        campaign_id: str,
        job_id: str | None = None,
    ) -> CampaignStatus:
        """Derive evaluation lifecycle from durable RAGAS job items."""
        await init_db()
        # Cancellation is terminal.  Workers may finish their cancellation
        # cleanup after the campaign endpoint has marked the campaign
        # cancelled; never let that late derivation turn it into completed or
        # failed based on the now-terminal job items.
        campaign = await self.get(user_id=user_id, campaign_id=campaign_id)
        if campaign.status is CampaignLifecycleStatus.CANCELLED:
            return campaign
        async with connect_db() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN item.status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
                           SUM(CASE WHEN item.status = 'failed' THEN 1 ELSE 0 END) AS failed,
                           SUM(CASE WHEN item.status IN ('pending', 'running', 'retry_wait') THEN 1 ELSE 0 END) AS unresolved,
                           SUM(CASE WHEN item.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled
                    FROM evaluation_job_items AS item
                    JOIN evaluation_jobs AS job ON job.id = item.job_id
                    JOIN evaluation_work_items AS work ON work.id = item.work_item_id
                    WHERE job.user_id = ? AND job.campaign_id = ?
                      AND work.work_type = 'ragas_metric'
                      AND (? IS NULL OR job.id = ?)
                    """,
                    (user_id, campaign_id, job_id, job_id),
                )
            ).fetchone()
        total = int(row["total"] or 0)
        succeeded = int(row["succeeded"] or 0)
        failed = int(row["failed"] or 0)
        unresolved = int(row["unresolved"] or 0)
        cancelled = int(row["cancelled"] or 0)
        if campaign.cancel_requested and cancelled and not unresolved:
            return await self.mark_cancelled(user_id=user_id, campaign_id=campaign_id)
        if total == 0:
            return await self.mark_completed(user_id=user_id, campaign_id=campaign_id, phase="evaluation")
        if unresolved:
            return await self._update_campaign(
                user_id=user_id,
                campaign_id=campaign_id,
                status=CampaignLifecycleStatus.EVALUATING,
                phase="evaluation",
                evaluation_completed_units=succeeded,
                evaluation_total_units=total,
            )
        if failed and succeeded:
            return await self._update_campaign(
                user_id=user_id,
                campaign_id=campaign_id,
                status=CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
                phase="evaluation",
                evaluation_completed_units=succeeded,
                evaluation_total_units=total,
                error_message="Some RAGAS metrics failed; valid checkpoints were retained.",
                completed_at=_utc_now_iso(),
                current_question_id=None,
                current_mode=None,
            )
        if failed:
            return await self.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message="No RAGAS metric result was produced.",
                phase="evaluation",
            )
        return await self.mark_completed(user_id=user_id, campaign_id=campaign_id, phase="evaluation")

    async def mark_failed(
        self,
        *,
        user_id: str,
        campaign_id: str,
        error_message: str,
        phase: Optional[str] = None,
    ) -> CampaignStatus:
        now = _utc_now_iso()
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.FAILED,
            error_message=error_message,
            phase=phase,
            completed_at=now,
            current_question_id=None,
            current_mode=None,
        )

    async def mark_cancelled(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        now = _utc_now_iso()
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            status=CampaignLifecycleStatus.CANCELLED,
            completed_at=now,
            current_question_id=None,
            current_mode=None,
        )

    async def request_cancel(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        return await self._update_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            cancel_requested=True,
        )

    async def is_cancel_requested(self, *, user_id: str, campaign_id: str) -> bool:
        campaign = await self.get(user_id=user_id, campaign_id=campaign_id)
        return campaign.cancel_requested

    async def _update_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        status: Optional[CampaignLifecycleStatus] = None,
        phase: Optional[str] = None,
        completed_units: Optional[int] = None,
        evaluation_completed_units: Optional[int] = None,
        evaluation_total_units: Optional[int] = None,
        current_question_id: Optional[str] | object = _UNSET,
        current_mode: Optional[str] | object = _UNSET,
        error_message: Optional[str] | object = _UNSET,
        cancel_requested: Optional[bool] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> CampaignStatus:
        await init_db()
        updates: list[str] = ["updated_at = ?"]
        values: list[Any] = [_utc_now_iso()]

        if status is not None:
            updates.append("status = ?")
            values.append(status.value)
        if phase is not None:
            updates.append("phase = ?")
            values.append(phase)
        if completed_units is not None:
            updates.append("completed_units = ?")
            values.append(completed_units)
        if evaluation_completed_units is not None:
            updates.append("evaluation_completed_units = ?")
            values.append(evaluation_completed_units)
        if evaluation_total_units is not None:
            updates.append("evaluation_total_units = ?")
            values.append(evaluation_total_units)
        if current_question_id is not _UNSET:
            updates.append("current_question_id = ?")
            values.append(current_question_id)
        if current_mode is not _UNSET:
            updates.append("current_mode = ?")
            values.append(current_mode)
        if error_message is not _UNSET:
            updates.append("error_message = ?")
            values.append(error_message)
        if cancel_requested is not None:
            updates.append("cancel_requested = ?")
            values.append(1 if cancel_requested else 0)
        if started_at is not None:
            updates.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at)

        values.extend([campaign_id, user_id])

        async with connect_db() as connection:
            cursor = await connection.execute(
                f"UPDATE campaigns SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                values,
            )
            await connection.commit()
            if cursor.rowcount == 0:
                raise AppError(
                    code=ErrorCode.NOT_FOUND,
                    message="Campaign not found",
                    status_code=404,
                )
        return await self.get(user_id=user_id, campaign_id=campaign_id)


class CampaignResultRepository:
    """Persistence for per-unit campaign outputs."""

    async def list_for_campaign_analytics(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list["CampaignAnalyticsResult"]:
        """Read only the bounded fields needed by campaign analytics.

        Answers, contexts, ground truth, and snapshot JSON are intentionally
        excluded from this projection. Explicit result/detail/export paths use
        ``list_for_campaign`` or ``get`` when full payloads are required.
        """
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    id,
                    campaign_id,
                    question_id,
                    question,
                    mode,
                    execution_profile,
                    context_policy_version,
                    run_number,
                    latency_ms,
                    total_latency_ms,
                    total_tokens,
                    category,
                    difficulty,
                    question_version,
                    status,
                    error_message,
                    derived_metrics_json,
                    substr(answer, 1, 240) AS answer_preview,
                    created_at
                FROM campaign_results
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at ASC, question_id ASC, mode ASC, run_number ASC, id ASC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()
        return [
            CampaignAnalyticsResult(
                id=row["id"],
                campaign_id=row["campaign_id"],
                question_id=row["question_id"],
                question=row["question"],
                mode=row["mode"],
                execution_profile=row["execution_profile"],
                context_policy_version=row["context_policy_version"],
                run_number=row["run_number"],
                latency_ms=row["latency_ms"],
                total_latency_ms=row["total_latency_ms"],
                total_tokens=row["total_tokens"],
                category=row["category"],
                difficulty=row["difficulty"],
                question_version=row["question_version"],
                status=CampaignResultStatus(row["status"]),
                error_message=row["error_message"],
                derived_metrics=_json_loads(row["derived_metrics_json"], {}),
                answer_preview=row["answer_preview"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def create(
        self,
        *,
        result_id: Optional[str] = None,
        user_id: str,
        campaign_id: str,
        question_id: str,
        question: str,
        ground_truth: str,
        ground_truth_short: Optional[str],
        key_points: list[str],
        ragas_focus: list[str],
        mode: str,
        execution_profile: Optional[str],
        context_policy_version: Optional[str],
        run_number: int,
        answer: str,
        contexts: list[str],
        source_doc_ids: list[str],
        expected_sources: list[str],
        latency_ms: float,
        token_usage: dict[str, Any],
        category: Optional[str],
        difficulty: Optional[str],
        status: CampaignResultStatus,
        error_message: Optional[str] = None,
        question_version: Optional[str] = None,
        request_id: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        total_latency_ms: Optional[float] = None,
        total_tokens: Optional[int] = None,
        question_snapshot: Optional[dict[str, Any]] = None,
        model_config_snapshot: Optional[dict[str, Any]] = None,
        system_version_snapshot: Optional[dict[str, Any]] = None,
        derived_metrics: Optional[dict[str, Any]] = None,
        final_answer_hash: Optional[str] = None,
    ) -> CampaignResult:
        await init_db()
        result_id = result_id or str(uuid4())
        created_at = _utc_now_iso()
        async with connect_db() as connection:
            try:
                await connection.execute(
                    """
                    INSERT INTO campaign_results (
                        id, campaign_id, user_id, question_id, question, ground_truth,
                        ground_truth_short, key_points_json, ragas_focus_json, mode, execution_profile,
                        context_policy_version, run_number, answer, contexts_json, source_doc_ids_json,
                        expected_sources_json, latency_ms, token_usage_json, category,
                        difficulty, status, error_message, question_version, request_id,
                        started_at, completed_at, total_latency_ms, total_tokens,
                        question_snapshot_json, model_config_snapshot_json,
                        system_version_snapshot_json, derived_metrics_json, final_answer_hash,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result_id,
                        campaign_id,
                        user_id,
                        question_id,
                        question,
                        ground_truth,
                        ground_truth_short,
                        _json_dumps(key_points),
                        _json_dumps(ragas_focus),
                        mode,
                        execution_profile,
                        context_policy_version,
                        run_number,
                        answer,
                        _json_dumps(contexts),
                        _json_dumps(source_doc_ids),
                        _json_dumps(expected_sources),
                        latency_ms,
                        _json_dumps(token_usage),
                        category,
                        difficulty,
                        status.value,
                        error_message,
                        question_version,
                        request_id,
                        started_at,
                        completed_at,
                        total_latency_ms,
                        0 if total_tokens is None else total_tokens,
                        _json_dumps(question_snapshot or {}),
                        _json_dumps(model_config_snapshot or {}),
                        _json_dumps(system_version_snapshot or {}),
                        _json_dumps(derived_metrics or {}),
                        final_answer_hash,
                        created_at,
                    ),
                )
                await connection.commit()
            except aiosqlite.IntegrityError as exc:
                existing = await self.get_by_unit(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    question_id=question_id,
                    mode=mode,
                    run_number=run_number,
                )
                if existing is not None:
                    return existing
                raise exc
        return await self.get(user_id=user_id, campaign_id=campaign_id, result_id=result_id)

    async def get_by_unit(
        self,
        *,
        user_id: str,
        campaign_id: str,
        question_id: str,
        mode: str,
        run_number: int,
    ) -> CampaignResult | None:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_results.*,
                       EXISTS(
                           SELECT 1
                           FROM agent_traces
                           WHERE agent_traces.campaign_result_id = campaign_results.id
                       ) AS has_trace
                FROM campaign_results
                WHERE campaign_id = ?
                  AND user_id = ?
                  AND question_id = ?
                  AND mode = ?
                  AND run_number = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (campaign_id, user_id, question_id, mode, run_number),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_campaign_result(row)

    async def get(
        self,
        *,
        user_id: str,
        campaign_id: str,
        result_id: str,
    ) -> CampaignResult:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_results.*,
                       EXISTS(
                           SELECT 1
                           FROM agent_traces
                           WHERE agent_traces.campaign_result_id = campaign_results.id
                       ) AS has_trace
                FROM campaign_results
                WHERE id = ? AND campaign_id = ? AND user_id = ?
                """,
                (result_id, campaign_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Campaign result not found",
                status_code=404,
            )
        return _row_to_campaign_result(row)

    async def list_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list[CampaignResult]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_results.*,
                       EXISTS(
                           SELECT 1
                           FROM agent_traces
                           WHERE agent_traces.campaign_result_id = campaign_results.id
                       ) AS has_trace
                FROM campaign_results
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at ASC, question_id ASC, mode ASC, run_number ASC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()
        return [_row_to_campaign_result(row) for row in rows]


class AgentTraceRepository:
    """Persistence helpers for campaign-linked agent traces."""

    async def replace_for_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_result_id: str,
        trace_payload: dict[str, Any],
    ) -> AgentTraceDetail:
        await init_db()
        normalized_payload = _normalize_trace_route_profiles(trace_payload)
        steps = normalized_payload.get("steps", [])
        tool_call_count = sum(len(step.get("tool_calls", [])) for step in steps)
        total_tokens = sum(int(step.get("token_usage", {}).get("total_tokens", 0) or 0) for step in steps)
        detail = AgentTraceDetail.model_validate(
            {
                "trace_id": normalized_payload.get("trace_id") or str(uuid4()),
                "campaign_id": campaign_id,
                "campaign_result_id": campaign_result_id,
                "question_id": normalized_payload.get("question_id", ""),
                "question": normalized_payload.get("question", ""),
                "mode": normalized_payload.get("mode"),
                "execution_profile": (
                    normalized_payload.get("execution_profile")
                    or (LEGACY_SHARED_PROFILE if normalized_payload.get("mode") == "agentic" else None)
                ),
                "question_intent": normalized_payload.get("question_intent"),
                "strategy_tier": normalized_payload.get("strategy_tier"),
                "route_profile": normalized_payload.get("route_profile"),
                "required_coverage": normalized_payload.get("required_coverage", []),
                "coverage_gaps": normalized_payload.get("coverage_gaps", []),
                "subtask_coverage_status": normalized_payload.get("subtask_coverage_status", {}),
                "supported_claim_count": normalized_payload.get("supported_claim_count", 0),
                "unsupported_claim_count": normalized_payload.get("unsupported_claim_count", 0),
                "claims": normalized_payload.get("claims", []),
                "visual_verification_attempted": normalized_payload.get("visual_verification_attempted", False),
                "visual_tool_call_count": normalized_payload.get("visual_tool_call_count", 0),
                "visual_force_fallback_used": normalized_payload.get("visual_force_fallback_used", False),
                "run_number": normalized_payload.get("run_number", 1),
                "trace_status": normalized_payload.get("trace_status", "completed"),
                "summary": normalized_payload.get("summary", ""),
                "step_count": len(steps),
                "tool_call_count": tool_call_count,
                "total_tokens": total_tokens,
                "created_at": normalized_payload.get("created_at") or _utc_now_iso(),
                "steps": steps,
            }
        )
        async with connect_db() as connection:
            await connection.execute(
                """
                DELETE FROM agent_traces
                WHERE campaign_result_id = ? AND user_id = ?
                """,
                (campaign_result_id, user_id),
            )
            await connection.execute(
                """
                INSERT INTO agent_traces (
                    id, campaign_id, campaign_result_id, user_id, trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    detail.trace_id,
                    campaign_id,
                    campaign_result_id,
                    user_id,
                    _json_dumps(detail.model_dump(mode="json")),
                    detail.created_at.isoformat(),
                ),
            )
            await connection.commit()
        return detail

    async def list_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list[AgentTraceSummary]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM agent_traces
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at DESC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()
        return [summarize_agent_trace(_row_to_agent_trace_detail(row)) for row in rows]

    async def get_for_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_result_id: str,
    ) -> AgentTraceDetail:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM agent_traces
                WHERE campaign_id = ? AND campaign_result_id = ? AND user_id = ?
                """,
                (campaign_id, campaign_result_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Agent trace not found",
                status_code=404,
            )
        return _row_to_agent_trace_detail(row)


class RagasScoreRepository:
    """Persistence for per-result RAGAS metrics."""

    async def _insert_score_rows(
        self,
        *,
        connection: Any,
        user_id: str,
        campaign_id: str,
        score_rows: list[dict[str, Any]],
    ) -> None:
        for row in score_rows:
            await connection.execute(
                """
                INSERT INTO ragas_scores (
                    id, campaign_id, campaign_result_id, user_id, metric_name,
                    metric_value, details_json, source_attempt_id, evaluation_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_result_id, metric_name) DO UPDATE SET
                    metric_value = excluded.metric_value,
                    details_json = excluded.details_json,
                    source_attempt_id = COALESCE(excluded.source_attempt_id, ragas_scores.source_attempt_id),
                    evaluation_signature = COALESCE(excluded.evaluation_signature, ragas_scores.evaluation_signature),
                    created_at = excluded.created_at
                WHERE ragas_scores.evaluation_signature IS NOT NULL
                  AND excluded.evaluation_signature IS NOT NULL
                  AND ragas_scores.evaluation_signature = excluded.evaluation_signature
                """,
                (
                    str(uuid4()),
                    campaign_id,
                    row["campaign_result_id"],
                    user_id,
                    row["metric_name"],
                    row["metric_value"],
                    _json_dumps(row.get("details", {})),
                    row.get("source_attempt_id"),
                    row.get("evaluation_signature"),
                    _utc_now_iso(),
                ),
            )

    async def replace_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        score_rows: list[dict[str, Any]],
    ) -> None:
        await init_db()
        async with connect_db() as connection:
            await self._insert_score_rows(
                connection=connection,
                user_id=user_id,
                campaign_id=campaign_id,
                score_rows=score_rows,
            )
            await connection.commit()

    async def replace_for_campaign_subset(
        self,
        *,
        user_id: str,
        campaign_id: str,
        selected_result_ids: list[str],
        score_rows: list[dict[str, Any]],
    ) -> None:
        if not selected_result_ids:
            return
        await init_db()
        async with connect_db() as connection:
            await self._insert_score_rows(
                connection=connection,
                user_id=user_id,
                campaign_id=campaign_id,
                score_rows=score_rows,
            )
            await connection.commit()

    async def list_for_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> list[dict[str, Any]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_result_id, metric_name, metric_value, details_json,
                       evaluation_signature, source_attempt_id
                FROM ragas_scores
                WHERE campaign_id = ? AND user_id = ?
                ORDER BY created_at ASC
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()

        return [
            {
                "campaign_result_id": row["campaign_result_id"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "details": _json_loads(row["details_json"], {}),
                "evaluation_signature": row["evaluation_signature"],
                "source_attempt_id": row["source_attempt_id"],
            }
            for row in rows
        ]



