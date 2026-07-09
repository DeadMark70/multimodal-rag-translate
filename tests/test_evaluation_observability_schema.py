import sqlite3
from datetime import datetime, timezone

import pytest

from evaluation import db as evaluation_db
from evaluation.trace_schemas import EvaluationClaim, EvaluationTraceEvent

EXPECTED_TABLES = {
    "evaluation_trace_events",
    "evaluation_llm_calls",
    "evaluation_retrieval_events",
    "evaluation_retrieval_chunks",
    "evaluation_context_packs",
    "evaluation_tool_calls",
    "evaluation_routing_decisions",
    "evaluation_claims",
    "evaluation_human_ratings",
}

EXPECTED_CAMPAIGN_RESULT_COLUMNS = {
    "question_version",
    "request_id",
    "started_at",
    "completed_at",
    "total_latency_ms",
    "total_tokens",
    "estimated_cost_usd",
    "estimated_cost_twd",
    "test_suite_id",
    "test_case_hash",
    "ground_truth_hash",
    "expected_evidence_hash",
    "knowledge_base_id",
    "index_version",
    "retriever_config_hash",
    "prompt_pack_version",
    "price_snapshot_id",
    "question_snapshot_json",
    "model_config_snapshot_json",
    "system_version_snapshot_json",
    "ablation_flags_json",
    "derived_metrics_json",
    "final_answer_hash",
}

COMMON_DETAIL_COLUMNS = {
    "run_id",
    "campaign_id",
    "span_id",
    "payload_json",
    "created_at",
}

COMMON_DETAIL_TABLES = EXPECTED_TABLES - {
    "evaluation_trace_events",
    "evaluation_retrieval_chunks",
}


async def _table_names() -> set[str]:
    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        rows = await cursor.fetchall()
    return {str(row["name"]) for row in rows}


async def _table_columns(table_name: str) -> set[str]:
    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(f"PRAGMA table_info({table_name})")
        rows = await cursor.fetchall()
    return {str(row["name"]) for row in rows}


async def _index_names() -> set[str]:
    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
        rows = await cursor.fetchall()
    return {str(row["name"]) for row in rows}


async def _index_columns(index_name: str) -> list[str]:
    async with evaluation_db.connect_db() as connection:
        cursor = await connection.execute(f"PRAGMA index_info({index_name})")
        rows = await cursor.fetchall()
    return [str(row["name"]) for row in rows]


@pytest.mark.asyncio
async def test_observability_tables_columns_and_indexes_are_created(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")

    await evaluation_db.init_db()

    assert EXPECTED_TABLES.issubset(await _table_names())

    campaign_result_columns = await _table_columns("campaign_results")
    assert EXPECTED_CAMPAIGN_RESULT_COLUMNS.issubset(campaign_result_columns)

    for table_name in COMMON_DETAIL_TABLES:
        assert COMMON_DETAIL_COLUMNS.issubset(await _table_columns(table_name))

    trace_columns = await _table_columns("evaluation_trace_events")
    assert {
        "event_id",
        "run_id",
        "campaign_id",
        "span_id",
        "parent_span_id",
        "event_schema_version",
        "sequence",
        "stage_type",
        "duration_ms",
        "status",
    }.issubset(trace_columns)

    retrieval_chunk_columns = await _table_columns("evaluation_retrieval_chunks")
    assert {
        "retrieval_chunk_id",
        "retrieval_event_id",
        "rank_before_rerank",
        "rank_after_rerank",
        "used_in_context",
        "used_in_answer",
        "expected_evidence_match",
    }.issubset(retrieval_chunk_columns)

    context_pack_columns = await _table_columns("evaluation_context_packs")
    assert "retrieved_but_not_packed_evidence_json" in context_pack_columns

    index_names = await _index_names()
    assert {
        "idx_eval_trace_events_run_started",
        "idx_eval_llm_calls_run_purpose",
        "idx_eval_retrieval_events_run_span",
        "idx_eval_claims_run_created",
        "idx_eval_human_ratings_run_created",
        "idx_eval_retrieval_chunks_event",
    }.issubset(index_names)

    campaign_index_prefixes = {
        "idx_eval_trace_events_campaign_run": ["campaign_id", "run_id"],
        "idx_eval_llm_calls_campaign_run": ["campaign_id", "run_id"],
        "idx_eval_retrieval_chunks_campaign_run": ["campaign_id", "run_id"],
        "idx_eval_claims_campaign_run": ["campaign_id", "run_id"],
        "idx_eval_human_ratings_campaign_run": ["campaign_id", "run_id"],
    }
    assert campaign_index_prefixes.keys() <= index_names
    for index_name, expected_prefix in campaign_index_prefixes.items():
        assert (await _index_columns(index_name))[: len(expected_prefix)] == expected_prefix


@pytest.mark.asyncio
async def test_observability_migration_repairs_partial_tables(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "evaluation.db"
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evaluation_trace_events (event_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE evaluation_claims (claim_id TEXT PRIMARY KEY)")
        connection.commit()

    await evaluation_db.init_db()

    trace_columns = await _table_columns("evaluation_trace_events")
    assert {"run_id", "campaign_id", "span_id", "event_schema_version", "duration_ms"}.issubset(
        trace_columns
    )
    claim_columns = await _table_columns("evaluation_claims")
    assert {"run_id", "campaign_id", "span_id", "evidence_json", "payload_json"}.issubset(
        claim_columns
    )


def test_evaluation_trace_event_allows_running_nullable_duration() -> None:
    event = EvaluationTraceEvent(
        event_id="evt-1",
        run_id="run-1",
        campaign_id="campaign-1",
        span_id="span-1",
        parent_event_id=None,
        parent_span_id=None,
        event_type="span_started",
        event_schema_version="1.0",
        sequence=1,
        stage_type="retrieval",
        stage_name="retrieve contexts",
        started_at=datetime.now(timezone.utc),
        ended_at=None,
        duration_ms=None,
        status="running",
        retry_count=0,
        payload={},
        error={},
        created_at=datetime.now(timezone.utc),
    )

    assert event.status == "running"
    assert event.duration_ms is None


def test_evaluation_claim_exposes_decoded_evidence_field() -> None:
    claim = EvaluationClaim(
        claim_id="claim-1",
        run_id="run-1",
        campaign_id="campaign-1",
        span_id=None,
        claim_text="The reported value is 0.9079.",
        evidence=[{"doc_id": "paper-a.pdf", "page": 5}],
        created_at=datetime.now(timezone.utc),
    )

    assert claim.evidence[0]["doc_id"] == "paper-a.pdf"
