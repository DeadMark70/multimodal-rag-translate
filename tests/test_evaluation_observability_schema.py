import sqlite3
from datetime import datetime, timezone

import pytest

from evaluation import db as evaluation_db
from evaluation.trace_schemas import (
    AgentTraceDetail,
    EvaluationClaim,
    EvaluationEvidencePacket,
    EvaluationSlotResolution,
    EvaluationTraceEvent,
)

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
    "evaluation_evidence_packets",
    "evaluation_slot_resolutions",
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
    "evaluation_evidence_packets",
    "evaluation_slot_resolutions",
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
        connection.execute(
            "CREATE TABLE evaluation_evidence_packets (evidence_packet_row_id TEXT PRIMARY KEY)"
        )
        connection.execute(
            "CREATE TABLE evaluation_slot_resolutions (slot_resolution_row_id TEXT PRIMARY KEY)"
        )
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
    evidence_columns = await _table_columns("evaluation_evidence_packets")
    resolution_columns = await _table_columns("evaluation_slot_resolutions")
    assert {"attempt_id", "schema_version", "evidence_id", "packet_json"}.issubset(
        evidence_columns
    )
    assert {"attempt_id", "schema_version", "slot_id", "resolution_json"}.issubset(
        resolution_columns
    )


@pytest.mark.asyncio
async def test_v9_evidence_tables_have_identity_schema_and_idempotency_indexes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")

    await evaluation_db.init_db()

    evidence_columns = await _table_columns("evaluation_evidence_packets")
    resolution_columns = await _table_columns("evaluation_slot_resolutions")
    required_columns = {
        "attempt_id",
        "run_id",
        "campaign_id",
        "condition_id",
        "schema_version",
        "created_at",
    }
    assert required_columns.issubset(evidence_columns)
    assert required_columns.issubset(resolution_columns)
    assert {"evidence_id", "packet_json"}.issubset(evidence_columns)
    assert {"slot_id", "resolution_stage", "resolution_json"}.issubset(resolution_columns)

    index_names = await _index_names()
    assert "idx_eval_evidence_packets_attempt_evidence" in index_names
    assert "idx_eval_slot_resolutions_attempt_slot_stage" in index_names
    assert await _index_columns("idx_eval_evidence_packets_attempt_evidence") == [
        "attempt_id",
        "evidence_id",
    ]
    assert await _index_columns("idx_eval_slot_resolutions_attempt_slot_stage") == [
        "attempt_id",
        "slot_id",
        "resolution_stage",
    ]


def test_v9_records_and_legacy_agent_trace_have_safe_defaults() -> None:
    now = datetime.now(timezone.utc)
    evidence = EvaluationEvidencePacket(
        attempt_id="attempt-1",
        run_id="run-1",
        campaign_id="campaign-1",
        condition_id="condition-1",
        evidence_id="evidence-1",
        packet={"statement": "Bounded evidence."},
        created_at=now,
    )
    resolution = EvaluationSlotResolution(
        attempt_id="attempt-1",
        run_id="run-1",
        campaign_id="campaign-1",
        condition_id="condition-1",
        slot_id="slot-1",
        resolution_stage="sufficiency",
        resolution={"status": "supported"},
        created_at=now,
    )
    legacy_trace = AgentTraceDetail(
        trace_id="trace-1",
        campaign_id="campaign-1",
        campaign_result_id="result-1",
        question_id="question-1",
        question="Question?",
        mode="agentic",
        run_number=1,
        trace_status="completed",
        created_at=now,
    )

    assert evidence.schema_version == "1"
    assert resolution.schema_version == "1"
    assert legacy_trace.agentic_v9 is None


def test_existing_claim_and_context_pack_accept_optional_v9_attempt_metadata() -> None:
    now = datetime.now(timezone.utc)
    claim = EvaluationClaim(
        claim_id="claim-v9",
        run_id="run-1",
        campaign_id="campaign-1",
        claim_text="Claim.",
        attempt_id="attempt-1",
        condition_id="condition-1",
        schema_version="1",
        created_at=now,
    )
    from evaluation.trace_schemas import EvaluationContextPack

    context_pack = EvaluationContextPack(
        context_pack_id="pack-v9",
        run_id="run-1",
        campaign_id="campaign-1",
        attempt_id="attempt-1",
        condition_id="condition-1",
        schema_version="1",
        created_at=now,
    )

    assert claim.attempt_id == "attempt-1"
    assert context_pack.condition_id == "condition-1"


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
