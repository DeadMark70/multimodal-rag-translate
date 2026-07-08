from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from evaluation import db as evaluation_db
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationContextPack,
    EvaluationHumanRating,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationRetrievalEvent,
    EvaluationRoutingDecision,
    EvaluationToolCall,
    EvaluationTraceEvent,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _seed_campaign(campaign_id: str) -> None:
    now = _now().isoformat()
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO campaigns (
                id, user_id, name, status, phase, config_json, completed_units, total_units,
                evaluation_completed_units, evaluation_total_units, current_question_id,
                current_mode, error_message, cancel_requested, created_at, started_at,
                completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
            """,
            (
                campaign_id,
                "user-a",
                "Observability repository test",
                "pending",
                "execution",
                "{}",
                now,
                now,
            ),
        )
        await connection.commit()


@pytest.mark.asyncio
async def test_observability_repository_round_trips_run_details(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    repository = EvaluationObservabilityRepository()
    created_at = _now()
    await _seed_campaign("campaign-1")

    await repository.record_trace_event(
        EvaluationTraceEvent(
            event_id="event-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            parent_event_id=None,
            parent_span_id=None,
            event_type="span_completed",
            event_schema_version="1.0",
            sequence=1,
            stage_type="retrieval",
            stage_name="retrieve",
            started_at=created_at,
            ended_at=created_at,
            duration_ms=12.5,
            status="success",
            retry_count=0,
            payload={"query_hash": "hash-1"},
            error={},
            created_at=created_at,
        )
    )
    await repository.record_llm_call(
        EvaluationLlmCall(
            llm_call_id="llm-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            provider="gemini",
            model_name="gemini-2.5-flash",
            purpose="generation",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            estimated_cost_usd=0.01,
            estimated_cost_twd=0.32,
            prompt_hash="prompt-hash",
            prompt_preview="prompt preview",
            response_hash="response-hash",
            latency_ms=30.5,
            status="success",
            error={},
            payload={"temperature": 0.2},
            created_at=created_at,
        )
    )
    await repository.record_retrieval_event(
        EvaluationRetrievalEvent(
            retrieval_event_id="retrieval-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            query="question",
            query_hash="query-hash",
            retriever_name="hybrid",
            top_k=5,
            result_count=1,
            latency_ms=9.5,
            payload={"fusion": "rrf"},
            created_at=created_at,
        )
    )
    await repository.record_retrieval_chunk(
        EvaluationRetrievalChunk(
            retrieval_chunk_id="chunk-row-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            retrieval_event_id="retrieval-1",
            chunk_id="chunk-1",
            doc_id="paper-a.pdf",
            page_start=5,
            page_end=5,
            modality="table",
            rank_before_rerank=3,
            rank_after_rerank=1,
            dense_score=0.7,
            bm25_score=0.3,
            rerank_score=0.91,
            used_in_context=True,
            used_in_answer=True,
            expected_evidence_match=True,
            excerpt="reported value 0.9079",
            content_hash="content-hash",
            payload={"section": "results"},
            created_at=created_at,
        )
    )
    await repository.record_context_pack(
        EvaluationContextPack(
            context_pack_id="context-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            input_chunk_count=3,
            packed_chunk_count=1,
            token_count=256,
            retrieved_but_not_packed_evidence=[{"evidence_id": "E2"}],
            payload={"policy": "default"},
            created_at=created_at,
        )
    )
    await repository.record_tool_call(
        EvaluationToolCall(
            tool_call_id="tool-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            tool_name="visual_reexamine",
            action="inspect_table",
            latency_ms=50,
            status="success",
            payload={"page": 5},
            created_at=created_at,
        )
    )
    await repository.record_routing_decision(
        EvaluationRoutingDecision(
            routing_decision_id="route-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            selected_mode="agentic",
            analysis_type="retrospective",
            confidence=0.82,
            reason="multi-modal evidence required",
            payload={"complexity": "high"},
            created_at=created_at,
        )
    )
    await repository.record_claim(
        EvaluationClaim(
            claim_id="claim-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id="span-1",
            claim_text="The value is 0.9079.",
            claim_type="numeric",
            support_status="supported",
            evidence=[{"doc_id": "paper-a.pdf", "page": 5}],
            unsupported_reason=None,
            payload={"verifier": "auto"},
            created_at=created_at,
        )
    )
    await repository.record_human_rating(
        EvaluationHumanRating(
            human_rating_id="rating-1",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id=None,
            rater_id_hash="hash-rater",
            rubric_version="v1",
            correctness_score=0.9,
            faithfulness_score=0.8,
            completeness_score=0.7,
            citation_quality_score=0.6,
            usefulness_score=0.5,
            comments="Looks supported.",
            is_blinded=True,
            shown_mode_label=False,
            payload={"source": "panel"},
            created_at=created_at,
        )
    )

    assert (await repository.list_trace_events_for_run("run-1"))[0].payload["query_hash"] == "hash-1"
    assert (await repository.list_llm_calls_for_run("run-1"))[0].total_tokens == 30
    assert (await repository.list_retrieval_events_for_run("run-1"))[0].payload["fusion"] == "rrf"
    assert (await repository.list_retrieval_chunks_for_run("run-1"))[0].expected_evidence_match is True
    assert (await repository.list_context_packs_for_run("run-1"))[0].retrieved_but_not_packed_evidence == [
        {"evidence_id": "E2"}
    ]
    assert (await repository.list_tool_calls_for_run("run-1"))[0].payload["page"] == 5
    assert (await repository.list_routing_decisions_for_run("run-1"))[0].selected_mode == "agentic"
    assert (await repository.list_claims_for_run("run-1"))[0].evidence[0]["doc_id"] == "paper-a.pdf"
    assert (await repository.list_human_ratings_for_run("run-1"))[0].is_blinded is True


@pytest.mark.asyncio
async def test_observability_repository_serializes_common_non_json_payload_values(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await _seed_campaign("campaign-json-safe")
    repository = EvaluationObservabilityRepository()

    created_at = datetime(2026, 7, 8, tzinfo=timezone.utc)
    await repository.record_trace_event(
        EvaluationTraceEvent(
            event_id="event-json-safe",
            run_id="run-json-safe",
            campaign_id="campaign-json-safe",
            span_id="span-json-safe",
            parent_event_id=None,
            parent_span_id=None,
            event_type="span",
            event_schema_version="1.0",
            sequence=1,
            stage_type="retrieval",
            stage_name="retrieve",
            started_at=created_at,
            ended_at=None,
            duration_ms=None,
            status="running",
            retry_count=0,
            payload={
                "observed_at": created_at,
                "request_uuid": UUID("12345678-1234-5678-1234-567812345678"),
                "price": Decimal("0.125"),
            },
            error={},
            created_at=created_at,
        )
    )

    events = await repository.list_trace_events_for_run("run-json-safe")
    assert events[0].payload == {
        "observed_at": "2026-07-08T00:00:00+00:00",
        "request_uuid": "12345678-1234-5678-1234-567812345678",
        "price": "0.125",
    }
