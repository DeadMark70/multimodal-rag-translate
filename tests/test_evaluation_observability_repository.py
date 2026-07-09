from datetime import datetime, timedelta, timezone
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


@pytest.mark.asyncio
async def test_observability_repository_lists_campaign_details_grouped_by_run(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    await _seed_campaign("campaign-bulk")
    await _seed_campaign("campaign-other")
    repository = EvaluationObservabilityRepository()
    created_at = datetime(2026, 7, 8, tzinfo=timezone.utc)

    await repository.record_trace_events(
        [
            EvaluationTraceEvent(
                event_id="trace-run-2",
                run_id="run-2",
                campaign_id="campaign-bulk",
                span_id="span-2",
                parent_event_id=None,
                parent_span_id=None,
                event_type="span_completed",
                event_schema_version="1.0",
                sequence=2,
                stage_type="generation",
                stage_name="generate",
                started_at=created_at + timedelta(seconds=2),
                ended_at=None,
                duration_ms=None,
                status="success",
                retry_count=0,
                payload={},
                error={},
                created_at=created_at + timedelta(seconds=2),
            ),
            EvaluationTraceEvent(
                event_id="trace-run-1",
                run_id="run-1",
                campaign_id="campaign-bulk",
                span_id="span-1",
                parent_event_id=None,
                parent_span_id=None,
                event_type="span_completed",
                event_schema_version="1.0",
                sequence=1,
                stage_type="retrieval",
                stage_name="retrieve",
                started_at=created_at,
                ended_at=None,
                duration_ms=None,
                status="success",
                retry_count=0,
                payload={},
                error={},
                created_at=created_at,
            ),
            EvaluationTraceEvent(
                event_id="trace-other",
                run_id="run-other",
                campaign_id="campaign-other",
                span_id="span-other",
                parent_event_id=None,
                parent_span_id=None,
                event_type="span_completed",
                event_schema_version="1.0",
                sequence=1,
                stage_type="retrieval",
                stage_name="retrieve",
                started_at=created_at,
                ended_at=None,
                duration_ms=None,
                status="success",
                retry_count=0,
                payload={},
                error={},
                created_at=created_at,
            ),
        ]
    )
    for run_id, campaign_id in (
        ("run-1", "campaign-bulk"),
        ("run-2", "campaign-bulk"),
        ("run-other", "campaign-other"),
    ):
        await repository.record_llm_call(
            EvaluationLlmCall(
                llm_call_id=f"llm-{run_id}",
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=f"span-{run_id}",
                provider="gemini",
                model_name="gemini-2.5-flash",
                purpose="generation",
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                estimated_cost_usd=0.01,
                estimated_cost_twd=0.32,
                prompt_hash=f"prompt-{run_id}",
                prompt_preview="preview",
                response_hash=f"response-{run_id}",
                latency_ms=10,
                status="success",
                error={},
                payload={},
                created_at=created_at + timedelta(seconds=1 if run_id == "run-2" else 0),
            )
        )
        await repository.record_retrieval_event(
            EvaluationRetrievalEvent(
                retrieval_event_id=f"retrieval-{run_id}",
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=f"span-{run_id}",
                query="question",
                query_hash=f"query-{run_id}",
                retriever_name="hybrid",
                top_k=5,
                result_count=1,
                latency_ms=8,
                payload={},
                created_at=created_at,
            )
        )
        await repository.record_retrieval_chunk(
            EvaluationRetrievalChunk(
                retrieval_chunk_id=f"chunk-row-{run_id}",
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=f"span-{run_id}",
                retrieval_event_id=f"retrieval-{run_id}",
                chunk_id=f"chunk-{run_id}",
                doc_id="paper.pdf",
                page_start=1,
                page_end=1,
                modality="text",
                rank_before_rerank=1,
                rank_after_rerank=1,
                dense_score=None,
                bm25_score=None,
                rerank_score=None,
                used_in_context=True,
                used_in_answer=True,
                expected_evidence_match=False,
                excerpt="excerpt",
                content_hash=f"content-{run_id}",
                payload={},
                created_at=created_at,
            )
        )
        await repository.record_claim(
            EvaluationClaim(
                claim_id=f"claim-{run_id}",
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=f"span-{run_id}",
                claim_text=f"Claim {run_id}",
                claim_type="fact",
                support_status="supported",
                evidence=[],
                unsupported_reason=None,
                payload={},
                created_at=created_at,
            )
        )
        await repository.record_human_rating(
            EvaluationHumanRating(
                human_rating_id=f"rating-{run_id}",
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=None,
                rater_id_hash="rater-hash",
                rubric_version="v1",
                correctness_score=0.9,
                faithfulness_score=0.8,
                completeness_score=0.7,
                citation_quality_score=0.6,
                usefulness_score=0.5,
                comments=None,
                is_blinded=True,
                shown_mode_label=False,
                payload={},
                created_at=created_at,
            )
        )

    trace_by_run = await repository.list_trace_events_for_campaign("campaign-bulk")
    llm_by_run = await repository.list_llm_calls_for_campaign("campaign-bulk")
    chunks_by_run = await repository.list_retrieval_chunks_for_campaign("campaign-bulk")
    claims_by_run = await repository.list_claims_for_campaign("campaign-bulk")
    ratings_by_run = await repository.list_human_ratings_for_campaign("campaign-bulk")

    assert set(trace_by_run) == {"run-1", "run-2"}
    assert set(llm_by_run) == {"run-1", "run-2"}
    assert set(chunks_by_run) == {"run-1", "run-2"}
    assert set(claims_by_run) == {"run-1", "run-2"}
    assert set(ratings_by_run) == {"run-1", "run-2"}
    assert trace_by_run["run-1"][0].event_id == "trace-run-1"
    assert trace_by_run["run-2"][0].event_id == "trace-run-2"
    assert llm_by_run["run-1"][0].llm_call_id == "llm-run-1"
    assert chunks_by_run["run-2"][0].retrieval_chunk_id == "chunk-row-run-2"
    assert claims_by_run["run-1"][0].claim_text == "Claim run-1"
    assert ratings_by_run["run-2"][0].human_rating_id == "rating-run-2"
