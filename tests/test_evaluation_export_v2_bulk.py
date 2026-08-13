from datetime import datetime, timezone
from math import inf, nan
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.errors import AppError
from evaluation import observability_storage, research_analytics
from evaluation.accounting_schemas import (
    AccountingScope,
    AccountingScopeTarget,
    UsageEvent,
)
from evaluation.accounting_store import (
    CampaignAccountingSnapshot,
    EvaluationAccountingStore,
)
from evaluation.campaign_schemas import CampaignResult
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem
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
    EvaluationV9AttemptMaterialization,
)


NOW = datetime(2026, 8, 13, tzinfo=timezone.utc)


def _result(index: int = 1, *, source_attempt_id: str | None = None) -> CampaignResult:
    return CampaignResult(
        id=f"run-{index}",
        campaign_id="campaign-1",
        question_id=f"question-{index}",
        question="Question?",
        ground_truth="Answer.",
        mode="agentic",
        run_number=index,
        agentic_execution_version="v9" if source_attempt_id else "v8",
        answer="Safe answer.",
        latency_ms=12,
        total_tokens=9 if source_attempt_id else None,
        source_attempt_id=source_attempt_id,
        status="completed",
        created_at=NOW,
    )


def _scope(run_id: str) -> AccountingScope:
    return AccountingScope(
        scope_id=f"scope-{run_id}",
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key=run_id,
        run_id=run_id,
        accounting_schema_version="2",
        status="completed",
        observed_call_count=1,
        measured_call_count=1,
        started_at=NOW,
        completed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        targets=[
            AccountingScopeTarget(
                campaign_result_id=run_id,
                job_id=f"job-{run_id}",
                work_item_id=f"work-{run_id}",
                attempt_id=f"attempt-{run_id}",
                is_official=True,
            )
        ],
    )


def _usage_event(run_id: str, *, total: int = 9) -> UsageEvent:
    return UsageEvent(
        usage_event_id=f"usage-{run_id}",
        scope_id=f"scope-{run_id}",
        campaign_id="campaign-1",
        scope_type="execution_run",
        scope_key=run_id,
        run_id=run_id,
        phase="answer_generation",
        purpose="generation",
        input_tokens=5,
        output_text_tokens=total - 5,
        usage_status="measured",
        reconciliation_status="balanced",
        pricing_status="missing_price",
        created_at=NOW,
    )


def _accounting_snapshot(results: list[CampaignResult]) -> CampaignAccountingSnapshot:
    return CampaignAccountingSnapshot(
        scopes_by_run_id={result.id: [_scope(result.id)] for result in results},
        events_by_scope_id={
            f"scope-{result.id}": [_usage_event(result.id)] for result in results
        },
    )


def _empty_snapshot(*, run_ids: tuple[str, ...] = ("run-1",)):
    snapshot_type = observability_storage.CampaignObservabilitySnapshot

    def grouped() -> dict[str, list]:
        return {run_id: [] for run_id in run_ids}

    return snapshot_type(
        trace_events_by_run_id=grouped(),
        llm_calls_by_run_id=grouped(),
        retrieval_events_by_run_id=grouped(),
        retrieval_chunks_by_run_id=grouped(),
        context_packs_by_run_id=grouped(),
        tool_calls_by_run_id=grouped(),
        routing_decisions_by_run_id=grouped(),
        graph_events_by_run_id=grouped(),
        graph_evidence_items_by_run_id=grouped(),
        claims_by_run_id=grouped(),
        human_ratings_by_run_id=grouped(),
        materializations_by_run_id=grouped(),
        evidence_packets_by_run_id=grouped(),
        slot_resolutions_by_run_id=grouped(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("run_count", [1, 50])
async def test_bulk_canonical_observability_uses_one_campaign_snapshot_without_n_plus_one(
    run_count: int,
) -> None:
    results = [_result(index) for index in range(1, run_count + 1)]
    observability = AsyncMock(spec_set=EvaluationObservabilityRepository)
    accounting = AsyncMock(spec_set=EvaluationAccountingStore)
    observability.load_campaign_observability_snapshot.return_value = _empty_snapshot(
        run_ids=tuple(result.id for result in results)
    )
    accounting.load_campaign_snapshot.return_value = _accounting_snapshot(results)
    for method_name in (
        "list_trace_events_for_run",
        "list_llm_calls_for_run",
        "list_retrieval_events_for_run",
        "list_retrieval_chunks_for_run",
        "list_context_packs_for_run",
        "list_tool_calls_for_run",
        "list_routing_decisions_for_run",
        "list_graph_events_for_run",
        "list_graph_evidence_items_for_run",
        "list_claims_for_run",
        "list_human_ratings_for_run",
        "get_v9_attempt_materialization",
        "list_evidence_packets_for_attempt",
        "list_slot_resolutions_for_attempt",
    ):
        getattr(observability, method_name).side_effect = AssertionError(
            f"per-entity loader called: {method_name}"
        )
    service = research_analytics.ResearchAnalyticsService(
        campaigns=SimpleNamespace(get=AsyncMock()),
        observability=observability,
        accounting=accounting,
    )

    details = await service.get_campaign_run_observability(
        user_id="user-1", campaign_id="campaign-1", results=results
    )

    assert list(details) == [result.id for result in results]
    observability.load_campaign_observability_snapshot.assert_awaited_once_with(
        "campaign-1"
    )
    accounting.load_campaign_snapshot.assert_awaited_once_with("campaign-1")


def _many_rows_snapshot():
    snapshot = _empty_snapshot()
    trace_events = [
        EvaluationTraceEvent(
            event_id=f"trace-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            span_id=f"span-{index}",
            event_type="span_completed",
            sequence=index + 1,
            stage_type="retrieval",
            stage_name="retrieve",
            started_at=NOW,
            status="success",
            created_at=NOW,
        )
        for index in range(101)
    ]
    llm_calls = [
        EvaluationLlmCall(
            llm_call_id=f"llm-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            created_at=NOW,
        )
        for index in range(101)
    ]
    retrieval_events = [
        EvaluationRetrievalEvent(
            retrieval_event_id=f"retrieval-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            created_at=NOW,
        )
        for index in range(101)
    ]
    retrieval_chunks = [
        EvaluationRetrievalChunk(
            retrieval_chunk_id=f"chunk-row-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            retrieval_event_id=f"retrieval-{index}",
            chunk_id=f"chunk-{index}",
            created_at=NOW,
        )
        for index in range(101)
    ]
    context_packs = [
        EvaluationContextPack(
            context_pack_id=f"context-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            created_at=NOW,
        )
        for index in range(101)
    ]
    tool_calls = [
        EvaluationToolCall(
            tool_call_id=f"tool-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            tool_name="search",
            created_at=NOW,
        )
        for index in range(101)
    ]
    routing_decisions = [
        EvaluationRoutingDecision(
            routing_decision_id=f"routing-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            selected_mode="agentic",
            created_at=NOW,
        )
        for index in range(101)
    ]
    graph_events = [
        EvaluationGraphEvent(
            graph_event_id=f"graph-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            graph_query="question",
            graph_search_mode="local",
            graph_route="local",
            created_at=NOW,
        )
        for index in range(101)
    ]
    graph_evidence = [
        EvaluationGraphEvidenceItem(
            graph_evidence_item_id=f"graph-evidence-{index}",
            graph_event_id=f"graph-{index}",
            created_at=NOW,
        )
        for index in range(101)
    ]
    claims = [
        EvaluationClaim(
            claim_id=f"claim-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            claim_text=f"Claim {index}",
            created_at=NOW,
        )
        for index in range(101)
    ]
    ratings = [
        EvaluationHumanRating(
            human_rating_id=f"rating-{index}",
            run_id="run-1",
            campaign_id="campaign-1",
            rater_id_hash="rater",
            rubric_version="v1",
            correctness_score=1,
            faithfulness_score=1,
            completeness_score=1,
            citation_quality_score=1,
            usefulness_score=1,
            created_at=NOW,
        )
        for index in range(101)
    ]
    return snapshot.__class__(
        trace_events_by_run_id={"run-1": trace_events},
        llm_calls_by_run_id={"run-1": llm_calls},
        retrieval_events_by_run_id={"run-1": retrieval_events},
        retrieval_chunks_by_run_id={"run-1": retrieval_chunks},
        context_packs_by_run_id={"run-1": context_packs},
        tool_calls_by_run_id={"run-1": tool_calls},
        routing_decisions_by_run_id={"run-1": routing_decisions},
        graph_events_by_run_id={"run-1": graph_events},
        graph_evidence_items_by_run_id={"run-1": graph_evidence},
        claims_by_run_id={"run-1": claims},
        human_ratings_by_run_id={"run-1": ratings},
        materializations_by_run_id={"run-1": []},
        evidence_packets_by_run_id={"run-1": []},
        slot_resolutions_by_run_id={"run-1": []},
    )


def test_canonical_builder_never_truncates_normalized_event_families() -> None:
    canonical = research_analytics._build_canonical_run_observability(
        result=_result(),
        observability=_many_rows_snapshot(),
        accounting=_accounting_snapshot([_result()]),
    )

    assert canonical.trace_events[-1].event_id == "trace-100"
    assert canonical.llm_calls[-1].llm_call_id == "llm-100"
    assert canonical.retrieval_events[-1].retrieval_event_id == "retrieval-100"
    assert canonical.retrieval_chunks[-1].retrieval_chunk_id == "chunk-row-100"
    assert canonical.context_packs[-1].context_pack_id == "context-100"
    assert canonical.tool_calls[-1].tool_call_id == "tool-100"
    assert canonical.routing_decisions[-1].routing_decision_id == "routing-100"
    assert canonical.graph_events[-1].graph_event_id == "graph-100"
    assert (
        canonical.graph_evidence_items[-1].graph_evidence_item_id
        == "graph-evidence-100"
    )
    assert canonical.claims[-1].claim_id == "claim-100"
    assert canonical.human_ratings[-1].human_rating_id == "rating-100"


def test_canonical_builder_selects_exact_source_attempt_independent_of_order() -> None:
    result = _result(source_attempt_id="attempt-current")
    snapshot = _empty_snapshot()
    wrong = EvaluationV9AttemptMaterialization(
        attempt_id="attempt-wrong",
        run_id=result.id,
        campaign_id=result.campaign_id,
        trace_payload={"comparison": {"attempt": "wrong"}},
        created_at=NOW,
    )
    current = EvaluationV9AttemptMaterialization(
        attempt_id="attempt-current",
        run_id=result.id,
        campaign_id=result.campaign_id,
        trace_payload={"comparison": {"attempt": "current"}},
        created_at=NOW,
    )
    snapshot.materializations_by_run_id[result.id].extend([wrong, current])
    canonical = research_analytics._build_canonical_run_observability(
        result=result,
        observability=snapshot,
        accounting=_accounting_snapshot([result]),
    )

    assert canonical.agentic_v9 is not None
    assert canonical.agentic_v9.comparison == {"attempt": "current"}


@pytest.mark.asyncio
async def test_selected_and_bulk_token_breakdown_match_with_v9_provider_attempts() -> (
    None
):
    result = _result(source_attempt_id="attempt-current")
    snapshot = _empty_snapshot()
    snapshot.materializations_by_run_id[result.id].append(
        EvaluationV9AttemptMaterialization(
            attempt_id="attempt-current",
            run_id=result.id,
            campaign_id=result.campaign_id,
            trace_payload={},
            created_at=NOW,
        )
    )
    snapshot.llm_calls_by_run_id[result.id].extend(
        [
            EvaluationLlmCall(
                llm_call_id="llm-attempt-1",
                run_id=result.id,
                campaign_id=result.campaign_id,
                phase="final_answer",
                reservation_id="reservation-1",
                provider_attempt=1,
                prompt_tokens=2,
                completion_tokens=2,
                total_tokens=4,
                reasoning_tokens=0,
                other_tokens=0,
                payload={"usage_status": "measured", "official_total_tokens": 4},
                created_at=NOW,
            ),
            EvaluationLlmCall(
                llm_call_id="llm-attempt-2",
                run_id=result.id,
                campaign_id=result.campaign_id,
                phase="final_answer",
                reservation_id="reservation-1",
                provider_attempt=2,
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                reasoning_tokens=0,
                other_tokens=0,
                payload={"usage_status": "measured", "official_total_tokens": 5},
                created_at=NOW,
            ),
        ]
    )
    observability = AsyncMock(spec_set=EvaluationObservabilityRepository)
    accounting = AsyncMock(spec_set=EvaluationAccountingStore)
    observability.load_campaign_observability_snapshot.return_value = snapshot
    accounting.load_campaign_snapshot.return_value = _accounting_snapshot([result])
    service = research_analytics.ResearchAnalyticsService(
        campaigns=SimpleNamespace(get=AsyncMock()),
        results=SimpleNamespace(get=AsyncMock(return_value=result)),
        observability=observability,
        accounting=accounting,
    )

    selected = await service.get_run_observability(
        user_id="user-1", campaign_id="campaign-1", run_id=result.id
    )
    bulk = await service.get_campaign_run_observability(
        user_id="user-1", campaign_id="campaign-1", results=[result]
    )

    assert selected.accounting_diagnostics == bulk[result.id].token_breakdown
    assert selected.accounting_diagnostics.by_phase == {"final_answer": 9}
    assert selected.accounting_diagnostics.phase_attribution_status == "complete"


def test_canonical_builder_rejects_missing_run_container() -> None:
    result = _result()
    with pytest.raises(AppError, match="observability"):
        research_analytics._build_canonical_run_observability(
            result=result,
            observability=_empty_snapshot(run_ids=()),
            accounting=_accounting_snapshot([result]),
        )


def test_canonical_builder_rejects_malformed_current_v9_materialization() -> None:
    result = _result(source_attempt_id="attempt-current")
    snapshot = _empty_snapshot()
    snapshot.materializations_by_run_id[result.id].append(
        EvaluationV9AttemptMaterialization(
            attempt_id="attempt-current",
            run_id=result.id,
            campaign_id=result.campaign_id,
            trace_payload={"query_contract": {"malformed": True}},
            created_at=NOW,
        )
    )
    with pytest.raises(AppError, match="materialization"):
        research_analytics._build_canonical_run_observability(
            result=result,
            observability=snapshot,
            accounting=_accounting_snapshot([result]),
        )


def test_official_ragas_filters_attempt_identity_and_nonfinite_values() -> None:
    results = [
        SimpleNamespace(
            id="run-1", source_attempt_id="attempt-1", context_policy_version=None
        ),
        SimpleNamespace(
            id="run-2", source_attempt_id="attempt-2", context_policy_version=None
        ),
        SimpleNamespace(
            id="run-3", source_attempt_id="attempt-3", context_policy_version=None
        ),
    ]

    def score(run_id: str, attempt: str, value: float, signature: str) -> dict:
        return {
            "campaign_result_id": run_id,
            "metric_name": "faithfulness",
            "metric_value": value,
            "source_attempt_id": attempt,
            "evaluation_signature": f"{run_id}-{attempt}-{signature}-{value}",
            "details": {
                "evaluator_model": "judge-v1",
                "metric_version": "v1",
                "compatibility_signature": signature,
                "compatibility_signature_version": "v2",
            },
        }

    scores = [
        score("run-1", "attempt-1", 0.8, "canonical"),
        score("run-2", "attempt-2", 0.7, "canonical"),
        score("run-3", "attempt-3", 0.2, "noncanonical"),
        score("run-1", "wrong-attempt", 0.1, "canonical"),
        score("run-1", "attempt-1", nan, "canonical"),
        score("run-2", "attempt-2", inf, "canonical"),
    ]

    official = research_analytics._official_ragas_by_run(
        results=results, scores=scores, work_metadata=[]
    )

    assert official == {
        "run-1": {"faithfulness": 0.8},
        "run-2": {"faithfulness": 0.7},
    }
