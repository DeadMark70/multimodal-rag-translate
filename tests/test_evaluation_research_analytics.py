"""Strict research-summary contract regression tests using durable repositories."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from evaluation import db as evaluation_db
from evaluation.accounting_schemas import AccountingScopeStart, TokenBreakdown, UsageEventCreate
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.campaign_schemas import CampaignConfig, CampaignResultStatus
from evaluation.db import (
    CampaignRepository,
    CampaignResultRepository,
    RagasScoreRepository,
)
from evaluation.research_analytics import (
    ResearchAnalyticsService,
    _claim_extraction_status,
    _evaluator_identity,
    _tokens,
    nearest_rank,
)
from evaluation.job_store import build_legacy_evaluator_compatibility_signature
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.schemas import ModelConfig
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationHumanRating,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationRetrievalEvent,
    EvaluationTraceEvent,
    EvaluationV9AttemptMaterialization,
)
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem


def test_nearest_rank_percentiles_are_observed_values() -> None:
    assert nearest_rank([100, 200, 300, 400, 500], 0.50) == 300
    assert nearest_rank([100, 200, 300, 400, 500], 0.95) == 500


def test_claim_extraction_status_distinguishes_empty_from_not_instrumented() -> None:
    assert _claim_extraction_status({"claim_extraction_status": "empty"}) == "empty"
    assert _claim_extraction_status({}) == "not_instrumented"


def test_token_breakdown_requires_provider_phase_rows_to_match_runtime_total() -> None:
    scope = SimpleNamespace(
        status="completed",
        observed_call_count=1,
        measured_call_count=1,
        missing_usage_call_count=0,
    )
    event = SimpleNamespace(
        usage_status="measured",
        reconciliation_status="balanced",
        phase="answer_generation",
        purpose="evaluation",
        provider="google",
        input_tokens=6,
        output_text_tokens=2,
        reasoning_tokens=1,
        other_tokens=1,
    )
    provider_call = SimpleNamespace(
        llm_call_id="call-1",
        phase="final_answer",
        reservation_id="reservation-1",
        provider_attempt=1,
        total_tokens=9,
        prompt_tokens=5,
        completion_tokens=2,
        reasoning_tokens=1,
        other_tokens=1,
        payload={"usage_status": "measured", "official_total_tokens": 9},
    )

    breakdown = _tokens(
        [scope],
        [event],
        provider_attempts=[provider_call],
        runtime_total_tokens=10,
    )

    assert breakdown.total_tokens == 10
    assert breakdown.phase_attribution_status == "partial"
    assert "provider_runtime_total_mismatch" in breakdown.phase_attribution_reasons


def test_legacy_identity_preserves_evaluator_config_differences() -> None:
    result = SimpleNamespace(context_policy_version="v3")

    def identity_for(config: dict) -> tuple[str, str, str]:
        signature = build_legacy_evaluator_compatibility_signature(
            evaluator_model="judge-v1",
            evaluator_config=config,
            metric_name="faithfulness",
            metric_version="v1",
            context_policy_version="v3",
            context_metrics_enabled=False,
        )
        return _evaluator_identity(
            {
                "metric_name": "faithfulness",
                "details": {
                    "evaluator_model": "judge-v1",
                    "metric_version": "v1",
                    "compatibility_signature": signature,
                },
                "_work_metadata": [
                    {
                        "compatibility_signature": signature,
                        "evaluator_model": "judge-v1",
                        "metric_version": "v1",
                        "evaluator_config": config,
                    }
                ],
            },
            result,
        )

    assert identity_for({"temperature": 0}) != identity_for({"temperature": 1})


def test_legacy_identity_fails_closed_on_score_metadata_mismatch() -> None:
    result = SimpleNamespace(context_policy_version="v3")
    signature = build_legacy_evaluator_compatibility_signature(
        evaluator_model="judge-v2",
        evaluator_config={},
        metric_name="faithfulness",
        metric_version="v1",
        context_policy_version="v3",
        context_metrics_enabled=False,
    )

    identity = _evaluator_identity(
        {
            "metric_name": "faithfulness",
            "details": {
                "evaluator_model": "judge-v1",
                "metric_version": "v1",
                "compatibility_signature": signature,
            },
            "_work_metadata": [
                {
                    "compatibility_signature": signature,
                    "evaluator_model": "judge-v2",
                    "metric_version": "v1",
                    "evaluator_config": {},
                }
            ],
        },
        result,
    )

    assert identity == ("judge-v1", "v1", signature)


@pytest_asyncio.fixture
async def research_service(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", tmp_path / "evaluation.db")
    return ResearchAnalyticsService()


async def _campaign(campaign_id: str, modes: list[str]) -> None:
    repository = CampaignRepository()
    await repository.create(
        user_id="user-1",
        name=campaign_id,
        config=CampaignConfig(
            test_case_ids=["q-1"],
            modes=modes,
            model_config=ModelConfig(id="model-1", name="Model", model_name="model"),
        ),
    )
    # Repository-generated IDs are intentionally not assumed by the fixture.
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE campaigns SET id = ? WHERE name = ?", (campaign_id, campaign_id)
        )
        await connection.commit()


async def _result(
    campaign_id: str,
    mode: str,
    attempt: str,
    *,
    latency: float = 100.0,
    run_number: int = 1,
    status: CampaignResultStatus = CampaignResultStatus.COMPLETED,
    context_policy_version: str = "v2",
) -> str:
    result = await CampaignResultRepository().create(
        user_id="user-1",
        campaign_id=campaign_id,
        question_id=f"q-{mode}",
        question="Q",
        ground_truth="A",
        ground_truth_short=None,
        key_points=[],
        ragas_focus=[],
        mode=mode,
        execution_profile="v2",
        context_policy_version=context_policy_version,
        run_number=run_number,
        answer="A",
        contexts=[],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=latency,
        token_usage={},
        category=None,
        difficulty=None,
        status=status,
        total_latency_ms=latency,
        source_attempt_id=attempt,
    )
    return result.id


async def _execution_scope(
    campaign_id: str,
    result_id: str,
    attempt: str,
    *,
    official: bool,
    scope_status: str = "completed",
    tokens: int = 15,
    cost: float | None = 0.1,
    pricing_status: str = "priced",
    scope_run_id: str | None = None,
    usage_status: str = "measured",
    reconciliation_status: str = "balanced",
) -> None:
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id=f"scope-{attempt}",
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=attempt,
            run_id=scope_run_id or result_id,
            targets=[
                {
                    "job_id": "job-1",
                    "work_item_id": "work-1",
                    "attempt_id": attempt,
                    "campaign_result_id": result_id,
                    "is_official": official,
                }
            ],
        )
    )
    await store.record_event(
        UsageEventCreate(
            usage_event_id=f"event-{attempt}",
            scope_id=f"scope-{attempt}",
            campaign_id=campaign_id,
            scope_type="execution_run",
            scope_key=attempt,
            run_id=scope_run_id or result_id,
            phase="answer_generation",
            purpose="evaluation",
            input_tokens=tokens - 5 if usage_status == "measured" else 0,
            output_text_tokens=5 if usage_status == "measured" else 0,
            reported_total_tokens=tokens if usage_status == "measured" else None,
            usage_status=usage_status,
            reconciliation_status=reconciliation_status,
            estimated_cost_usd=cost,
            pricing_status=pricing_status,
            created_at=datetime.now(timezone.utc),
        )
    )
    await store.finalize_scope(f"scope-{attempt}", scope_status)


async def _official_scope(
    campaign_id: str,
    result_id: str,
    attempt: str,
    *,
    tokens: int = 15,
) -> None:
    await _execution_scope(
        campaign_id,
        result_id,
        attempt,
        official=True,
        tokens=tokens,
    )


@pytest.mark.asyncio
async def test_get_run_observability_projects_owned_v9_normalized_data() -> None:
    now = datetime.now(timezone.utc)

    class Campaigns:
        async def get(self, *, user_id: str, campaign_id: str):
            assert (user_id, campaign_id) == ("user-1", "cmp-1")

    result = SimpleNamespace(
        id="run-1",
        question_id="q-1",
        mode="agentic",
        repeat_number=1,
        answer="safe answer",
        total_latency_ms=12.0,
        latency_ms=12.0,
        created_at=now,
        agentic_execution_version="v9",
        source_attempt_id="attempt-1",
        derived_metrics={"gold_fact_attrition": []},
    )

    class Results:
        async def get(self, *, user_id: str, campaign_id: str, result_id: str):
            assert (user_id, campaign_id, result_id) == ("user-1", "cmp-1", "run-1")
            return result

    class Observability:
        async def list_trace_events_for_run(self, run_id):
            return [
                EvaluationTraceEvent(
                    event_id="trace-1", run_id=run_id, campaign_id="cmp-1",
                    span_id="span-1", event_type="retrieval", sequence=1,
                    stage_type="retrieval", stage_name="retrieve", started_at=now,
                    status="success", payload={"secret": "redact"}, error={"secret": "redact"},
                    created_at=now,
                )
            ]

        async def list_llm_calls_for_run(self, run_id):
            return [
                EvaluationLlmCall(
                    llm_call_id="llm-1", run_id=run_id, campaign_id="cmp-1",
                    prompt_preview="safe prompt", payload={"secret": "redact"},
                    error={"secret": "redact"}, created_at=now,
                )
            ]

        async def list_retrieval_events_for_run(self, run_id):
            return [EvaluationRetrievalEvent(
                retrieval_event_id="retrieval-1", run_id=run_id, campaign_id="cmp-1",
                created_at=now,
            )]

        async def list_retrieval_chunks_for_run(self, run_id):
            return [
                EvaluationRetrievalChunk(
                    retrieval_chunk_id="chunk-derived", run_id=run_id, campaign_id="cmp-1",
                    retrieval_event_id="retrieval-1", chunk_id="chunk-derived",
                    excerpt="safe excerpt", used_in_context=True, used_in_answer=True,
                    expected_evidence_match=True,
                    payload={
                        "observation_provenance": "derived",
                        "availability_status": "partial",
                        "availability_reasons": ["result_context_reconstruction"],
                        "used_in_answer_provenance": "heuristic",
                        "provider_body": {"secret": "redact"},
                    },
                    created_at=now,
                ),
                EvaluationRetrievalChunk(
                    retrieval_chunk_id="chunk-measured", run_id=run_id, campaign_id="cmp-1",
                    retrieval_event_id="retrieval-1", chunk_id="chunk-measured",
                    used_in_context=True, used_in_answer=False,
                    expected_evidence_match=True,
                    payload={
                        "observation_provenance": "measured",
                        "availability_status": "complete",
                        "availability_reasons": [],
                    },
                    created_at=now,
                ),
                EvaluationRetrievalChunk(
                    retrieval_chunk_id="chunk-historical", run_id=run_id, campaign_id="cmp-1",
                    retrieval_event_id="retrieval-1", chunk_id="chunk-historical",
                    used_in_context=True, used_in_answer=True,
                    expected_evidence_match=True, payload={"provider_body": "secret"},
                    created_at=now,
                ),
            ]

        async def list_graph_events_for_run(self, run_id):
            return [EvaluationGraphEvent(
                graph_event_id="graph-1", run_id=run_id, campaign_id="cmp-1",
                graph_query="query", graph_search_mode="local", graph_route="local",
                created_at=now,
            )]

        async def list_graph_evidence_items_for_run(self, run_id):
            return [EvaluationGraphEvidenceItem(
                graph_evidence_item_id="evidence-1", graph_event_id="graph-1",
                provenance_status="full", created_at=now,
            )]

        async def list_context_packs_for_run(self, run_id):
            return []

        async def list_tool_calls_for_run(self, run_id):
            return []

        async def list_routing_decisions_for_run(self, run_id):
            return []

        async def list_claims_for_run(self, run_id):
            return [EvaluationClaim(
                claim_id="claim-1", run_id=run_id, campaign_id="cmp-1",
                claim_text="safe claim",
                evidence=[{
                    "evidence_id": "evidence-1", "doc_id": "doc-1",
                    "chunk_id": "chunk-1", "page": 8, "provider_body": {"secret": "redact"},
                }],
                payload={
                    "repair_action": "requery", "post_repair_status": "supported",
                    "provider_body": {"secret": "redact"},
                },
                created_at=now,
            )]

        async def list_human_ratings_for_run(self, run_id):
            return [EvaluationHumanRating(
                human_rating_id="rating-1", run_id=run_id, campaign_id="cmp-1",
                rater_id_hash="rater", rubric_version="v1", correctness_score=1,
                faithfulness_score=1, completeness_score=1, citation_quality_score=1,
                usefulness_score=1, created_at=now,
            )]

        async def get_v9_attempt_materialization(self, attempt_id):
            assert attempt_id == "attempt-1"
            return EvaluationV9AttemptMaterialization(
                attempt_id=attempt_id, run_id="run-1", campaign_id="cmp-1",
                trace_payload={}, created_at=now,
            )

        async def list_evidence_packets_for_attempt(self, attempt_id):
            return []

        async def list_slot_resolutions_for_attempt(self, attempt_id):
            return []

    service = ResearchAnalyticsService(
        campaigns=Campaigns(), results=Results(), observability=Observability()
    )
    service.get_run_token_breakdown = AsyncMock(return_value=TokenBreakdown(
        total_tokens=7, accounting_status="complete", phase_attribution_status="complete"
    ))

    detail = await service.get_run_observability(
        user_id="user-1", campaign_id="cmp-1", run_id="run-1"
    )

    assert detail.run_id == "run-1"
    assert detail.run_summary is not None
    assert detail.accounting_diagnostics.accounting_status == "complete"
    assert [row.human_rating_id for row in detail.human_ratings] == ["rating-1"]
    assert detail.agentic_v9 is not None
    assert detail.agentic_v9.schema_version == "1"
    derived, measured, historical = detail.retrieval_chunks
    assert derived.provenance == "derived"
    assert derived.availability.status == "partial"
    assert derived.availability.reasons == ["result_context_reconstruction"]
    assert derived.used_in_context is True
    assert derived.used_in_answer is True
    assert derived.expected_evidence_match is True
    assert derived.payload == {}
    assert measured.provenance == "measured"
    assert measured.availability.status == "complete"
    assert measured.used_in_context is True
    assert measured.used_in_answer is False
    assert measured.expected_evidence_match is True
    assert historical.availability.status == "not_available"
    assert historical.availability.reasons == ["provenance_not_recorded"]
    assert historical.used_in_context is None
    assert historical.used_in_answer is None
    assert historical.expected_evidence_match is None
    assert historical.payload == {}
    projected_claim = detail.claims[0]
    assert projected_claim.evidence_refs[0].model_dump() == {
        "evidence_id": "evidence-1", "doc_id": "doc-1", "chunk_id": "chunk-1", "page": 8,
    }
    assert projected_claim.repair_action == "requery"
    assert projected_claim.post_repair_status == "supported"
    assert projected_claim.extraction_status == "recorded"
    assert projected_claim.payload == {}
    assert "secret" not in projected_claim.model_dump_json()


@pytest.mark.asyncio
async def test_production_research_paths_reconcile_v9_provider_attempts(
    research_service,
) -> None:
    campaign_id = "production-token-reconciliation"
    await _campaign(campaign_id, ["agentic"])
    result_id = await _result(campaign_id, "agentic", "attempt-v9")
    await _official_scope(campaign_id, result_id, "attempt-v9")
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            UPDATE campaign_results
            SET system_version_snapshot_json = ?
            WHERE id = ?
            """,
            (json.dumps({"agentic_execution_version": "v9"}), result_id),
        )
        await connection.commit()
    await EvaluationObservabilityRepository().record_llm_call(
        EvaluationLlmCall(
            llm_call_id="provider-attempt-v9",
            run_id=result_id,
            campaign_id=campaign_id,
            provider="google",
            model_name="gemini-2.5-flash",
            phase="final_answer",
            purpose="synthesizer",
            reservation_id="reservation-v9",
            provider_attempt=1,
            prompt_tokens=9,
            completion_tokens=5,
            total_tokens=14,
            reasoning_tokens=0,
            other_tokens=0,
            payload={"usage_status": "measured", "official_total_tokens": 14},
            created_at=datetime.now(timezone.utc),
        )
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id=campaign_id
    )
    selected = await research_service.get_run_token_breakdown(
        campaign_id=campaign_id,
        run_id=result_id,
        agentic_execution_version="v9",
    )
    comparison = await research_service.get_question_comparison(
        user_id="user-1", campaign_id=campaign_id
    )
    behavior = await research_service.get_agent_behavior(
        user_id="user-1", campaign_id=campaign_id
    )

    assert summary.tokens.phase_attribution_status == "partial"
    assert summary.modes[0].tokens.phase_attribution_status == "partial"
    assert "provider_runtime_total_mismatch" in (
        summary.tokens.phase_attribution_reasons
    )
    assert selected.phase_attribution_status == "partial"
    assert comparison.rows[0].by_mode[0].accounting_status == "partial"
    assert behavior.rows[0].accounting_status == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_totals", "expected_status"),
    [((9, 11), "partial"), ((10, 10), "complete")],
)
async def test_research_analytics_reconciles_v9_attempts_per_run_before_aggregation(
    research_service,
    provider_totals,
    expected_status,
) -> None:
    campaign_id = "per-run-token-reconciliation"
    await _campaign(campaign_id, ["agentic"])
    first_result_id = await _result(
        campaign_id, "agentic", "attempt-v9-first", run_number=1
    )
    second_result_id = await _result(
        campaign_id, "agentic", "attempt-v9-second", run_number=2
    )
    await _official_scope(
        campaign_id, first_result_id, "attempt-v9-first", tokens=10
    )
    await _official_scope(
        campaign_id, second_result_id, "attempt-v9-second", tokens=10
    )
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            UPDATE campaign_results
            SET system_version_snapshot_json = ?
            WHERE campaign_id = ?
            """,
            (json.dumps({"agentic_execution_version": "v9"}), campaign_id),
        )
        await connection.commit()

    repository = EvaluationObservabilityRepository()
    for run_id, attempt, provider_total in zip(
        (first_result_id, second_result_id),
        ("first", "second"),
        provider_totals,
        strict=True,
    ):
        await repository.record_llm_call(
            EvaluationLlmCall(
                llm_call_id=f"provider-{attempt}",
                run_id=run_id,
                campaign_id=campaign_id,
                provider="google",
                model_name="gemini-2.5-flash",
                phase="final_answer",
                purpose="synthesizer",
                reservation_id=f"reservation-{attempt}",
                provider_attempt=1,
                prompt_tokens=provider_total - 5,
                completion_tokens=5,
                total_tokens=provider_total,
                reasoning_tokens=0,
                other_tokens=0,
                payload={
                    "usage_status": "measured",
                    "official_total_tokens": provider_total,
                },
                created_at=datetime.now(timezone.utc),
            )
        )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id=campaign_id
    )
    selected = await research_service.get_run_token_breakdown(
        campaign_id=campaign_id,
        run_id=first_result_id,
        agentic_execution_version="v9",
    )

    assert summary.tokens.accounting_status == expected_status
    assert summary.tokens.phase_attribution_status == expected_status
    assert summary.modes[0].tokens.accounting_status == expected_status
    assert summary.modes[0].tokens.phase_attribution_status == expected_status
    assert selected.accounting_status == (
        "partial" if provider_totals[0] != 10 else "complete"
    )
    assert selected.phase_attribution_status == (
        "partial" if provider_totals[0] != 10 else "complete"
    )
    if expected_status == "partial":
        assert "provider_runtime_total_mismatch" in (
            summary.tokens.phase_attribution_reasons
        )


@pytest.mark.asyncio
async def test_research_analytics_marks_v9_run_partial_when_attempt_mapping_is_missing(
    research_service,
) -> None:
    campaign_id = "missing-v9-provider-attempt-mapping"
    await _campaign(campaign_id, ["agentic"])
    result_id = await _result(campaign_id, "agentic", "attempt-v9")
    await _official_scope(campaign_id, result_id, "attempt-v9", tokens=10)
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            UPDATE campaign_results
            SET system_version_snapshot_json = ?
            WHERE id = ?
            """,
            (json.dumps({"agentic_execution_version": "v9"}), result_id),
        )
        await connection.commit()

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id=campaign_id
    )
    selected = await research_service.get_run_token_breakdown(
        campaign_id=campaign_id,
        run_id=result_id,
        agentic_execution_version="v9",
    )

    assert summary.tokens.accounting_status == "partial"
    assert summary.tokens.phase_attribution_status == "partial"
    assert summary.modes[0].tokens.accounting_status == "partial"
    assert selected.accounting_status == "partial"
    assert "provider_attempts_missing" in summary.tokens.phase_attribution_reasons


def _primary_score_rows(
    result_attempts: list[tuple[str, str]],
    *,
    evaluator_model: str = "judge-v1",
    compatibility_signature: str = "policy-a",
) -> list[dict]:
    return [
        {
            "campaign_result_id": result_id,
            "metric_name": metric_name,
            "metric_value": 0.8,
            "source_attempt_id": attempt,
            "evaluation_signature": f"{attempt}-{metric_name}",
            "details": {
                "evaluator_model": evaluator_model,
                "metric_version": "v1",
                "compatibility_signature": compatibility_signature,
            },
        }
        for result_id, attempt in result_attempts
        for metric_name in (
            "answer_correctness",
            "faithfulness",
            "answer_relevancy",
        )
    ]


@pytest.mark.asyncio
async def test_legacy_campaign_is_visible_but_not_comparable(research_service) -> None:
    await _campaign("legacy", ["naive"])
    await _result("legacy", "naive", "legacy-attempt")

    summary = await research_service.get_summary(user_id="user-1", campaign_id="legacy")

    assert summary.token_accounting_status == "incomplete_legacy"
    assert summary.tokens.total_tokens is None
    assert summary.execution_cost.benchmark_usd is None
    assert summary.evaluation_overhead.retry_count == 0
    assert summary.modes[0].comparable is False
    assert "legacy_accounting" in summary.modes[0].not_comparable_reasons


@pytest.mark.asyncio
async def test_ragas_retry_count_uses_durable_values_and_warns_when_legacy_unknown(
    research_service,
) -> None:
    await _campaign("ragas-retries", ["naive"])
    store = EvaluationAccountingStore()
    for scope_id in ("known-retries", "unknown-retries"):
        await store.start_scope(
            AccountingScopeStart(
                scope_id=scope_id,
                campaign_id="ragas-retries",
                scope_type="ragas_batch",
                scope_key=scope_id,
                metric_name="faithfulness",
                targets=[
                    {
                        "job_id": "ragas",
                        "work_item_id": scope_id,
                        "attempt_id": scope_id,
                    }
                ],
            )
        )
    await store.increment_scope_retry("known-retries")
    known_summary = await research_service.get_summary(
        user_id="user-1", campaign_id="ragas-retries"
    )
    assert known_summary.evaluation_overhead.retry_count == 1
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE evaluation_accounting_scopes SET retry_count = NULL WHERE scope_id = 'unknown-retries'"
        )
        await connection.commit()

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="ragas-retries"
    )

    assert summary.evaluation_overhead.retry_count is None
    assert any(
        warning.code == "unknown_ragas_retry_count" for warning in summary.warnings
    )


@pytest.mark.asyncio
async def test_missing_usage_keeps_all_token_categories_nullable(
    research_service,
) -> None:
    await _campaign("missing-usage", ["naive"])
    result_id = await _result("missing-usage", "naive", "missing-attempt")
    await _execution_scope(
        "missing-usage",
        result_id,
        "missing-attempt",
        official=True,
        tokens=0,
        cost=None,
        pricing_status="unavailable_usage",
        usage_status="missing",
        reconciliation_status="unavailable",
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="missing-usage"
    )

    assert summary.tokens.input_tokens is None
    assert summary.tokens.output_text_tokens is None
    assert summary.tokens.reasoning_tokens is None
    assert summary.tokens.other_tokens is None
    assert summary.tokens.total_tokens is None
    assert summary.tokens.by_phase == {}
    assert summary.tokens.accounting_status == "partial"
    assert summary.tokens.phase_attribution_status == "not_available"
    assert summary.tokens.observed_call_count == 1
    assert summary.tokens.measured_call_count == 0
    assert summary.tokens.missing_usage_call_count == 1
    assert summary.tokens.unbalanced_call_count == 0
    assert summary.tokens.unclassified_phase_call_count == 0


@pytest.mark.asyncio
async def test_agent_behavior_projects_materialized_v9_evidence_not_legacy_step_zeros(
    research_service,
) -> None:
    await _campaign("agent-behavior-v9", ["agentic"])
    result_id = await _result("agent-behavior-v9", "agentic", "attempt-v9")
    await _official_scope("agent-behavior-v9", result_id, "attempt-v9")

    class Observability:
        async def list_v9_attempt_materializations_for_campaign(self, campaign_id):
            assert campaign_id == "agent-behavior-v9"
            return {
                result_id: EvaluationV9AttemptMaterialization(
                    attempt_id="attempt-v9",
                    run_id=result_id,
                    campaign_id=campaign_id,
                    trace_payload={
                        "query_contract": {
                            "contract_version": "2",
                            "route": "multi_hop",
                            "graph_policy": "required_locator",
                            "visual_requested": True,
                            "visual_required": False,
                            "evidence_extraction_required": True,
                            "required_slots": [{"slot_id": "S1"}],
                        },
                        "metrics": {
                            "retrieval_query_count": 2,
                            "provider_attempt_count": 1,
                            "final_generation_count": 1,
                            "reconciled_tokens": 15,
                        },
                        "sufficiency": {
                            "evidence_complete": True,
                            "supported_slot_ids": ["S1"],
                        },
                        "context_pack": {"packed_evidence_ids": ["E1"]},
                        "budget_reservations": [{"reserved_tokens": 32}],
                        "repairs": [],
                        "final_claims": [{"claim_id": "C1"}],
                        "visual_execution": {
                            "state": "attempted_without_evidence",
                            "attempted": True,
                        },
                    },
                )
            }

        async def list_v9_behavior_counts_for_campaign(self, campaign_id):
            return {
                result_id: {"evidence_packet_count": 13, "slot_resolution_count": 1}
            }

        async def list_graph_events_for_campaign(self, campaign_id):
            return {}

    research_service._observability = Observability()

    response = await research_service.get_agent_behavior(
        user_id="user-1", campaign_id="agent-behavior-v9"
    )

    row = response.rows[0]
    assert response.behavior_schema_version == "2"
    assert row.behavior_schema == "v9"
    assert row.trace_status == "not_instrumented"
    assert row.legacy is None
    assert row.v9 is not None
    assert row.v9.evidence_packet_count == 13
    assert row.v9.slot_resolution_count == 1
    assert row.v9.slot_semantics == "heuristic_experimental"
    assert row.v9.atomic_completeness is None
    assert row.v9.atomic_completeness_reason == "atomic_slot_matching_experimental"
    assert row.v9.graph_execution == "required_but_not_satisfied"
    assert row.v9.visual_requested is True
    assert row.v9.visual_required is False
    assert row.v9.visual_execution == "attempted_without_evidence"


@pytest.mark.asyncio
async def test_agent_behavior_projects_v1_contract_as_non_atomic_na(
    research_service,
) -> None:
    await _campaign("agent-behavior-v1-contract", ["agentic"])
    result_id = await _result(
        "agent-behavior-v1-contract", "agentic", "attempt-v1-contract"
    )
    await _official_scope(
        "agent-behavior-v1-contract", result_id, "attempt-v1-contract"
    )

    class Observability:
        async def list_v9_attempt_materializations_for_campaign(self, campaign_id):
            return {
                result_id: EvaluationV9AttemptMaterialization(
                    attempt_id="attempt-v1-contract",
                    run_id=result_id,
                    campaign_id=campaign_id,
                    trace_payload={
                        "query_contract": {
                            "contract_version": "1",
                            "route": "single_lookup",
                            "intent": "legacy",
                            "required_slots": [
                                {"slot_id": "fact", "description": "generic fact"}
                            ],
                        }
                    },
                )
            }

        async def list_v9_behavior_counts_for_campaign(self, campaign_id):
            return {result_id: {}}

        async def list_graph_events_for_campaign(self, campaign_id):
            return {}

    research_service._observability = Observability()

    response = await research_service.get_agent_behavior(
        user_id="user-1", campaign_id="agent-behavior-v1-contract"
    )

    assert response.rows[0].v9.contract_version == "1"
    assert response.rows[0].v9.slot_semantics == "legacy_generic"
    assert response.rows[0].v9.atomic_completeness is None
    assert response.rows[0].v9.atomic_completeness_reason is None


@pytest.mark.asyncio
async def test_agent_behavior_projects_failed_v9_reason_without_endpoint_failure(
    research_service,
) -> None:
    await _campaign("agent-behavior-failed-v9", ["agentic"])
    result_id = await _result(
        "agent-behavior-failed-v9",
        "agentic-v9",
        "attempt-v9-failed",
        status=CampaignResultStatus.FAILED,
    )
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            "UPDATE campaign_results SET error_message = '' WHERE id = ?",
            (result_id,),
        )
        await connection.commit()

    response = await research_service.get_agent_behavior(
        user_id="user-1", campaign_id="agent-behavior-failed-v9"
    )

    assert response.rows[0].trace_status == "failed"
    assert response.rows[0].failure_reason == "failure_reason_not_recorded"


@pytest.mark.asyncio
async def test_mixed_usage_reports_measured_subtotals_without_total(
    research_service,
) -> None:
    await _campaign("mixed-usage", ["naive"])
    result_id = await _result("mixed-usage", "naive", "mixed-attempt")
    await _official_scope("mixed-usage", result_id, "mixed-attempt")
    await EvaluationAccountingStore().record_event(
        UsageEventCreate(
            usage_event_id="event-mixed-missing",
            scope_id="scope-mixed-attempt",
            campaign_id="mixed-usage",
            scope_type="execution_run",
            scope_key="mixed-attempt",
            run_id=result_id,
            phase="answer_generation",
            purpose="evaluation",
            usage_status="missing",
            reconciliation_status="unavailable",
            pricing_status="unavailable_usage",
            created_at=datetime.now(timezone.utc),
        )
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="mixed-usage"
    )

    assert summary.tokens.input_tokens == 10
    assert summary.tokens.output_text_tokens == 5
    assert summary.tokens.total_tokens is None
    assert summary.tokens.by_phase == {"answer_generation": 15}
    assert summary.tokens.accounting_status == "partial"
    assert summary.tokens.observed_call_count == 2
    assert summary.tokens.measured_call_count == 1
    assert summary.tokens.missing_usage_call_count == 1
    assert summary.tokens.unbalanced_call_count == 0
    assert summary.tokens.missing_usage_by_phase == {"answer_generation": 1}
    assert summary.tokens.missing_usage_by_purpose == {"evaluation": 1}
    assert summary.tokens.missing_usage_by_provider == {"unknown": 1}
    assert any(w.code == "missing_usage" for w in summary.warnings)


@pytest.mark.asyncio
async def test_campaign_cohort_rejects_modes_scored_by_different_evaluators(
    research_service,
) -> None:
    await _campaign("mixed-cohorts", ["naive", "graph"])
    naive = await _result("mixed-cohorts", "naive", "naive-attempt")
    graph = await _result("mixed-cohorts", "graph", "graph-attempt")
    await _official_scope("mixed-cohorts", naive, "naive-attempt")
    await _official_scope("mixed-cohorts", graph, "graph-attempt")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="mixed-cohorts",
        score_rows=(
            _primary_score_rows([(naive, "naive-attempt")])
            + _primary_score_rows(
                [(graph, "graph-attempt")],
                evaluator_model="judge-v2",
                compatibility_signature="policy-b",
            )
        ),
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="mixed-cohorts"
    )

    assert sum(mode.comparable for mode in summary.modes) <= 1
    assert any(
        "evaluator_metadata_mismatch" in mode.not_comparable_reasons
        for mode in summary.modes
    )


@pytest.mark.asyncio
async def test_campaign_cohort_keeps_modes_comparable_when_identity_is_shared(
    research_service,
) -> None:
    await _campaign("shared-cohort", ["naive", "graph"])
    naive = await _result("shared-cohort", "naive", "naive-attempt")
    graph = await _result("shared-cohort", "graph", "graph-attempt")
    await _official_scope("shared-cohort", naive, "naive-attempt")
    await _official_scope("shared-cohort", graph, "graph-attempt")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="shared-cohort",
        score_rows=_primary_score_rows(
            [(naive, "naive-attempt"), (graph, "graph-attempt")]
        ),
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="shared-cohort"
    )

    assert all(mode.comparable for mode in summary.modes)


@pytest.mark.asyncio
async def test_unknown_monetary_pricing_does_not_block_token_only_comparison(
    research_service,
) -> None:
    await _campaign("token-only", ["graph"])
    result_id = await _result("token-only", "graph", "graph-attempt")
    await _execution_scope(
        "token-only",
        result_id,
        "graph-attempt",
        official=True,
        cost=None,
        pricing_status="unknown_model",
    )
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="token-only",
        score_rows=_primary_score_rows([(result_id, "graph-attempt")]),
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="token-only"
    )
    mode = summary.modes[0]

    assert mode.comparable is True
    assert "incomplete_pricing" not in mode.not_comparable_reasons
    assert mode.execution_cost.pricing_status == "partial"
    assert mode.tokens.total_tokens == 15


@pytest.mark.asyncio
async def test_legacy_context_policy_signatures_do_not_hide_agentic_scores(
    research_service, monkeypatch
) -> None:
    """Historical v3/v4 context-policy hashes remain visible to the summary."""
    await _campaign("legacy-context-cohort", ["naive", "agentic"])
    naive = await _result(
        "legacy-context-cohort",
        "naive",
        "naive-attempt",
        context_policy_version="v3",
    )
    agentic = await _result(
        "legacy-context-cohort",
        "agentic",
        "agentic-attempt",
        context_policy_version="v4",
    )
    await _official_scope("legacy-context-cohort", naive, "naive-attempt")
    await _official_scope("legacy-context-cohort", agentic, "agentic-attempt")

    legacy_rows = []
    work_metadata = []
    for result_id, attempt, context_policy in (
        (naive, "naive-attempt", "v3"),
        (agentic, "agentic-attempt", "v4"),
    ):
        for metric_name in (
            "answer_correctness",
            "faithfulness",
            "answer_relevancy",
        ):
            signature = build_legacy_evaluator_compatibility_signature(
                evaluator_model="judge-v1",
                evaluator_config={},
                metric_name=metric_name,
                metric_version="v1",
                context_policy_version=context_policy,
                context_metrics_enabled=False,
            )
            legacy_rows.append(
                {
                    "campaign_result_id": result_id,
                    "metric_name": metric_name,
                    "metric_value": 0.8,
                    "source_attempt_id": attempt,
                    "evaluation_signature": f"{attempt}-{metric_name}",
                    "details": {
                        "evaluator_model": "judge-v1",
                        "metric_version": "v1",
                        "compatibility_signature": signature,
                    },
                }
            )
            work_metadata.append(
                {
                    "campaign_result_id": result_id,
                    "metric_name": metric_name,
                    "metric_version": "v1",
                    "compatibility_signature": signature,
                    "evaluator_model": "judge-v1",
                    "evaluator_config": {},
                }
            )
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="legacy-context-cohort",
        score_rows=legacy_rows,
    )

    async def legacy_metadata(*, user_id: str, campaign_id: str) -> list[dict]:
        return work_metadata

    monkeypatch.setattr(
        research_service._ragas_scores,
        "list_work_metadata_for_campaign",
        legacy_metadata,
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="legacy-context-cohort"
    )

    agentic_summary = next(mode for mode in summary.modes if mode.mode == "agentic")
    assert agentic_summary.quality["answer_correctness"].valid_samples == 1
    assert agentic_summary.quality["faithfulness"].valid_samples == 1
    assert agentic_summary.quality["answer_relevancy"].valid_samples == 1
    assert "evaluator_metadata_mismatch" not in agentic_summary.not_comparable_reasons


@pytest.mark.asyncio
async def test_terminal_ragas_failure_is_counted_per_result_not_as_missing(
    research_service,
) -> None:
    await _campaign("per-result-failure", ["naive"])
    results = [
        await _result(
            "per-result-failure", "naive", f"attempt-{index}", run_number=index
        )
        for index in range(1, 6)
    ]
    for index, result_id in enumerate(results, start=1):
        await _official_scope("per-result-failure", result_id, f"attempt-{index}")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="per-result-failure",
        score_rows=[
            {
                "campaign_result_id": result_id,
                "metric_name": "faithfulness",
                "metric_value": 0.8,
                "source_attempt_id": f"attempt-{index}",
                "evaluation_signature": f"attempt-{index}-faithfulness",
                "details": {
                    "evaluator_model": "judge-v1",
                    "metric_version": "v1",
                    "compatibility_signature": "policy-a",
                },
            }
            for index, result_id in enumerate(results[:4], start=1)
        ],
    )
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-fifth-faithfulness",
            campaign_id="per-result-failure",
            scope_type="ragas_batch",
            scope_key="faithfulness-fifth",
            metric_name="faithfulness",
            targets=[
                {
                    "campaign_result_id": results[4],
                    "job_id": "ragas",
                    "work_item_id": "faithfulness-fifth",
                    "attempt_id": "attempt-5",
                }
            ],
        )
    )
    await store.finalize_scope("failed-fifth-faithfulness", "failed")

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="per-result-failure"
    )
    observation = summary.quality["faithfulness"]

    assert observation.valid_samples == 4
    assert observation.failed_samples == 1
    assert observation.missing_samples == 0
    assert observation.status == "partial"


@pytest.mark.asyncio
async def test_targeted_ragas_failure_does_not_leak_across_modes(
    research_service,
) -> None:
    await _campaign("isolated-ragas-failure", ["naive", "graph"])
    naive = await _result("isolated-ragas-failure", "naive", "naive-attempt")
    graph = await _result("isolated-ragas-failure", "graph", "graph-attempt")
    await _official_scope("isolated-ragas-failure", naive, "naive-attempt")
    await _official_scope("isolated-ragas-failure", graph, "graph-attempt")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="isolated-ragas-failure",
        score_rows=_primary_score_rows(
            [(naive, "naive-attempt"), (graph, "graph-attempt")]
        ),
    )
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-naive-faithfulness",
            campaign_id="isolated-ragas-failure",
            scope_type="ragas_batch",
            scope_key="naive-faithfulness",
            metric_name="faithfulness",
            targets=[
                {
                    "campaign_result_id": naive,
                    "job_id": "ragas",
                    "work_item_id": "naive-faithfulness",
                    "attempt_id": "naive-attempt",
                }
            ],
        )
    )
    await store.finalize_scope("failed-naive-faithfulness", "failed")

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="isolated-ragas-failure"
    )
    by_mode = {mode.mode: mode for mode in summary.modes}

    assert by_mode["naive"].quality["faithfulness"].failed_samples == 1
    assert by_mode["graph"].quality["faithfulness"].valid_samples == 1
    assert by_mode["graph"].quality["faithfulness"].failed_samples == 0
    assert by_mode["graph"].quality["faithfulness"].value == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_failed_shared_ragas_scope_preserves_official_target_mode(
    research_service,
) -> None:
    await _campaign("shared-ragas-failure", ["naive", "graph"])
    naive = await _result("shared-ragas-failure", "naive", "naive-attempt")
    graph = await _result("shared-ragas-failure", "graph", "graph-attempt")
    await _official_scope("shared-ragas-failure", naive, "naive-attempt")
    await _official_scope("shared-ragas-failure", graph, "graph-attempt")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="shared-ragas-failure",
        score_rows=_primary_score_rows([(naive, "naive-attempt")]),
    )
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-shared-faithfulness",
            campaign_id="shared-ragas-failure",
            scope_type="ragas_batch",
            scope_key="shared-faithfulness",
            metric_name="faithfulness",
            targets=[
                {
                    "campaign_result_id": naive,
                    "job_id": "ragas",
                    "work_item_id": "naive-faithfulness",
                    "attempt_id": "naive-attempt",
                    "is_official": True,
                },
                {
                    "campaign_result_id": graph,
                    "job_id": "ragas",
                    "work_item_id": "graph-faithfulness",
                    "attempt_id": "graph-attempt",
                },
            ],
        )
    )
    await store.finalize_scope("failed-shared-faithfulness", "failed")

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="shared-ragas-failure"
    )
    by_mode = {mode.mode: mode for mode in summary.modes}

    naive_quality = by_mode["naive"].quality["faithfulness"]
    graph_quality = by_mode["graph"].quality["faithfulness"]
    assert naive_quality.valid_samples == 1
    assert naive_quality.failed_samples == 0
    assert naive_quality.missing_samples == 0
    assert naive_quality.status == "complete"
    assert graph_quality.valid_samples == 0
    assert graph_quality.failed_samples == 1
    assert graph_quality.missing_samples == 0
    assert graph_quality.status == "failed"


@pytest.mark.asyncio
async def test_unattributable_older_v2_execution_scope_fails_closed(
    research_service,
) -> None:
    await _campaign("missing-mode", ["naive"])
    result_id = await _result("missing-mode", "naive", "official-attempt")
    await _official_scope("missing-mode", result_id, "official-attempt")
    await _execution_scope(
        "missing-mode",
        "unknown-result",
        "unknown-attempt",
        official=False,
        scope_status="failed",
        scope_run_id="unknown-run",
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="missing-mode"
    )

    assert summary.token_accounting_status == "partial"
    assert summary.phase_attribution_status == "partial"
    assert summary.modes[0].tokens.accounting_status == "partial"
    assert summary.modes[0].comparable is False
    assert "incomplete_accounting" in summary.modes[0].not_comparable_reasons
    assert "incomplete_pricing" not in summary.modes[0].not_comparable_reasons
    assert summary.modes[0].execution_cost.benchmark_usd == pytest.approx(0.1)
    assert summary.modes[0].execution_cost.operational_usd is None
    assert summary.modes[0].execution_cost.pricing_status == "partial"
    assert summary.modes[0].execution_cost.priced_call_count == 1
    assert summary.modes[0].execution_cost.unpriced_call_count == 0
    assert summary.execution_cost.benchmark_usd == pytest.approx(0.1)
    assert summary.execution_cost.operational_usd == pytest.approx(0.2)
    assert summary.execution_cost.pricing_status == "complete"
    assert summary.execution_cost.priced_call_count == 2
    assert summary.execution_cost.unpriced_call_count == 0
    assert any(
        warning.code == "missing_mode_attribution" for warning in summary.warnings
    )


@pytest.mark.asyncio
async def test_missing_faithfulness_stays_null_and_primary_metrics_are_present(
    research_service,
) -> None:
    await _campaign("partial", ["naive"])
    result_id = await _result("partial", "naive", "attempt-1")
    await _official_scope("partial", result_id, "attempt-1")
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-faithfulness",
            campaign_id="partial",
            scope_type="ragas_batch",
            scope_key="faithfulness",
            metric_name="faithfulness",
            targets=[
                {
                    "job_id": "ragas",
                    "work_item_id": "faithfulness",
                    "attempt_id": "faithfulness-attempt",
                }
            ],
        )
    )
    await store.finalize_scope("failed-faithfulness", "failed")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="partial",
        score_rows=[
            {
                "campaign_result_id": result_id,
                "metric_name": "answer_correctness",
                "metric_value": 0.8,
                "source_attempt_id": "attempt-1",
                "evaluation_signature": "ragas-v1",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
            {
                "campaign_result_id": result_id,
                "metric_name": "answer_relevancy",
                "metric_value": 0.7,
                "source_attempt_id": "attempt-1",
                "evaluation_signature": "ragas-v1",
                "details": {"evaluator_model": "judge", "metric_version": "v1"},
            },
        ],
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="partial"
    )

    assert set(summary.modes[0].quality) == {
        "answer_correctness",
        "faithfulness",
        "answer_relevancy",
    }
    faithfulness = summary.modes[0].quality["faithfulness"]
    assert faithfulness.value is None
    assert faithfulness.status == "failed"
    assert summary.quality["faithfulness"].value is None
    assert summary.quality["faithfulness"].status == "failed"


@pytest.mark.asyncio
async def test_mixed_campaign_keeps_execution_and_ragas_accounting_separate(
    research_service,
) -> None:
    await _campaign("mixed", ["naive", "graph"])
    naive = await _result("mixed", "naive", "naive-official", latency=100)
    graph = await _result("mixed", "graph", "graph-official", latency=500)
    failed = await _result(
        "mixed",
        "naive",
        "naive-retry",
        run_number=2,
        status=CampaignResultStatus.FAILED,
    )
    await _execution_scope(
        "mixed", naive, "naive-official", official=True, tokens=15, cost=0.1
    )
    await _execution_scope(
        "mixed", graph, "graph-official", official=True, tokens=30, cost=0.2
    )
    await _execution_scope(
        "mixed",
        failed,
        "naive-retry",
        official=False,
        scope_status="failed",
        tokens=45,
        cost=None,
        pricing_status="missing_price",
        scope_run_id="execution-worker-retry-uuid",
    )
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="ragas-batch",
            campaign_id="mixed",
            scope_type="ragas_batch",
            scope_key="faithfulness",
            metric_name="faithfulness",
            targets=[
                {"job_id": "ragas", "work_item_id": "faith", "attempt_id": "ragas-1"}
            ],
        )
    )
    await store.record_event(
        UsageEventCreate(
            usage_event_id="ragas-event",
            scope_id="ragas-batch",
            campaign_id="mixed",
            scope_type="ragas_batch",
            scope_key="faithfulness",
            phase="evaluation",
            purpose="ragas",
            metric_name="faithfulness",
            model_name="judge-v1",
            input_tokens=100,
            output_text_tokens=20,
            reported_total_tokens=120,
            usage_status="measured",
            reconciliation_status="balanced",
            estimated_cost_usd=0.5,
            pricing_status="priced",
            created_at=datetime.now(timezone.utc),
        )
    )
    await store.finalize_scope("ragas-batch", "completed")

    summary = await research_service.get_summary(user_id="user-1", campaign_id="mixed")

    assert summary.tokens.total_tokens == 45
    assert summary.execution_cost.benchmark_usd == pytest.approx(0.3)
    assert summary.execution_cost.operational_usd is None
    assert summary.execution_cost.pricing_status == "partial"
    assert (
        summary.execution_cost.priced_call_count,
        summary.execution_cost.unpriced_call_count,
    ) == (2, 1)
    assert summary.evaluation_overhead.tokens.total_tokens == 120
    assert summary.evaluation_overhead.cost_usd == pytest.approx(0.5)
    graph_mode = next(mode for mode in summary.modes if mode.mode == "graph")
    naive_mode = next(mode for mode in summary.modes if mode.mode == "naive")
    assert graph_mode.execution_cost.operational_usd == pytest.approx(0.2)
    assert graph_mode.execution_cost.pricing_status == "complete"
    assert naive_mode.execution_cost.operational_usd is None
    assert naive_mode.execution_cost.pricing_status == "partial"


@pytest.mark.asyncio
async def test_failed_context_batch_requests_null_optional_metric(
    research_service,
) -> None:
    await _campaign("requested-context", ["naive"])
    result_id = await _result("requested-context", "naive", "attempt")
    await _official_scope("requested-context", result_id, "attempt")
    store = EvaluationAccountingStore()
    await store.start_scope(
        AccountingScopeStart(
            scope_id="failed-context-batch",
            campaign_id="requested-context",
            scope_type="ragas_batch",
            scope_key="context-recall",
            metric_name="context_recall",
            targets=[
                {"job_id": "ragas", "work_item_id": "context", "attempt_id": "batch"}
            ],
        )
    )
    await store.finalize_scope("failed-context-batch", "failed")

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="requested-context"
    )

    context_recall = summary.quality["context_recall"]
    assert context_recall.value is None
    assert context_recall.status == "failed"


@pytest.mark.asyncio
async def test_running_ragas_batches_mark_primary_and_optional_metrics_evaluating(
    research_service,
) -> None:
    await _campaign("active-work", ["naive"])
    result_id = await _result("active-work", "naive", "attempt")
    await _official_scope("active-work", result_id, "attempt")
    store = EvaluationAccountingStore()
    for metric_name in ("answer_correctness", "context_precision"):
        await store.start_scope(
            AccountingScopeStart(
                scope_id=f"running-{metric_name}",
                campaign_id="active-work",
                scope_type="ragas_batch",
                scope_key=metric_name,
                metric_name=metric_name,
                targets=[
                    {
                        "job_id": "ragas",
                        "work_item_id": metric_name,
                        "attempt_id": f"attempt-{metric_name}",
                    }
                ],
            )
        )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="active-work"
    )

    assert summary.quality["answer_correctness"].status == "evaluating"
    assert summary.quality["context_precision"].status == "evaluating"
    assert summary.quality["faithfulness"].status == "not_requested"


@pytest.mark.asyncio
async def test_cross_linked_score_attempt_is_excluded_from_its_result(
    research_service,
) -> None:
    await _campaign("stale-score", ["naive"])
    first = await _result("stale-score", "naive", "first")
    second = await _result("stale-score", "naive", "second", run_number=2)
    await _official_scope("stale-score", first, "first")
    await _official_scope("stale-score", second, "second")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="stale-score",
        score_rows=[
            {
                "campaign_result_id": first,
                "metric_name": "answer_correctness",
                "metric_value": 0.9,
                "source_attempt_id": "second",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge"},
            },
            {
                "campaign_result_id": second,
                "metric_name": "answer_correctness",
                "metric_value": 0.4,
                "source_attempt_id": "second",
                "evaluation_signature": "sig",
                "details": {"evaluator_model": "judge"},
            },
        ],
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="stale-score"
    )
    observation = summary.quality["answer_correctness"]

    assert observation.value == pytest.approx(0.4)
    assert observation.valid_samples == 1
    assert observation.metric_version is None


@pytest.mark.asyncio
async def test_mixed_evaluator_metadata_excludes_incompatible_scores_and_marks_mode(
    research_service,
) -> None:
    await _campaign("compat", ["naive"])
    first = await _result("compat", "naive", "first")
    second = await _result("compat", "naive", "second", run_number=2)
    await _official_scope("compat", first, "first")
    await _official_scope("compat", second, "second")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="compat",
        score_rows=[
            {
                "campaign_result_id": first,
                "metric_name": "answer_correctness",
                "metric_value": 0.8,
                "source_attempt_id": "first",
                "evaluation_signature": "input-sig-first",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "policy-a",
                    "compatibility_signature_version": "v2",
                },
            },
            {
                "campaign_result_id": second,
                "metric_name": "answer_correctness",
                "metric_value": 0.2,
                "source_attempt_id": "second",
                "evaluation_signature": "input-sig-second",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "policy-b",
                    "compatibility_signature_version": "v2",
                },
            },
        ],
    )

    summary = await research_service.get_summary(user_id="user-1", campaign_id="compat")
    mode = summary.modes[0]

    assert mode.quality["answer_correctness"].value == pytest.approx(0.8)
    assert mode.quality["answer_correctness"].valid_samples == 1
    assert mode.comparable is False
    assert "evaluator_metadata_mismatch" in mode.not_comparable_reasons
    assert any(
        warning.code == "evaluator_metadata_mismatch" for warning in summary.warnings
    )


@pytest.mark.asyncio
async def test_fully_scored_primary_metrics_compare_each_evaluator_cohort_separately(
    research_service,
) -> None:
    await _campaign("fully-scored", ["naive"])
    results = [
        await _result(
            "fully-scored",
            "naive",
            f"attempt-{index}",
            run_number=index,
        )
        for index in range(1, 6)
    ]
    for index, result_id in enumerate(results, start=1):
        await _official_scope("fully-scored", result_id, f"attempt-{index}")

    metrics = ("answer_correctness", "faithfulness", "answer_relevancy")
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="fully-scored",
        score_rows=[
            {
                "campaign_result_id": result_id,
                "metric_name": metric_name,
                "metric_value": 0.8,
                "source_attempt_id": f"attempt-{index}",
                "evaluation_signature": f"input-{metric_name}-{index}",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": f"policy-{metric_name}",
                },
            }
            for index, result_id in enumerate(results, start=1)
            for metric_name in metrics
        ],
    )

    summary = await research_service.get_summary(
        user_id="user-1", campaign_id="fully-scored"
    )
    mode = summary.modes[0]

    assert all(mode.quality[metric].status == "complete" for metric in metrics)
    assert all(mode.quality[metric].valid_samples == 5 for metric in metrics)
    assert mode.comparable is True
    assert "evaluator_metadata_mismatch" not in mode.not_comparable_reasons
    assert not any(
        warning.code == "evaluator_metadata_mismatch" for warning in summary.warnings
    )


@pytest.mark.asyncio
async def test_campaign_aggregate_uses_raw_runs_and_optional_metrics_are_requested_only(
    research_service,
) -> None:
    await _campaign("raw", ["naive", "graph"])
    naive_one = await _result("raw", "naive", "n1", latency=10)
    naive_two = await _result("raw", "naive", "n2", latency=20, run_number=2)
    graph = await _result("raw", "graph", "g1", latency=100, run_number=3)
    for result_id, attempt in ((naive_one, "n1"), (naive_two, "n2"), (graph, "g1")):
        await _official_scope("raw", result_id, attempt)
    await RagasScoreRepository().replace_for_campaign(
        user_id="user-1",
        campaign_id="raw",
        score_rows=[
            {
                "campaign_result_id": naive_one,
                "metric_name": "answer_correctness",
                "metric_value": 0.1,
                "source_attempt_id": "n1",
                "evaluation_signature": "input-sig-n1",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "answer-policy",
                },
            },
            {
                "campaign_result_id": naive_two,
                "metric_name": "answer_correctness",
                "metric_value": 0.1,
                "source_attempt_id": "n2",
                "evaluation_signature": "input-sig-n2",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "answer-policy",
                },
            },
            {
                "campaign_result_id": graph,
                "metric_name": "answer_correctness",
                "metric_value": 0.9,
                "source_attempt_id": "g1",
                "evaluation_signature": "input-sig-g1",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "answer-policy",
                },
            },
            {
                "campaign_result_id": graph,
                "metric_name": "context_precision",
                "metric_value": 0.6,
                "source_attempt_id": "g1",
                "evaluation_signature": "input-sig-context-g1",
                "details": {
                    "evaluator_model": "judge",
                    "metric_version": "v1",
                    "compatibility_signature": "context-policy",
                },
            },
        ],
    )

    summary = await research_service.get_summary(user_id="user-1", campaign_id="raw")

    assert summary.quality["answer_correctness"].value == pytest.approx(
        (0.1 + 0.1 + 0.9) / 3
    )
    assert summary.latency.mean_ms == pytest.approx(130 / 3)
    assert summary.latency.p50_ms == 20
    assert summary.latency.p95_ms == 100
    assert summary.latency.low_sample_size is True
    assert set(summary.quality) == {
        "answer_correctness",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    }


@pytest.mark.asyncio
async def test_research_aggregates_use_bounded_result_projection_for_large_payloads(
    research_service, monkeypatch
) -> None:
    """Research aggregate responses must not materialize detail/export payloads."""
    campaign_id = "bounded-research-results"
    await _campaign(campaign_id, ["naive"])
    result_id = await _result(campaign_id, "naive", "bounded-attempt")
    large_payload = "x" * (2 * 1024 * 1024)
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            UPDATE campaign_results
            SET answer = ?, contexts_json = ?, ground_truth = ?
            WHERE id = ?
            """,
            (large_payload, json.dumps([large_payload]), large_payload, result_id),
        )
        await connection.commit()

    # Preserve the existing response contract before injecting a bounded-loader spy.
    expected_summary = await research_service.get_summary(
        user_id="user-1", campaign_id=campaign_id
    )
    expected_comparison = await research_service.get_question_comparison(
        user_id="user-1", campaign_id=campaign_id
    )
    expected_behavior = await research_service.get_agent_behavior(
        user_id="user-1", campaign_id=campaign_id
    )

    class ResultRepositorySpy:
        def __init__(self) -> None:
            self.research_calls = 0
            self.full_calls = 0
            self._delegate = CampaignResultRepository()

        async def list_for_campaign_research(self, *, user_id: str, campaign_id: str):
            self.research_calls += 1
            return await self._delegate.list_for_campaign_research(
                user_id=user_id, campaign_id=campaign_id
            )

        async def list_for_campaign(self, *, user_id: str, campaign_id: str):
            self.full_calls += 1
            raise AssertionError("research aggregate loaded full campaign results")

    campaign_result_queries: list[str] = []
    original_execute = evaluation_db.aiosqlite.Connection.execute

    async def capture_execute(connection, sql, parameters=()):
        if "from campaign_results" in sql.lower():
            campaign_result_queries.append(sql.lower())
        return await original_execute(connection, sql, parameters)

    monkeypatch.setattr(evaluation_db.aiosqlite.Connection, "execute", capture_execute)
    results = ResultRepositorySpy()
    service = ResearchAnalyticsService(results=results)

    summary = await service.get_summary(user_id="user-1", campaign_id=campaign_id)
    comparison = await service.get_question_comparison(
        user_id="user-1", campaign_id=campaign_id
    )
    behavior = await service.get_agent_behavior(
        user_id="user-1", campaign_id=campaign_id
    )

    assert summary.model_dump(mode="json") == expected_summary.model_dump(mode="json")
    assert comparison.model_dump(mode="json") == expected_comparison.model_dump(
        mode="json"
    )
    assert behavior.model_dump(mode="json") == expected_behavior.model_dump(mode="json")
    assert results.research_calls == 3
    assert results.full_calls == 0
    assert len(campaign_result_queries) == 3
    assert all("answer" not in query for query in campaign_result_queries)
    assert all("contexts_json" not in query for query in campaign_result_queries)
    assert all("ground_truth" not in query for query in campaign_result_queries)
