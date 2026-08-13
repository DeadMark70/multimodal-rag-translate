from __future__ import annotations

import asyncio
import itertools
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from evaluation.router import get_evaluation_export_service
from evaluation import db as evaluation_db
from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    CostSummary,
    EvaluationOverheadSummary,
    LatencySummary,
    TokenBreakdown,
)
from evaluation.campaign_engine import CampaignEngine
from evaluation.campaign_schemas import (
    AblationResponse,
    AgentBehaviorResponse,
    CampaignErrorsResponse,
    CampaignResult,
    CampaignStageWarningsResponse,
    HumanEvalQueueResponse,
    HumanVsAutoResponse,
    ResearchQuestionComparisonResponse,
    RouterAnalysisResponse,
    V9EvidencePacket,
    V9ExecutionObservability,
)
from evaluation.evidence import content_hash
from evaluation.export_schemas import (
    ExportCampaignRequest as ExportCampaignRequestV2,
    resolve_export_content_policy,
)
from evaluation.export_service import (
    EvaluationExportService,
    _project_export_run_observability,
)
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.release_metrics import ReleaseMetricsReport
from evaluation.research_analytics import CanonicalRunObservability
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationTraceEvent,
)
from data_base.agentic_v9.schemas import EvidencePacket, FinalClaim
from main import app


CONTENT_POLICY_ROWS = list(itertools.product((False, True), repeat=5))
NOW = datetime(2026, 8, 13, tzinfo=timezone.utc)
FREE_TEXT_SECRET = "api_key=export-free-text-secret"


@pytest.mark.parametrize(
    (
        "include_raw_trace_payloads",
        "include_prompt_previews",
        "include_full_prompts",
        "include_answers",
        "include_retrieved_excerpts",
    ),
    CONTENT_POLICY_ROWS,
)
def test_content_policy_covers_all_32_flag_combinations(
    include_raw_trace_payloads: bool,
    include_prompt_previews: bool,
    include_full_prompts: bool,
    include_answers: bool,
    include_retrieved_excerpts: bool,
) -> None:
    request = ExportCampaignRequestV2(
        include_run_observability=False,
        include_raw_trace_payloads=include_raw_trace_payloads,
        include_prompt_previews=include_prompt_previews,
        include_full_prompts=include_full_prompts,
        include_answers=include_answers,
        include_retrieved_excerpts=include_retrieved_excerpts,
    )

    captured = resolve_export_content_policy(request, captured_at_execution=True)
    not_captured = resolve_export_content_policy(request, captured_at_execution=False)

    assert len(CONTENT_POLICY_ROWS) == 32
    assert captured.raw_trace_allowed is include_raw_trace_payloads
    assert captured.prompt_preview_allowed is include_prompt_previews
    assert captured.full_prompt_allowed is include_full_prompts
    assert not_captured.full_prompt_allowed is False
    assert captured.answer_text_allowed is include_answers
    assert captured.excerpt_text_allowed is include_retrieved_excerpts
    assert captured.provider_bodies_allowed is False
    assert captured.credentials_allowed is False
    assert captured.authorization_headers_allowed is False
    assert captured.stack_traces_allowed is False
    assert captured.unrestricted_errors_allowed is False
    assert captured.non_trace_payloads_allowed is False


def _aggregate(model_type):
    return model_type(
        campaign_id="campaign-1",
        analysis_unit="execution",
        sample_count=1,
        independent_question_count=1,
        repeat_count=1,
        sample_note="one execution",
        warnings=[],
        rows=[],
        summaries={},
    )


def _export_result() -> CampaignResult:
    return CampaignResult(
        id="run-1",
        campaign_id="campaign-1",
        question_id="question-1",
        question="Question?",
        ground_truth="Ground truth",
        ground_truth_short="Truth",
        mode="agentic",
        run_number=1,
        repeat_number=1,
        agentic_execution_version="v9",
        answer=f"Answer text {FREE_TEXT_SECRET}",
        contexts=[f"Context text {FREE_TEXT_SECRET}"],
        source_doc_ids=["doc-1"],
        latency_ms=12,
        total_latency_ms=15,
        total_tokens=17,
        source_attempt_id="attempt-1",
        status="completed",
        created_at=NOW,
    )


def _canonical_export_run() -> CanonicalRunObservability:
    result = _export_result()
    return CanonicalRunObservability(
        result=result,
        token_breakdown=TokenBreakdown(
            input_tokens=10,
            output_text_tokens=7,
            total_tokens=17,
            accounting_status="complete",
            phase_attribution_status="complete",
        ),
        trace_events=[
            EvaluationTraceEvent(
                event_id="trace-1",
                run_id=result.id,
                campaign_id=result.campaign_id,
                span_id="span-1",
                event_type="generation",
                sequence=1,
                stage_type="generation",
                stage_name="final_answer",
                started_at=NOW,
                status="success",
                payload={
                    "safe": f"value {FREE_TEXT_SECRET}",
                    "access_token": "trace-secret-sentinel",
                    "nested": {
                        "answer": f"Nested answer {FREE_TEXT_SECRET}",
                        "statement": f"Nested statement {FREE_TEXT_SECRET}",
                        "context": f"Nested context {FREE_TEXT_SECRET}",
                        "excerpt": f"Nested excerpt {FREE_TEXT_SECRET}",
                        "doc_id": "doc-identity-must-survive",
                    },
                    "provider_response": {"body": "provider-body-sentinel"},
                    "stack_trace": "stack-trace-sentinel",
                    "error": {"message": "unrestricted-error-sentinel"},
                },
                created_at=NOW,
            )
        ],
        llm_calls=[
            EvaluationLlmCall(
                llm_call_id="llm-1",
                run_id=result.id,
                campaign_id=result.campaign_id,
                phase="final_answer",
                purpose="generation",
                prompt_tokens=10,
                completion_tokens=7,
                total_tokens=17,
                prompt_capture_status="captured",
                full_prompt_capture_status="captured",
                prompt_preview=f"Prompt preview {FREE_TEXT_SECRET}",
                payload={"full_prompt": f"Full prompt {FREE_TEXT_SECRET}"},
                created_at=NOW,
            )
        ],
        retrieval_events=[],
        retrieval_chunks=[
            EvaluationRetrievalChunk(
                retrieval_chunk_id="chunk-row-1",
                run_id=result.id,
                campaign_id=result.campaign_id,
                retrieval_event_id="retrieval-1",
                chunk_id="chunk-1",
                doc_id="doc-1",
                excerpt=f"Retrieved excerpt {FREE_TEXT_SECRET}",
                created_at=NOW,
            )
        ],
        context_packs=[],
        tool_calls=[],
        routing_decisions=[],
        graph_events=[],
        graph_evidence_items=[],
        claims=[
            EvaluationClaim(
                claim_id="claim-1",
                run_id=result.id,
                campaign_id=result.campaign_id,
                claim_text=f"Claim statement {FREE_TEXT_SECRET}",
                created_at=NOW,
            )
        ],
        human_ratings=[],
        agentic_v9=V9ExecutionObservability(
            evidence_packets=[
                V9EvidencePacket(
                    evidence_id="evidence-row-1",
                    packet=EvidencePacket(
                        schema_version="1",
                        evidence_id="evidence-1",
                        task_id="task-1",
                        round_id="round-1",
                        query_id="query-1",
                        slot_ids=["slot-1"],
                        statement=f"Evidence statement {FREE_TEXT_SECRET}",
                        support_type="direct",
                        source={"doc_id": "doc-1", "chunk_id": "chunk-1"},
                        scope={"dataset": "benchmark"},
                        locator={"pdf_page_index": 0},
                    ),
                )
            ],
            final_claims=[
                FinalClaim(
                    claim_id="final-claim-1",
                    statement=f"Final claim statement {FREE_TEXT_SECRET}",
                    support_type="direct",
                    evidence_ids=["evidence-1"],
                )
            ],
            comparison={
                "planner_status": "planned",
                "subjects": [
                    {
                        "subject_id": "subject-1",
                        "display_name": f"Subject {FREE_TEXT_SECRET}",
                    }
                ],
                "task_diagnostics": [
                    {
                        "task_id": "task-1",
                        "subject_id": "subject-1",
                        "query_preview": f"Query {FREE_TEXT_SECRET}",
                        "selected": [{"doc_id": "doc-1", "chunk_id": "chunk-1"}],
                    }
                ],
                "provider_body": "comparison-provider-body-sentinel",
                "stack_trace": "comparison-stack-sentinel",
            },
        ),
        graph_observability_status="not_instrumented",
        claim_extraction_status="recorded",
        evidence_coverage=[
            {
                "atomic_fact_id": "fact-1",
                "fact_text": f"Atomic fact text {FREE_TEXT_SECRET}",
                "retrieved": True,
                "packed": True,
                "mentioned": True,
                "cited": True,
                "expected_doc_ids": ["doc-1"],
            }
        ],
        evidence_coverage_status="complete",
    )


@pytest.mark.parametrize(
    (
        "include_raw_trace_payloads",
        "include_prompt_previews",
        "include_full_prompts",
        "include_answers",
        "include_retrieved_excerpts",
    ),
    CONTENT_POLICY_ROWS,
)
def test_export_projector_applies_all_nested_content_policies(
    include_raw_trace_payloads: bool,
    include_prompt_previews: bool,
    include_full_prompts: bool,
    include_answers: bool,
    include_retrieved_excerpts: bool,
) -> None:
    projected = _project_export_run_observability(
        canonical=_canonical_export_run(),
        request=ExportCampaignRequestV2(
            include_run_observability=True,
            include_raw_trace_payloads=include_raw_trace_payloads,
            include_prompt_previews=include_prompt_previews,
            include_full_prompts=include_full_prompts,
            include_answers=include_answers,
            include_retrieved_excerpts=include_retrieved_excerpts,
        ),
    )

    assert projected.trace_events[0].payload == (
        {
            "safe": "value [redacted]",
            "access_token": "[redacted]",
            "nested": {
                "answer": "Nested answer [redacted]" if include_answers else None,
                "statement": (
                    "Nested statement [redacted]"
                    if include_retrieved_excerpts
                    else None
                ),
                "context": (
                    "Nested context [redacted]"
                    if include_retrieved_excerpts
                    else None
                ),
                "excerpt": (
                    "Nested excerpt [redacted]"
                    if include_retrieved_excerpts
                    else None
                ),
                "doc_id": "doc-identity-must-survive",
            },
        }
        if include_raw_trace_payloads
        else {}
    )
    assert projected.llm_calls[0].prompt_preview == (
        "Prompt preview [redacted]" if include_prompt_previews else None
    )
    assert projected.llm_calls[0].full_prompt == (
        "Full prompt [redacted]" if include_full_prompts else None
    )
    assert projected.run_summary.answer_preview == (
        "Answer text [redacted]" if include_answers else None
    )
    assert projected.claims[0].claim_text == (
        "Claim statement [redacted]" if include_answers else None
    )
    assert projected.evidence_coverage[0].fact_text == (
        "Atomic fact text [redacted]" if include_answers else None
    )
    assert projected.retrieval_chunks[0].excerpt == (
        "Retrieved excerpt [redacted]" if include_retrieved_excerpts else None
    )
    assert projected.agentic_v9 is not None
    assert projected.agentic_v9.final_claims[0].statement == (
        "Final claim statement [redacted]" if include_answers else None
    )
    assert projected.agentic_v9.evidence_packets[0].packet.statement == (
        "Evidence statement [redacted]" if include_retrieved_excerpts else None
    )
    assert projected.agentic_v9.evidence_packets[0].packet.locator.pdf_page_index == 0
    assert projected.agentic_v9.comparison is not None
    assert projected.agentic_v9.comparison.subjects[0].display_name == "Subject [redacted]"
    assert (
        projected.agentic_v9.comparison.task_diagnostics[0].query_preview
        == "Query [redacted]"
    )
    serialized = projected.model_dump_json()
    for forbidden in (
        "export-free-text-secret",
        "provider-body-sentinel",
        "stack-trace-sentinel",
        "unrestricted-error-sentinel",
        "comparison-provider-body-sentinel",
        "comparison-stack-sentinel",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_export_summary_composes_named_sections_without_loading_observability() -> None:
    oversized_name = f"Campaign {FREE_TEXT_SECRET} " + ("x" * 100_000)
    campaign = SimpleNamespace(
        id="campaign-1",
        name=oversized_name,
        status="completed",
        config=SimpleNamespace(
            benchmark_id=None,
            modes=["agentic"],
            repeat_count=1,
        ),
        created_at=NOW,
        updated_at=NOW,
    )
    result = _export_result()
    analytics = SimpleNamespace(
        router_analysis=AsyncMock(return_value=_aggregate(RouterAnalysisResponse)),
        ablation=AsyncMock(return_value=_aggregate(AblationResponse)),
        human_vs_auto=AsyncMock(return_value=_aggregate(HumanVsAutoResponse)),
        human_eval_queue=AsyncMock(
            return_value=HumanEvalQueueResponse(campaign_id="campaign-1", rows=[])
        ),
        campaign_errors=AsyncMock(
            return_value=CampaignErrorsResponse(campaign_id="campaign-1", rows=[])
        ),
        campaign_stage_warnings=AsyncMock(
            return_value=CampaignStageWarningsResponse(
                campaign_id="campaign-1", rows=[]
            )
        ),
    )
    research = SimpleNamespace(
        get_summary=AsyncMock(
            return_value=CampaignResearchSummaryResponse(
                campaign_id="campaign-1",
                completed_run_count=1,
                total_run_count=1,
                failed_run_count=0,
                quality_status="partial",
                token_accounting_status="incomplete_legacy",
                pricing_status="unknown",
                phase_attribution_status="not_available",
                sample_count=1,
                latency=LatencySummary(mean_ms=15, sample_count=1),
                tokens=TokenBreakdown(
                    total_tokens=None,
                    accounting_status="incomplete_legacy",
                    phase_attribution_status="not_available",
                ),
                execution_cost=CostSummary(pricing_status="unknown"),
                modes=[],
                evaluation_overhead=EvaluationOverheadSummary(
                    tokens=TokenBreakdown(
                        total_tokens=None,
                        accounting_status="incomplete_legacy",
                        phase_attribution_status="not_available",
                    ),
                    pricing_status="unknown",
                ),
            )
        ),
        get_question_comparison=AsyncMock(
            return_value=_aggregate(ResearchQuestionComparisonResponse)
        ),
        get_agent_behavior=AsyncMock(
            return_value=_aggregate(AgentBehaviorResponse)
        ),
        get_official_ragas_by_run=AsyncMock(
            return_value={result.id: {"faithfulness": 0.9}}
        ),
        get_campaign_run_accounting=AsyncMock(
            return_value={
                result.id: TokenBreakdown(
                    total_tokens=None,
                    accounting_status="incomplete_legacy",
                    phase_attribution_status="not_available",
                )
            }
        ),
        get_campaign_run_observability=AsyncMock(
            side_effect=AssertionError("summary export loaded observability")
        ),
    )
    release = SimpleNamespace(
        get_report=AsyncMock(
            return_value=ReleaseMetricsReport(
                benchmark_id="",
                benchmark_kind="not_applicable",
                comparable=False,
                availability="not_applicable",
                not_applicable_reason="benchmark_not_configured",
                gate_reasons=["benchmark_not_configured"],
            )
        )
    )
    service = EvaluationExportService(
        campaigns=SimpleNamespace(get=AsyncMock(return_value=campaign)),
        results=SimpleNamespace(list_for_campaign=AsyncMock(return_value=[result])),
        analytics=analytics,
        research=research,
        release=release,
    )

    response = await service.export_campaign(
        user_id="user-1",
        campaign_id="campaign-1",
        request=ExportCampaignRequestV2(),
    )

    assert set(type(response.sections).model_fields) == {
        "overview",
        "question_analysis",
        "agent_behavior",
        "router_analysis",
        "ablation",
        "human_evaluation",
        "diagnostics",
    }
    assert response.sections.overview.data.release_metrics.availability.status == (
        "not_applicable"
    )
    assert response.runs[0].ragas_metrics == {"faithfulness": 0.9}
    assert response.runs[0].accounting.total_tokens is None
    assert response.runs[0].latency.total_latency_ms == 15
    assert response.runs[0].observability.included is False
    assert response.runs[0].observability.data is None
    assert response.campaign.name is not None
    assert "export-free-text-secret" not in response.model_dump_json()
    assert len(response.campaign.name) < len(oversized_name)
    assert response.campaign.name == "[REDACTED]"
    research.get_campaign_run_observability.assert_not_awaited()

    research.get_campaign_run_observability.side_effect = None
    research.get_campaign_run_observability.return_value = {}
    with pytest.raises(ValueError, match="result IDs"):
        await service.export_campaign(
            user_id="user-1",
            campaign_id="campaign-1",
            request=ExportCampaignRequestV2(include_run_observability=True),
        )


def test_export_projector_does_not_truncate_event_tail() -> None:
    canonical = _canonical_export_run()
    trace_events = [
        canonical.trace_events[0].model_copy(
            update={"event_id": f"trace-{index}", "sequence": index + 1}
        )
        for index in range(101)
    ]

    projected = _project_export_run_observability(
        canonical=replace(canonical, trace_events=trace_events),
        request=ExportCampaignRequestV2(include_run_observability=True),
    )

    assert len(projected.trace_events) == 101
    assert projected.trace_events[-1].event_id == "trace-100"


def test_export_required_section_failure_returns_no_partial_v2_body() -> None:
    class FailingExportService:
        async def export_campaign(self, **_: object) -> object:
            raise AppError(
                code=ErrorCode.INTERNAL_ERROR,
                message="required export section unavailable",
                status_code=500,
            )

    engine = CampaignEngine(runner=AsyncMock(), ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export_all_or_error")
    with _build_client("user-a", upload_root, db_path, engine) as client:
        app.dependency_overrides[get_evaluation_export_service] = (
            lambda: FailingExportService()
        )
        response = client.post(
            "/api/evaluation/campaigns/campaign-1/export", json={}
        )

    assert response.status_code == 500
    assert "schema_version" not in response.json()
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


class FakeRagasEvaluator:
    async def evaluate_campaign(self, *, on_progress=None, **kwargs) -> str:
        if on_progress:
            await on_progress(1, 1, "Q-EXPORT", "agentic")
        return "fake-ragas"


@contextmanager
def _build_client(
    user_id: str, upload_root: Path, db_path: Path, engine: CampaignEngine
):
    process_worker = Mock(is_configured=False)
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.storage.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("evaluation.db.EVALUATION_DB_PATH", db_path),
        patch("evaluation.campaign_engine.get_campaign_engine", return_value=engine),
        patch(
            "evaluation.job_worker.get_evaluation_job_worker",
            return_value=process_worker,
        ),
        patch("evaluation.router.get_campaign_engine", return_value=engine),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: user_id
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def _wait_for_completed(client: TestClient, campaign_id: str) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        response = client.get("/api/evaluation/campaigns")
        assert response.status_code == 200
        current = next(item for item in response.json() if item["id"] == campaign_id)
        if current["status"] == "completed":
            return
        time.sleep(0.05)
    raise AssertionError(f"campaign {campaign_id} did not complete")


async def _seed_export_rows(*, run_id: str, campaign_id: str, attempt_id: str) -> None:
    repository = EvaluationObservabilityRepository()
    now = datetime.now(timezone.utc)
    await repository.record_trace_event(
        EvaluationTraceEvent(
            event_id=f"{run_id}-export-error",
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=f"{run_id}-span",
            parent_event_id=None,
            parent_span_id=None,
            event_type="generation",
            event_schema_version="1.0",
            sequence=10,
            stage_type="generation",
            stage_name="answer_generation",
            started_at=now,
            ended_at=now,
            duration_ms=10,
            status="failed",
            retry_count=0,
            payload={"provider": "test"},
            error={
                "code": "PROVIDER_ERROR",
                "message": "apiKey=sk-secret exploded with stack trace",
            },
            created_at=now,
        )
    )
    await repository.record_llm_call(
        EvaluationLlmCall(
            llm_call_id=f"{run_id}-llm",
            run_id=run_id,
            campaign_id=campaign_id,
            phase="final_answer",
            purpose="campaign_generation",
            provider="google",
            model_name="gemini-2.5-flash",
            prompt_tokens=10,
            completion_tokens=6,
            total_tokens=16,
            prompt_hash="prompt-hash",
            prompt_preview="Question: preview only",
            prompt_capture_status="captured",
            full_prompt_capture_status="captured",
            response_hash="response-hash",
            latency_ms=12,
            status="failed",
            error={"message": "apiKey=sk-secret exploded with stack trace"},
            payload={
                "full_prompt": (
                    '{"content":{"password":"hunter2","note":"safe",'
                    '"access_token":"export-access-token-sentinel",'
                    '"client_secret":"export-client-secret-sentinel",'
                    '"refresh_token":"export-refresh-token-sentinel",'
                    '"id_token":"export-id-token-sentinel",'
                    '"private_key":"export-private-key-sentinel"},'
                    '"authorization":"Bearer quoted-credential"}'
                ),
                "structured": {
                    "client_id": "export-public-client-id",
                    "api_key": "quoted-api-key",
                    "token": "quoted-token",
                    "credentials": [
                        {"access_token": "export-list-access-token-sentinel"},
                        {"client_secret": "export-list-client-secret-sentinel"},
                    ],
                },
                "other_field": "kept",
            },
            created_at=now,
        )
    )
    await repository.materialize_v9_attempt(
        attempt_id=attempt_id,
        run_id=run_id,
        campaign_id=campaign_id,
        condition_id="",
        schema_version="1",
        trace_payload={
            "requirement_guidance": {
                "enabled": True,
                "source": "setup_snapshot",
                "applied_task_count": 2,
                "fallback_reason": None,
                "_advisory_suffix": "MUST NOT BE EXPORTED",
                "applied_task_ids": ["task-a", "task-b"],
            },
            "requirement_shadow": {
                "schema_version": "shadow_requirements_v2",
                "behavior_influence": False,
                "support_assessment": "candidate_only",
                "requirements": [
                    {
                        "requirement_id": "R1",
                        "text": "What failed?",
                        "answer_kind": "text",
                        "information_need": "plain_text",
                        "information_needs": ["plain_text"],
                        "decomposition_method": "fallback",
                        "decomposition_confidence": "low",
                        "visual_precision": "none",
                        "visual_decision": "not_requested",
                        "visual_reason": "text_representation_expected",
                        "importance": "core",
                        "coverage_status": "candidate",
                        "available_representations": ["plain_text"],
                        "candidate_evidence_refs": ["doc-1:runtime-export-chunk"],
                    }
                ],
                "response_constraints": [
                    {
                        "constraint_id": "C1",
                        "kind": "conditional_scope",
                        "text": "若不能，必須按 claim scope 分開回答。",
                    }
                ],
                "truncated": False,
                "summary": {
                    "requirement_count": 1,
                    "candidate_count": 1,
                    "missing_count": 0,
                    "supported_count": 0,
                    "visual_required_count": 0,
                    "constraint_count": 1,
                    "low_confidence_count": 1,
                    "truncated_requirement_count": 0,
                    "truncated_constraint_count": 0,
                },
            },
        },
        evidence_packets=[],
        slot_resolutions=[],
        claims=[],
    )
    await repository.record_llm_call(
        EvaluationLlmCall(
            llm_call_id=f"{run_id}-not-captured",
            run_id=run_id,
            campaign_id=campaign_id,
            phase="evidence_extract",
            purpose="evidence_extraction",
            reservation_id="reservation-not-captured",
            provider_attempt=1,
            provider="google",
            model_name="gemini-2.5-flash",
            prompt_hash="not-captured-hash",
            prompt_capture_status="not_captured_at_execution",
            full_prompt_capture_status="not_captured_at_execution",
            payload={"full_prompt": "MUST NEVER BE EXPORTED"},
            created_at=now,
        )
    )


async def _seed_stage_warning(*, run_id: str, campaign_id: str) -> None:
    now = datetime.now(timezone.utc)
    await EvaluationObservabilityRepository().record_trace_event(
        EvaluationTraceEvent(
            event_id=f"{run_id}-graph-warning",
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=f"{run_id}-graph-span",
            parent_event_id=None,
            parent_span_id=None,
            event_type="graph_locator",
            event_schema_version="1.0",
            sequence=11,
            stage_type="graph",
            stage_name="agentic_v9_graph_locator",
            started_at=now,
            ended_at=now,
            duration_ms=10,
            status="partial",
            retry_count=0,
            payload={"execution_state": "required_but_not_satisfied"},
            error={"reason": "no_eligible_graph_source_evidence"},
            created_at=now,
        )
    )


def _campaign_payload() -> dict:
    return {
        "name": "Export",
        "test_case_ids": ["Q-EXPORT"],
        "modes": ["agentic"],
        "model_config": {
            "id": "cfg-1",
            "name": "Balanced",
            "model_name": "gemini-2.5-flash",
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "thinking_mode": False,
            "thinking_budget": 8192,
        },
        "repeat_count": 1,
        "batch_size": 1,
        "rpm_limit": 60,
    }


def _condition_campaign_payload() -> dict:
    payload = _campaign_payload()
    payload["ablation_conditions"] = [
        {
            "condition_id": "v9-baseline",
            "label": "Requirement guidance off",
            "mode": "agentic",
            "ablation_flags": {"requirement_guidance": False},
        },
        {
            "condition_id": "v9-guided",
            "label": "Requirement guidance on",
            "mode": "agentic",
            "ablation_flags": {"requirement_guidance": True},
        },
    ]
    return payload


async def _seed_condition_ragas_scores(
    *, campaign_id: str, user_id: str, result_ids_by_condition: dict[str, str]
) -> None:
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        for condition_id, result_id in result_ids_by_condition.items():
            await connection.execute(
                """
                INSERT INTO ragas_scores (
                    id, campaign_id, campaign_result_id, user_id, metric_name,
                    metric_value, details_json, created_at
                ) VALUES (?, ?, ?, ?, 'answer_correctness', ?, '{}', ?)
                """,
                (
                    f"{result_id}-correctness",
                    campaign_id,
                    result_id,
                    user_id,
                    0.7 if condition_id == "v9-baseline" else 0.9,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        await connection.commit()


def _make_workspace_paths(prefix: str) -> tuple[Path, Path]:
    root = Path.cwd() / "output" / "test_tmp" / f"{prefix}_{uuid4().hex}"
    return root / "uploads", root / "evaluation.db"


def test_export_defaults_redact_full_prompts_and_errors_are_sanitized() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="Grounded answer",
            contexts=["SECRET RETRIEVED CONTEXT"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 16},
            category=test_case.category,
            difficulty=test_case.difficulty,
            agent_trace={
                "agentic_v9": {
                    "retrieval_diagnostics": [
                        {
                            "task_id": "export-retrieval-task",
                            "status": "executed",
                            "fallback_reason": None,
                            "candidate_count": 8,
                            "selected_count": 4,
                            "selected": [
                                {
                                    "doc_id": "doc-1",
                                    "chunk_id": "runtime-export-chunk",
                                    "content_hash": content_hash(
                                        "SECRET RETRIEVED CONTEXT"
                                    ),
                                    "pre_rerank_rank": 7,
                                    "post_rerank_rank": 2,
                                    "rerank_score": 0.73,
                                }
                            ],
                        }
                    ]
                }
            },
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export")

    with _build_client("user-a", upload_root, db_path, engine) as client:
        created_case = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-EXPORT",
                "question": "What failed?",
                "ground_truth": "A safe answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        created = client.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client, campaign_id)
        result_row = client.get(
            f"/api/evaluation/campaigns/{campaign_id}/results"
        ).json()["results"][0]
        run_id = result_row["id"]
        asyncio.run(
            _seed_export_rows(
                run_id=run_id,
                campaign_id=campaign_id,
                attempt_id=result_row["source_attempt_id"],
            )
        )
        asyncio.run(_seed_stage_warning(run_id=run_id, campaign_id=campaign_id))

        errors_response = client.get(f"/api/evaluation/campaigns/{campaign_id}/errors")
        assert errors_response.status_code == 200
        error_row = errors_response.json()["rows"][0]
        assert error_row["run_id"] == run_id
        assert error_row["stage_name"] == "answer_generation"
        assert "sk-secret" not in error_row["message"]
        assert "stack trace" not in error_row["message"].lower()

        warnings_response = client.get(
            f"/api/evaluation/campaigns/{campaign_id}/stage-warnings"
        )
        assert warnings_response.status_code == 200
        assert warnings_response.json()["rows"] == [
            {
                "run_id": run_id,
                "campaign_id": campaign_id,
                "question_id": "Q-EXPORT",
                "mode": "agentic",
                "stage_name": "agentic_v9_graph_locator",
                "status": "required_but_not_satisfied",
                "failure_reason": "no_eligible_graph_source_evidence",
                "created_at": warnings_response.json()["rows"][0]["created_at"],
            }
        ]

        default_export = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export", json={}
        )
        assert default_export.status_code == 200
        export_body = default_export.json()
        assert export_body["schema_version"] == "2.0"
        assert set(export_body) == {
            "schema_version",
            "export_metadata",
            "campaign",
            "sections",
            "runs",
        }
        assert export_body["export_metadata"]["options"]["include_run_observability"] is False
        assert export_body["export_metadata"]["redaction"] == {
            "provider_errors": "excluded",
            "stack_traces": "excluded",
            "credentials": "redacted",
        }
        assert export_body["runs"][0]["observability"] == {
            "included": False,
            "availability": {
                "status": "not_applicable",
                "reasons": ["not_requested"],
            },
            "data": None,
        }

        full_export = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export",
            json={"include_run_observability": True, "include_full_prompts": True},
        )
        assert full_export.status_code == 200
        full_llm_call = next(
            item
            for item in full_export.json()["runs"][0]["observability"]["data"]["llm_calls"]
            if item["llm_call_id"] == f"{run_id}-llm"
        )
        assert '"note":"safe"' in full_llm_call["full_prompt"]
        assert "hunter2" not in full_export.text
        assert "quoted-api-key" not in full_export.text
        assert "quoted-token" not in full_export.text
        assert "quoted-credential" not in full_export.text
        for sentinel in (
            "export-access-token-sentinel",
            "export-client-secret-sentinel",
            "export-refresh-token-sentinel",
            "export-id-token-sentinel",
            "export-private-key-sentinel",
            "export-list-access-token-sentinel",
            "export-list-client-secret-sentinel",
        ):
            assert sentinel not in full_export.text
        unavailable = next(
            item
            for item in full_export.json()["runs"][0]["observability"]["data"]["llm_calls"]
            if item["llm_call_id"] == f"{run_id}-not-captured"
        )
        assert unavailable["full_prompt_capture_status"] == (
            "not_captured_at_execution"
        )
        assert unavailable["full_prompt"] is None
        assert "MUST NEVER BE EXPORTED" not in full_export.text

        redacted_export = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export",
            json={"include_answers": False, "include_retrieved_excerpts": False},
        )
        assert redacted_export.status_code == 200
        redacted_text = redacted_export.text
        assert "Grounded answer" not in redacted_text
        assert "A safe answer" not in redacted_text
        assert "SECRET RETRIEVED CONTEXT" not in redacted_text
        assert redacted_export.json()["runs"][0]["result"]["answer"] is None
        assert redacted_export.json()["runs"][0]["result"]["contexts"] is None


def test_export_includes_condition_comparison_and_excludes_unattributed_ragas() -> (
    None
):
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="Grounded answer",
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 16},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export_condition")

    with _build_client("user-a", upload_root, db_path, engine) as client:
        created_case = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-EXPORT",
                "question": "What failed?",
                "ground_truth": "A safe answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        created = client.post(
            "/api/evaluation/campaigns", json=_condition_campaign_payload()
        )
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client, campaign_id)

        result_rows = client.get(
            f"/api/evaluation/campaigns/{campaign_id}/results"
        ).json()["results"]
        result_ids_by_condition = {
            row["condition_id"]: row["id"] for row in result_rows
        }
        asyncio.run(
            _seed_condition_ragas_scores(
                campaign_id=campaign_id,
                user_id="user-a",
                result_ids_by_condition=result_ids_by_condition,
            )
        )

        export_response = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export", json={}
        )
        assert export_response.status_code == 200
        payload = export_response.json()
        runs_by_id = {run["result"]["run_id"]: run for run in payload["runs"]}

        assert (
            payload["sections"]["ablation"]["data"]["summaries"]
            ["condition_comparison"]["paired"]["completed_pair_count"]
            == 1
        )
        assert runs_by_id[result_ids_by_condition["v9-baseline"]]["ragas_metrics"] == {}
        assert runs_by_id[result_ids_by_condition["v9-guided"]]["ragas_metrics"] == {}
        assert payload["runs"][0]["observability"]["data"] is None


def test_user_cannot_export_another_users_campaign() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="Grounded answer",
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 16},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export")

    with _build_client("user-a", upload_root, db_path, engine) as client_a:
        created_case = client_a.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-EXPORT",
                "question": "What failed?",
                "ground_truth": "A safe answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        created = client_a.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client_a, campaign_id)

    with _build_client("user-b", upload_root, db_path, engine) as client_b:
        denied = client_b.post(
            f"/api/evaluation/campaigns/{campaign_id}/export", json={}
        )
        assert denied.status_code == 404
