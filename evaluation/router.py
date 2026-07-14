"""Evaluation API router (Phase 1 foundation)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from evaluation.analytics import EvaluationAnalyticsService
from evaluation.campaign_engine import get_campaign_engine
from evaluation.campaign_schemas import (
    AblationResponse,
    CampaignAnalyticsDashboardResponse,
    CampaignErrorsResponse,
    CampaignCreateRequest,
    CampaignCreateResponse,
    CampaignEvaluateRequest,
    CampaignLifecycleStatus,
    CampaignMetricsResponse,
    CampaignOverviewResponse,
    CampaignProgressEvent,
    CampaignResultsResponse,
    CampaignStatus,
    CostLatencyResponse,
    EvaluationRunListResponse,
    ExportCampaignRequest,
    ExportCampaignResponse,
    HumanEvalQueueResponse,
    HumanRatingRequest,
    HumanRatingResponse,
    HumanVsAutoResponse,
    ModeComparisonResponse,
    QuestionComparisonResponse,
    RepeatStabilitySummary,
    RouterAnalysisResponse,
    RunClaimsResponse,
    RunContextResponse,
    RunDiffResponse,
    RunLlmCallsResponse,
    RunMetricsResponse,
    RunRetrievalResponse,
    RunToolsResponse,
    RunTraceResponse,
)
from evaluation.model_capabilities import normalize_model_config_for_storage
from evaluation.model_discovery import list_available_models
from evaluation.db import CampaignResultRepository
from evaluation.job_schemas import (
    EvaluationAttempt,
    EvaluationJob,
    EvaluationJobItemSummary,
    EvaluationRerunRequest,
)
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.schemas import (
    AvailableModel,
    DeleteResult,
    ImportResult,
    ModelConfig,
    TestCase,
    ThinkingLevel,
    normalize_test_case_difficulty,
)
from evaluation.storage import (
    create_model_config,
    create_test_case,
    delete_model_config,
    delete_test_case,
    import_test_cases,
    list_model_configs,
    list_test_cases,
    update_model_config,
    update_test_case,
)
from evaluation.trace_schemas import (
    AgentTraceDetail,
    AgentTraceSummary,
    EvaluationRunObservabilityDetail,
)

router = APIRouter()
_TERMINAL_CAMPAIGN_STATUSES = {
    CampaignLifecycleStatus.COMPLETED,
    CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
    CampaignLifecycleStatus.FAILED,
    CampaignLifecycleStatus.CANCELLED,
}
_TEST_CASE_RESEARCH_METADATA_FIELDS = (
    "question_version",
    "required_modalities",
    "atomic_facts",
    "expected_evidence",
)


class TestCaseCreateRequest(BaseModel):
    """Create one test case."""

    id: str | None = Field(default=None, min_length=1, max_length=128)
    question: str = Field(..., min_length=1, max_length=5000)
    ground_truth: str = Field(..., min_length=1)
    ground_truth_short: str | None = Field(default=None, min_length=1)
    category: str | None = None
    difficulty: str | None = None
    source_docs: list[str] = Field(default_factory=list)
    requires_multi_doc_reasoning: bool = False
    test_objective: str | None = None
    key_points: list[str] = Field(default_factory=list)
    ragas_focus: list[str] = Field(default_factory=list)
    question_version: str | None = None
    required_modalities: list[str] = Field(default_factory=list)
    atomic_facts: list[dict[str, Any]] = Field(default_factory=list)
    expected_evidence: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: Any) -> Any:
        return normalize_test_case_difficulty(value)


class TestCaseImportRequest(BaseModel):
    """Import test cases using golden dataset shape."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    questions: list[TestCaseCreateRequest] = Field(default_factory=list)


class ModelConfigCreateRequest(BaseModel):
    """Create model config preset."""

    id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1, max_length=200)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=100)
    max_input_tokens: int = Field(8192, ge=1)
    max_output_tokens: int = Field(8192, ge=1)
    thinking_mode: bool = False
    thinking_budget: int | None = Field(default=8192, ge=-1, le=32768)
    thinking_level: ThinkingLevel | None = None
    thinking_include_thoughts: bool = False


def _to_sse_event(event_name: str, payload: BaseModel | dict[str, Any]) -> dict[str, str]:
    body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return {
        "event": event_name,
        "data": json.dumps(body, ensure_ascii=False),
    }


_ANALYTICS_SERVICE = EvaluationAnalyticsService()


def get_evaluation_analytics_service() -> EvaluationAnalyticsService:
    """Factory for evaluation analytics service."""
    return _ANALYTICS_SERVICE


async def _preserve_omitted_test_case_metadata(
    *,
    user_id: str,
    test_case_id: str,
    payload: TestCaseCreateRequest,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    omitted_metadata_fields = [
        field_name
        for field_name in _TEST_CASE_RESEARCH_METADATA_FIELDS
        if field_name not in payload.model_fields_set
    ]
    if not omitted_metadata_fields:
        return candidate

    existing_cases = await list_test_cases(user_id)
    existing = next(
        (item for item in existing_cases if isinstance(item, dict) and item.get("id") == test_case_id),
        None,
    )
    if not existing:
        return candidate

    preserved = dict(candidate)
    for field_name in omitted_metadata_fields:
        if field_name in existing:
            preserved[field_name] = existing[field_name]
    return preserved


@router.get("/test-cases", response_model=list[TestCase])
async def get_test_cases(
    user_id: str = Depends(get_current_user_id),
) -> list[TestCase]:
    """List test cases for current user."""
    items = await list_test_cases(user_id)
    return [TestCase.model_validate(item) for item in items]


@router.post("/test-cases", response_model=TestCase | ImportResult)
async def create_or_import_test_case(
    payload: TestCaseCreateRequest | TestCaseImportRequest,
    user_id: str = Depends(get_current_user_id),
) -> TestCase | ImportResult:
    """
    Create one test case or import many test cases.

    - Single create: payload is `TestCaseCreateRequest`
    - Batch import: payload is `TestCaseImportRequest`
    """
    if isinstance(payload, TestCaseImportRequest):
        imported, total = await import_test_cases(
            user_id=user_id,
            questions_to_import=[item.model_dump(exclude_none=True) for item in payload.questions],
            metadata=payload.metadata,
        )
        return ImportResult(imported=imported, total=total)

    created = await create_test_case(
        user_id=user_id,
        test_case=payload.model_dump(exclude_none=True),
    )
    return TestCase.model_validate(created)


@router.put("/test-cases/{test_case_id}", response_model=TestCase)
async def put_test_case(
    test_case_id: str,
    payload: TestCaseCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> TestCase:
    """Replace one test case."""
    if payload.id and payload.id != test_case_id:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Payload id must match path id",
            status_code=400,
        )

    candidate = await _preserve_omitted_test_case_metadata(
        user_id=user_id,
        test_case_id=test_case_id,
        payload=payload,
        candidate=payload.model_dump(exclude_none=True),
    )

    updated = await update_test_case(
        user_id=user_id,
        test_case_id=test_case_id,
        test_case=candidate,
    )
    return TestCase.model_validate(updated)


@router.delete("/test-cases/{test_case_id}", response_model=DeleteResult)
async def remove_test_case(
    test_case_id: str,
    user_id: str = Depends(get_current_user_id),
) -> DeleteResult:
    """Delete one test case."""
    total = await delete_test_case(user_id=user_id, test_case_id=test_case_id)
    return DeleteResult(deleted_id=test_case_id, total=total)


@router.get("/models", response_model=list[AvailableModel])
async def get_available_models(
    force_refresh: bool = Query(default=False, description="Bypass model cache"),
    _user_id: str = Depends(get_current_user_id),
) -> list[AvailableModel]:
    """List models available to current Gemini API key."""
    return await list_available_models(force_refresh=force_refresh)


@router.get("/model-configs", response_model=list[ModelConfig])
async def get_model_configs(
    user_id: str = Depends(get_current_user_id),
) -> list[ModelConfig]:
    """List saved model config presets for current user."""
    items = await list_model_configs(user_id)
    return [ModelConfig.model_validate(item) for item in items]


@router.post("/model-configs", response_model=ModelConfig)
async def post_model_config(
    payload: ModelConfigCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> ModelConfig:
    """Create one model config preset."""
    created = await create_model_config(
        user_id=user_id,
        model_config=normalize_model_config_for_storage(payload.model_dump(exclude_none=False)),
    )
    return ModelConfig.model_validate(created)


@router.put("/model-configs/{config_id}", response_model=ModelConfig)
async def put_model_config(
    config_id: str,
    payload: ModelConfigCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> ModelConfig:
    """Replace one model config preset."""
    if payload.id and payload.id != config_id:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Payload id must match path id",
            status_code=400,
        )

    updated = await update_model_config(
        user_id=user_id,
        config_id=config_id,
        model_config=normalize_model_config_for_storage(payload.model_dump(exclude_none=False)),
    )
    return ModelConfig.model_validate(updated)


@router.delete("/model-configs/{config_id}", response_model=DeleteResult)
async def remove_model_config(
    config_id: str,
    user_id: str = Depends(get_current_user_id),
) -> DeleteResult:
    """Delete one model config preset."""
    total = await delete_model_config(user_id=user_id, config_id=config_id)
    return DeleteResult(deleted_id=config_id, total=total)


@router.post("/campaigns", response_model=CampaignCreateResponse)
async def create_campaign(
    payload: CampaignCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> CampaignCreateResponse:
    """Create and start an evaluation campaign."""
    engine = get_campaign_engine()
    return await engine.create_and_start(
        user_id=user_id,
        name=payload.name,
        config=payload.to_config(),
    )


@router.get("/campaigns", response_model=list[CampaignStatus])
async def get_campaigns(
    user_id: str = Depends(get_current_user_id),
) -> list[CampaignStatus]:
    """List evaluation campaigns for current user."""
    engine = get_campaign_engine()
    return await engine.list_campaigns(user_id=user_id)


@router.get("/campaigns/{campaign_id}/results", response_model=CampaignResultsResponse)
async def get_campaign_results(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> CampaignResultsResponse:
    """Fetch persisted campaign results."""
    engine = get_campaign_engine()
    return await engine.get_results(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/overview", response_model=CampaignOverviewResponse)
async def get_campaign_research_overview(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> CampaignOverviewResponse:
    """Fetch research analytics overview for one campaign."""
    return await analytics.campaign_overview(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/runs", response_model=EvaluationRunListResponse)
async def get_campaign_research_runs(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> EvaluationRunListResponse:
    """List execution samples for one campaign."""
    return await analytics.list_campaign_runs(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/analytics-dashboard", response_model=CampaignAnalyticsDashboardResponse)
async def get_campaign_analytics_dashboard(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> CampaignAnalyticsDashboardResponse:
    """Fetch the Evaluation Center dashboard analytics in one bundled request."""
    return await analytics.analytics_dashboard(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/mode-comparison", response_model=ModeComparisonResponse)
async def get_campaign_mode_comparison(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> ModeComparisonResponse:
    """Fetch mode-level research comparison aggregates."""
    return await analytics.mode_comparison(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/question-comparison", response_model=QuestionComparisonResponse)
async def get_campaign_question_comparison(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> QuestionComparisonResponse:
    """Fetch question-level research comparison aggregates."""
    return await analytics.question_comparison(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/cost-latency", response_model=CostLatencyResponse)
async def get_campaign_cost_latency(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> CostLatencyResponse:
    """Fetch cost and latency aggregates for one campaign."""
    return await analytics.cost_latency(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/router-analysis", response_model=RouterAnalysisResponse)
async def get_campaign_router_analysis(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RouterAnalysisResponse:
    """Fetch retrospective router decision analytics for one campaign."""
    return await analytics.router_analysis(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/ablation", response_model=AblationResponse)
async def get_campaign_ablation(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> AblationResponse:
    """Fetch ablation grouping aggregates for one campaign."""
    return await analytics.ablation(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/human-vs-auto", response_model=HumanVsAutoResponse)
async def get_campaign_human_vs_auto(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> HumanVsAutoResponse:
    """Fetch human-rating versus automated metric aggregates."""
    return await analytics.human_vs_auto(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/human-eval-queue", response_model=HumanEvalQueueResponse)
async def get_campaign_human_eval_queue(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> HumanEvalQueueResponse:
    """List runs in one campaign for human review."""
    return await analytics.human_eval_queue(user_id=user_id, campaign_id=campaign_id)


@router.post("/runs/{run_id}/human-ratings", response_model=HumanRatingResponse)
async def post_run_human_rating(
    run_id: str,
    payload: HumanRatingRequest,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> HumanRatingResponse:
    """Store one human rubric score row for a run owned by the current user."""
    return await analytics.create_human_rating(user_id=user_id, run_id=run_id, request=payload)


@router.get("/campaigns/{campaign_id}/repeat-stability", response_model=RepeatStabilitySummary)
async def get_campaign_repeat_stability(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RepeatStabilitySummary:
    """Fetch repeat-run stability aggregates for one campaign."""
    return await analytics.repeat_stability(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/errors", response_model=CampaignErrorsResponse)
async def get_campaign_errors(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> CampaignErrorsResponse:
    """Fetch sanitized error rows for one campaign."""
    return await analytics.campaign_errors(user_id=user_id, campaign_id=campaign_id)


@router.post("/campaigns/{campaign_id}/export", response_model=ExportCampaignResponse)
async def post_campaign_export(
    campaign_id: str,
    payload: ExportCampaignRequest,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> ExportCampaignResponse:
    """Export one campaign with explicit redaction controls."""
    return await analytics.export_campaign(user_id=user_id, campaign_id=campaign_id, request=payload)


@router.get("/runs/{run_id}/trace", response_model=RunTraceResponse)
async def get_evaluation_run_trace(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunTraceResponse:
    """Fetch stage trace and routing observability for one run."""
    return await analytics.run_trace(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/retrieval", response_model=RunRetrievalResponse)
async def get_evaluation_run_retrieval(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunRetrievalResponse:
    """Fetch retrieval events and chunks for one run."""
    return await analytics.run_retrieval(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/context", response_model=RunContextResponse)
async def get_evaluation_run_context(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunContextResponse:
    """Fetch context packing details for one run."""
    return await analytics.run_context(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/llm-calls", response_model=RunLlmCallsResponse)
async def get_evaluation_run_llm_calls(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunLlmCallsResponse:
    """Fetch token, cost, and model call details for one run."""
    return await analytics.run_llm_calls(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/tools", response_model=RunToolsResponse)
async def get_evaluation_run_tools(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunToolsResponse:
    """Fetch all tool calls for one run."""
    return await analytics.run_tools(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/visual", response_model=RunToolsResponse)
async def get_evaluation_run_visual_tools(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunToolsResponse:
    """Fetch visual tool calls for one run."""
    return await analytics.run_tools(user_id=user_id, run_id=run_id, tool_type="visual")


@router.get("/runs/{run_id}/graph", response_model=RunToolsResponse)
async def get_evaluation_run_graph_tools(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunToolsResponse:
    """Fetch graph tool calls for one run."""
    return await analytics.run_tools(user_id=user_id, run_id=run_id, tool_type="graph")


@router.get("/runs/{run_id}/claims", response_model=RunClaimsResponse)
async def get_evaluation_run_claims(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunClaimsResponse:
    """Fetch claim-evidence verification rows for one run."""
    return await analytics.run_claims(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/metrics", response_model=RunMetricsResponse)
async def get_evaluation_run_metrics(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunMetricsResponse:
    """Fetch derived metrics for one run."""
    return await analytics.run_metrics(user_id=user_id, run_id=run_id)


@router.get("/runs/{run_id}/diff", response_model=RunDiffResponse)
async def get_evaluation_run_diff(
    run_id: str,
    baseline_run_id: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user_id),
    analytics: EvaluationAnalyticsService = Depends(get_evaluation_analytics_service),
) -> RunDiffResponse:
    """Compare one run against a baseline run owned by the same user."""
    return await analytics.run_diff(
        user_id=user_id,
        run_id=run_id,
        baseline_run_id=baseline_run_id,
    )


@router.get("/campaigns/{campaign_id}/traces", response_model=list[AgentTraceSummary])
async def get_campaign_traces(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[AgentTraceSummary]:
    """List available persisted agent traces for a campaign."""
    engine = get_campaign_engine()
    return await engine.list_traces(user_id=user_id, campaign_id=campaign_id)


@router.get(
    "/campaigns/{campaign_id}/results/{campaign_result_id}/trace",
    response_model=AgentTraceDetail,
)
async def get_campaign_result_trace(
    campaign_id: str,
    campaign_result_id: str,
    user_id: str = Depends(get_current_user_id),
) -> AgentTraceDetail:
    """Fetch one persisted agent trace by campaign result id."""
    engine = get_campaign_engine()
    return await engine.get_trace(
        user_id=user_id,
        campaign_id=campaign_id,
        campaign_result_id=campaign_result_id,
    )


@router.get(
    "/campaigns/{campaign_id}/runs/{run_id}/observability",
    response_model=EvaluationRunObservabilityDetail,
)
async def get_campaign_run_observability(
    campaign_id: str,
    run_id: str,
    user_id: str = Depends(get_current_user_id),
) -> EvaluationRunObservabilityDetail:
    """Fetch normalized observability details for one campaign run."""
    engine = get_campaign_engine()
    await engine.get_campaign(user_id=user_id, campaign_id=campaign_id)
    await CampaignResultRepository().get(user_id=user_id, campaign_id=campaign_id, result_id=run_id)

    repository = EvaluationObservabilityRepository()
    trace_events = [
        item
        for item in await repository.list_trace_events_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    llm_calls = [
        item
        for item in await repository.list_llm_calls_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    retrieval_events = [
        item
        for item in await repository.list_retrieval_events_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    retrieval_chunks = [
        item
        for item in await repository.list_retrieval_chunks_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    context_packs = [
        item
        for item in await repository.list_context_packs_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    tool_calls = [
        item
        for item in await repository.list_tool_calls_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    routing_decisions = [
        item
        for item in await repository.list_routing_decisions_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    claims = [
        item
        for item in await repository.list_claims_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    human_ratings = [
        item
        for item in await repository.list_human_ratings_for_run(run_id)
        if item.campaign_id == campaign_id
    ]
    return EvaluationRunObservabilityDetail(
        run_id=run_id,
        campaign_id=campaign_id,
        trace_events=trace_events,
        llm_calls=llm_calls,
        retrieval_events=retrieval_events,
        retrieval_chunks=retrieval_chunks,
        context_packs=context_packs,
        tool_calls=tool_calls,
        routing_decisions=routing_decisions,
        claims=claims,
        human_ratings=human_ratings,
    )


@router.get("/campaigns/{campaign_id}/metrics", response_model=CampaignMetricsResponse)
async def get_campaign_metrics(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> CampaignMetricsResponse:
    """Fetch aggregated RAGAS metrics for a campaign."""
    engine = get_campaign_engine()
    return await engine.get_metrics(user_id=user_id, campaign_id=campaign_id)


@router.post("/campaigns/{campaign_id}/evaluate", response_model=CampaignStatus)
async def evaluate_campaign(
    campaign_id: str,
    payload: CampaignEvaluateRequest | None = None,
    user_id: str = Depends(get_current_user_id),
) -> CampaignStatus:
    """Trigger a manual RAGAS rerun for an existing campaign."""
    engine = get_campaign_engine()
    return await engine.evaluate_campaign(
        user_id=user_id,
        campaign_id=campaign_id,
        question_ids=(payload.question_ids if payload else None),
    )


@router.post("/campaigns/{campaign_id}/reruns", response_model=EvaluationJob)
async def create_campaign_rerun(
    campaign_id: str,
    payload: EvaluationRerunRequest,
    user_id: str = Depends(get_current_user_id),
) -> EvaluationJob:
    """Create a durable execution and/or RAGAS rerun job."""
    engine = get_campaign_engine()
    return await engine.create_rerun(
        user_id=user_id,
        campaign_id=campaign_id,
        request=payload,
    )


@router.get("/campaigns/{campaign_id}/jobs", response_model=list[EvaluationJob])
async def list_campaign_jobs(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[EvaluationJob]:
    """List durable jobs for one owned campaign."""
    engine = get_campaign_engine()
    return await engine.list_jobs(user_id=user_id, campaign_id=campaign_id)


@router.get("/jobs/{job_id}", response_model=EvaluationJob)
async def get_evaluation_job(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
) -> EvaluationJob:
    """Fetch one owned durable evaluation job."""
    engine = get_campaign_engine()
    return await engine.get_job(user_id=user_id, job_id=job_id)


@router.get("/jobs/{job_id}/items", response_model=list[EvaluationJobItemSummary])
async def list_evaluation_job_items(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[EvaluationJobItemSummary]:
    """List one owned durable job's work items and latest safe attempts."""
    engine = get_campaign_engine()
    return await engine.list_job_items(user_id=user_id, job_id=job_id)


@router.post("/jobs/{job_id}/cancel", response_model=EvaluationJob)
async def cancel_evaluation_job(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
) -> EvaluationJob:
    """Cancel active work for one owned durable job."""
    engine = get_campaign_engine()
    return await engine.cancel_job(user_id=user_id, job_id=job_id)


@router.get("/work-items/{work_item_id}/attempts", response_model=list[EvaluationAttempt])
async def get_work_item_attempts(
    work_item_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[EvaluationAttempt]:
    """List append-only attempts for one owned work item."""
    engine = get_campaign_engine()
    return await engine.list_attempts(user_id=user_id, work_item_id=work_item_id)


@router.post("/campaigns/{campaign_id}/cancel", response_model=CampaignStatus)
async def cancel_campaign(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> CampaignStatus:
    """Request cancellation for a running campaign."""
    engine = get_campaign_engine()
    return await engine.cancel_campaign(user_id=user_id, campaign_id=campaign_id)


@router.get("/campaigns/{campaign_id}/stream")
async def stream_campaign(
    campaign_id: str,
    user_id: str = Depends(get_current_user_id),
) -> EventSourceResponse:
    """Stream campaign progress with authenticated SSE."""
    engine = get_campaign_engine()
    await engine.ensure_campaign_task(user_id=user_id, campaign_id=campaign_id)

    async def event_generator():
        snapshot = await engine.get_campaign(user_id=user_id, campaign_id=campaign_id)
        yield _to_sse_event("campaign_snapshot", snapshot)
        last_progress = (
            snapshot.status,
            snapshot.phase,
            snapshot.completed_units,
            snapshot.evaluation_completed_units,
            snapshot.evaluation_total_units,
            snapshot.current_question_id,
            snapshot.current_mode,
            snapshot.error_message,
        )

        if snapshot.status in _TERMINAL_CAMPAIGN_STATUSES:
            yield _to_sse_event(f"campaign_{snapshot.status.value}", snapshot)
            return

        while True:
            await asyncio.sleep(1)
            current = await engine.get_campaign(user_id=user_id, campaign_id=campaign_id)
            progress_state = (
                current.status,
                current.phase,
                current.completed_units,
                current.evaluation_completed_units,
                current.evaluation_total_units,
                current.current_question_id,
                current.current_mode,
                current.error_message,
            )
            if progress_state != last_progress:
                yield _to_sse_event(
                    "campaign_progress",
                    CampaignProgressEvent(
                        campaign_id=current.id,
                        status=current.status,
                        phase=current.phase,
                        completed_units=current.completed_units,
                        total_units=current.total_units,
                        evaluation_completed_units=current.evaluation_completed_units,
                        evaluation_total_units=current.evaluation_total_units,
                        current_question_id=current.current_question_id,
                        current_mode=current.current_mode,
                    ),
                )
                last_progress = progress_state

            if current.status in _TERMINAL_CAMPAIGN_STATUSES:
                yield _to_sse_event(f"campaign_{current.status.value}", current)
                return

    return EventSourceResponse(event_generator())






