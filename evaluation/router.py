"""Evaluation API router (Phase 1 foundation)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from evaluation.campaign_engine import get_campaign_engine
from evaluation.campaign_schemas import (
    CampaignCreateRequest,
    CampaignCreateResponse,
    CampaignEvaluateRequest,
    CampaignLifecycleStatus,
    CampaignMetricsResponse,
    CampaignProgressEvent,
    CampaignResultsResponse,
    CampaignStatus,
)
from evaluation.model_discovery import list_available_models
from evaluation.schemas import (
    AvailableModel,
    DeleteResult,
    ImportResult,
    ModelConfig,
    TestCase,
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
from evaluation.trace_schemas import AgentTraceDetail, AgentTraceSummary

router = APIRouter()
_TERMINAL_CAMPAIGN_STATUSES = {
    CampaignLifecycleStatus.COMPLETED,
    CampaignLifecycleStatus.FAILED,
    CampaignLifecycleStatus.CANCELLED,
}


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
    thinking_budget: int = Field(8192, ge=1024, le=32768)


def _to_sse_event(event_name: str, payload: BaseModel | dict[str, Any]) -> dict[str, str]:
    body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return {
        "event": event_name,
        "data": json.dumps(body, ensure_ascii=False),
    }


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

    updated = await update_test_case(
        user_id=user_id,
        test_case_id=test_case_id,
        test_case=payload.model_dump(exclude_none=True),
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
        model_config=payload.model_dump(exclude_none=True),
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
        model_config=payload.model_dump(exclude_none=True),
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
