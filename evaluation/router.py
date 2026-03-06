"""Evaluation API router (Phase 1 foundation)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
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

router = APIRouter()


class TestCaseCreateRequest(BaseModel):
    """Create one test case."""

    id: str | None = Field(default=None, min_length=1, max_length=128)
    question: str = Field(..., min_length=1, max_length=5000)
    ground_truth: str = Field(..., min_length=1)
    category: str | None = None
    difficulty: str | None = None
    source_docs: list[str] = Field(default_factory=list)
    requires_multi_doc_reasoning: bool = False
    test_objective: str | None = None


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
