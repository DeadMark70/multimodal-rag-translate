"""Schemas for evaluation campaigns and SSE progress updates."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.schemas import ModelConfig

CampaignMode = Literal["naive", "advanced", "graph", "agentic"]


class CampaignLifecycleStatus(str, Enum):
    """Campaign status persisted in SQLite."""

    PENDING = "pending"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CampaignResultStatus(str, Enum):
    """Per-unit execution result status."""

    COMPLETED = "completed"
    FAILED = "failed"


class CampaignConfig(BaseModel):
    """User-supplied campaign configuration."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    test_case_ids: list[str] = Field(default_factory=list, min_length=1)
    modes: list[CampaignMode] = Field(default_factory=list, min_length=1)
    model_preset: ModelConfig = Field(alias="model_config")
    model_config_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    repeat_count: int = Field(default=1, ge=1, le=10)
    batch_size: int = Field(default=1, ge=1, le=4)
    rpm_limit: int = Field(default=60, ge=1, le=600)

    @model_validator(mode="after")
    def dedupe_modes(self) -> "CampaignConfig":
        ordered_unique_modes: list[CampaignMode] = []
        for mode in self.modes:
            if mode not in ordered_unique_modes:
                ordered_unique_modes.append(mode)
        self.modes = ordered_unique_modes
        return self


class CampaignCreateRequest(BaseModel):
    """Create-and-start campaign payload."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    name: Optional[str] = Field(default=None, max_length=200)
    test_case_ids: list[str] = Field(default_factory=list, min_length=1)
    modes: list[CampaignMode] = Field(default_factory=list, min_length=1)
    model_preset: ModelConfig = Field(alias="model_config")
    model_config_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    repeat_count: int = Field(default=1, ge=1, le=10)
    batch_size: int = Field(default=1, ge=1, le=4)
    rpm_limit: int = Field(default=60, ge=1, le=600)

    def to_config(self) -> CampaignConfig:
        return CampaignConfig(
            test_case_ids=self.test_case_ids,
            modes=self.modes,
            model_preset=self.model_preset,
            model_config_id=self.model_config_id,
            repeat_count=self.repeat_count,
            batch_size=self.batch_size,
            rpm_limit=self.rpm_limit,
        )


class CampaignCreateResponse(BaseModel):
    """Create campaign response."""

    campaign_id: str
    status: CampaignLifecycleStatus


class CampaignStatus(BaseModel):
    """Campaign snapshot returned by REST and SSE APIs."""

    id: str
    name: Optional[str] = None
    status: CampaignLifecycleStatus
    config: CampaignConfig
    completed_units: int = Field(default=0, ge=0)
    total_units: int = Field(default=0, ge=0)
    current_question_id: Optional[str] = None
    current_mode: Optional[CampaignMode] = None
    error_message: Optional[str] = None
    cancel_requested: bool = False
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime


class CampaignResult(BaseModel):
    """Persisted result for a single question-mode-run unit."""

    id: str
    campaign_id: str
    question_id: str
    question: str
    ground_truth: str
    mode: CampaignMode
    run_number: int = Field(ge=1)
    answer: str
    contexts: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    status: CampaignResultStatus
    error_message: Optional[str] = None
    created_at: datetime


class CampaignResultsResponse(BaseModel):
    """Campaign snapshot with all unit results."""

    campaign: CampaignStatus
    results: list[CampaignResult] = Field(default_factory=list)


class CampaignProgressEvent(BaseModel):
    """Incremental SSE progress payload."""

    campaign_id: str
    completed_units: int = Field(default=0, ge=0)
    total_units: int = Field(default=0, ge=0)
    current_question_id: Optional[str] = None
    current_mode: Optional[CampaignMode] = None
    latest_result_id: Optional[str] = None
