"""Typed contracts for durable evaluation jobs and their attempts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class EvaluationJobType(str, Enum):
    INITIAL = "initial"
    RERUN = "rerun"


class EvaluationWorkType(str, Enum):
    DATASET_EXECUTION = "dataset_execution"
    RAGAS_METRIC = "ragas_metric"


class EvaluationJobItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class EvaluationAttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


EvaluationRerunScope = Literal["failed_only", "selected", "all"]
EvaluationRerunStages = Literal["execution", "ragas", "execution_and_ragas"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EvaluationRerunRequest(BaseModel):
    scope: EvaluationRerunScope
    stages: EvaluationRerunStages
    question_ids: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_selection(self) -> EvaluationRerunRequest:
        self.question_ids = list(
            dict.fromkeys(value.strip() for value in self.question_ids if value.strip())
        )
        self.metric_names = list(
            dict.fromkeys(value.strip() for value in self.metric_names if value.strip())
        )
        if self.scope == "selected" and not self.question_ids:
            raise ValueError("selected reruns require question_ids")
        return self


class WorkItemSpec(BaseModel):
    """The immutable input needed to execute one unit of evaluation work."""

    work_item_id: str
    work_type: EvaluationWorkType
    question_id: str
    metric_name: str | None = None
    input_snapshot: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationJob(BaseModel):
    job_id: str
    job_type: EvaluationJobType
    rerun_request: EvaluationRerunRequest | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class EvaluationJobItem(BaseModel):
    job_item_id: str
    job_id: str
    work_item_id: str
    status: EvaluationJobItemStatus = EvaluationJobItemStatus.PENDING
    retry_after: datetime | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class EvaluationAttempt(BaseModel):
    attempt_id: str
    job_id: str
    job_item_id: str
    work_item_id: str
    attempt_number: int
    status: EvaluationAttemptStatus = EvaluationAttemptStatus.RUNNING
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None
    error_type: str | None = None
    safe_error_message: str | None = None


class ClaimedEvaluationWork(BaseModel):
    """A worker claim bound to the exact input snapshot it must execute."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    job_item_id: str
    work_item_id: str
    attempt_id: str
    input_snapshot: dict[str, JsonValue]
