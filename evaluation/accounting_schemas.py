"""Pydantic models for evaluation usage accounting persistence."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ScopeType = Literal["execution_run", "ragas_batch"]
ScopeStatus = Literal["running", "completed", "failed", "interrupted", "cancelled"]
UsageStatus = Literal["measured", "missing", "failed"]
ReconciliationStatus = Literal["balanced", "partial", "unavailable"]
PricingStatus = Literal["priced", "unknown_model", "missing_price", "unavailable_usage"]


class AccountingScopeTarget(BaseModel):
    campaign_result_id: str | None = None
    job_id: str
    work_item_id: str
    attempt_id: str
    metric_name: str | None = None
    is_official: bool = False


class AccountingScopeStart(BaseModel):
    scope_id: str
    campaign_id: str
    scope_type: ScopeType
    scope_key: str
    run_id: str | None = None
    metric_name: str | None = None
    accounting_schema_version: str = "2"
    targets: list[AccountingScopeTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope_shape(self) -> "AccountingScopeStart":
        if self.scope_type == "execution_run" and (not self.run_id or len(self.targets) != 1):
            raise ValueError("execution_run requires run_id and exactly one target")
        if self.scope_type == "ragas_batch" and not self.metric_name:
            raise ValueError("ragas_batch requires metric_name")
        return self


class AccountingScope(BaseModel):
    scope_id: str
    campaign_id: str
    scope_type: ScopeType
    scope_key: str
    run_id: str | None = None
    metric_name: str | None = None
    accounting_schema_version: str
    status: ScopeStatus
    observed_call_count: int = 0
    measured_call_count: int = 0
    missing_usage_call_count: int = 0
    unclassified_phase_call_count: int = 0
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    targets: list[AccountingScopeTarget] = Field(default_factory=list)


class UsageEventCreate(BaseModel):
    usage_event_id: str
    scope_id: str
    campaign_id: str
    scope_type: ScopeType
    scope_key: str
    run_id: str | None = None
    provider_run_id: str | None = None
    phase: str
    purpose: str
    metric_name: str | None = None
    provider: str | None = None
    model_name: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_text_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    other_tokens: int = Field(default=0, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    usage_status: UsageStatus
    reconciliation_status: ReconciliationStatus
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_twd: float | None = Field(default=None, ge=0)
    pricing_status: PricingStatus
    price_snapshot_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    status: Literal["success", "failed"] = "success"
    error: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class UsageEvent(UsageEventCreate):
    pass
