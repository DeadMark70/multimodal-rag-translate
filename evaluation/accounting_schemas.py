"""Pydantic models for evaluation usage accounting persistence."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ScopeType = Literal["execution_run", "ragas_batch"]
ScopeStatus = Literal["running", "completed", "failed", "interrupted", "cancelled"]
UsageStatus = Literal["measured", "missing", "failed"]
ReconciliationStatus = Literal["balanced", "partial", "unavailable"]
PricingStatus = Literal["priced", "unknown_model", "missing_price", "unavailable_usage"]
QualityStatus = Literal["complete", "evaluating", "partial", "failed", "not_requested"]
TokenAccountingStatus = Literal["complete", "partial", "incomplete_legacy"]
ResearchPricingStatus = Literal["complete", "partial", "unknown"]
PhaseAttributionStatus = Literal["complete", "partial", "not_available"]


class MetricObservation(BaseModel):
    value: float | None = None
    status: QualityStatus
    valid_samples: int = 0
    missing_samples: int = 0
    failed_samples: int = 0
    evaluator_model: str | None = None
    metric_version: str | None = None


class LatencySummary(BaseModel):
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    sample_count: int = 0
    method: Literal["nearest_rank"] = "nearest_rank"
    low_sample_size: bool = False


class TokenBreakdown(BaseModel):
    input_tokens: int | None = None
    output_text_tokens: int | None = None
    reasoning_tokens: int | None = None
    other_tokens: int | None = None
    total_tokens: int | None = None
    by_phase: dict[str, int] = Field(default_factory=dict)
    accounting_status: TokenAccountingStatus
    phase_attribution_status: PhaseAttributionStatus


class CostSummary(BaseModel):
    benchmark_usd: float | None = None
    operational_usd: float | None = None
    pricing_status: ResearchPricingStatus
    priced_call_count: int = 0
    unpriced_call_count: int = 0


class ModeResearchSummary(BaseModel):
    mode: str
    sample_count: int
    comparable: bool
    not_comparable_reasons: list[str] = Field(default_factory=list)
    quality: dict[str, MetricObservation] = Field(default_factory=dict)
    latency: LatencySummary
    tokens: TokenBreakdown
    execution_cost: CostSummary


class EvaluationOverheadSummary(BaseModel):
    tokens: TokenBreakdown
    cost_usd: float | None = None
    pricing_status: ResearchPricingStatus
    evaluator_models: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    batch_count: int = 0
    retry_count: int | None = Field(default=None, ge=0)


class ResearchWarning(BaseModel):
    code: str
    message: str
    mode: str | None = None


class CampaignResearchSummaryResponse(BaseModel):
    campaign_id: str
    research_schema_version: Literal["2"] = "2"
    completed_run_count: int
    total_run_count: int
    failed_run_count: int
    quality_status: QualityStatus
    token_accounting_status: TokenAccountingStatus
    pricing_status: ResearchPricingStatus
    phase_attribution_status: PhaseAttributionStatus
    sample_count: int
    quality: dict[str, MetricObservation] = Field(default_factory=dict)
    latency: LatencySummary
    tokens: TokenBreakdown
    execution_cost: CostSummary
    modes: list[ModeResearchSummary]
    evaluation_overhead: EvaluationOverheadSummary
    warnings: list[ResearchWarning] = Field(default_factory=list)


class AccountingScopeTarget(BaseModel):
    campaign_result_id: str | None = None
    job_id: str
    work_item_id: str
    attempt_id: str
    mode: str | None = None
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
        if self.scope_type == "execution_run" and (
            not self.run_id or len(self.targets) != 1
        ):
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
    retry_count: int | None = Field(default=None, ge=0)
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
