"""Schemas for evaluation campaigns and SSE progress updates."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.schemas import ModelConfig

CampaignMode = Literal["naive", "advanced", "graph", "agentic", "router"]
CampaignEvaluationPhase = Literal["execution", "evaluation"]


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
    ragas_batch_size: int = Field(default=8, ge=1, le=8)
    ragas_parallel_batches: int = Field(default=8, ge=1, le=8)
    ragas_rpm_limit: int = Field(default=1000, ge=1, le=1000)
    actual_router_execution_enabled: bool = False

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
    ragas_batch_size: int = Field(default=8, ge=1, le=8)
    ragas_parallel_batches: int = Field(default=8, ge=1, le=8)
    ragas_rpm_limit: int = Field(default=1000, ge=1, le=1000)
    actual_router_execution_enabled: bool = False

    def to_config(self) -> CampaignConfig:
        return CampaignConfig(
            test_case_ids=self.test_case_ids,
            modes=self.modes,
            model_preset=self.model_preset,
            model_config_id=self.model_config_id,
            repeat_count=self.repeat_count,
            batch_size=self.batch_size,
            rpm_limit=self.rpm_limit,
            ragas_batch_size=self.ragas_batch_size,
            ragas_parallel_batches=self.ragas_parallel_batches,
            ragas_rpm_limit=self.ragas_rpm_limit,
            actual_router_execution_enabled=self.actual_router_execution_enabled,
        )


class CampaignCreateResponse(BaseModel):
    """Create campaign response."""

    campaign_id: str
    status: CampaignLifecycleStatus


class CampaignEvaluateRequest(BaseModel):
    """Manual evaluation rerun payload."""

    question_ids: Optional[list[str]] = None

    @model_validator(mode="after")
    def normalize_question_ids(self) -> "CampaignEvaluateRequest":
        if self.question_ids is None:
            return self
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in self.question_ids:
            question_id = str(raw or "").strip()
            if not question_id or question_id in seen:
                continue
            seen.add(question_id)
            normalized.append(question_id)
        self.question_ids = normalized
        return self


class CampaignStatus(BaseModel):
    """Campaign snapshot returned by REST and SSE APIs."""

    id: str
    name: Optional[str] = None
    status: CampaignLifecycleStatus
    phase: CampaignEvaluationPhase = "execution"
    config: CampaignConfig
    completed_units: int = Field(default=0, ge=0)
    total_units: int = Field(default=0, ge=0)
    evaluation_completed_units: int = Field(default=0, ge=0)
    evaluation_total_units: int = Field(default=0, ge=0)
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
    ground_truth_short: Optional[str] = None
    key_points: list[str] = Field(default_factory=list)
    ragas_focus: list[str] = Field(default_factory=list)
    mode: CampaignMode
    execution_profile: Optional[str] = None
    context_policy_version: Optional[str] = None
    run_number: int = Field(ge=1)
    answer: str
    contexts: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0, ge=0)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    question_version: Optional[str] = None
    request_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_latency_ms: Optional[float] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    question_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    system_version_snapshot: dict[str, Any] = Field(default_factory=dict)
    derived_metrics: dict[str, Any] = Field(default_factory=dict)
    final_answer_hash: Optional[str] = None
    status: CampaignResultStatus
    error_message: Optional[str] = None
    has_trace: bool = False
    created_at: datetime


class CampaignResultsResponse(BaseModel):
    """Campaign snapshot with all unit results."""

    campaign: CampaignStatus
    results: list[CampaignResult] = Field(default_factory=list)


class CampaignOverviewResponse(BaseModel):
    """Research analytics overview for one campaign."""

    campaign_id: str
    analysis_unit: Literal["execution", "question", "category"] = "execution"
    sample_count: int = Field(default=0, ge=0)
    independent_question_count: int = Field(default=0, ge=0)
    repeat_count: int = Field(default=0, ge=0)
    sample_note: str = ""
    mode_counts: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: Optional[float] = Field(default=None, ge=0)
    total_cost_twd: Optional[float] = Field(default=None, ge=0)
    cost_status: Literal["complete", "partial", "unknown"] = "unknown"
    priced_call_count: int = Field(default=0, ge=0)
    unpriced_call_count: int = Field(default=0, ge=0)
    avg_latency_ms: Optional[float] = Field(default=None, ge=0)


class AnalyticsAggregateResponse(BaseModel):
    """Base shape for aggregate research analytics responses."""

    campaign_id: str
    analysis_unit: Literal["execution", "question", "category"]
    sample_count: int = Field(default=0, ge=0)
    independent_question_count: int = Field(default=0, ge=0)
    repeat_count: int = Field(default=0, ge=0)
    sample_note: str = ""
    warnings: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    summaries: dict[str, Any] = Field(default_factory=dict)


class ModeComparisonResponse(AnalyticsAggregateResponse):
    """Mode-level comparison analytics."""


class QuestionComparisonResponse(AnalyticsAggregateResponse):
    """Question-level comparison analytics."""


class CostLatencyResponse(AnalyticsAggregateResponse):
    """Cost and latency analytics."""


class RouterAnalysisResponse(AnalyticsAggregateResponse):
    """Router retrospective or actual-router analytics."""

    analysis_type: Literal["retrospective", "actual"] = "retrospective"


class AblationResponse(AnalyticsAggregateResponse):
    """Ablation grouping analytics."""


class HumanVsAutoResponse(AnalyticsAggregateResponse):
    """Human rating versus automated metric calibration analytics."""


class RepeatStabilitySummary(AnalyticsAggregateResponse):
    """Repeat-run stability analytics."""


class EvaluationRunListItem(BaseModel):
    """List row for one persisted evaluation execution."""

    run_id: str
    campaign_id: str
    question_id: str
    question: str
    mode: CampaignMode
    run_number: int = Field(ge=1)
    status: CampaignResultStatus
    total_tokens: int = Field(default=0, ge=0)
    total_latency_ms: Optional[float] = Field(default=None, ge=0)
    created_at: datetime


class EvaluationRunListResponse(BaseModel):
    """Run list for one campaign."""

    campaign_id: str
    runs: list[EvaluationRunListItem] = Field(default_factory=list)


class RunTraceResponse(BaseModel):
    run_id: str
    campaign_id: str
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    routing_decisions: list[dict[str, Any]] = Field(default_factory=list)


class RunRetrievalResponse(BaseModel):
    run_id: str
    campaign_id: str
    retrieval_events: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_chunks: list[dict[str, Any]] = Field(default_factory=list)


class RunContextResponse(BaseModel):
    run_id: str
    campaign_id: str
    context_packs: list[dict[str, Any]] = Field(default_factory=list)


class RunLlmCallsResponse(BaseModel):
    run_id: str
    campaign_id: str
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)


class RunToolsResponse(BaseModel):
    run_id: str
    campaign_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class RunClaimsResponse(BaseModel):
    run_id: str
    campaign_id: str
    claims: list[dict[str, Any]] = Field(default_factory=list)


class RunMetricsResponse(BaseModel):
    run_id: str
    campaign_id: str
    derived_metrics: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    total_latency_ms: Optional[float] = Field(default=None, ge=0)


class RunDiffResponse(BaseModel):
    run_id: str
    baseline_run_id: str
    campaign_id: str
    baseline_campaign_id: str
    token_delta: int
    latency_delta_ms: float
    comparable: bool = True
    comparison_scope: Literal["same_run", "same_campaign_question", "cross_campaign"] = "same_campaign_question"
    answer_changed: bool
    answer_change_status: Literal["changed", "unchanged", "unknown"] = "unknown"
    derived_metric_delta: dict[str, float] = Field(default_factory=dict)


class RunDetailResponse(BaseModel):
    """Full run detail assembled from all research observability views."""

    run_id: str
    campaign_id: str
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_events: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_chunks: list[dict[str, Any]] = Field(default_factory=list)
    context_packs: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    routing_decisions: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    human_ratings: list[dict[str, Any]] = Field(default_factory=list)


class RunVisualResponse(RunToolsResponse):
    """Visual-tool subset for one run."""


class RunGraphResponse(RunToolsResponse):
    """Graph-tool subset for one run."""


class CampaignProgressEvent(BaseModel):
    """Incremental SSE progress payload."""

    campaign_id: str
    status: CampaignLifecycleStatus
    phase: CampaignEvaluationPhase = "execution"
    completed_units: int = Field(default=0, ge=0)
    total_units: int = Field(default=0, ge=0)
    evaluation_completed_units: int = Field(default=0, ge=0)
    evaluation_total_units: int = Field(default=0, ge=0)
    current_question_id: Optional[str] = None
    current_mode: Optional[CampaignMode] = None
    latest_result_id: Optional[str] = None


class MetricAggregate(BaseModel):
    """Aggregate statistics for one metric."""

    mean: float = Field(default=0, ge=0)
    max: float = Field(default=0, ge=0)
    stddev: float = Field(default=0, ge=0)


class CampaignMetricRow(BaseModel):
    """One evaluated row used by tables and charts."""

    campaign_result_id: str
    question_id: str
    question: str
    mode: CampaignMode
    run_number: int = Field(ge=1)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    ragas_focus: list[str] = Field(default_factory=list)
    reference_source: Optional[str] = None
    context_policy_version: Optional[str] = None
    total_tokens: int = Field(default=0, ge=0)
    metric_values: dict[str, float] = Field(default_factory=dict)
    invalid_metrics: dict[str, bool] = Field(default_factory=dict)
    invalid_reasons: dict[str, str] = Field(default_factory=dict)
    faithfulness: float = Field(default=0, ge=0)
    answer_correctness: float = Field(default=0, ge=0)


class GroupMetricsSummary(BaseModel):
    """Aggregated metrics for one grouping dimension."""

    group_key: str
    sample_count: int = Field(default=0, ge=0)
    metric_summaries: dict[str, MetricAggregate] = Field(default_factory=dict)
    total_tokens: MetricAggregate = Field(default_factory=MetricAggregate)


class ModeMetricsSummary(BaseModel):
    """Aggregated metrics for a single RAG mode."""

    mode: CampaignMode
    sample_count: int = Field(default=0, ge=0)
    metric_summaries: dict[str, MetricAggregate] = Field(default_factory=dict)
    faithfulness: MetricAggregate = Field(default_factory=MetricAggregate)
    answer_correctness: MetricAggregate = Field(default_factory=MetricAggregate)
    total_tokens: MetricAggregate = Field(default_factory=MetricAggregate)
    delta_answer_correctness: float = 0
    delta_faithfulness: float = 0
    delta_total_tokens: float = 0
    ecr: Optional[float] = None
    ecr_note: Optional[str] = None
    ecr_faithfulness: Optional[float] = None
    ecr_faithfulness_note: Optional[str] = None
    ecr_direction_correctness: str = "neutral"
    ecr_direction_faithfulness: str = "neutral"


class DeltaModeSummary(BaseModel):
    """Per-mode delta summary under one grouping key."""

    mode: CampaignMode
    sample_count: int = Field(default=0, ge=0)
    answer_correctness_mean: float = Field(default=0, ge=0)
    faithfulness_mean: float = Field(default=0, ge=0)
    total_tokens_mean: float = Field(default=0, ge=0)
    delta_answer_correctness: Optional[float] = None
    delta_faithfulness: Optional[float] = None
    delta_total_tokens: Optional[float] = None
    ecr: Optional[float] = None
    ecr_note: Optional[str] = None
    ecr_faithfulness: Optional[float] = None
    ecr_faithfulness_note: Optional[str] = None
    ecr_direction_correctness: str = "neutral"
    ecr_direction_faithfulness: str = "neutral"


class DeltaGroupSummary(BaseModel):
    """Grouped delta summaries keyed by mode."""

    group_key: str
    by_mode: dict[CampaignMode, DeltaModeSummary] = Field(default_factory=dict)


class EvaluationWarnings(BaseModel):
    """RAGAS scoring health warnings for the current metrics payload."""

    total_metric_rows: int = Field(default=0, ge=0)
    invalid_metric_rows: int = Field(default=0, ge=0)
    invalid_ratio: float = Field(default=0, ge=0)
    invalid_by_metric: dict[str, int] = Field(default_factory=dict)


class CampaignMetricsResponse(BaseModel):
    """Campaign-level metrics response for result analysis."""

    campaign: CampaignStatus
    evaluator_model: str
    available_metrics: list[str] = Field(default_factory=list)
    summary_by_mode: dict[CampaignMode, ModeMetricsSummary] = Field(default_factory=dict)
    summary_by_category: dict[str, GroupMetricsSummary] = Field(default_factory=dict)
    summary_by_focus: dict[str, GroupMetricsSummary] = Field(default_factory=dict)
    delta_by_category: dict[str, DeltaGroupSummary] = Field(default_factory=dict)
    delta_by_difficulty: dict[str, DeltaGroupSummary] = Field(default_factory=dict)
    delta_by_question: dict[str, DeltaGroupSummary] = Field(default_factory=dict)
    evaluation_warnings: EvaluationWarnings = Field(default_factory=EvaluationWarnings)
    rows: list[CampaignMetricRow] = Field(default_factory=list)
