"""Typed, allow-listed models and pure content policy for Export Schema v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from math import isfinite
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    TokenBreakdown,
)
from evaluation.campaign_schemas import (
    AgentBehaviorRow,
    CampaignLifecycleStatus,
    CampaignMode,
    CampaignResultStatus,
    ConditionMetricSummary,
    LegacyAgentBehaviorMetrics,
    QuestionComparisonRow,
    QuestionModeComparison,
    RouterAnalysisRow,
    SanitizedErrorRow,
    StageWarningRow,
    V9AgentBehaviorMetrics,
    V9ContextPack,
    V9SlotResolution,
)
from data_base.agentic_v9.repair import RepairPlan
from data_base.agentic_v9.schemas import (
    BudgetReservation,
    ClaimSupportType,
    ConflictCandidate,
    EvidenceScope,
    EvidenceSource,
    EvidenceSupportType,
    EvidenceValidationStatus,
    QueryContract,
    SourceLocator,
    SufficiencyReport,
    V9ExecutionMetrics,
)

T = TypeVar("T")
AvailabilityStatus = Literal[
    "complete", "partial", "not_instrumented", "not_available", "not_applicable"
]
ObservationProvenance = Literal["measured", "persisted", "derived", "heuristic"]
PromptCaptureStatus = Literal[
    "unknown",
    "captured",
    "redacted",
    "not_captured_at_execution",
    "capture_failed",
]
TraceEventStatus = Literal[
    "running", "success", "failed", "skipped", "timeout", "partial"
]
TraceStageType = Literal[
    "routing",
    "planning",
    "retrieval",
    "rerank",
    "graph",
    "visual",
    "tool",
    "context_packing",
    "generation",
    "claim_verification",
    "evaluation",
    "export",
]


class ExportModel(BaseModel):
    """Reject fields outside an export model's explicit allow list."""

    model_config = ConfigDict(extra="forbid")


class ExportCampaignRequest(ExportModel):
    """Explicit export scope and content controls."""

    include_run_observability: bool = False
    include_raw_trace_payloads: bool = False
    include_prompt_previews: bool = True
    include_full_prompts: bool = False
    include_answers: bool = True
    include_retrieved_excerpts: bool = True
    format: Literal["json"] = "json"


class ExportAvailability(ExportModel):
    status: AvailabilityStatus
    reasons: list[str] = Field(default_factory=list)


class ExportEmptyObjectV2(ExportModel):
    """A serialized empty object with no extensible payload surface."""


class ExportSection(ExportModel, Generic[T]):
    availability: ExportAvailability
    data: T | None


class ExportCampaignIdentityV2(ExportModel):
    id: str
    name: str | None
    status: CampaignLifecycleStatus
    benchmark_id: str | None
    modes: list[CampaignMode]
    repeat_count: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ExportResultV2(ExportModel):
    run_id: str
    campaign_id: str
    question_id: str
    question: str
    mode: CampaignMode
    run_number: int = Field(ge=1)
    repeat_number: int = Field(ge=1)
    condition_id: str | None
    execution_profile: str | None
    context_policy_version: str | None
    agentic_execution_version: Literal["v8", "v9"]
    execution_identity: str | None
    response_status: str | None
    status: CampaignResultStatus
    answer: str | None
    ground_truth: str | None
    ground_truth_short: str | None
    contexts: list[str] | None
    source_doc_ids: list[str]
    latency_ms: float | None = Field(default=None, ge=0)
    total_latency_ms: float | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    created_at: datetime


class ExportRunLatencyV2(ExportModel):
    latency_ms: float | None = Field(default=None, ge=0)
    total_latency_ms: float | None = Field(default=None, ge=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExportTraceEventV2(ExportModel):
    event_id: str
    run_id: str
    campaign_id: str
    span_id: str
    parent_event_id: str | None = None
    parent_span_id: str | None = None
    event_type: str
    event_schema_version: str
    sequence: int = Field(ge=1)
    stage_type: TraceStageType
    stage_name: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    status: TraceEventStatus
    retry_count: int = Field(ge=0)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    error: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportLlmCallV2(ExportModel):
    llm_call_id: str
    run_id: str
    campaign_id: str
    span_id: str | None = None
    provider: str | None = None
    model_name: str | None = None
    phase: str
    purpose: str
    reservation_id: str | None = None
    provider_attempt: int | None = Field(default=None, ge=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    other_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_twd: float | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    status: TraceEventStatus
    prompt_hash: str | None = None
    response_hash: str | None = None
    prompt_capture_status: PromptCaptureStatus
    full_prompt_capture_status: PromptCaptureStatus
    prompt_preview: str | None = None
    full_prompt: str | None = None
    error: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportRetrievalEventV2(ExportModel):
    retrieval_event_id: str
    run_id: str
    campaign_id: str
    span_id: str | None = None
    query: str | None = None
    query_hash: str | None = None
    retriever_name: str | None = None
    top_k: int | None = Field(default=None, ge=0)
    result_count: int = Field(ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportRetrievalChunkV2(ExportModel):
    retrieval_chunk_id: str
    run_id: str
    campaign_id: str
    span_id: str | None = None
    retrieval_event_id: str
    chunk_id: str
    doc_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    modality: str | None = None
    rank_before_rerank: int | None = None
    rank_after_rerank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    used_in_context: bool | None = None
    used_in_answer: bool | None = None
    expected_evidence_match: bool | None = None
    excerpt: str | None = None
    content_hash: str | None = None
    provenance: ObservationProvenance
    availability: ExportAvailability
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportEvidenceReferenceV2(ExportModel):
    evidence_id: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None


class ExportContextPackV2(ExportModel):
    context_pack_id: str
    run_id: str
    campaign_id: str
    attempt_id: str | None = None
    condition_id: str
    schema_version: str
    span_id: str | None = None
    input_chunk_count: int = Field(ge=0)
    packed_chunk_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    retrieved_but_not_packed_evidence: list[ExportEvidenceReferenceV2]
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportToolCallV2(ExportModel):
    tool_call_id: str
    run_id: str
    campaign_id: str
    span_id: str | None = None
    tool_name: str
    action: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    status: TraceEventStatus
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportRoutingDecisionV2(ExportModel):
    routing_decision_id: str
    run_id: str
    campaign_id: str
    span_id: str | None = None
    selected_mode: CampaignMode
    analysis_type: Literal["retrospective", "actual"]
    decision_source: Literal["deterministic", "llm_planner", "safe_fallback"] | None
    candidate_routes: list[str]
    matched_rules: list[str]
    fallback_reason: str | None
    confidence: float | None
    reason: str | None
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportGraphEventV2(ExportModel):
    graph_event_id: str
    run_id: str
    campaign_id: str | None
    span_id: str | None
    graph_query: str
    graph_search_mode: str
    graph_evidence_mode: str
    graph_route: str
    router_reason: str | None
    graph_feature_flags: ExportEmptyObjectV2 = Field(
        default_factory=ExportEmptyObjectV2
    )
    graph_snapshot_version: str | None
    graph_schema_version: str | None
    graph_extraction_prompt_version: str | None
    matched_entity_ids: list[str]
    community_ids: list[int]
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    path_count: int = Field(ge=0)
    graph_latency_ms: int | None = Field(default=None, ge=0)
    graph_context_tokens: int = Field(ge=0)
    graph_to_chunk_success_rate: float | None = Field(default=None, ge=0, le=1)
    graph_noise_ratio: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime


class ExportGraphEvidenceItemV2(ExportModel):
    graph_evidence_item_id: str
    graph_event_id: str
    node_ids: list[str]
    edge_ids: list[str]
    relation_path: list[str]
    source_doc_ids: list[str]
    source_chunk_ids: list[str]
    pages: list[int]
    asset_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    provenance_status: Literal["full", "partial", "missing"]
    used_as_locator: bool
    packed_in_context: bool
    used_in_answer: bool
    supported_claim_ids: list[str]
    created_at: datetime


class ExportClaimV2(ExportModel):
    claim_id: str
    run_id: str
    campaign_id: str
    attempt_id: str | None
    condition_id: str
    schema_version: str
    span_id: str | None
    claim_text: str | None
    claim_type: str | None
    support_status: Literal[
        "supported", "partially_supported", "unsupported", "contradicted"
    ]
    evidence: list[ExportEvidenceReferenceV2] = Field(default_factory=list)
    evidence_refs: list[ExportEvidenceReferenceV2]
    unsupported_reason: str | None
    repair_action: str | None
    post_repair_status: str | None
    extraction_status: Literal["recorded", "empty", "not_instrumented"]
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportHumanRatingV2(ExportModel):
    human_rating_id: str
    run_id: str
    campaign_id: str
    span_id: str | None
    rater_id_hash: str
    rubric_version: str
    correctness_score: float = Field(ge=0, le=1)
    faithfulness_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    citation_quality_score: float = Field(ge=0, le=1)
    usefulness_score: float = Field(ge=0, le=1)
    comments: str | None
    is_blinded: bool
    shown_mode_label: bool
    payload: ExportEmptyObjectV2 = Field(default_factory=ExportEmptyObjectV2)
    created_at: datetime


class ExportEvidenceCoverageV2(ExportModel):
    atomic_fact_id: str
    fact_text: str | None
    retrieved: bool
    packed: bool
    mentioned: bool
    cited: bool
    expected_doc_ids: list[str]


class ExportV9EvidencePacketDataV2(ExportModel):
    """Export-owned packet shape whose content may be policy-suppressed."""

    schema_version: str
    evidence_id: str
    task_id: str
    round_id: str
    query_id: str
    slot_ids: list[str]
    statement: str | None
    support_type: EvidenceSupportType
    source: EvidenceSource
    scope: EvidenceScope
    locator: SourceLocator
    raw_value: Decimal | None = None
    normalized_value: Decimal | None = None
    unit: str | None = None
    calculation_operation: str | None = None
    premise_evidence_ids: list[str] = Field(default_factory=list)
    display_precision: int | None = Field(default=None, ge=0)
    rounding_mode: str | None = None
    extractor_version: str | None = None
    prompt_version: str | None = None
    validation_status: EvidenceValidationStatus = "deterministic_valid"


class ExportV9EvidencePacketV2(ExportModel):
    evidence_id: str
    packet: ExportV9EvidencePacketDataV2


class ExportV9FinalClaimV2(ExportModel):
    claim_id: str
    slot_id: str | None = None
    statement: str | None
    support_type: ClaimSupportType
    evidence_ids: list[str] = Field(default_factory=list)
    premise_evidence_ids: list[str] = Field(default_factory=list)
    qualified_reason: str | None = None


class ExportV9ComparisonValidationIssueV2(ExportModel):
    path: str
    type: str


class ExportV9ComparisonSubjectV2(ExportModel):
    subject_id: str
    display_name: str
    aliases: list[str]


class ExportV9ComparisonSelectedV2(ExportModel):
    doc_id: str | None = None
    chunk_id: str | None = None


class ExportV9ComparisonTaskDiagnosticV2(ExportModel):
    task_id: str
    subject_id: str
    query_hash: str
    query_preview: str
    status: Literal["executed", "fallback", "not_instrumented"]
    fallback_reason: Literal[
        "reranker_unavailable",
        "reranker_error",
        "reranker_empty_result",
        "unknown",
    ] | None
    candidate_count: int = Field(ge=0)
    pre_subject_limit_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    selected: list[ExportV9ComparisonSelectedV2]


class ExportV9ComparisonFinalEvidenceV2(ExportModel):
    evidence_id: str
    doc_id: str
    chunk_id: str | None
    subject_ids: list[str]


class ExportV9ComparisonV2(ExportModel):
    planner_status: Literal["not_requested", "planned", "fallback", "unknown"]
    planner_latency_ms: float = Field(ge=0)
    planner_fallback_reason: Literal[
        "timeout",
        "provider_error",
        "invalid_response",
        "schema_violation",
        "invalid_subjects",
        "not_comparison",
        "unknown",
    ] | None
    fallback_stage: Literal[
        "response_decode",
        "transport_schema",
        "subject_validation",
        "trusted_plan_validation",
        "numeric_guard",
        "unknown",
    ] | None
    validation_issues: list[ExportV9ComparisonValidationIssueV2]
    is_comparison: bool
    subjects: list[ExportV9ComparisonSubjectV2]
    dimensions: list[str]
    task_diagnostics: list[ExportV9ComparisonTaskDiagnosticV2]
    coverage_before_repair: list[str]
    missing_before_repair: list[str]
    repair_executed: bool
    coverage_after_repair: list[str]
    missing_after_repair: list[str]
    final_status: Literal[
        "complete", "qualified_partial", "insufficient", "unknown"
    ]
    final_evidence_subjects: list[str]
    final_evidence_count: int = Field(ge=0)
    final_evidence: list[ExportV9ComparisonFinalEvidenceV2]


class ExportV9ExecutionObservabilityV2(ExportModel):
    schema_version: str = "1"
    contract: QueryContract | None = None
    slot_resolutions: list[V9SlotResolution] = Field(default_factory=list)
    evidence_packets: list[ExportV9EvidencePacketV2] = Field(default_factory=list)
    sufficiency: SufficiencyReport | None = None
    context_pack: V9ContextPack | None = None
    budget: list[BudgetReservation] = Field(default_factory=list)
    repairs: list[RepairPlan] = Field(default_factory=list)
    conflicts: list[ConflictCandidate] = Field(default_factory=list)
    final_claims: list[ExportV9FinalClaimV2] = Field(default_factory=list)
    metrics: V9ExecutionMetrics = Field(default_factory=V9ExecutionMetrics)
    comparison: ExportV9ComparisonV2 | None = None


class ExportRunSummaryV2(ExportModel):
    run_id: str
    campaign_id: str
    question_id: str
    mode: CampaignMode
    repeat_number: int = Field(ge=1)
    answer_preview: str | None
    latency_ms: float | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    accounting_status: Literal["complete", "partial", "not_available"]
    created_at: datetime


class ExportRunObservabilityDataV2(ExportModel):
    run_id: str
    campaign_id: str
    run_summary: ExportRunSummaryV2
    accounting_diagnostics: TokenBreakdown
    trace_events: list[ExportTraceEventV2]
    llm_calls: list[ExportLlmCallV2]
    retrieval_events: list[ExportRetrievalEventV2]
    retrieval_chunks: list[ExportRetrievalChunkV2]
    context_packs: list[ExportContextPackV2]
    tool_calls: list[ExportToolCallV2]
    routing_decisions: list[ExportRoutingDecisionV2]
    graph_events: list[ExportGraphEventV2]
    graph_evidence_items: list[ExportGraphEvidenceItemV2]
    graph_observability_status: Literal["recorded", "fallback", "not_instrumented"]
    claims: list[ExportClaimV2]
    claim_extraction_status: Literal["recorded", "empty", "not_instrumented"]
    human_ratings: list[ExportHumanRatingV2]
    evidence_coverage: list[ExportEvidenceCoverageV2] | None
    evidence_coverage_status: Literal[
        "complete", "partial", "not_available", "not_instrumented"
    ]
    agentic_v9: ExportV9ExecutionObservabilityV2 | None


class ExportRunObservabilityV2(ExportModel):
    included: bool
    availability: ExportAvailability
    data: ExportRunObservabilityDataV2 | None


class ExportRunV2(ExportModel):
    result: ExportResultV2
    ragas_metrics: dict[str, float]
    accounting: TokenBreakdown
    latency: ExportRunLatencyV2
    observability: ExportRunObservabilityV2

    @field_validator("ragas_metrics")
    @classmethod
    def require_finite_ragas_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not isfinite(metric) for metric in value.values()):
            raise ValueError("ragas_metrics must contain only finite values")
        return value


class ExportReleaseMetricV2(ExportModel):
    value: float | int | None = None
    reason: str | None = None


class ExportReleaseArmV2(ExportModel):
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None = None
    response_status_counts: dict[str, int] = Field(default_factory=dict)
    run_count: int = Field(ge=0)
    complete_run_count: int = Field(ge=0)
    accounting_complete_run_count: int = Field(ge=0)


class ExportReleaseManifestBlockV2(ExportModel):
    question_id: str
    repeat_number: int = Field(ge=1)
    mode: str
    condition_id: str
    execution_profile: str
    agentic_execution_version: str
    shadow_evaluation_policy: str | None = None
    golden_question_fingerprint: str | None = None


class ExportReleaseEvaluatorBlindingV2(ExportModel):
    enabled: bool
    shown_mode_label: bool
    method: str


class ExportReleaseEmptyObjectV2(ExportModel):
    """Closed representation of an intentionally empty release detail block."""


class ExportReleaseManifestV2(ExportModel):
    benchmark_id: str
    kind: Literal["smoke", "formal", "insufficient"]
    arm_order_seed: int
    ordered_blocks: list[ExportReleaseManifestBlockV2]
    evaluator_blinding: ExportReleaseEvaluatorBlindingV2
    environment_fingerprint: str | None = None
    evaluator_fingerprint: str | None = None
    non_blocking_ablations: list[str]


class ExportReleaseStatisticsV2(ExportModel):
    method: str
    availability: Literal["release_gate_blocked"] | None = None
    seed: int | None = None
    resamples: int | None = Field(default=None, ge=0)
    cluster_count: int | None = Field(default=None, ge=0)
    repeat_aggregation: str | None = None
    token_ratio_method: str | None = None
    final_generation_count_aggregation: str | None = None


class ExportReleaseMetricsV2(ExportModel):
    benchmark_id: str
    benchmark_kind: str
    comparable: bool
    availability: Literal["available", "not_applicable"] = "available"
    not_applicable_reason: str | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    manifest: ExportReleaseManifestV2 | ExportReleaseEmptyObjectV2 = Field(
        default_factory=ExportReleaseEmptyObjectV2
    )
    arms: list[ExportReleaseArmV2] = Field(default_factory=list)
    required_slot_coverage: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    important_unsupported_claim_rate: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    provenance_failure_rate: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    pack_efficiency: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    graph_locator_success: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    graph_locator_fallback: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    final_generation_count: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    latency_p95_ms: ExportReleaseMetricV2 = Field(default_factory=ExportReleaseMetricV2)
    token_ratio: ExportReleaseMetricV2 = Field(default_factory=ExportReleaseMetricV2)
    paired_quality_delta: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    paired_quality_ci_lower: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    paired_quality_ci_upper: ExportReleaseMetricV2 = Field(
        default_factory=ExportReleaseMetricV2
    )
    category_quality_deltas: dict[str, ExportReleaseMetricV2] = Field(
        default_factory=dict
    )
    per_question_quality_deltas: dict[str, ExportReleaseMetricV2] = Field(
        default_factory=dict
    )
    statistics: ExportReleaseStatisticsV2 | ExportReleaseEmptyObjectV2 = Field(
        default_factory=ExportReleaseEmptyObjectV2
    )


class ExportResearchSummaryV2(CampaignResearchSummaryResponse):
    model_config = ConfigDict(extra="forbid")


class ExportQuestionModeComparisonV2(QuestionModeComparison):
    model_config = ConfigDict(extra="forbid")


class ExportQuestionComparisonRowV2(QuestionComparisonRow):
    model_config = ConfigDict(extra="forbid")

    by_mode: list[ExportQuestionModeComparisonV2]


class ExportLegacyAgentBehaviorMetricsV2(LegacyAgentBehaviorMetrics):
    model_config = ConfigDict(extra="forbid")


class ExportV9AgentBehaviorMetricsV2(V9AgentBehaviorMetrics):
    model_config = ConfigDict(extra="forbid")


class ExportAgentBehaviorRowV2(AgentBehaviorRow):
    model_config = ConfigDict(extra="forbid")

    legacy: ExportLegacyAgentBehaviorMetricsV2 | None
    v9: ExportV9AgentBehaviorMetricsV2 | None


class ExportRouterAnalysisRowV2(RouterAnalysisRow):
    model_config = ConfigDict(extra="forbid")


class ExportAnalyticsBaseV2(ExportModel):
    campaign_id: str
    analysis_unit: Literal["execution", "question", "category"]
    sample_count: int = Field(ge=0)
    independent_question_count: int = Field(ge=0)
    repeat_count: int = Field(ge=0)
    sample_note: str
    warnings: list[str]


class ExportQuestionAnalysisV2(ExportAnalyticsBaseV2):
    rows: list[ExportQuestionComparisonRowV2]
    summaries: dict[str, ExportQuestionComparisonRowV2]


class ExportAgentBehaviorV2(ExportAnalyticsBaseV2):
    behavior_schema_version: Literal["2"]
    rows: list[ExportAgentBehaviorRowV2]
    summaries: dict[str, ExportAgentBehaviorRowV2]


class ExportRouterAnalysisSummaryV2(ExportModel):
    decision_count: int = Field(default=0, ge=0)


class ExportRouterAnalysisV2(ExportAnalyticsBaseV2):
    analysis_type: Literal["retrospective"]
    rows: list[ExportRouterAnalysisRowV2]
    summaries: ExportRouterAnalysisSummaryV2


ExportAblationFlagScalarV2 = bool | int | float | str | None
ExportAblationFlagNestedV2 = dict[str, ExportAblationFlagScalarV2]
ExportAblationFlagValueV2 = (
    ExportAblationFlagScalarV2
    | ExportAblationFlagNestedV2
    | dict[str, ExportAblationFlagNestedV2]
)


class ExportConditionMetricSummaryV2(ConditionMetricSummary):
    model_config = ConfigDict(extra="forbid")


class ExportConditionAggregateV2(ExportModel):
    condition_id: str
    label: str
    ablation_flags: dict[str, ExportAblationFlagValueV2]
    execution_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    quality: dict[str, ExportConditionMetricSummaryV2]
    mean_tokens: float | None = Field(ge=0)
    mean_latency_ms: float | None = Field(ge=0)


class ExportConditionPairedComparisonV2(ExportModel):
    baseline_condition_id: str
    guided_condition_id: str
    completed_pair_count: int = Field(ge=0)
    metric_pair_counts: dict[str, int]
    delta: dict[str, ExportConditionMetricSummaryV2]
    excluded_pairs: dict[str, int]


class ExportConditionMetricAvailabilityV2(ExportModel):
    ragas_rows_found: bool
    valid_metric_row_count: int = Field(ge=0)
    warning: str | None


class ExportConditionComparisonV2(ExportModel):
    conditions: dict[str, ExportConditionAggregateV2]
    paired: ExportConditionPairedComparisonV2 | None
    availability: ExportConditionMetricAvailabilityV2


class ExportAblationSummariesV2(ExportModel):
    condition_counts: dict[str, int] = Field(default_factory=dict)
    condition_labels: dict[str, str] = Field(default_factory=dict)
    conditions_by_ablation_family: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    graph_metrics_by_ablation_family: dict[str, dict[str, float | None]] = Field(
        default_factory=dict
    )
    condition_comparison: ExportConditionComparisonV2 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ExportEmptyAnalyticsRowV2(ExportModel):
    pass


class ExportAblationV2(ExportAnalyticsBaseV2):
    rows: list[ExportEmptyAnalyticsRowV2]
    summaries: ExportAblationSummariesV2


class ExportHumanVsAutoRowV2(ExportModel):
    run_id: str
    question_id: str
    mode: CampaignMode
    rating_count: int = Field(ge=0)
    human_correctness_mean: float = Field(ge=0, le=1)
    human_faithfulness_mean: float = Field(ge=0, le=1)
    ragas_answer_correctness: float | None = Field(ge=0, le=1)
    ragas_faithfulness: float | None = Field(ge=0, le=1)


class ExportHumanVsAutoSummaryV2(ExportModel):
    human_rating_count: int = Field(default=0, ge=0)
    paired_sample_count: int = Field(default=0, ge=0)
    human_correctness_mean: float | None = Field(default=None, ge=0, le=1)
    human_faithfulness_mean: float | None = Field(default=None, ge=0, le=1)
    ragas_human_pearson_r: float | None = Field(default=None, ge=-1, le=1)
    ragas_human_spearman_r: float | None = Field(default=None, ge=-1, le=1)
    inter_rater_agreement: float | None = Field(default=None, ge=0, le=1)


class ExportHumanVsAutoV2(ExportAnalyticsBaseV2):
    rows: list[ExportHumanVsAutoRowV2]
    summaries: ExportHumanVsAutoSummaryV2


class ExportSanitizedErrorRowV2(SanitizedErrorRow):
    model_config = ConfigDict(extra="forbid")


class ExportCampaignErrorsV2(ExportModel):
    campaign_id: str
    rows: list[ExportSanitizedErrorRowV2]


class ExportStageWarningRowV2(StageWarningRow):
    model_config = ConfigDict(extra="forbid")


class ExportCampaignStageWarningsV2(ExportModel):
    campaign_id: str
    rows: list[ExportStageWarningRowV2]


class ExportOverviewDataV2(ExportModel):
    research_summary: ExportResearchSummaryV2
    release_metrics: ExportSection[ExportReleaseMetricsV2]


class ExportHumanEvalQueueItemV2(ExportModel):
    run_id: str
    campaign_id: str
    question_id: str
    question: str
    mode: CampaignMode
    run_number: int = Field(ge=1)
    repeat_number: int = Field(ge=1)
    answer_preview: str | None
    existing_rating_count: int = Field(ge=0)
    already_rated_by_current_user: bool


class ExportHumanEvalQueueV2(ExportModel):
    campaign_id: str
    rows: list[ExportHumanEvalQueueItemV2] = Field(default_factory=list)


class ExportHumanEvaluationDataV2(ExportModel):
    comparison: ExportHumanVsAutoV2
    queue: ExportHumanEvalQueueV2


class ExportDiagnosticsDataV2(ExportModel):
    errors: ExportCampaignErrorsV2
    stage_warnings: ExportCampaignStageWarningsV2


class ExportSectionsV2(ExportModel):
    overview: ExportSection[ExportOverviewDataV2]
    question_analysis: ExportSection[ExportQuestionAnalysisV2]
    agent_behavior: ExportSection[ExportAgentBehaviorV2]
    router_analysis: ExportSection[ExportRouterAnalysisV2]
    ablation: ExportSection[ExportAblationV2]
    human_evaluation: ExportSection[ExportHumanEvaluationDataV2]
    diagnostics: ExportSection[ExportDiagnosticsDataV2]


class ExportRedactionMetadata(ExportModel):
    provider_errors: Literal["excluded"] = "excluded"
    stack_traces: Literal["excluded"] = "excluded"
    credentials: Literal["redacted"] = "redacted"


class ExportMetadataV2(ExportModel):
    exported_at: datetime
    options: ExportCampaignRequest
    redaction: ExportRedactionMetadata
    availability_warnings: list[str]


class ExportCampaignResponse(ExportModel):
    schema_version: Literal["2.0"] = "2.0"
    export_metadata: ExportMetadataV2
    campaign: ExportCampaignIdentityV2
    sections: ExportSectionsV2
    runs: list[ExportRunV2]


@dataclass(frozen=True, slots=True)
class ExportContentPolicy:
    raw_trace_allowed: bool
    prompt_preview_allowed: bool
    full_prompt_allowed: bool
    answer_text_allowed: bool
    excerpt_text_allowed: bool
    provider_bodies_allowed: Literal[False] = False
    credentials_allowed: Literal[False] = False
    authorization_headers_allowed: Literal[False] = False
    stack_traces_allowed: Literal[False] = False
    unrestricted_errors_allowed: Literal[False] = False
    non_trace_payloads_allowed: Literal[False] = False


def resolve_export_content_policy(
    request: ExportCampaignRequest, *, captured_at_execution: bool
) -> ExportContentPolicy:
    """Resolve content flags without changing export scope or stored data."""

    return ExportContentPolicy(
        raw_trace_allowed=request.include_raw_trace_payloads,
        prompt_preview_allowed=request.include_prompt_previews,
        full_prompt_allowed=(request.include_full_prompts and captured_at_execution),
        answer_text_allowed=request.include_answers,
        excerpt_text_allowed=request.include_retrieved_excerpts,
    )
