"""Typed, allow-listed models and pure content policy for Export Schema v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    TokenBreakdown,
)
from evaluation.campaign_schemas import (
    AblationResponse,
    AgentBehaviorResponse,
    CampaignErrorsResponse,
    CampaignLifecycleStatus,
    CampaignMode,
    CampaignResultStatus,
    CampaignStageWarningsResponse,
    HumanEvalQueueResponse,
    HumanVsAutoResponse,
    ResearchQuestionComparisonResponse,
    RouterAnalysisResponse,
    V9ExecutionObservability,
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
    evidence_refs: list[ExportEvidenceReferenceV2]
    unsupported_reason: str | None
    repair_action: str | None
    post_repair_status: str | None
    extraction_status: Literal["recorded", "empty", "not_instrumented"]
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
    created_at: datetime


class ExportEvidenceCoverageV2(ExportModel):
    atomic_fact_id: str
    fact_text: str | None
    retrieved: bool
    packed: bool
    mentioned: bool
    cited: bool
    expected_doc_ids: list[str]


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
    agentic_v9: V9ExecutionObservability | None


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
    manifest: ExportReleaseManifestV2 | None = None
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
    statistics: ExportReleaseStatisticsV2 | None = None


class ExportOverviewDataV2(ExportModel):
    research_summary: CampaignResearchSummaryResponse
    release_metrics: ExportSection[ExportReleaseMetricsV2]


class ExportHumanEvaluationDataV2(ExportModel):
    comparison: HumanVsAutoResponse
    queue: HumanEvalQueueResponse


class ExportDiagnosticsDataV2(ExportModel):
    errors: CampaignErrorsResponse
    stage_warnings: CampaignStageWarningsResponse


class ExportSectionsV2(ExportModel):
    overview: ExportSection[ExportOverviewDataV2]
    question_analysis: ExportSection[ResearchQuestionComparisonResponse]
    agent_behavior: ExportSection[AgentBehaviorResponse]
    router_analysis: ExportSection[RouterAnalysisResponse]
    ablation: ExportSection[AblationResponse]
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
