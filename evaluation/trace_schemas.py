"""Schemas for persisted agent traces."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from evaluation.campaign_schemas import CampaignMode

TracePhase = Literal["planning", "execution", "drilldown", "evaluation", "synthesis"]
TraceStatus = Literal["completed", "partial", "failed"]
TraceEventStatus = Literal["running", "success", "failed", "skipped", "timeout", "partial"]
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


class EvaluationTraceEvent(BaseModel):
    """Generic span/event row for evaluation run observability."""

    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    parent_event_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    event_type: str = Field(min_length=1)
    event_schema_version: str = Field(default="1.0", min_length=1)
    sequence: int = Field(ge=1)
    stage_type: TraceStageType
    stage_name: str = Field(min_length=1)
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    status: TraceEventStatus
    retry_count: int = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationLlmCall(BaseModel):
    """Normalized LLM usage row for a run or evaluator call."""

    llm_call_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    purpose: str = Field(default="unknown", min_length=1)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
    estimated_cost_twd: Optional[float] = Field(default=None, ge=0)
    prompt_hash: Optional[str] = None
    prompt_preview: Optional[str] = None
    response_hash: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    status: TraceEventStatus = "success"
    error: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationRetrievalEvent(BaseModel):
    """Normalized retrieval request/response summary."""

    retrieval_event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    query: Optional[str] = None
    query_hash: Optional[str] = None
    retriever_name: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=0)
    result_count: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationRetrievalChunk(BaseModel):
    """One retrieved chunk plus evidence matching flags."""

    retrieval_chunk_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    retrieval_event_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    doc_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    modality: Optional[str] = None
    rank_before_rerank: Optional[int] = None
    rank_after_rerank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    used_in_context: bool = False
    used_in_answer: bool = False
    expected_evidence_match: bool = False
    excerpt: Optional[str] = None
    content_hash: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationContextPack(BaseModel):
    """Context-packing summary for retrieved evidence."""

    context_pack_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    input_chunk_count: int = Field(default=0, ge=0)
    packed_chunk_count: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)
    retrieved_but_not_packed_evidence: list[dict[str, Any]] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationToolCall(BaseModel):
    """External tool invocation made during a run."""

    tool_call_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    tool_name: str = Field(min_length=1)
    action: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    status: TraceEventStatus = "success"
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationRoutingDecision(BaseModel):
    """Router or retrospective routing analysis for one run."""

    routing_decision_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    selected_mode: CampaignMode
    analysis_type: Literal["retrospective", "actual"] = "retrospective"
    confidence: Optional[float] = None
    reason: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationClaim(BaseModel):
    """Generated claim and evidence alignment status."""

    claim_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    claim_text: str = Field(min_length=1)
    claim_type: Optional[str] = None
    support_status: Literal["supported", "partially_supported", "unsupported", "contradicted"] = "unsupported"
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_reason: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationRunSummary(BaseModel):
    """Selected-run identity and strict token projection."""

    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    mode: CampaignMode
    repeat_number: int = Field(default=1, ge=1)
    answer_preview: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    accounting_status: Literal["complete", "partial", "not_available"] = "not_available"
    created_at: datetime


class EvaluationHumanRating(BaseModel):
    """Human rating row used for RAGAS calibration."""

    human_rating_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    span_id: Optional[str] = None
    rater_id_hash: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    correctness_score: float = Field(ge=0, le=1)
    faithfulness_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    citation_quality_score: float = Field(ge=0, le=1)
    usefulness_score: float = Field(ge=0, le=1)
    comments: Optional[str] = None
    is_blinded: bool = True
    shown_mode_label: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvaluationRunObservabilityDetail(BaseModel):
    """Normalized observability payload for one evaluation run."""

    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    trace_events: list[EvaluationTraceEvent] = Field(default_factory=list)
    llm_calls: list[EvaluationLlmCall] = Field(default_factory=list)
    retrieval_events: list[EvaluationRetrievalEvent] = Field(default_factory=list)
    retrieval_chunks: list[EvaluationRetrievalChunk] = Field(default_factory=list)
    context_packs: list[EvaluationContextPack] = Field(default_factory=list)
    tool_calls: list[EvaluationToolCall] = Field(default_factory=list)
    routing_decisions: list[EvaluationRoutingDecision] = Field(default_factory=list)
    claims: list[EvaluationClaim] = Field(default_factory=list)
    human_ratings: list[EvaluationHumanRating] = Field(default_factory=list)
    evidence_coverage: Optional[list[dict[str, Any]]] = None
    run_summary: Optional[EvaluationRunSummary] = None


class AgentTraceToolCall(BaseModel):
    """Normalized tool invocation recorded inside one trace step."""

    index: int = Field(default=0, ge=0)
    action: str = Field(min_length=1)
    status: TraceStatus = "completed"
    payload: dict[str, Any] = Field(default_factory=dict)
    result_preview: Optional[str] = None


class AgentTraceStep(BaseModel):
    """One traceable unit in the agent flow."""

    step_id: str = Field(min_length=1)
    phase: TracePhase
    step_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    status: TraceStatus = "completed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    raw_text: Optional[str] = None
    tool_calls: list[AgentTraceToolCall] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTraceDetail(BaseModel):
    """Full trace payload persisted for one campaign result."""

    trace_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    campaign_result_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    mode: CampaignMode
    execution_profile: Optional[str] = None
    question_intent: Optional[str] = None
    strategy_tier: Optional[str] = None
    route_profile: Optional[str] = None
    required_coverage: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    subtask_coverage_status: dict[str, bool] = Field(default_factory=dict)
    supported_claim_count: int = Field(default=0, ge=0)
    unsupported_claim_count: int = Field(default=0, ge=0)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    visual_verification_attempted: bool = False
    visual_tool_call_count: int = Field(default=0, ge=0)
    visual_force_fallback_used: bool = False
    classifier_decision: dict[str, Any] = Field(default_factory=dict)
    complexity_score: Optional[int] = None
    tier_shift: Optional[str] = None
    pruned_followups: int = Field(default=0, ge=0)
    semantic_gate_score: Optional[float] = None
    run_number: int = Field(ge=1)
    trace_status: TraceStatus
    summary: str = ""
    step_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    created_at: datetime
    steps: list[AgentTraceStep] = Field(default_factory=list)


class AgentTraceSummary(BaseModel):
    """Compact trace data for campaign-level listing and selection."""

    trace_id: str = Field(min_length=1)
    campaign_result_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    mode: CampaignMode
    execution_profile: Optional[str] = None
    question_intent: Optional[str] = None
    strategy_tier: Optional[str] = None
    route_profile: Optional[str] = None
    run_number: int = Field(ge=1)
    trace_status: TraceStatus
    summary: str = ""
    step_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    subtask_count: int = Field(default=0, ge=0)
    drilldown_depth: int = Field(default=0, ge=0)
    graph_tool_call_count: int = Field(default=0, ge=0)
    visual_verification_attempted: bool = False
    visual_tool_call_count: int = Field(default=0, ge=0)
    visual_force_fallback_used: bool = False
    complexity_score: Optional[int] = None
    tier_shift: Optional[str] = None
    pruned_followups: int = Field(default=0, ge=0)
    semantic_gate_score: Optional[float] = None
    total_tokens: int = Field(default=0, ge=0)
    created_at: datetime


def summarize_agent_trace(detail: AgentTraceDetail) -> AgentTraceSummary:
    """Derive a list-friendly summary from a persisted detail payload."""

    subtask_steps = [step for step in detail.steps if step.step_type == "sub_task_execution"]
    drilldown_iterations: list[int] = []
    for step in detail.steps:
        if step.phase != "drilldown":
            continue
        try:
            drilldown_iterations.append(int(step.metadata.get("iteration", 0) or 0))
        except (TypeError, ValueError):
            continue
    drilldown_depth = max(drilldown_iterations or [0])
    graph_tool_call_count = sum(
        1
        for step in detail.steps
        for tool_call in step.tool_calls
        if "graph" in f"{tool_call.action} {tool_call.payload}".lower()
    )

    return AgentTraceSummary(
        trace_id=detail.trace_id,
        campaign_result_id=detail.campaign_result_id,
        question_id=detail.question_id,
        question=detail.question,
        mode=detail.mode,
        execution_profile=detail.execution_profile,
        question_intent=detail.question_intent,
        strategy_tier=detail.strategy_tier,
        route_profile=detail.route_profile,
        run_number=detail.run_number,
        trace_status=detail.trace_status,
        summary=detail.summary,
        step_count=detail.step_count,
        tool_call_count=detail.tool_call_count,
        subtask_count=len(subtask_steps),
        drilldown_depth=drilldown_depth,
        graph_tool_call_count=graph_tool_call_count,
        visual_verification_attempted=detail.visual_verification_attempted,
        visual_tool_call_count=detail.visual_tool_call_count,
        visual_force_fallback_used=detail.visual_force_fallback_used,
        complexity_score=detail.complexity_score,
        tier_shift=detail.tier_shift,
        pruned_followups=detail.pruned_followups,
        semantic_gate_score=detail.semantic_gate_score,
        total_tokens=detail.total_tokens,
        created_at=detail.created_at,
    )
