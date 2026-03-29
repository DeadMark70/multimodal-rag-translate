"""Schemas for persisted agent traces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from evaluation.campaign_schemas import CampaignMode

TracePhase = Literal["planning", "execution", "drilldown", "evaluation", "synthesis"]
TraceStatus = Literal["completed", "partial", "failed"]


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
    visual_verification_attempted: bool = False
    visual_tool_call_count: int = Field(default=0, ge=0)
    visual_force_fallback_used: bool = False
    total_tokens: int = Field(default=0, ge=0)
    created_at: datetime


def summarize_agent_trace(detail: AgentTraceDetail) -> AgentTraceSummary:
    """Derive a list-friendly summary from a persisted detail payload."""

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
        visual_verification_attempted=detail.visual_verification_attempted,
        visual_tool_call_count=detail.visual_tool_call_count,
        visual_force_fallback_used=detail.visual_force_fallback_used,
        total_tokens=detail.total_tokens,
        created_at=detail.created_at,
    )
