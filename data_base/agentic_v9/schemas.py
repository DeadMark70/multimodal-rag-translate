"""Serializable execution contracts for evidence-first Agentic v9."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Awaitable, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgenticV9Route = Literal[
    "single_lookup",
    "bounded_compare",
    "exact_structured",
    "multi_document_exact",
    "multi_hop",
    "graph_relational",
]
GraphPolicy = Literal["never", "locator_fallback", "required_locator"]
ResponseStatus = Literal["complete", "qualified_partial", "insufficient"]
EvidenceSupportType = Literal[
    "direct", "calculated", "scope_constraint", "contradictory"
]
ClaimSupportType = Literal["direct", "calculated", "comparative_inference", "qualified"]
ScopeMatch = Literal["same", "different", "unknown"]
SlotResolutionStatus = Literal[
    "supported", "conflicted", "explicitly_unavailable", "not_found"
]

ROUTE_GRAPH_POLICIES: dict[AgenticV9Route, GraphPolicy] = {
    "single_lookup": "never",
    "bounded_compare": "never",
    "exact_structured": "locator_fallback",
    "multi_document_exact": "locator_fallback",
    "multi_hop": "locator_fallback",
    "graph_relational": "required_locator",
}


def default_graph_policy(route: AgenticV9Route) -> GraphPolicy:
    """Return the frozen graph policy default for a v9 route."""
    return ROUTE_GRAPH_POLICIES[route]


class RequiredSlot(BaseModel):
    """One fact, comparison, or locator the answer must resolve."""

    slot_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool = True
    entity_ids: list[str] = Field(default_factory=list)
    locator_hints: list[str] = Field(default_factory=list)


class ResolvedSourceScope(BaseModel):
    """Resolver output; the sole v9 contract that contains authorized IDs."""

    requested_doc_ids: list[str] = Field(default_factory=list)
    requested_source_names: list[str] = Field(default_factory=list)
    resolved_doc_ids: list[str] = Field(default_factory=list)
    authorized_doc_ids: list[str] = Field(default_factory=list)
    rejected_source_names: list[str] = Field(default_factory=list)


class QueryContract(BaseModel):
    """Routing authority for a bounded evidence-first execution."""

    contract_version: str = Field(default="1", min_length=1)
    route: AgenticV9Route
    intent: str = Field(min_length=1)
    required_slots: list[RequiredSlot] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    locator_hints: list[str] = Field(default_factory=list)
    graph_policy: GraphPolicy | None = None
    visual_required: bool = False
    evidence_extraction_required: bool = False
    max_retrieval_rounds: int = Field(default=0, ge=0)
    max_repair_rounds: int = Field(default=0, ge=0)
    max_llm_calls: int = Field(default=0, ge=0)
    runtime_token_budget: int = Field(default=0, ge=0)
    resolved_source_scope: ResolvedSourceScope | None = None
    strategy_tier: str | None = None

    @model_validator(mode="after")
    def apply_route_graph_policy(self) -> QueryContract:
        """Keep route defaults deterministic unless a caller explicitly overrides them."""
        if self.graph_policy is None:
            self.graph_policy = default_graph_policy(self.route)
        if self.runtime_token_budget and not self.max_llm_calls:
            raise ValueError("runtime_token_budget requires max_llm_calls")
        return self


class RetrievalTask(BaseModel):
    """A retrieval-only unit that targets slots and never contains an answer."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    target_slot_ids: list[str] = Field(min_length=1)
    source_scope: ResolvedSourceScope
    source_group_id: str = Field(default="source-group-1", min_length=1)
    locator_hints: list[str] = Field(default_factory=list)
    graph_policy: GraphPolicy = "never"
    visual_required: bool = False
    depends_on_task_ids: list[str] = Field(default_factory=list)


class RagRetrievalResult(BaseModel):
    """Generic retrieval boundary result with no generated answer field."""

    retrieval_id: str = Field(min_length=1)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class TaskRetrievalResult(BaseModel):
    """One v9 retrieval task result, still constrained to retrieved evidence."""

    task_id: str = Field(min_length=1)
    retrieval: RagRetrievalResult


class EvidenceSource(BaseModel):
    """Canonical source identity for a positive evidence packet."""

    doc_id: str = Field(min_length=1)
    chunk_id: str | None = None
    parent_id: str | None = None
    asset_id: str | None = None
    document_name: str | None = None
    source_span_hash: str | None = None


class EvidenceScope(BaseModel):
    """Experimental/publication scope that prevents false evidence equivalence."""

    dataset: str | None = None
    split: str | None = None
    metric: str | None = None
    model_variant: str | None = None
    training_protocol: str | None = None
    prompt_setting: str | None = None
    noise_level: str | None = None
    publication_year: int | None = Field(default=None, ge=0)


class SourceLocator(BaseModel):
    """Versioned source location with distinct PDF and printed-page coordinates."""

    pdf_page_index: int | None = Field(default=None, ge=0)
    printed_page_label: str | None = None
    section: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    citation_format_version: str = Field(default="1", min_length=1)

    @model_validator(mode="after")
    def require_location(self) -> SourceLocator:
        """Reject packets that cannot be located in their declared source."""
        if not any(
            (
                self.pdf_page_index is not None,
                self.printed_page_label,
                self.section,
                self.table_id,
                self.figure_id,
                self.bbox,
            )
        ):
            raise ValueError("source locator requires at least one location field")
        return self


class EvidencePacket(BaseModel):
    """Positive, provenance-bound fact state; absence belongs to slot resolution."""

    schema_version: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    slot_ids: list[str] = Field(min_length=1)
    statement: str = Field(min_length=1)
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


class SlotResolution(BaseModel):
    """The separately persisted resolution state for a required slot."""

    slot_id: str = Field(min_length=1)
    status: SlotResolutionStatus
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    resolution_stage: str | None = None

    @model_validator(mode="after")
    def require_status_consistent_evidence(self) -> SlotResolution:
        """Keep positive evidence links separate from absence resolutions."""
        if self.status == "supported" and not self.evidence_ids:
            raise ValueError("supported slot resolutions require evidence IDs")
        if self.status == "conflicted" and len(self.evidence_ids) < 2:
            raise ValueError("conflicted slot resolutions require two evidence IDs")
        if self.status in {"explicitly_unavailable", "not_found"} and self.evidence_ids:
            raise ValueError("absence slot resolutions cannot link positive evidence")
        return self


class SufficiencyReport(BaseModel):
    """Evidence completeness and response usability are deliberately independent."""

    evidence_complete: bool
    answerable: bool
    response_status: ResponseStatus
    supported_slot_ids: list[str] = Field(default_factory=list)
    conflicted_slot_ids: list[str] = Field(default_factory=list)
    explicitly_unavailable_slot_ids: list[str] = Field(default_factory=list)
    not_found_slot_ids: list[str] = Field(default_factory=list)
    stop_reason: str | None = None

    @model_validator(mode="after")
    def require_consistent_completion_state(self) -> SufficiencyReport:
        """Prevent absence or unresolved conflict from being labeled complete."""
        unavailable_slots = (
            self.explicitly_unavailable_slot_ids or self.not_found_slot_ids
        )
        if self.evidence_complete and unavailable_slots:
            raise ValueError("unavailable or not-found slots preclude complete evidence")
        if self.response_status == "complete" and (
            not self.evidence_complete
            or not self.answerable
            or self.conflicted_slot_ids
            or unavailable_slots
        ):
            raise ValueError("complete responses require fully answerable evidence")
        return self


class ConflictCandidate(BaseModel):
    """A persisted scope-aware conflict requiring qualification or arbitration."""

    candidate_id: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=2)
    scope_match: ScopeMatch
    reason: str = Field(min_length=1)
    unresolved: bool = True


class FinalClaim(BaseModel):
    """A rendered answer claim linked to its supporting evidence packet IDs."""

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    support_type: ClaimSupportType
    evidence_ids: list[str] = Field(default_factory=list)
    premise_evidence_ids: list[str] = Field(default_factory=list)
    qualified_reason: str | None = None


class FinalAnswerResult(BaseModel):
    """One final answer or a deterministic partial/insufficiency projection."""

    response_status: ResponseStatus
    answer: str = ""
    claims: list[FinalClaim] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    final_generation_count: int = Field(default=0, ge=0, le=1)


class RetrievalPolicy(BaseModel):
    """Generic retrieval controls exposed to the v9 retrieval adapter."""

    top_k: int = Field(default=5, ge=1)
    use_reranker: bool = True
    allow_graph_locator: bool = False


class AnswerPolicy(BaseModel):
    """Final answer rendering controls, independent of retrieval behavior."""

    citation_format_version: str = Field(default="1", min_length=1)
    verify_high_risk_claims: bool = True
    max_final_generations: int = Field(default=1, ge=0, le=1)


class GeneratedRagAnswer(BaseModel):
    """Generic generation boundary result prior to v9 final-answer projection."""

    text: str = ""
    citations: list[str] = Field(default_factory=list)
    provider_call_count: int = Field(default=0, ge=0)


class EffectivePhasePolicy(BaseModel):
    """Setup-constrained model sampling policy for one v9 provider phase."""

    phase: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(ge=0, le=1)
    top_k: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)


class BudgetReservation(BaseModel):
    """Atomic budget reservation made before a provider attempt."""

    reservation_id: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    estimated_input_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    reserved_reasoning_tokens: int = Field(default=0, ge=0)
    provider_attempt: int = Field(default=1, ge=1)


class BudgetExceededError(RuntimeError):
    """Raised when a pre-invoke provider reservation cannot be admitted."""


class ExecutionPolicy(BaseModel):
    """Run-level concurrency and timeout bounds, without runtime dependencies."""

    max_retrieval_concurrency: int = Field(default=3, ge=1)
    max_llm_concurrency: int = Field(default=2, ge=1)
    max_visual_concurrency: int = Field(default=1, ge=1)
    total_deadline_s: float = Field(default=24.0, gt=0)
    phase_timeouts_s: dict[str, float] = Field(default_factory=dict)


class V9ExecutionRequest(BaseModel):
    """Serializable request body; authorization and runtime are adapter injected."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    requested_doc_ids: list[str] = Field(default_factory=list)
    requested_source_names: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    setup_snapshot: dict[str, Any] = Field(default_factory=dict)
    agentic_execution_version: Literal["v8", "v9"] = "v9"
    trace_id: str = Field(min_length=1)


class V9ExecutionMetrics(BaseModel):
    """Run-level observability metrics with provider and tool work kept separate."""

    provider_attempt_count: int = Field(default=0, ge=0)
    tool_operation_count: int = Field(default=0, ge=0)
    retrieval_query_count: int = Field(default=0, ge=0)
    final_generation_count: int = Field(default=0, ge=0, le=1)
    reserved_tokens: int = Field(default=0, ge=0)
    reconciled_tokens: int = Field(default=0, ge=0)


class V9ExecutionResult(BaseModel):
    """Whole-run projection assembled from evidence-only task results."""

    trace_id: str = Field(min_length=1)
    task_results: list[TaskRetrievalResult] = Field(default_factory=list)
    final_answer: FinalAnswerResult | None = None
    sufficiency: SufficiencyReport | None = None
    metrics: V9ExecutionMetrics = Field(default_factory=V9ExecutionMetrics)


class V9ExecutionEvent(BaseModel):
    """Serializable execution event suitable for a versioned trace payload."""

    event_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class LlmInvoker(Protocol):
    """Async-only model boundary injected into v9 runtime code."""

    async def invoke(
        self, *, phase: str, purpose: str, messages: list[dict[str, Any]]
    ) -> Any:
        """Invoke a provider through the runtime's budgeted boundary."""


@dataclass(slots=True)
class V9RuntimeContext:
    """Non-serializable adapter dependencies for ``core.execute`` only."""

    cancellation_token: Any
    event_sink: Callable[[V9ExecutionEvent], Awaitable[None]]
    budget_controller: Any
    deadline: Any
    clock: Callable[[], datetime]
    llm_invoker: LlmInvoker
