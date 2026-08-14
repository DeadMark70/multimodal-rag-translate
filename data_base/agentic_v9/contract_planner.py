"""Question-only planning of answer-free atomic Agentic v9 contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.prompt_loader import PromptRegistry
from data_base.agentic_v9.requirement_decomposition import (
    DecomposedRequirement,
    QuestionDecomposition,
    decompose_question,
)
from data_base.agentic_v9.schemas import (
    ATOMIC_SLOT_MATCHING_EXPERIMENTAL,
    AtomicPlannerDiagnostics,
    BudgetExceededError,
    ComparisonPlan,
    ComparisonSubject,
    ExpectedAnswerType,
    LlmInvoker,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    ResponseConstraint,
    ResponseConstraintKind,
    SlotPlanSource,
    SlotPlanStatus,
    SynthesisObligation,
    SynthesisObligationKind,
    VisualPolicy,
    validate_active_atomic_contract,
)
from data_base.agentic_v9.retrieval_tasks import compile_retrieval_tasks
from data_base.agentic_v9.slot_constraints import canonical_structured_locator

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "agentic_v9_contract_planner.json"
)
_PROMPT_KEY = "atomic_contract_planning"

_EXACT_LOCATOR_PATTERN = re.compile(
    r"\b(Figure|Fig\.|Table|Appendix|Formula|Equation|Theorem|Page|Section)"
    r"\s*(\(?[A-Za-z0-9](?:[A-Za-z0-9_:-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9_:-]+)*(?:\([A-Za-z0-9]{1,3}\))?\)?)",
    re.IGNORECASE,
)


class _PlannerEvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=512)
    source_name_hints: list[str] = Field(default_factory=list, max_length=8)
    locator_hints: list[str] = Field(default_factory=list, max_length=8)
    expected_answer_type: ExpectedAnswerType = "text"
    depends_on_requirement_indexes: list[int] = Field(
        default_factory=list, max_length=8
    )
    visual_policy: VisualPolicy = "never"


class _PlannerSynthesisObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SynthesisObligationKind
    description: str = Field(min_length=1, max_length=512)
    depends_on_requirement_indexes: list[int] = Field(
        default_factory=list, max_length=8
    )


class _PlannerResponseConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ResponseConstraintKind
    description: str = Field(min_length=1, max_length=512)


class _PlannerComparisonSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=160)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    retrieval_query: str = Field(min_length=1, max_length=512)
    evidence_requirement_indexes: list[int] = Field(
        default_factory=list, max_length=8
    )


class _PlannerComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjects: list[_PlannerComparisonSubject] = Field(
        min_length=2, max_length=4
    )
    dimensions: list[str] = Field(default_factory=list, max_length=12)
    qualification: str | None = Field(default=None, max_length=512)


class _PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_requirements: list[_PlannerEvidenceRequirement] = Field(
        min_length=1, max_length=8
    )
    synthesis_obligations: list[_PlannerSynthesisObligation] = Field(
        default_factory=list, max_length=8
    )
    response_constraints: list[_PlannerResponseConstraint] = Field(
        default_factory=list, max_length=8
    )
    comparison: _PlannerComparison | None = None
    confidence: float = Field(ge=0.0, le=1.0)


def atomic_contract_planner_response_schema() -> dict[str, Any]:
    """Return the JSON schema for the consolidated atomic planner response."""
    return _PlannerDecision.model_json_schema()


@dataclass(frozen=True, slots=True)
class AtomicContractPreparation:
    """Pure deterministic pre-planning analysis of question and base contract."""

    decomposition: QuestionDecomposition
    semantic_planning_requested: bool
    comparison_candidate: bool


@dataclass(frozen=True, slots=True)
class AtomicContractPlanningOutcome:
    """Consolidated outcome of atomic contract planning."""

    contract: QueryContract
    planner_call_count: Literal[0, 1]
    latency_ms: float
    planner_diagnostics: AtomicPlannerDiagnostics


def apply_atomic_contract_overlay(
    base_contract: QueryContract,
    *,
    required_slots: list[RequiredSlot],
    synthesis_obligations: list[SynthesisObligation] | None = None,
    response_constraints: list[ResponseConstraint] | None = None,
    comparison_plan: ComparisonPlan | None = None,
    slot_plan_status: SlotPlanStatus,
    slot_plan_source: SlotPlanSource,
    slot_plan_confidence: Literal["high", "medium", "low"] | None = None,
    slot_plan_fallback_reason: str | None = None,
    truncated_requirement_count: int | None = None,
) -> QueryContract:
    """Apply atomic contract fields over an immutable base contract."""
    overlaid = base_contract.model_copy(
        update={
            "contract_version": "2",
            "slot_semantics": "heuristic_experimental",
            "atomic_completeness": None,
            "atomic_completeness_reason": ATOMIC_SLOT_MATCHING_EXPERIMENTAL,
            "required_slots": list(required_slots),
            "synthesis_obligations": (
                list(synthesis_obligations)
                if synthesis_obligations is not None
                else []
            ),
            "response_constraints": (
                list(response_constraints)
                if response_constraints is not None
                else []
            ),
            "comparison_plan": comparison_plan,
            "slot_plan_status": slot_plan_status,
            "slot_plan_source": slot_plan_source,
            "slot_plan_confidence": slot_plan_confidence,
            "slot_plan_fallback_reason": slot_plan_fallback_reason,
            "truncated_requirement_count": truncated_requirement_count,
        }
    )
    return validate_active_atomic_contract(overlaid)


class _UnauthorizedSourceExpansion(ValueError):
    """Planner attempted to add a source outside the authorized scope."""


class PlannerProviderInvocationError(RuntimeError):
    """The admitted provider attempt did not yield a usable response."""


class PlannerProviderEmptyResponseError(ValueError):
    """The provider completed but did not return planner content."""


class PlannerResponseDecodeError(ValueError):
    """The provider response was not valid JSON."""


class PlannerSchemaValidationError(ValueError):
    """The decoded response did not meet the planner transport schema."""


class PlannerSemanticValidationError(ValueError):
    """The schema-valid decision violated planner semantic constraints."""


class QuestionContractPlanner:
    """Plan answer-free atomic v2 contracts over an immutable base contract."""

    def __init__(self, *, llm_invoker: LlmInvoker | None = None) -> None:
        self._llm_invoker = llm_invoker

    @classmethod
    def prepare(
        cls,
        *,
        question: str,
        base_contract: QueryContract,
        decomposition: QuestionDecomposition | None = None,
    ) -> AtomicContractPreparation:
        """Analyze question deterministically to determine if semantic planning is needed."""
        decomp = (
            decomposition
            if decomposition is not None
            else decompose_question(question)
        )
        is_comparison_candidate = (
            base_contract.route == "bounded_compare"
            or len(decomp.comparison_subjects) >= 2
            or any(o.kind == "comparison" for o in decomp.synthesis_obligations)
            or _contains_any(
                question.casefold(),
                ("compare", "versus", " vs ", "比較", "是否優於"),
            )
        )

        deterministic_comparison_usable = False
        if is_comparison_candidate:
            subjects = decomp.comparison_subjects
            if 2 <= len(subjects) <= 4:
                subject_slot_mapping: list[list[int]] = []
                for subject_name in subjects:
                    mapped = [
                        idx
                        for idx, req in enumerate(decomp.requirements)
                        if subject_name.casefold() in req.text.casefold()
                        or subject_name in req.entity_ids
                    ]
                    subject_slot_mapping.append(mapped)
                if all(len(mapped) >= 1 for mapped in subject_slot_mapping):
                    deterministic_comparison_usable = True

        semantic_planning_requested = False
        if decomp.requires_semantic_planning:
            semantic_planning_requested = True
        elif decomp.confidence == "low":
            semantic_planning_requested = True
        elif (
            decomp.truncated_requirement_count > 0
            or decomp.truncated_constraint_count > 0
            or decomp.truncated_synthesis_count > 0
        ):
            semantic_planning_requested = True
        elif not (1 <= len(decomp.requirements) <= 8):
            semantic_planning_requested = True
        elif is_comparison_candidate and not deterministic_comparison_usable:
            semantic_planning_requested = True

        return AtomicContractPreparation(
            decomposition=decomp,
            semantic_planning_requested=semantic_planning_requested,
            comparison_candidate=is_comparison_candidate,
        )

    async def plan(
        self,
        *,
        question: str,
        base_contract: QueryContract,
        preparation: AtomicContractPreparation | None = None,
        allow_semantic_planning: bool = True,
    ) -> AtomicContractPlanningOutcome:
        """Plan atomic contract requirements using deterministic or budgeted semantic planning."""
        start_time = time.perf_counter()
        prep = (
            preparation
            if preparation is not None
            else self.prepare(question=question, base_contract=base_contract)
        )

        if prep.semantic_planning_requested:
            if not allow_semantic_planning:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="semantic_planning_not_admitted",
                    planner_call_count=0,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
            if self._llm_invoker is None:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="planner_unavailable",
                    planner_call_count=0,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )

            requested_source_names = (
                list(base_contract.resolved_source_scope.requested_source_names)
                if base_contract.resolved_source_scope
                else []
            )
            prompt = PromptRegistry(_PROMPT_PATH).format(
                _PROMPT_KEY,
                question=question,
                authorized_source_names=json.dumps(
                    requested_source_names, ensure_ascii=False
                ),
            )
            try:
                try:
                    response = await self._llm_invoker.invoke(
                        phase="contract_planning",
                        purpose="atomic_contract_planning",
                        messages=[{"role": "user", "content": prompt}],
                    )
                except (BudgetExceededError, TimeoutError):
                    raise
                except Exception as error:
                    raise PlannerProviderInvocationError from error
                decision = _parse_decision(response)
                try:
                    _validate_decision_scope(
                        decision,
                        allowed_source_names=set(requested_source_names),
                    )
                    _validate_decision_indexes(decision)
                    _validate_answer_free(
                        decision,
                        question=question,
                        authorized_source_names=requested_source_names,
                    )

                    slots = _build_slots_from_decision(
                        decision,
                        scope=base_contract.resolved_source_scope,
                    )
                    obligations = _build_obligations_from_decision(decision)
                    constraints = _build_constraints_from_decision(decision)
                    comparison_plan = _build_comparison_from_decision(
                        decision.comparison
                    )
                except (_UnauthorizedSourceExpansion, PlannerSemanticValidationError):
                    raise
                except (TypeError, ValueError, ValidationError) as error:
                    raise PlannerSemanticValidationError from error

                contract = apply_atomic_contract_overlay(
                    base_contract,
                    required_slots=slots,
                    synthesis_obligations=obligations,
                    response_constraints=constraints,
                    comparison_plan=comparison_plan,
                    slot_plan_status="complete",
                    slot_plan_source="llm_planner",
                    slot_plan_confidence=_confidence_label(decision.confidence),
                    slot_plan_fallback_reason=None,
                    truncated_requirement_count=None,
                )
                return AtomicContractPlanningOutcome(
                    contract=contract,
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    planner_diagnostics=_planner_diagnostics(
                        contract=contract,
                        outcome="planned",
                        provider_response_received=True,
                    ),
                )
            except TimeoutError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="planner_timeout",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="provider_invocation",
                    failure_code="provider_attempt_failed",
                )
            except BudgetExceededError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="planner_budget_rejected",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="budget_rejected",
                    failure_code="budget_rejected",
                )
            except PlannerProviderInvocationError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="invalid_planner_output",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="provider_invocation",
                    failure_code="provider_attempt_failed",
                )
            except PlannerProviderEmptyResponseError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="invalid_planner_output",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="provider_empty_response",
                    failure_code="empty_response",
                    provider_response_received=True,
                )
            except PlannerResponseDecodeError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="invalid_planner_output",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="response_decode",
                    failure_code="invalid_json",
                    provider_response_received=True,
                )
            except PlannerSchemaValidationError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="invalid_planner_output",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="schema_validation",
                    failure_code="pydantic_validation_failed",
                    provider_response_received=True,
                )
            except PlannerSemanticValidationError:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="invalid_planner_output",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="semantic_validation",
                    failure_code="planner_semantic_rejection",
                    provider_response_received=True,
                )
            except _UnauthorizedSourceExpansion:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="unauthorized_source_expansion",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    failure_stage="semantic_validation",
                    failure_code="planner_semantic_rejection",
                    provider_response_received=True,
                )
            except Exception:
                return _safe_fallback_outcome(
                    base_contract=base_contract,
                    fallback_reason="invalid_planner_output",
                    planner_call_count=1,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    provider_response_received=True,
                )

        # Deterministic path
        try:
            slots = _build_slots_from_decomposition(
                prep.decomposition.requirements,
                scope=base_contract.resolved_source_scope,
            )
            if not (1 <= len(slots) <= 8):
                raise ValueError("deterministic slots must be 1 to 8")

            obligations = _build_obligations_from_decomposition(
                prep.decomposition.synthesis_obligations,
                slot_count=len(slots),
            )
            constraints = _build_constraints_from_decomposition(
                prep.decomposition.response_constraints
            )
            comparison_plan = None
            if prep.comparison_candidate:
                comparison_plan = _build_deterministic_comparison_plan(
                    prep.decomposition,
                    slots=slots,
                )

            contract = apply_atomic_contract_overlay(
                base_contract,
                required_slots=slots,
                synthesis_obligations=obligations,
                response_constraints=constraints,
                comparison_plan=comparison_plan,
                slot_plan_status="complete",
                slot_plan_source="deterministic",
                slot_plan_confidence=prep.decomposition.confidence,
                slot_plan_fallback_reason=None,
                truncated_requirement_count=(
                    prep.decomposition.truncated_requirement_count or None
                ),
            )
            return AtomicContractPlanningOutcome(
                contract=contract,
                planner_call_count=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                planner_diagnostics=_planner_diagnostics(
                    contract=contract,
                    outcome="deterministic",
                    provider_response_received=False,
                ),
            )
        except Exception:
            return _safe_fallback_outcome(
                base_contract=base_contract,
                fallback_reason="deterministic_unusable",
                planner_call_count=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                failure_code="deterministic_unusable",
            )


def _safe_fallback_outcome(
    *,
    base_contract: QueryContract,
    fallback_reason: str,
    planner_call_count: Literal[0, 1],
    latency_ms: float,
    failure_stage: str | None = None,
    failure_code: str | None = None,
    provider_response_received: bool = False,
) -> AtomicContractPlanningOutcome:
    requested_names = (
        list(base_contract.resolved_source_scope.requested_source_names)
        if base_contract.resolved_source_scope
        else []
    )
    authorized_ids = (
        list(base_contract.resolved_source_scope.authorized_doc_ids)
        if base_contract.resolved_source_scope
        else []
    )
    slot = RequiredSlot(
        slot_id="S1",
        description="Resolve the complete source-bound requirement in the original question.",
        source_name_hints=requested_names,
        authorized_source_doc_ids=authorized_ids,
        locator_hints=[],
        expected_answer_type="text",
        depends_on_slot_ids=[],
        visual_policy="never",
    )
    contract = apply_atomic_contract_overlay(
        base_contract,
        required_slots=[slot],
        synthesis_obligations=[],
        response_constraints=[],
        comparison_plan=None,
        slot_plan_status="degraded",
        slot_plan_source="safe_fallback",
        slot_plan_confidence="low",
        slot_plan_fallback_reason=fallback_reason,
        truncated_requirement_count=None,
    )
    return AtomicContractPlanningOutcome(
        contract=contract,
        planner_call_count=planner_call_count,
        latency_ms=latency_ms,
        planner_diagnostics=_planner_diagnostics(
            contract=contract,
            outcome="degraded",
            failure_stage=failure_stage,
            failure_code=failure_code,
            provider_response_received=provider_response_received,
        ),
    )


def _parse_decision(response: Any) -> _PlannerDecision:
    content = response
    if isinstance(response, dict) and "content" in response:
        content = response["content"]
    elif hasattr(response, "content"):
        content = response.content
    if not isinstance(content, str) or not content.strip():
        raise PlannerProviderEmptyResponseError
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise PlannerResponseDecodeError from error
    try:
        return _PlannerDecision.model_validate(decoded)
    except ValidationError as error:
        raise PlannerSchemaValidationError from error


def _planner_diagnostics(
    *,
    contract: QueryContract,
    outcome: Literal["deterministic", "planned", "degraded"],
    provider_response_received: bool,
    failure_stage: str | None = None,
    failure_code: str | None = None,
) -> AtomicPlannerDiagnostics:
    """Return a bounded projection that never includes provider exception text."""
    return AtomicPlannerDiagnostics(
        outcome=outcome,
        failure_stage=failure_stage,
        failure_code=failure_code,
        provider_response_received=provider_response_received,
        retrieval_query_strategy=(
            "safe_fallback_original_question"
            if outcome == "degraded"
            else "atomic_slots"
        ),
        compiled_retrieval_task_count=_compiled_retrieval_task_count(contract),
    )


def _compiled_retrieval_task_count(contract: QueryContract) -> int:
    """Measure the existing pure compiler without changing its task plan."""
    try:
        return len(
            compile_retrieval_tasks(
                question="planner diagnostics",
                query_id="planner-diagnostics",
                contract=contract,
            ).tasks
        )
    except ValueError:
        return 0


def _validate_decision_scope(
    decision: _PlannerDecision,
    *,
    allowed_source_names: set[str],
) -> None:
    if allowed_source_names:
        for req in decision.evidence_requirements:
            if req.source_name_hints and not set(req.source_name_hints) <= allowed_source_names:
                raise _UnauthorizedSourceExpansion("unauthorized source name")


def _validate_decision_indexes(decision: _PlannerDecision) -> None:
    req_count = len(decision.evidence_requirements)
    if not (1 <= req_count <= 8):
        raise ValueError(f"evidence requirements must be 1 to 8, got {req_count}")

    for idx, req in enumerate(decision.evidence_requirements):
        for dep in req.depends_on_requirement_indexes:
            if not (0 <= dep < idx):
                raise ValueError(
                    f"requirement index {idx} has invalid dependency index {dep}"
                )

    for ob_idx, ob in enumerate(decision.synthesis_obligations):
        if not ob.depends_on_requirement_indexes:
            raise ValueError(
                f"synthesis obligation {ob_idx} must specify at least one requirement dependency"
            )
        for dep in ob.depends_on_requirement_indexes:
            if not (0 <= dep < req_count):
                raise ValueError(
                    f"synthesis obligation {ob_idx} references invalid requirement index {dep}"
                )

    if decision.comparison is not None:
        sub_count = len(decision.comparison.subjects)
        if not (2 <= sub_count <= 4):
            raise ValueError(f"comparison must have 2 to 4 subjects, got {sub_count}")
        subject_ids = [s.subject_id for s in decision.comparison.subjects]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("comparison subject IDs must be unique")
        for sub in decision.comparison.subjects:
            if not sub.evidence_requirement_indexes:
                raise ValueError(
                    f"comparison subject {sub.subject_id} must have evidence requirement indexes"
                )
            for dep in sub.evidence_requirement_indexes:
                if not (0 <= dep < req_count):
                    raise ValueError(
                        f"comparison subject {sub.subject_id} references invalid requirement index {dep}"
                    )


def _validate_answer_free(
    decision: _PlannerDecision,
    *,
    question: str,
    authorized_source_names: list[str] | None = None,
) -> None:
    question_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
    if authorized_source_names:
        for name in authorized_source_names:
            question_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", name))

    for req in decision.evidence_requirements:
        for locator in req.locator_hints:
            if not _valid_locator_hint(locator):
                raise ValueError(f"invalid locator hint: {locator}")
            question_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", locator))

    planner_texts: list[str] = []
    for req in decision.evidence_requirements:
        planner_texts.append(req.description)
    for ob in decision.synthesis_obligations:
        planner_texts.append(ob.description)
    for con in decision.response_constraints:
        planner_texts.append(con.description)
    if decision.comparison is not None:
        if decision.comparison.qualification:
            planner_texts.append(decision.comparison.qualification)
        for dim in decision.comparison.dimensions:
            planner_texts.append(dim)
        for sub in decision.comparison.subjects:
            planner_texts.append(sub.display_name)
            planner_texts.append(sub.retrieval_query)

    for text in planner_texts:
        authored_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        if not authored_numbers <= question_numbers:
            raise ValueError("planner text contains an answer-like value")


def _valid_locator_hint(value: str) -> bool:
    """Admit only bounded source-location descriptions, never free-form values."""
    normalized = value.strip().casefold()
    if not normalized or len(normalized) > 120:
        return False
    if normalized.startswith("section"):
        return canonical_structured_locator(value) is not None
    return bool(
        re.match(
            r"^(figure|fig\.?|table|appendix|formula|equation|theorem|page|section)\b",
            normalized,
        )
        or any(
            term in normalized
            for term in ("source passage", "regional impurity equation", "matrix")
        )
    )


def _confidence_label(score: float) -> Literal["high", "medium", "low"]:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _promote_source_hints(
    hints: list[str],
    scope: ResolvedSourceScope | None,
) -> tuple[list[str], list[str]]:
    if scope is None:
        return list(hints), []
    requested = scope.requested_source_names
    valid_hints = [h for h in hints if h in requested] or list(requested)
    doc_ids: list[str] = []
    if scope.source_name_to_doc_ids:
        for name in valid_hints:
            for doc_id in scope.source_name_to_doc_ids.get(name, ()):
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
    if not doc_ids:
        doc_ids = list(scope.authorized_doc_ids)
    return valid_hints, doc_ids


def _build_slots_from_decision(
    decision: _PlannerDecision,
    *,
    scope: ResolvedSourceScope | None,
) -> list[RequiredSlot]:
    slots: list[RequiredSlot] = []
    for idx, req in enumerate(decision.evidence_requirements, 1):
        source_names, doc_ids = _promote_source_hints(req.source_name_hints, scope)
        slots.append(
            RequiredSlot(
                slot_id=f"S{idx}",
                description=req.description,
                source_name_hints=source_names,
                authorized_source_doc_ids=doc_ids,
                locator_hints=list(req.locator_hints),
                expected_answer_type=req.expected_answer_type,
                depends_on_slot_ids=[
                    f"S{dep + 1}" for dep in req.depends_on_requirement_indexes
                ],
                visual_policy=req.visual_policy,
            )
        )
    return slots


def _build_obligations_from_decision(
    decision: _PlannerDecision,
) -> list[SynthesisObligation]:
    obligations: list[SynthesisObligation] = []
    for idx, ob in enumerate(decision.synthesis_obligations, 1):
        obligations.append(
            SynthesisObligation(
                obligation_id=f"O{idx}",
                kind=ob.kind,
                description=ob.description,
                depends_on_slot_ids=[
                    f"S{dep + 1}" for dep in ob.depends_on_requirement_indexes
                ],
            )
        )
    return obligations


def _build_constraints_from_decision(
    decision: _PlannerDecision,
) -> list[ResponseConstraint]:
    constraints: list[ResponseConstraint] = []
    for idx, con in enumerate(decision.response_constraints, 1):
        constraints.append(
            ResponseConstraint(
                constraint_id=f"C{idx}",
                kind=con.kind,
                description=con.description,
            )
        )
    return constraints


def _build_comparison_from_decision(
    comparison: _PlannerComparison | None,
) -> ComparisonPlan | None:
    if comparison is None:
        return None
    subjects: list[ComparisonSubject] = []
    for sub in comparison.subjects:
        subjects.append(
            ComparisonSubject(
                subject_id=sub.subject_id,
                display_name=sub.display_name,
                aliases=list(sub.aliases),
                retrieval_query=sub.retrieval_query,
                evidence_slot_ids=[
                    f"S{dep + 1}" for dep in sub.evidence_requirement_indexes
                ],
            )
        )
    return ComparisonPlan(
        subjects=subjects,
        dimensions=list(comparison.dimensions),
        qualification=comparison.qualification,
    )


def _build_slots_from_decomposition(
    requirements: tuple[DecomposedRequirement, ...],
    *,
    scope: ResolvedSourceScope | None,
) -> list[RequiredSlot]:
    slots: list[RequiredSlot] = []
    requested_names = list(scope.requested_source_names) if scope else []
    for idx, req in enumerate(requirements[:8], 1):
        hints = _source_hints(req.text, requested_names)
        source_names, doc_ids = _promote_source_hints(hints, scope)
        locators = _exact_locators(req.text)
        visual_policy: VisualPolicy = (
            "preferred"
            if any(loc.casefold().startswith(("figure", "table")) for loc in locators)
            or bool(re.search(r"\b(?:figure|table)\b", req.text, re.I))
            else "never"
        )
        slots.append(
            RequiredSlot(
                slot_id=f"S{idx}",
                description=req.text,
                source_name_hints=source_names,
                authorized_source_doc_ids=doc_ids,
                locator_hints=locators,
                expected_answer_type=_answer_type(req.text),
                depends_on_slot_ids=[],
                visual_policy=visual_policy,
            )
        )
    return slots


def _build_obligations_from_decomposition(
    obligations: tuple[Any, ...],
    *,
    slot_count: int,
) -> list[SynthesisObligation]:
    result: list[SynthesisObligation] = []
    all_slot_ids = [f"S{i}" for i in range(1, slot_count + 1)]
    for idx, ob in enumerate(obligations[:8], 1):
        deps = [
            f"S{dep + 1}"
            for dep in getattr(ob, "depends_on_requirement_indexes", ())
            if 0 <= dep < slot_count
        ]
        result.append(
            SynthesisObligation(
                obligation_id=f"O{idx}",
                kind=getattr(ob, "kind", "comparison"),
                description=getattr(ob, "text", str(ob)),
                depends_on_slot_ids=deps or all_slot_ids,
            )
        )
    return result


def _build_constraints_from_decomposition(
    constraints: tuple[Any, ...],
) -> list[ResponseConstraint]:
    result: list[ResponseConstraint] = []
    for idx, con in enumerate(constraints[:8], 1):
        result.append(
            ResponseConstraint(
                constraint_id=f"C{idx}",
                kind=getattr(con, "kind", "output_format"),
                description=getattr(con, "text", str(con)),
            )
        )
    return result


def _build_deterministic_comparison_plan(
    decomposition: QuestionDecomposition,
    *,
    slots: list[RequiredSlot],
) -> ComparisonPlan | None:
    subjects_raw = decomposition.comparison_subjects
    if not (2 <= len(subjects_raw) <= 4):
        return None
    subjects: list[ComparisonSubject] = []
    for subject_name in subjects_raw:
        safe_id = re.sub(
            r"[^a-z0-9_-]", "", subject_name.strip().casefold().replace(" ", "_")
        ) or "subject"
        display_name = subject_name.strip()
        mapped_slot_ids = [
            slot.slot_id
            for slot in slots
            if display_name.casefold() in slot.description.casefold()
        ]
        if not mapped_slot_ids:
            mapped_slot_ids = [slots[0].slot_id]
        mapped_req_texts = [
            s.description for s in slots if s.slot_id in mapped_slot_ids
        ]
        retrieval_query = f"{display_name} {' '.join(mapped_req_texts)}".strip()
        subjects.append(
            ComparisonSubject(
                subject_id=safe_id,
                display_name=display_name,
                aliases=[],
                retrieval_query=retrieval_query[:512],
                evidence_slot_ids=mapped_slot_ids,
            )
        )
    dimensions = [
        ob.description
        for ob in _build_obligations_from_decomposition(
            decomposition.synthesis_obligations, slot_count=len(slots)
        )
        if ob.kind == "comparison"
    ]
    return ComparisonPlan(subjects=subjects, dimensions=dimensions)


def _exact_locators(text: str) -> list[str]:
    locators: list[str] = []
    for match in _EXACT_LOCATOR_PATTERN.finditer(text):
        locator = (
            f"{match.group(1).title().replace('Fig.', 'Figure')} {match.group(2)}"
        )
        if canonical_structured_locator(locator) is None:
            continue
        if match.group(1).casefold() == "section":
            if re.match(r"\s+[A-Za-z]", text[match.end() :]):
                continue
        locators.append(locator)
    return list(dict.fromkeys(locators))


def _source_hints(text: str, source_names: list[str]) -> list[str]:
    matched = [
        name
        for name in source_names
        if Path(name).stem.casefold() in text.casefold()
        or text.casefold() in Path(name).stem.casefold()
    ]
    return matched or source_names


def _answer_type(text: str) -> ExpectedAnswerType:
    normalized = text.casefold()
    if any(term in normalized for term in ("equation", "formula")):
        return "equation"
    if any(term in normalized for term in ("meaning", "define", "what is ")):
        return "definition"
    if any(term in normalized for term in ("compare", "which")):
        return "comparison"
    if any(term in normalized for term in ("why", "reason", "explain")):
        return "explanation"
    if any(term in normalized for term in ("dice", "score", "value", "how many")):
        return "number"
    return "text"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.casefold() in text for term in terms)


__all__ = [
    "AtomicContractPlanningOutcome",
    "AtomicContractPreparation",
    "QuestionContractPlanner",
    "apply_atomic_contract_overlay",
    "atomic_contract_planner_response_schema",
]
