"""Question-only deterministic and budgeted atomic contract planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data_base.agentic_v9.contract_planner import (
    AtomicContractPlanningOutcome,
    AtomicContractPreparation,
    QuestionContractPlanner,
    apply_atomic_contract_overlay,
    atomic_contract_planner_response_schema,
)
from data_base.agentic_v9.requirement_decomposition import (
    DecomposedRequirement,
    QuestionDecomposition,
)
from data_base.agentic_v9.route_planner import RoutePlanner
from data_base.agentic_v9.schemas import (
    AgenticV9Route,
    BudgetExceededError,
    ComparisonPlan,
    ComparisonSubject,
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
    ResponseConstraint,
    RouteDecision,
    SynthesisObligation,
)

QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "golden"
    / "agentic_v9_questions_v2.json"
)


def _questions() -> dict[str, dict]:
    rows = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    return {row["id"]: row for row in rows}


class _MockInvoker:
    def __init__(
        self,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _make_base_contract(
    *,
    route: AgenticV9Route = "single_lookup",
    question: str = "What is reported in Table 1?",
    source_names: list[str] | None = None,
    source_doc_ids: list[str] | None = None,
    source_mapping: dict[str, list[str]] | None = None,
    max_llm_calls: int = 3,
    evidence_extraction_required: bool = False,
) -> QueryContract:
    names = source_names or ["paper.pdf"]
    doc_ids = source_doc_ids or ["doc-1"]
    mapping = source_mapping or {names[0]: [doc_ids[0]]}
    return QueryContract(
        contract_version="1",
        route=route,
        intent=f"Locate one source-bound fact: {question}",
        required_slots=[
            RequiredSlot(
                slot_id="base-slot-1",
                description="Resolve the base requirement.",
                source_name_hints=names,
                authorized_source_doc_ids=doc_ids,
            )
        ],
        resolved_source_scope=ResolvedSourceScope(
            requested_source_names=names,
            resolved_doc_ids=doc_ids,
            authorized_doc_ids=doc_ids,
            source_name_to_doc_ids=mapping,
        ),
        route_decision=RouteDecision(
            selected_route=route,
            decision_source="deterministic",
            route_reason="Deterministic test route.",
            confidence=1.0,
        ),
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=max_llm_calls,
        runtime_token_budget=30000,
        evidence_extraction_required=evidence_extraction_required,
    )


def test_prepare_returns_pure_deterministic_preparation() -> None:
    base_contract = _make_base_contract(route="single_lookup")
    preparation = QuestionContractPlanner.prepare(
        question="Using paper.pdf: 1. report Table 1 score; 2. report Table 2 score.",
        base_contract=base_contract,
    )
    assert isinstance(preparation, AtomicContractPreparation)
    assert isinstance(preparation.decomposition, QuestionDecomposition)
    assert preparation.semantic_planning_requested is False
    assert preparation.comparison_candidate is False


def test_prepare_flags_semantic_planning_on_low_confidence_or_ambiguity() -> None:
    base_contract = _make_base_contract(route="single_lookup")
    custom_decomp = QuestionDecomposition(
        requirements=(
            DecomposedRequirement(
                text="Unclear requirement.",
                method="fallback",
                confidence="low",
            ),
        ),
        confidence="low",
        semantic_planning_reasons=("unclear_clauses",),
    )
    preparation = QuestionContractPlanner.prepare(
        question="Unclear question.",
        base_contract=base_contract,
        decomposition=custom_decomp,
    )
    assert preparation.semantic_planning_requested is True


def test_prepare_identifies_comparison_candidate() -> None:
    base_contract = _make_base_contract(route="bounded_compare")
    preparation = QuestionContractPlanner.prepare(
        question="Compare Model A and Model B Dice scores.",
        base_contract=base_contract,
    )
    assert preparation.comparison_candidate is True


@pytest.mark.asyncio
async def test_plan_accepted_deterministic_uses_zero_calls_and_deterministic_source() -> (
    None
):
    question = "Using paper.pdf: 1. report Table 1 score; 2. report Table 2 score."
    base_contract = _make_base_contract(
        question=question,
        source_names=["paper.pdf"],
        source_doc_ids=["doc-1"],
    )
    preparation = QuestionContractPlanner.prepare(
        question=question,
        base_contract=base_contract,
    )
    invoker = _MockInvoker()
    planner = QuestionContractPlanner(llm_invoker=invoker)

    outcome = await planner.plan(
        question=question,
        base_contract=base_contract,
        preparation=preparation,
        allow_semantic_planning=True,
    )

    assert isinstance(outcome, AtomicContractPlanningOutcome)
    assert outcome.planner_call_count == 0
    assert len(invoker.calls) == 0
    assert outcome.contract.contract_version == "2"
    assert outcome.contract.slot_plan_source == "deterministic"
    assert outcome.contract.slot_plan_status == "complete"
    assert len(outcome.contract.required_slots) >= 2
    assert outcome.planner_diagnostics.model_dump() == {
        "outcome": "deterministic",
        "failure_stage": None,
        "failure_code": None,
        "provider_response_received": False,
        "retrieval_query_strategy": "atomic_slots",
        "compiled_retrieval_task_count": len(outcome.contract.required_slots),
    }
    assert [s.slot_id for s in outcome.contract.required_slots] == [
        f"S{i}" for i in range(1, len(outcome.contract.required_slots) + 1)
    ]


@pytest.mark.asyncio
async def test_plan_preserves_base_contract_immutability() -> None:
    question = "From paper.pdf, report the value in Table 1."
    base_contract = _make_base_contract(
        route="exact_structured",
        question=question,
        max_llm_calls=5,
        evidence_extraction_required=False,
    )
    outcome = await QuestionContractPlanner().plan(
        question=question,
        base_contract=base_contract,
    )

    assert outcome.contract.contract_version == "2"
    assert outcome.contract.route == base_contract.route
    assert outcome.contract.route_decision == base_contract.route_decision
    assert outcome.contract.max_llm_calls == base_contract.max_llm_calls
    assert outcome.contract.evidence_extraction_required == (
        base_contract.evidence_extraction_required
    )
    assert outcome.contract.intent == base_contract.intent
    assert outcome.contract.max_retrieval_rounds == (
        base_contract.max_retrieval_rounds
    )
    assert outcome.contract.max_repair_rounds == base_contract.max_repair_rounds
    assert outcome.contract.runtime_token_budget == (
        base_contract.runtime_token_budget
    )
    assert outcome.contract.resolved_source_scope == (
        base_contract.resolved_source_scope
    )


def test_apply_atomic_contract_overlay_only_modifies_allow_listed_fields() -> None:
    base = _make_base_contract()
    slot = RequiredSlot(
        slot_id="S1",
        description="Requirement 1",
        source_name_hints=["paper.pdf"],
        authorized_source_doc_ids=["doc-1"],
    )
    obligation = SynthesisObligation(
        obligation_id="O1",
        kind="comparison",
        description="Compare S1",
        depends_on_slot_ids=["S1"],
    )
    constraint = ResponseConstraint(
        constraint_id="C1",
        kind="output_format",
        description="Strict format",
    )
    comparison_plan = ComparisonPlan(
        subjects=[
            ComparisonSubject(
                subject_id="model_a",
                display_name="Model A",
                retrieval_query="Model A query",
                evidence_slot_ids=["S1"],
            ),
            ComparisonSubject(
                subject_id="model_b",
                display_name="Model B",
                retrieval_query="Model B query",
                evidence_slot_ids=["S1"],
            ),
        ],
        dimensions=["accuracy"],
    )

    overlaid = apply_atomic_contract_overlay(
        base,
        required_slots=[slot],
        synthesis_obligations=[obligation],
        response_constraints=[constraint],
        comparison_plan=comparison_plan,
        slot_plan_status="complete",
        slot_plan_source="llm_planner",
        slot_plan_confidence="high",
        slot_plan_fallback_reason=None,
        truncated_requirement_count=0,
    )

    assert overlaid.contract_version == "2"
    assert overlaid.required_slots == [slot]
    assert overlaid.synthesis_obligations == [obligation]
    assert overlaid.response_constraints == [constraint]
    assert overlaid.comparison_plan == comparison_plan
    assert overlaid.slot_plan_status == "complete"
    assert overlaid.slot_plan_source == "llm_planner"
    assert overlaid.slot_plan_confidence == "high"
    assert overlaid.slot_plan_fallback_reason is None
    assert overlaid.truncated_requirement_count == 0

    # Unchanged fields
    assert overlaid.route == base.route
    assert overlaid.route_decision == base.route_decision
    assert overlaid.intent == base.intent
    assert overlaid.max_llm_calls == base.max_llm_calls
    assert overlaid.evidence_extraction_required == (
        base.evidence_extraction_required
    )


@pytest.mark.asyncio
async def test_semantic_planning_low_confidence_invokes_llm_once_and_builds_combined_contract() -> (
    None
):
    provider_response = {
        "evidence_requirements": [
            {
                "description": "Retrieve Model A Dice score from Table 1.",
                "source_name_hints": ["paper.pdf"],
                "locator_hints": ["Table 1"],
                "expected_answer_type": "number",
                "depends_on_requirement_indexes": [],
                "visual_policy": "never",
            },
            {
                "description": "Retrieve Model B Dice score from Table 2.",
                "source_name_hints": ["paper.pdf"],
                "locator_hints": ["Table 2"],
                "expected_answer_type": "number",
                "depends_on_requirement_indexes": [],
                "visual_policy": "never",
            },
        ],
        "synthesis_obligations": [
            {
                "kind": "comparison",
                "description": "Compare Model A and Model B Dice scores.",
                "depends_on_requirement_indexes": [0, 1],
            }
        ],
        "response_constraints": [
            {
                "kind": "output_format",
                "description": "Report difference as a percentage.",
            }
        ],
        "comparison": {
            "subjects": [
                {
                    "subject_id": "model_a",
                    "display_name": "Model A",
                    "aliases": [],
                    "retrieval_query": "Model A Dice score Table 1",
                    "evidence_requirement_indexes": [0],
                },
                {
                    "subject_id": "model_b",
                    "display_name": "Model B",
                    "aliases": [],
                    "retrieval_query": "Model B Dice score Table 2",
                    "evidence_requirement_indexes": [1],
                },
            ],
            "dimensions": ["Dice score"],
            "qualification": None,
        },
        "confidence": 0.9,
    }

    invoker = _MockInvoker(response={"content": json.dumps(provider_response)})
    base_contract = _make_base_contract(
        route="bounded_compare",
        question="Compare Model A and Model B performance across Table 1 and Table 2.",
        source_names=["paper.pdf"],
        source_doc_ids=["doc-1"],
        source_mapping={"paper.pdf": ["doc-1"]},
    )
    decomp = QuestionDecomposition(
        requirements=(
            DecomposedRequirement(
                text="Compare Model A and Model B.",
                method="fallback",
                confidence="low",
            ),
        ),
        confidence="low",
        semantic_planning_reasons=("unclear_subquestions",),
    )
    preparation = AtomicContractPreparation(
        decomposition=decomp,
        semantic_planning_requested=True,
        comparison_candidate=True,
    )

    planner = QuestionContractPlanner(llm_invoker=invoker)
    outcome = await planner.plan(
        question="Compare Model A and Model B performance across Table 1 and Table 2.",
        base_contract=base_contract,
        preparation=preparation,
        allow_semantic_planning=True,
    )

    assert outcome.planner_call_count == 1
    assert len(invoker.calls) == 1
    assert outcome.contract.slot_plan_source == "llm_planner"
    assert outcome.contract.slot_plan_status == "complete"
    assert outcome.contract.slot_plan_confidence == "high"

    # Assigned slot IDs
    assert [s.slot_id for s in outcome.contract.required_slots] == ["S1", "S2"]
    assert outcome.contract.required_slots[0].authorized_source_doc_ids == ["doc-1"]
    assert outcome.contract.required_slots[0].locator_hints == ["Table 1"]

    # Assigned obligation IDs
    assert len(outcome.contract.synthesis_obligations) == 1
    assert outcome.contract.synthesis_obligations[0].obligation_id == "O1"
    assert outcome.contract.synthesis_obligations[0].depends_on_slot_ids == [
        "S1",
        "S2",
    ]

    # Assigned constraint IDs
    assert len(outcome.contract.response_constraints) == 1
    assert outcome.contract.response_constraints[0].constraint_id == "C1"

    # Comparison plan with slot bindings
    assert outcome.contract.comparison_plan is not None
    assert len(outcome.contract.comparison_plan.subjects) == 2
    assert outcome.contract.comparison_plan.subjects[0].evidence_slot_ids == ["S1"]
    assert outcome.contract.comparison_plan.subjects[1].evidence_slot_ids == ["S2"]


def test_atomic_contract_planner_response_schema_forbids_route_and_doc_ids() -> None:
    schema = atomic_contract_planner_response_schema()
    assert isinstance(schema, dict)
    properties = schema.get("properties", {})
    assert "selected_route" not in properties
    assert "route" not in properties
    assert "authorized_doc_ids" not in properties
    assert "evidence_requirements" in properties
    assert "synthesis_obligations" in properties
    assert "response_constraints" in properties
    assert "comparison" in properties


@pytest.mark.asyncio
async def test_provider_response_promotes_source_name_hints_authoritatively() -> (
    None
):
    provider_response = {
        "evidence_requirements": [
            {
                "description": "Retrieve Table 1 from nnMamba.pdf.",
                "source_name_hints": ["nnMamba.pdf"],
                "locator_hints": ["Table 1"],
                "expected_answer_type": "text",
                "depends_on_requirement_indexes": [],
                "visual_policy": "never",
            }
        ],
        "synthesis_obligations": [],
        "response_constraints": [],
        "comparison": None,
        "confidence": 0.85,
    }
    invoker = _MockInvoker(response={"content": json.dumps(provider_response)})
    base_contract = _make_base_contract(
        question="Check nnMamba.pdf.",
        source_names=["nnMamba.pdf", "Other.pdf"],
        source_doc_ids=["doc-a", "doc-z"],
        source_mapping={
            "nnMamba.pdf": ["doc-z"],
            "Other.pdf": ["doc-a"],
        },
    )
    decomp = QuestionDecomposition(
        requirements=(
            DecomposedRequirement(
                text="Check nnMamba.", method="fallback", confidence="low"
            ),
        ),
        confidence="low",
        semantic_planning_reasons=("unclear",),
    )
    preparation = AtomicContractPreparation(
        decomposition=decomp,
        semantic_planning_requested=True,
        comparison_candidate=False,
    )

    outcome = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Check nnMamba.pdf.",
        base_contract=base_contract,
        preparation=preparation,
        allow_semantic_planning=True,
    )

    assert outcome.contract.slot_plan_status == "complete"
    assert outcome.contract.required_slots[0].source_name_hints == ["nnMamba.pdf"]
    assert outcome.contract.required_slots[0].authorized_source_doc_ids == ["doc-z"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate_fn", "expected_fallback_reason", "call_count"),
    [
        (lambda p: None, "semantic_planning_not_admitted", 0),
        (lambda p: None, "planner_unavailable", 0),
        (lambda p: None, "planner_timeout", 1),
        (lambda p: None, "planner_budget_rejected", 1),
        (
            lambda p: p["evidence_requirements"][0].update(
                {"source_name_hints": ["unauthorized.pdf"]}
            ),
            "unauthorized_source_expansion",
            1,
        ),
        (
            lambda p: p["evidence_requirements"][0].update(
                {"description": "The score is 0.9079."}
            ),
            "invalid_planner_output",
            1,
        ),
        (
            lambda p: p["evidence_requirements"][0].update(
                {"depends_on_requirement_indexes": [99]}
            ),
            "invalid_planner_output",
            1,
        ),
        (
            lambda p: p["evidence_requirements"][0].update(
                {"locator_hints": ["invalid_prose_locator"]}
            ),
            "invalid_planner_output",
            1,
        ),
        (
            lambda p: p.update({"extra_field": "forbidden"}),
            "invalid_planner_output",
            1,
        ),
        (
            lambda p: p.update({"evidence_requirements": []}),
            "invalid_planner_output",
            1,
        ),
    ],
)
async def test_degraded_fallbacks_for_all_failure_modes(
    mutate_fn: Any,
    expected_fallback_reason: str,
    call_count: int,
) -> None:
    base_contract = _make_base_contract(
        source_names=["paper.pdf"],
        source_doc_ids=["doc-1"],
    )
    decomp = QuestionDecomposition(
        requirements=(
            DecomposedRequirement(
                text="Unclear.", method="fallback", confidence="low"
            ),
        ),
        confidence="low",
        semantic_planning_reasons=("unclear",),
    )
    preparation = AtomicContractPreparation(
        decomposition=decomp,
        semantic_planning_requested=True,
        comparison_candidate=False,
    )

    if expected_fallback_reason == "semantic_planning_not_admitted":
        outcome = await QuestionContractPlanner(llm_invoker=_MockInvoker()).plan(
            question="Unclear question.",
            base_contract=base_contract,
            preparation=preparation,
            allow_semantic_planning=False,
        )
    elif expected_fallback_reason == "planner_unavailable":
        outcome = await QuestionContractPlanner(llm_invoker=None).plan(
            question="Unclear question.",
            base_contract=base_contract,
            preparation=preparation,
            allow_semantic_planning=True,
        )
    elif expected_fallback_reason == "planner_timeout":
        outcome = await QuestionContractPlanner(
            llm_invoker=_MockInvoker(error=TimeoutError("slow"))
        ).plan(
            question="Unclear question.",
            base_contract=base_contract,
            preparation=preparation,
            allow_semantic_planning=True,
        )
    elif expected_fallback_reason == "planner_budget_rejected":
        outcome = await QuestionContractPlanner(
            llm_invoker=_MockInvoker(error=BudgetExceededError("exhausted"))
        ).plan(
            question="Unclear question.",
            base_contract=base_contract,
            preparation=preparation,
            allow_semantic_planning=True,
        )
    else:
        payload: dict[str, Any] = {
            "evidence_requirements": [
                {
                    "description": "Retrieve fact from paper.pdf.",
                    "source_name_hints": ["paper.pdf"],
                    "locator_hints": [],
                    "expected_answer_type": "text",
                    "depends_on_requirement_indexes": [],
                    "visual_policy": "never",
                }
            ],
            "synthesis_obligations": [],
            "response_constraints": [],
            "comparison": None,
            "confidence": 0.8,
        }
        mutate_fn(payload)
        outcome = await QuestionContractPlanner(
            llm_invoker=_MockInvoker(response={"content": json.dumps(payload)})
        ).plan(
            question="Unclear question.",
            base_contract=base_contract,
            preparation=preparation,
            allow_semantic_planning=True,
        )

    assert outcome.planner_call_count == call_count
    assert outcome.contract.slot_plan_status == "degraded"
    assert outcome.contract.slot_plan_source == "safe_fallback"
    assert outcome.contract.slot_plan_confidence == "low"
    assert outcome.contract.slot_plan_fallback_reason == expected_fallback_reason
    assert len(outcome.contract.required_slots) == 1
    assert outcome.contract.required_slots[0].slot_id == "S1"
    assert (
        outcome.contract.required_slots[0].description
        == "Resolve the complete source-bound requirement in the original question."
    )
    assert outcome.contract.required_slots[0].authorized_source_doc_ids == ["doc-1"]
    assert outcome.contract.synthesis_obligations == []
    assert outcome.contract.response_constraints == []
    assert outcome.contract.comparison_plan is None


@pytest.mark.asyncio
async def test_no_retry_after_invalid_output() -> None:
    invoker = _MockInvoker(response={"content": "not-valid-json"})
    base_contract = _make_base_contract()
    decomp = QuestionDecomposition(
        requirements=(
            DecomposedRequirement(
                text="Unclear.", method="fallback", confidence="low"
            ),
        ),
        confidence="low",
        semantic_planning_reasons=("unclear",),
    )
    preparation = AtomicContractPreparation(
        decomposition=decomp,
        semantic_planning_requested=True,
        comparison_candidate=False,
    )

    outcome = await QuestionContractPlanner(llm_invoker=invoker).plan(
        question="Unclear question.",
        base_contract=base_contract,
        preparation=preparation,
        allow_semantic_planning=True,
    )

    assert len(invoker.calls) == 1
    assert outcome.planner_call_count == 1
    assert outcome.contract.slot_plan_status == "degraded"
    assert outcome.contract.slot_plan_fallback_reason == "invalid_planner_output"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error", "expected_stage", "expected_code", "received"),
    [
        (None, RuntimeError("api_key=planner-provider-secret"), "provider_invocation", "provider_attempt_failed", False),
        ({"content": ""}, None, "provider_empty_response", "empty_response", True),
        ({"content": "{not json"}, None, "response_decode", "invalid_json", True),
        ({"content": "{}"}, None, "schema_validation", "pydantic_validation_failed", True),
        (
            {
                "content": json.dumps(
                    {
                        "evidence_requirements": [
                            {
                                "description": "The score is 0.9079.",
                                "source_name_hints": ["paper.pdf"],
                                "locator_hints": [],
                                "expected_answer_type": "text",
                                "depends_on_requirement_indexes": [],
                                "visual_policy": "never",
                            }
                        ],
                        "synthesis_obligations": [],
                        "response_constraints": [],
                        "comparison": None,
                        "confidence": 0.8,
                    }
                )
            },
            None,
            "semantic_validation",
            "planner_semantic_rejection",
            True,
        ),
    ],
)
async def test_planner_diagnostic_classifies_each_provider_failure_boundary(
    response: object,
    error: Exception | None,
    expected_stage: str,
    expected_code: str,
    received: bool,
) -> None:
    """Fails if a planner boundary is merged into a generic fallback."""
    decomp = QuestionDecomposition(
        requirements=(DecomposedRequirement(text="Unclear.", method="fallback", confidence="low"),),
        confidence="low",
        semantic_planning_reasons=("unclear",),
    )
    outcome = await QuestionContractPlanner(
        llm_invoker=_MockInvoker(response=response, error=error)
    ).plan(
        question="Unclear question.",
        base_contract=_make_base_contract(),
        preparation=AtomicContractPreparation(
            decomposition=decomp,
            semantic_planning_requested=True,
            comparison_candidate=False,
        ),
    )

    assert outcome.planner_diagnostics.model_dump() == {
        "outcome": "degraded",
        "failure_stage": expected_stage,
        "failure_code": expected_code,
        "provider_response_received": received,
        "retrieval_query_strategy": "safe_fallback_original_question",
        "compiled_retrieval_task_count": 1,
    }
    assert "planner-provider-secret" not in outcome.planner_diagnostics.model_dump_json()


@pytest.mark.asyncio
async def test_planner_diagnostic_reports_budget_rejection_without_provider_response() -> None:
    """Fails if rejected admission is reported as an invocation failure."""
    decomp = QuestionDecomposition(
        requirements=(DecomposedRequirement(text="Unclear.", method="fallback", confidence="low"),),
        confidence="low",
        semantic_planning_reasons=("unclear",),
    )
    outcome = await QuestionContractPlanner(
        llm_invoker=_MockInvoker(error=BudgetExceededError("budget detail"))
    ).plan(
        question="Unclear question.",
        base_contract=_make_base_contract(),
        preparation=AtomicContractPreparation(
            decomposition=decomp,
            semantic_planning_requested=True,
            comparison_candidate=False,
        ),
    )

    assert outcome.planner_diagnostics.model_dump() == {
        "outcome": "degraded",
        "failure_stage": "budget_rejected",
        "failure_code": "budget_rejected",
        "provider_response_received": False,
        "retrieval_query_strategy": "safe_fallback_original_question",
        "compiled_retrieval_task_count": 1,
    }


@pytest.mark.asyncio
async def test_deterministic_comparison_subject_binding() -> None:
    question = (
        "Compare Model A and Model B Dice score on Dataset 1. "
        "1. Report Model A Dice; 2. Report Model B Dice."
    )
    base_contract = _make_base_contract(
        route="bounded_compare",
        question=question,
    )
    decomp = QuestionDecomposition(
        requirements=(
            DecomposedRequirement(
                text="Report Model A Dice.",
                method="numbered",
                confidence="high",
                entity_ids=("Model A",),
            ),
            DecomposedRequirement(
                text="Report Model B Dice.",
                method="numbered",
                confidence="high",
                entity_ids=("Model B",),
            ),
        ),
        comparison_subjects=("Model A", "Model B"),
        confidence="high",
    )
    preparation = QuestionContractPlanner.prepare(
        question=question,
        base_contract=base_contract,
        decomposition=decomp,
    )
    assert preparation.comparison_candidate is True
    assert preparation.semantic_planning_requested is False

    outcome = await QuestionContractPlanner().plan(
        question=question,
        base_contract=base_contract,
        preparation=preparation,
    )

    assert outcome.contract.slot_plan_status == "complete"
    assert outcome.contract.comparison_plan is not None
    assert len(outcome.contract.comparison_plan.subjects) == 2
    assert outcome.contract.comparison_plan.subjects[0].evidence_slot_ids == ["S1"]
    assert outcome.contract.comparison_plan.subjects[1].evidence_slot_ids == ["S2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question_id",
    ["Q5", "Q7", "Q11", "Q14"],
)
async def test_formal_questions_use_bounded_experimental_answer_free_slots(
    question_id: str,
) -> None:
    case = _questions()[question_id]
    doc_ids = [f"doc-{index}" for index, _ in enumerate(case["source_docs"], 1)]
    mapping = {
        name: [doc_id]
        for name, doc_id in zip(case["source_docs"], doc_ids, strict=True)
    }

    base_contract = await RoutePlanner().plan(
        question=case["question"],
        resolved_source_scope=ResolvedSourceScope(
            requested_source_names=case["source_docs"],
            resolved_doc_ids=doc_ids,
            authorized_doc_ids=doc_ids,
            source_name_to_doc_ids=mapping,
        ),
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    preparation = QuestionContractPlanner.prepare(
        question=case["question"],
        base_contract=base_contract,
    )
    outcome = await QuestionContractPlanner().plan(
        question=case["question"],
        base_contract=base_contract,
        preparation=preparation,
    )

    contract = outcome.contract
    assert contract.contract_version == "2"
    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.atomic_completeness is None
    assert contract.slot_plan_status in {"complete", "degraded"}
    assert 1 <= len(contract.required_slots) <= 8
    assert [slot.slot_id for slot in contract.required_slots] == [
        f"S{index}" for index in range(1, len(contract.required_slots) + 1)
    ]
    assert all(slot.description.strip() for slot in contract.required_slots)
    assert all(slot.authorized_source_doc_ids for slot in contract.required_slots)


@pytest.mark.asyncio
async def test_production_contract_does_not_require_strict_evidence_extraction() -> (
    None
):
    case = _questions()["Q5"]
    doc_ids = [f"doc-{index}" for index, _ in enumerate(case["source_docs"], 1)]
    base_contract = await RoutePlanner().plan(
        question=case["question"],
        resolved_source_scope=ResolvedSourceScope(
            requested_source_names=case["source_docs"],
            resolved_doc_ids=doc_ids,
            authorized_doc_ids=doc_ids,
            source_name_to_doc_ids={
                name: [doc_id]
                for name, doc_id in zip(case["source_docs"], doc_ids, strict=True)
            },
        ),
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    outcome = await QuestionContractPlanner().plan(
        question=case["question"],
        base_contract=base_contract,
    )

    assert outcome.contract.evidence_extraction_required == (
        base_contract.evidence_extraction_required
    )


@pytest.mark.asyncio
async def test_q16_uses_generic_experimental_planning_without_benchmark_bundle() -> (
    None
):
    planner_source = (
        Path(__file__).resolve().parents[1]
        / "data_base"
        / "agentic_v9"
        / "contract_planner.py"
    ).read_text(encoding="utf-8")
    assert "q16_structured_bundle" not in planner_source
    assert "_known_question_slots" not in planner_source
    for benchmark_name in (
        "gepar3d",
        "segmentanybone",
        "weak-mamba-unet",
        "swinunetr",
    ):
        assert benchmark_name not in planner_source.casefold()

    case = _questions()["Q16"]
    doc_ids = ["gepar", "odes", "ukan"]
    mapping = {
        name: [doc_id]
        for name, doc_id in zip(case["source_docs"], doc_ids, strict=True)
    }
    base_contract = await RoutePlanner().plan(
        question=case["question"],
        resolved_source_scope=ResolvedSourceScope(
            requested_source_names=case["source_docs"],
            resolved_doc_ids=doc_ids,
            authorized_doc_ids=doc_ids,
            source_name_to_doc_ids=mapping,
        ),
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )

    outcome = await QuestionContractPlanner().plan(
        question=case["question"],
        base_contract=base_contract,
    )

    contract = outcome.contract
    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.atomic_completeness is None
    serialized = contract.model_dump_json()
    for forbidden in ("0.179", "0.4064", "0.9079", "0 ≤", "0 <="):
        assert forbidden not in serialized
    for forbidden in (
        "tooth 1 to tooth 32 penalty",
        "regional impurity equation",
        "|A^c(x,y)|",
    ):
        assert forbidden not in serialized
