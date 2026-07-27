"""Focused contracts for deterministic-first Agentic v9 route planning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_base.agentic_v9.budget_feasibility import (
    FeasibilityStatus,
    validate_post_contract_feasibility,
)
from data_base.agentic_v9.route_planner import RoutePlanner
from data_base.agentic_v9.retrieval_tasks import RetrievalTaskCompiler
from data_base.agentic_v9.schemas import ResolvedSourceScope


ROOT = Path(__file__).resolve().parents[1]
ROUTES_PATH = ROOT / "evaluation" / "golden" / "agentic_v9_route_regressions.json"
QUESTIONS_PATH = ROOT / "evaluation" / "golden" / "agentic_v9_questions_v2.json"


class _NeverInvoker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def invoke(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        raise AssertionError("deterministic route must not invoke the planner model")


class _PlannerInvoker:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def invoke(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def _scope() -> ResolvedSourceScope:
    return ResolvedSourceScope(
        requested_doc_ids=["doc-a", "doc-b", "doc-c"],
        authorized_doc_ids=["doc-a", "doc-b", "doc-c"],
    )


@pytest.mark.asyncio
async def test_legacy_multisource_scope_without_mapping_fails_closed() -> None:
    contract = await RoutePlanner().plan(
        question="From nnMamba.pdf, report the value in Table 2.",
        resolved_source_scope=ResolvedSourceScope(
            requested_source_names=["nnMamba.pdf", "Other.pdf"],
            authorized_doc_ids=["doc-a", "doc-z"],
        ),
    )

    assert contract.slot_plan_status == "degraded"
    assert (
        contract.route_decision.fallback_reason
        == "authoritative_source_mapping_missing"
    )
    assert all(
        slot.source_name_hints == ["nnMamba.pdf", "Other.pdf"]
        for slot in contract.required_slots
    )


@pytest.mark.asyncio
async def test_deterministic_regressions_emit_complete_retrieval_contracts() -> None:
    cases = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))["cases"]
    invoker = _NeverInvoker()
    planner = RoutePlanner(llm_invoker=invoker)

    for case in cases:
        contract = await planner.plan(
            question=case["question"],
            resolved_source_scope=_scope(),
        )

        assert contract.route == case["expected_route"]
        expected_graph_policy = {
            "single_lookup": "never",
            "bounded_compare": "never",
            "exact_structured": "locator_fallback",
            "multi_document_exact": "locator_fallback",
            "multi_hop": "locator_fallback",
            "graph_relational": "required_locator",
        }[case["expected_route"]]
        assert contract.graph_policy == expected_graph_policy
        assert contract.required_slots
        assert contract.locator_hints
        assert contract.resolved_source_scope == _scope()
        assert contract.max_retrieval_rounds >= 1
        assert contract.max_llm_calls >= 1
        assert contract.runtime_token_budget > 0
        assert "answer" not in contract.model_dump()

    assert invoker.calls == []


@pytest.mark.asyncio
async def test_generic_comparison_does_not_invent_a_qualification_round() -> None:
    contract = await RoutePlanner(llm_invoker=_NeverInvoker()).plan(
        question="SwinUNETR and nnU-Net: which performs better?",
        resolved_source_scope=_scope(),
    )

    plan = RetrievalTaskCompiler().compile(
        question="SwinUNETR and nnU-Net: which performs better?",
        query_id="R2",
        contract=contract,
    )

    assert contract.max_retrieval_rounds == 2
    assert [task.round_id for task in plan.tasks] == ["round-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected_entities"),
    [
        (
            "在 BraTS 類 3D 腦腫瘤分割場景，資料少且 GPU 資源受限時，SwinUNETR、MedNeXt、nnMamba 應如何取捨？",
            {"SwinUNETR", "MedNeXt", "nnMamba"},
        ),
        (
            "從 MedSAM、SAM-Med3D 到 MedSAM-2，3D 空間建模與 prompt 需求如何演進？",
            {"MedSAM", "SAM-Med3D", "MedSAM-2"},
        ),
    ],
)
async def test_named_model_questions_use_generic_fallback_without_benchmark_routes(
    question: str, expected_entities: set[str]
) -> None:
    planner = RoutePlanner()

    contract = await planner.plan(question=question, resolved_source_scope=_scope())

    assert expected_entities.issubset(contract.entities)
    assert contract.slot_semantics == "heuristic_experimental"
    assert contract.route_decision is not None
    assert "technical_entity_bundle" not in contract.route_decision.matched_rules
    assert 1 <= len(contract.required_slots) <= 8


@pytest.mark.asyncio
async def test_visual_and_graph_routes_reserve_their_required_provider_phases() -> None:
    planner = RoutePlanner(llm_invoker=_NeverInvoker())

    visual_contract = await planner.plan(
        question="What is the table score?",
        resolved_source_scope=_scope(),
    )
    graph_visual_contract = await planner.plan(
        question="What is the graph path in Figure 2?",
        resolved_source_scope=_scope(),
    )

    assert visual_contract.route == "exact_structured"
    assert visual_contract.visual_requested is True
    assert visual_contract.visual_required is False
    assert visual_contract.max_llm_calls == 3
    assert graph_visual_contract.route == "graph_relational"
    assert graph_visual_contract.visual_requested is True
    assert graph_visual_contract.visual_required is False
    assert graph_visual_contract.max_llm_calls == 4


@pytest.mark.asyncio
async def test_formal_question_set_fits_the_published_v9_preflight_envelope() -> None:
    """The UI's five-call admission envelope must admit every formal question.

    This includes one ambiguity-only route-plan call plus the resolved runtime
    phases, preventing a future route-budget reduction from making the setup
    preflight disagree with runtime admission again.
    """
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    planner = RoutePlanner()

    for case in cases:
        source_docs = case["source_docs"]
        contract = await planner.plan(
            question=case["question"],
            resolved_source_scope=ResolvedSourceScope(
                requested_doc_ids=source_docs,
                authorized_doc_ids=source_docs,
            ),
        )
        result = validate_post_contract_feasibility(
            contract=contract,
            setup_snapshot={"max_output_tokens": 8192, "thinking_mode": False},
            remaining_token_budget=50_000,
            remaining_llm_calls=5,
            route_plan_used=contract.strategy_tier == "budgeted_ambiguity",
        )

        assert result.status is FeasibilityStatus.FEASIBLE, (
            f"{case['id']}: {result.reason}"
        )


@pytest.mark.asyncio
async def test_only_ambiguous_question_uses_one_budgeted_route_plan_call() -> None:
    invoker = _PlannerInvoker(SimpleNamespace(content='{"route": "single_lookup"}'))
    planner = RoutePlanner(llm_invoker=invoker)

    contract = await planner.plan(
        question="Please help me understand this.",
        resolved_source_scope=_scope(),
    )

    assert contract.route == "single_lookup"
    assert len(invoker.calls) == 1
    assert invoker.calls[0]["phase"] == "contract_planning"
    assert invoker.calls[0]["purpose"] == "atomic_contract_planning"
    assert (
        contract.max_llm_calls == 4
    )  # route-plan + evidence extraction + visual reserve + final
    assert "answer" not in contract.model_dump()


@pytest.mark.asyncio
async def test_ambiguous_planner_output_with_an_answer_or_scope_is_rejected() -> None:
    invoker = _PlannerInvoker(
        {
            "content": '{"route":"bounded_compare","answer":"invented",'
            '"authorized_doc_ids":["outside"]}'
        }
    )
    contract = await RoutePlanner(llm_invoker=invoker).plan(
        question="Can you check this?", resolved_source_scope=_scope()
    )

    assert contract.slot_plan_status == "degraded"
    assert contract.route_decision.decision_source == "safe_fallback"
    assert contract.route_decision.fallback_reason == "invalid_planner_output"


@pytest.mark.asyncio
async def test_route_planner_delegates_to_v2_atomic_contract_planner() -> None:
    contract = await RoutePlanner().plan(
        question=(
            "Using Alpha.pdf and Beta.pdf, report the values in Table 2 "
            "and explain Equation 3."
        ),
        resolved_source_scope=ResolvedSourceScope(
            requested_source_names=["Alpha.pdf", "Beta.pdf"],
            resolved_doc_ids=["alpha", "beta"],
            authorized_doc_ids=["alpha", "beta"],
        ),
    )

    assert contract.contract_version == "2"
    assert [slot.slot_id for slot in contract.required_slots] == ["S1", "S2"]
    assert contract.route_decision is not None
