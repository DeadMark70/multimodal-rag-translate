"""Focused contracts for deterministic-first Agentic v9 route planning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_base.agentic_v9.route_planner import RoutePlanner
from data_base.agentic_v9.retrieval_tasks import RetrievalTaskCompiler
from data_base.agentic_v9.schemas import ResolvedSourceScope


ROOT = Path(__file__).resolve().parents[1]
ROUTES_PATH = ROOT / "evaluation" / "golden" / "agentic_v9_route_regressions.json"


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
async def test_r2_planner_contract_compiles_its_required_qualification_round() -> None:
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
    assert [task.round_id for task in plan.tasks] == ["round-1", "round-1", "round-2"]


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
async def test_q1_q2_shape_as_multi_hop_slot_contracts(
    question: str, expected_entities: set[str]
) -> None:
    planner = RoutePlanner(llm_invoker=_NeverInvoker())

    contract = await planner.plan(question=question, resolved_source_scope=_scope())

    assert contract.route == "multi_hop"
    assert expected_entities.issubset(contract.entities)
    assert len(contract.required_slots) >= 2
    assert contract.graph_policy == "locator_fallback"
    assert contract.max_retrieval_rounds == 2
    assert contract.max_repair_rounds == 1


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
    assert invoker.calls[0]["phase"] == "route_plan"
    assert invoker.calls[0]["purpose"] == "resolve_ambiguous_query_contract"
    assert contract.max_llm_calls == 3  # route-plan + evidence extraction + final
    assert "answer" not in contract.model_dump()


@pytest.mark.asyncio
async def test_ambiguous_planner_output_with_an_answer_or_scope_is_rejected() -> None:
    invoker = _PlannerInvoker(
        {
            "content": '{"route":"bounded_compare","answer":"invented",'
            '"authorized_doc_ids":["outside"]}'
        }
    )
    with pytest.raises(ValueError, match="valid route decision"):
        await RoutePlanner(llm_invoker=invoker).plan(
            question="Can you check this?", resolved_source_scope=_scope()
        )
