"""Contracts for bounded, source-authorized Agentic v9 evidence repair."""

from __future__ import annotations

from data_base.agentic_v9.repair import build_repair_plan
from data_base.agentic_v9.schemas import (
    QueryContract,
    RequiredSlot,
    ResolvedSourceScope,
)
from data_base.agentic_v9.sufficiency_gate import evaluate_sufficiency


def _contract(*, route: str = "exact_structured", repair_rounds: int = 1) -> QueryContract:
    return QueryContract(
        route=route,
        intent="Retrieve source-bound evidence only.",
        entities=["global entity that is not the original question"],
        locator_hints=["Appendix"],
        required_slots=[
            RequiredSlot(
                slot_id="theorem-range",
                description="Theorem 1 m range",
                entity_ids=["GEPAR3D"],
                locator_hints=["Theorem 1"],
            ),
            RequiredSlot(
                slot_id="noise-score",
                description="noise robustness score",
                entity_ids=["ODES"],
                locator_hints=["Table 3"],
            ),
        ],
        max_repair_rounds=repair_rounds,
        resolved_source_scope=ResolvedSourceScope(
            requested_doc_ids=["gepar", "odes"],
            resolved_doc_ids=["gepar", "odes"],
            authorized_doc_ids=["gepar", "odes"],
        ),
    )


def test_repair_query_derives_only_from_missing_slot_entity_and_locator() -> None:
    contract = _contract()
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-17",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert len(plan.tasks) == 2
    assert plan.tasks[0].target_slot_ids == ["theorem-range"]
    assert plan.tasks[0].query == "GEPAR3D Theorem 1 m range Theorem 1"
    assert plan.tasks[1].query == "ODES noise robustness score Table 3"
    assert "original question" not in " ".join(task.query for task in plan.tasks)
    assert all(task.source_scope.authorized_doc_ids == ["gepar", "odes"] for task in plan.tasks)
    assert all("answer" not in task.model_dump() for task in plan.tasks)


def test_repair_stops_at_route_or_contract_round_cap() -> None:
    contract = _contract(route="bounded_compare", repair_rounds=2)
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-18",
        repair_round_index=2,
        final_budget_available=True,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "repair_round_cap_reached"


def test_repair_never_runs_when_the_final_budget_is_not_protected() -> None:
    contract = _contract()
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-19",
        repair_round_index=1,
        final_budget_available=False,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "final_budget_protected"


def test_single_lookup_never_repairs_even_when_a_slot_is_missing() -> None:
    contract = _contract(route="single_lookup", repair_rounds=1)
    sufficiency = evaluate_sufficiency(contract, [])

    plan = build_repair_plan(
        contract=contract,
        sufficiency=sufficiency,
        query_id="query-20",
        repair_round_index=1,
        final_budget_available=True,
    )

    assert plan.tasks == []
    assert plan.stop_reason == "repair_round_cap_reached"
