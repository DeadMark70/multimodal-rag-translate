"""Tests for Agentic v9 configuration feasibility before provider admission."""

import pytest

from data_base.agentic_v9.budget_feasibility import (
    ADMISSION_PRIORITY,
    FeasibilityStatus,
    validate_post_contract_feasibility,
    validate_pre_route_feasibility,
)
from data_base.agentic_v9.schemas import QueryContract


def _setup(**overrides: object) -> dict[str, object]:
    setup: dict[str, object] = {
        "max_input_tokens": 32_768,
        "max_output_tokens": 8_192,
        "thinking_mode": True,
        "thinking_budget": 8_192,
    }
    setup.update(overrides)
    return setup


def test_preflight_reserves_worst_case_planning_and_downstream_calls() -> None:
    result = validate_pre_route_feasibility(
        setup_snapshot=_setup(),
        remaining_token_budget=50_000,
        remaining_llm_calls=5,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {
        "contract_planning": 1,
        "graph_route": 1,
        "visual_extract": 1,
        "evidence_extract": 1,
        "final_answer": 1,
    }
    assert result.reserved_tokens <= 50_000
    assert result.max_provider_calls_by_phase == {
        "contract_planning": 1,
        "route_plan": 1,
        "query_rewrite": 1,
        "retrieval_judge": 1,
        "graph_route": 1,
        "visual_extract": 1,
        "evidence_extract": 3,
        "conflict_arbitration": 1,
        "claim_verifier": 1,
        "final_answer": 1,
    }


def test_pre_route_rejects_thinking_budget_that_cannot_fit_route_reservation() -> None:
    result = validate_pre_route_feasibility(
        setup_snapshot=_setup(),
        remaining_token_budget=1,
        remaining_llm_calls=5,
    )

    assert result.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert (
        result.reason
        == "planning_or_downstream_reserve_exceeds_remaining_token_budget"
    )


def test_pre_route_rejects_dynamic_thinking_without_an_explicit_reserve() -> None:
    result = validate_pre_route_feasibility(
        setup_snapshot=_setup(thinking_budget=-1),
        remaining_token_budget=100_000,
        remaining_llm_calls=5,
    )

    assert result.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert result.reason == "thinking_reserve_unknown"


def test_post_contract_admits_high_thinking_single_lookup_and_charges_final() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Find the reported score.",
        max_llm_calls=1,
        runtime_token_budget=9_728,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {"final_answer": 1}
    assert result.max_tool_operations == 0
    assert result.reserved_tokens == 9_728


def test_post_contract_reserves_optional_contract_planner_only_when_requested() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Compare two models.",
        max_llm_calls=2,
        runtime_token_budget=18_688,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=18_688,
        remaining_llm_calls=2,
        contract_plan_requested=True,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {
        "contract_planning": 1,
        "final_answer": 1,
    }
    assert result.max_provider_calls_by_phase["contract_planning"] == 1
    assert "comparison_plan" not in result.required_provider_calls
    assert "comparison_plan" not in result.max_provider_calls_by_phase


def test_post_contract_default_does_not_reserve_contract_planner_or_comparison_planner() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Find one fact.",
        max_llm_calls=1,
        runtime_token_budget=9_728,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
    )

    assert "contract_planning" not in result.required_provider_calls
    assert "comparison_plan" not in result.required_provider_calls
    assert "comparison_plan" not in result.max_provider_calls_by_phase


def test_post_contract_rejects_unfunded_contract_planner_before_execution() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Compare two models.",
        max_llm_calls=1,
        runtime_token_budget=9_728,
    )

    with_planner = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=contract.runtime_token_budget,
        remaining_llm_calls=contract.max_llm_calls,
        contract_plan_requested=True,
    )
    without_planner = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=contract.runtime_token_budget,
        remaining_llm_calls=contract.max_llm_calls,
        contract_plan_requested=False,
    )

    assert with_planner.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert with_planner.reason == "required_provider_calls_exceed_call_budget"
    assert without_planner.status is FeasibilityStatus.FEASIBLE
    assert contract.max_llm_calls == 1


def test_post_contract_charges_used_contract_planning_to_exact_contract() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Find the reported score.",
        max_llm_calls=2,
        runtime_token_budget=18_688,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
        route_plan_used=True,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {
        "contract_planning": 1,
        "final_answer": 1,
    }
    assert result.reserved_tokens == 18_688


def test_post_contract_reserves_required_visual_and_graph_before_curation() -> None:
    contract = QueryContract(
        route="graph_relational",
        intent="Resolve a relation from the source graph.",
        visual_required=True,
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=2,
        max_llm_calls=4,
        runtime_token_budget=35_968,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=35_968,
        remaining_llm_calls=4,
        evidence_qualification_provider_calls=1,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {
        "graph_route": 1,
        "visual_extract": 1,
        "evidence_extract": 1,
        "final_answer": 1,
    }
    assert result.max_tool_operations == 5
    assert ADMISSION_PRIORITY == (
        "final_answer",
        "contract_planning",
        "route_plan",
        "visual_extract",
        "graph_route",
        "evidence_extract",
        "retrieval_judge",
        "query_rewrite",
        "conflict_arbitration",
        "claim_verifier",
    )


def test_post_contract_qualification_provider_calls_reserve_only_actual_work() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Find the reported score.",
        evidence_extraction_required=True,
        max_llm_calls=1,
        runtime_token_budget=9_728,
    )

    without_provider_qualification = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
        evidence_qualification_provider_calls=0,
    )
    default_qualification = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
    )
    with_provider_qualification = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
        evidence_qualification_provider_calls=1,
    )

    assert without_provider_qualification.status is FeasibilityStatus.FEASIBLE
    assert without_provider_qualification.required_provider_calls == {"final_answer": 1}
    assert default_qualification.status is FeasibilityStatus.FEASIBLE
    assert default_qualification.required_provider_calls == {"final_answer": 1}
    assert with_provider_qualification.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert with_provider_qualification.reason == "required_provider_calls_exceed_call_budget"
    assert with_provider_qualification.required_provider_calls == {
        "evidence_extract": 1,
        "final_answer": 1,
    }


@pytest.mark.parametrize("provider_calls", [-1, 2])
def test_post_contract_rejects_invalid_initial_qualification_provider_calls(
    provider_calls: int,
) -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Find the reported score.",
        max_llm_calls=2,
        runtime_token_budget=19_456,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=19_456,
        remaining_llm_calls=2,
        evidence_qualification_provider_calls=provider_calls,
    )

    assert result.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert result.reason == "invalid_initial_evidence_qualification_provider_calls"


def test_post_contract_rejects_route_call_budget_below_required_admission() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare the two reported scores.",
        max_llm_calls=1,
        runtime_token_budget=100_000,
        visual_required=True,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(thinking_mode=False, thinking_budget=None),
        remaining_token_budget=100_000,
        remaining_llm_calls=1,
    )

    assert result.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert result.reason == "required_provider_calls_exceed_call_budget"
