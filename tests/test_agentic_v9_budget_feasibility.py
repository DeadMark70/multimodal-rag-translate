"""Tests for Agentic v9 configuration feasibility before provider admission."""

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


def test_pre_route_reserves_route_and_final_calls_with_setup_thinking_budget() -> None:
    result = validate_pre_route_feasibility(
        setup_snapshot=_setup(),
        remaining_token_budget=18_304,
        remaining_llm_calls=2,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {"route_plan": 1, "final_answer": 1}
    assert result.reserved_tokens == 18_304
    assert result.max_provider_calls_by_phase == {
        "route_plan": 1,
        "query_rewrite": 1,
        "retrieval_judge": 1,
        "graph_route": 1,
        "visual_extract": 1,
        "evidence_extract": 1,
        "conflict_arbitration": 1,
        "claim_verifier": 1,
        "final_answer": 1,
    }


def test_pre_route_rejects_thinking_budget_that_cannot_fit_route_reservation() -> None:
    result = validate_pre_route_feasibility(
        setup_snapshot=_setup(),
        remaining_token_budget=18_303,
        remaining_llm_calls=2,
    )

    assert result.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE
    assert result.reason == "route_or_final_reserve_exceeds_remaining_token_budget"


def test_pre_route_rejects_dynamic_thinking_without_an_explicit_reserve() -> None:
    result = validate_pre_route_feasibility(
        setup_snapshot=_setup(thinking_budget=-1),
        remaining_token_budget=100_000,
        remaining_llm_calls=2,
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


def test_post_contract_charges_a_used_route_plan_to_the_resolved_route_budget() -> None:
    contract = QueryContract(
        route="single_lookup",
        intent="Find the reported score.",
        max_llm_calls=2,
        runtime_token_budget=18_304,
    )

    result = validate_post_contract_feasibility(
        contract=contract,
        setup_snapshot=_setup(),
        remaining_token_budget=9_728,
        remaining_llm_calls=1,
        route_plan_used=True,
    )

    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.required_provider_calls == {"route_plan": 1, "final_answer": 1}
    assert result.reserved_tokens == 18_304


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
        "route_plan",
        "visual_extract",
        "graph_route",
        "evidence_extract",
        "retrieval_judge",
        "query_rewrite",
        "conflict_arbitration",
        "claim_verifier",
    )


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
