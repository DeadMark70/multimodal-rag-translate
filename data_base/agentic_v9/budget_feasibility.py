"""Configuration-only admission checks for Agentic v9 provider work.

This module deliberately does not reserve or invoke providers.  It answers
whether an execution *can* be admitted before Task 3 introduces the atomic
runtime ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from data_base.agentic_v9.phase_policy import (
    MAX_PROVIDER_CALLS_BY_PHASE,
    provider_reservation_tokens,
)
from data_base.agentic_v9.schemas import QueryContract

ADMISSION_PRIORITY: tuple[str, ...] = (
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


class FeasibilityStatus(StrEnum):
    """Stable result statuses consumable by later runtime and API layers."""

    FEASIBLE = "feasible"
    CONFIGURATION_INCOMPATIBLE = "configuration_incompatible"


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    """A non-mutating feasibility decision and the reservation it proves."""

    status: FeasibilityStatus
    reason: str | None
    required_provider_calls: dict[str, int]
    max_provider_calls_by_phase: dict[str, int]
    max_tool_operations: int
    reserved_tokens: int


def _result(
    *,
    status: FeasibilityStatus,
    reason: str | None,
    required_provider_calls: dict[str, int],
    max_tool_operations: int,
    reserved_tokens: int,
) -> FeasibilityResult:
    return FeasibilityResult(
        status=status,
        reason=reason,
        required_provider_calls=required_provider_calls,
        max_provider_calls_by_phase=dict(MAX_PROVIDER_CALLS_BY_PHASE),
        max_tool_operations=max_tool_operations,
        reserved_tokens=reserved_tokens,
    )


def _positive_int(snapshot: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _setup_reasoning_reserve(snapshot: Mapping[str, Any]) -> int | None:
    """Return a known per-call thought reserve without changing Setup policy."""
    if not bool(snapshot.get("thinking_mode", snapshot.get("thinking_enabled", False))):
        return 0

    explicit_reserve = snapshot.get("thinking_token_reserve")
    if isinstance(explicit_reserve, int) and not isinstance(explicit_reserve, bool):
        return explicit_reserve if explicit_reserve >= 0 else None

    budget = snapshot.get("thinking_budget")
    if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
        return budget
    # Dynamic budgets and qualitative levels do not prove a bounded reservation
    # unless Setup carries a numeric reserve.  Do not weaken thinking to admit.
    return None


def _reservation_total(
    required_provider_calls: Mapping[str, int],
    *,
    setup_output_ceiling: int,
    reasoning_reserve: int,
) -> int:
    return sum(
        provider_reservation_tokens(
            phase,
            setup_output_ceiling=setup_output_ceiling,
            setup_reasoning_reserve=reasoning_reserve,
        )
        * calls
        for phase, calls in required_provider_calls.items()
    )


def _setup_feasibility(
    setup_snapshot: Mapping[str, Any],
) -> tuple[int | None, int | None]:
    output_ceiling = _positive_int(
        setup_snapshot, "setup_max_output_tokens", "max_output_tokens"
    )
    if output_ceiling is None:
        return None, None
    return output_ceiling, _setup_reasoning_reserve(setup_snapshot)


def validate_pre_route_feasibility(
    *,
    setup_snapshot: Mapping[str, Any],
    remaining_token_budget: int,
    remaining_llm_calls: int,
) -> FeasibilityResult:
    """Prove an ambiguous route can plan while preserving one final response.

    The deterministic insufficiency fallback has no provider cost, but v9 first
    reserves the final-answer envelope so an otherwise supported route is not
    stranded after planning.
    """
    required = {"route_plan": 1, "final_answer": 1}
    output_ceiling, reasoning_reserve = _setup_feasibility(setup_snapshot)
    if output_ceiling is None:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="setup_output_ceiling_missing",
            required_provider_calls=required,
            max_tool_operations=0,
            reserved_tokens=0,
        )
    if reasoning_reserve is None:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="thinking_reserve_unknown",
            required_provider_calls=required,
            max_tool_operations=0,
            reserved_tokens=0,
        )

    reserved_tokens = _reservation_total(
        required,
        setup_output_ceiling=output_ceiling,
        reasoning_reserve=reasoning_reserve,
    )
    if remaining_llm_calls < sum(required.values()):
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="route_or_final_reserve_exceeds_call_budget",
            required_provider_calls=required,
            max_tool_operations=0,
            reserved_tokens=reserved_tokens,
        )
    if remaining_token_budget < reserved_tokens:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="route_or_final_reserve_exceeds_remaining_token_budget",
            required_provider_calls=required,
            max_tool_operations=0,
            reserved_tokens=reserved_tokens,
        )
    return _result(
        status=FeasibilityStatus.FEASIBLE,
        reason=None,
        required_provider_calls=required,
        max_tool_operations=0,
        reserved_tokens=reserved_tokens,
    )


def validate_post_contract_feasibility(
    *,
    contract: QueryContract,
    setup_snapshot: Mapping[str, Any],
    remaining_token_budget: int,
    remaining_llm_calls: int,
    route_plan_used: bool = False,
) -> FeasibilityResult:
    """Validate a resolved route against the current non-mutating ledger view."""
    pending_provider_calls: dict[str, int] = {"final_answer": 1}
    charged_provider_calls: dict[str, int] = (
        {"route_plan": 1} if route_plan_used else {}
    )
    if contract.graph_policy == "required_locator":
        pending_provider_calls["graph_route"] = 1
    if contract.visual_required:
        pending_provider_calls["visual_extract"] = 1
    if contract.evidence_extraction_required:
        pending_provider_calls["evidence_extract"] = 1
    required = {**charged_provider_calls, **pending_provider_calls}

    max_tool_operations = (
        contract.max_retrieval_rounds
        + contract.max_repair_rounds
        + int(contract.graph_policy != "never")
        + int(contract.visual_required)
    )
    output_ceiling, reasoning_reserve = _setup_feasibility(setup_snapshot)
    if output_ceiling is None:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="setup_output_ceiling_missing",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=0,
        )
    if reasoning_reserve is None:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="thinking_reserve_unknown",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=0,
        )

    reserved_tokens = _reservation_total(
        required,
        setup_output_ceiling=output_ceiling,
        reasoning_reserve=reasoning_reserve,
    )
    pending_reserved_tokens = _reservation_total(
        pending_provider_calls,
        setup_output_ceiling=output_ceiling,
        reasoning_reserve=reasoning_reserve,
    )
    required_calls = sum(required.values())
    if any(
        calls > MAX_PROVIDER_CALLS_BY_PHASE[phase]
        for phase, calls in required.items()
    ):
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="provider_phase_call_limit_exceeded",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=reserved_tokens,
        )
    if contract.max_llm_calls < required_calls:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="required_provider_calls_exceed_call_budget",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=reserved_tokens,
        )
    if remaining_llm_calls < sum(pending_provider_calls.values()):
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="required_provider_calls_exceed_remaining_calls",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=reserved_tokens,
        )
    if contract.runtime_token_budget < reserved_tokens:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="required_provider_calls_exceed_route_token_budget",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=reserved_tokens,
        )
    if remaining_token_budget < pending_reserved_tokens:
        return _result(
            status=FeasibilityStatus.CONFIGURATION_INCOMPATIBLE,
            reason="required_provider_calls_exceed_remaining_token_budget",
            required_provider_calls=required,
            max_tool_operations=max_tool_operations,
            reserved_tokens=reserved_tokens,
        )
    return _result(
        status=FeasibilityStatus.FEASIBLE,
        reason=None,
        required_provider_calls=required,
        max_tool_operations=max_tool_operations,
        reserved_tokens=reserved_tokens,
    )
