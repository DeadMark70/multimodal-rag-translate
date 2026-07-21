"""Tests for Agentic v9's atomic pre-invoke provider budget ledger."""

import pytest

from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.schemas import BudgetExceededError


@pytest.mark.asyncio
async def test_optional_call_cannot_consume_the_protected_final_envelope() -> None:
    controller = RunBudgetController(
        max_llm_calls=2,
        runtime_token_budget=300,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )

    with pytest.raises(BudgetExceededError, match="final_envelope_protected"):
        await controller.reserve_call(
            phase="route_plan",
            purpose="planner",
            estimated_input_tokens=1,
        )

    reservation = await controller.reserve_call(
        phase="final_answer",
        purpose="synthesizer",
        estimated_input_tokens=100,
    )

    assert reservation.phase == "final_answer"


@pytest.mark.asyncio
async def test_reconciliation_uses_provider_total_once_without_double_counting_reasoning() -> (
    None
):
    controller = RunBudgetController(
        max_llm_calls=2,
        runtime_token_budget=500,
        setup_snapshot={
            "max_output_tokens": 100,
            "thinking_mode": True,
            "thinking_budget": 20,
        },
        final_input_tokens=100,
    )
    reservation = await controller.reserve_call(
        phase="route_plan",
        purpose="planner",
        estimated_input_tokens=10,
    )

    first = await controller.reconcile_usage(
        reservation.reservation_id,
        {
            "input_tokens": 80,
            "output_tokens": 40,
            "thoughts_token_count": 24,
            "total_tokens": 140,
        },
    )
    second = await controller.reconcile_usage(
        reservation.reservation_id,
        {"total_tokens": 999},
    )
    snapshot = await controller.snapshot()

    assert first.total_tokens == 140
    assert first.visible_output_tokens == 40
    assert first.reasoning_tokens == 24
    assert second == first
    assert snapshot.reconciled_tokens == 140


@pytest.mark.asyncio
async def test_missing_usage_reconciles_to_the_conservative_reservation() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )
    reservation = await controller.reserve_call(
        phase="final_answer",
        purpose="synthesizer",
        estimated_input_tokens=100,
    )

    usage = await controller.reconcile_usage(reservation.reservation_id, {})

    assert usage.usage_status == "estimated"
    assert usage.total_tokens == 200
