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


@pytest.mark.asyncio
async def test_reservations_expose_the_persistable_attempt_ledger() -> None:
    controller = RunBudgetController(
        max_llm_calls=2,
        runtime_token_budget=10_000,
        setup_snapshot={"max_output_tokens": 256, "thinking_mode": False},
        final_input_tokens=128,
    )

    reservation = await controller.reserve_call(
        phase="final_answer", purpose="answer", estimated_input_tokens=12
    )

    assert await controller.reservations() == (reservation,)


@pytest.mark.asyncio
async def test_controller_rejects_enabled_thinking_without_a_numeric_reserve() -> None:
    with pytest.raises(BudgetExceededError, match="thinking_reserve_unknown"):
        RunBudgetController(
            max_llm_calls=1,
            runtime_token_budget=1_000,
            setup_snapshot={
                "max_output_tokens": 100,
                "thinking_enabled": True,
                "thinking_level": "high",
            },
            final_input_tokens=100,
        )


@pytest.mark.asyncio
async def test_controller_caps_each_phase_and_numbers_provider_attempts() -> None:
    controller = RunBudgetController(
        max_llm_calls=2,
        runtime_token_budget=500,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )

    route = await controller.reserve_call(
        phase="route_plan", purpose="planner", estimated_input_tokens=1
    )
    final = await controller.reserve_call(
        phase="final_answer", purpose="synthesizer", estimated_input_tokens=100
    )

    assert (route.provider_attempt, final.provider_attempt) == (1, 2)
    with pytest.raises(
        BudgetExceededError, match="provider_phase_call_limit_exhausted"
    ):
        await controller.reserve_call(
            phase="route_plan", purpose="planner", estimated_input_tokens=1
        )
