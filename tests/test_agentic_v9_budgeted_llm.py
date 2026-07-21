"""Tests for the provider boundary that reserves before invoking."""

import pytest

from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.budgeted_llm import invoke_budgeted_llm
from data_base.agentic_v9.schemas import BudgetExceededError


class _NeverCalledProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: object) -> object:
        self.calls += 1
        return {"usage_metadata": {"total_tokens": 1}}


class _UnavailableProvider:
    async def ainvoke(self, messages: object) -> object:
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_rejected_reservation_prevents_provider_invocation() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )
    provider = _NeverCalledProvider()

    with pytest.raises(BudgetExceededError, match="final_envelope_protected"):
        await invoke_budgeted_llm(
            controller=controller,
            provider=provider,
            phase="route_plan",
            purpose="planner",
            messages=[{"role": "user", "content": "route this"}],
            estimated_input_tokens=1,
        )

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_final_provider_failure_returns_deterministic_qualified_partial() -> None:
    controller = RunBudgetController(
        max_llm_calls=1,
        runtime_token_budget=200,
        setup_snapshot={"max_output_tokens": 100, "thinking_mode": False},
        final_input_tokens=100,
    )

    result = await invoke_budgeted_llm(
        controller=controller,
        provider=_UnavailableProvider(),
        phase="final_answer",
        purpose="synthesizer",
        messages=[{"role": "user", "content": "answer"}],
        estimated_input_tokens=100,
    )

    assert result.response_status == "qualified_partial"
    assert result.final_generation_count == 0
    assert (
        result.answer
        == "Final generation was unavailable; evidence is returned as a qualified partial."
    )
