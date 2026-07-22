"""Single-attempt provider invocation behind the Agentic v9 budget gate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from core.llm_factory import get_flat_llm_usage
from core.llm_usage_context import (
    agentic_budget_reservation_scope,
    agentic_budget_scope,
    llm_accounting_phase,
)
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.phase_policy import agentic_phase_policy_scope
from data_base.agentic_v9.schemas import BudgetExceededError, FinalAnswerResult


class AsyncProvider(Protocol):
    """The minimal asynchronous provider surface used by the v9 boundary."""

    async def ainvoke(self, messages: list[dict[str, Any]]) -> Any:
        """Invoke one provider attempt."""


@dataclass(frozen=True, slots=True)
class BudgetedLlmInvoker:
    """Concrete v9 invoker that admits every provider call through one gate."""

    controller: RunBudgetController
    provider_factory: Callable[[str], AsyncProvider]

    async def invoke(
        self,
        *,
        phase: str,
        purpose: str,
        messages: list[dict[str, Any]],
    ) -> Any:
        """Resolve the provider only after the v9 caller chooses its purpose."""
        return await invoke_budgeted_llm(
            controller=self.controller,
            provider_factory=self.provider_factory,
            phase=phase,
            purpose=purpose,
            messages=messages,
        )


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Return a deliberately conservative, dependency-free prompt estimate."""
    rendered = "".join(str(message.get("content", "")) for message in messages)
    return (len(rendered) + 3) // 4


async def invoke_budgeted_llm(
    *,
    controller: RunBudgetController,
    provider: AsyncProvider | None = None,
    provider_factory: Callable[[str], AsyncProvider] | None = None,
    phase: str,
    purpose: str,
    messages: list[dict[str, Any]],
    estimated_input_tokens: int | None = None,
) -> Any:
    """Reserve before one provider attempt, then reconcile its terminal usage."""
    if (provider is None) == (provider_factory is None):
        raise ValueError("supply exactly one of provider or provider_factory")
    try:
        reservation = await controller.reserve_call(
            phase=phase,
            purpose=purpose,
            estimated_input_tokens=(
                estimate_message_tokens(messages)
                if estimated_input_tokens is None
                else estimated_input_tokens
            ),
        )
    except BudgetExceededError:
        if phase == "final_answer":
            return _final_qualified_partial()
        raise
    try:
        policy = await controller.phase_policy(phase)
        with (
            agentic_budget_scope(controller),
            agentic_budget_reservation_scope(reservation.reservation_id),
            agentic_phase_policy_scope(policy),
            llm_accounting_phase(phase),
        ):
            active_provider = (
                provider if provider is not None else provider_factory(purpose)
            )
            response = await active_provider.ainvoke(messages)
    except asyncio.CancelledError:
        await controller.reconcile_usage(reservation.reservation_id, {})
        raise
    except Exception:
        await controller.reconcile_usage(reservation.reservation_id, {})
        if phase == "final_answer":
            return _final_qualified_partial()
        raise
    await controller.reconcile_usage(
        reservation.reservation_id, get_flat_llm_usage(response)
    )
    return response


def _final_qualified_partial() -> FinalAnswerResult:
    """Return the stable non-LLM final fallback for an unavailable generation."""
    return FinalAnswerResult(
        response_status="qualified_partial",
        answer="Final generation was unavailable; evidence is returned as a qualified partial.",
        final_generation_count=0,
    )
