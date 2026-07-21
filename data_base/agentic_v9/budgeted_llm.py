"""Single-attempt provider invocation behind the Agentic v9 budget gate."""

from __future__ import annotations

from typing import Any, Protocol

from core.llm_factory import get_flat_llm_usage
from core.llm_usage_context import (
    agentic_budget_reservation_scope,
    agentic_budget_scope,
)
from data_base.agentic_v9.budget_controller import RunBudgetController
from data_base.agentic_v9.schemas import BudgetExceededError, FinalAnswerResult


class AsyncProvider(Protocol):
    """The minimal asynchronous provider surface used by the v9 boundary."""

    async def ainvoke(self, messages: list[dict[str, Any]]) -> Any:
        """Invoke one provider attempt."""


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Return a deliberately conservative, dependency-free prompt estimate."""
    rendered = "".join(str(message.get("content", "")) for message in messages)
    return (len(rendered) + 3) // 4


async def invoke_budgeted_llm(
    *,
    controller: RunBudgetController,
    provider: AsyncProvider,
    phase: str,
    purpose: str,
    messages: list[dict[str, Any]],
    estimated_input_tokens: int | None = None,
) -> Any:
    """Reserve before one provider attempt, then reconcile its terminal usage."""
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
        with (
            agentic_budget_scope(controller),
            agentic_budget_reservation_scope(reservation.reservation_id),
        ):
            response = await provider.ainvoke(messages)
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
