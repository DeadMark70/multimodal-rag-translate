"""Single-attempt provider invocation behind the Agentic v9 budget gate."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
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
class LlmAttemptObservation:
    """One admitted provider attempt at its terminal boundary."""

    phase: str
    purpose: str
    reservation_id: str
    provider_attempt: int
    provider: str
    model_name: str
    prompt_hash: str
    response_hash: str | None
    latency_ms: float
    status: str
    error: dict[str, str]
    usage: dict[str, int | str]


class LlmCallObserver(Protocol):
    """Best-effort sink for admitted provider terminal attempts."""

    async def on_terminal_attempt(
        self, observation: LlmAttemptObservation
    ) -> bool:
        """Persist one terminal attempt and return whether it was recorded."""

    def mark_partial(self, reason: str) -> None:
        """Mark the owning run's observability as incomplete."""


@dataclass(frozen=True, slots=True)
class BudgetedLlmInvoker:
    """Concrete v9 invoker that admits every provider call through one gate."""

    controller: RunBudgetController
    provider_factory: Callable[[str], AsyncProvider]
    observer: LlmCallObserver | None = None
    provider_name: str = "unknown"
    model_name: str = "unknown"

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
            observer=self.observer,
            provider_name=self.provider_name,
            model_name=self.model_name,
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
    observer: LlmCallObserver | None = None,
    provider_name: str = "unknown",
    model_name: str = "unknown",
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
    started_at = time.perf_counter()
    prompt_hash = _stable_hash(messages)
    active_provider: AsyncProvider | None = provider
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
        usage = await controller.reconcile_usage(reservation.reservation_id, {})
        await _observe_terminal(
            observer=observer,
            phase=phase,
            purpose=purpose,
            reservation=reservation,
            provider_name=provider_name,
            model_name=model_name,
            prompt_hash=prompt_hash,
            response_hash=None,
            started_at=started_at,
            status="failed",
            error={
                "type": "CancelledError",
                "message": "provider_attempt_cancelled",
            },
            usage=usage,
        )
        raise
    except Exception as exc:
        usage = await controller.reconcile_usage(reservation.reservation_id, {})
        await _observe_terminal(
            observer=observer,
            phase=phase,
            purpose=purpose,
            reservation=reservation,
            provider_name=provider_name,
            model_name=model_name,
            prompt_hash=prompt_hash,
            response_hash=None,
            started_at=started_at,
            status="timeout" if isinstance(exc, asyncio.TimeoutError) else "failed",
            error={
                "type": exc.__class__.__name__,
                "message": "provider_attempt_failed",
            },
            usage=usage,
        )
        if phase == "final_answer":
            return _final_qualified_partial()
        raise
    flat_usage = get_flat_llm_usage(response)
    flat_usage["other_tokens"] = max(
        flat_usage.get("total_tokens", 0)
        - flat_usage.get("input_tokens", 0)
        - flat_usage.get("output_tokens", 0)
        - flat_usage.get("reasoning_tokens", 0),
        0,
    )
    usage = await controller.reconcile_usage(reservation.reservation_id, flat_usage)
    await _observe_terminal(
        observer=observer,
        phase=phase,
        purpose=purpose,
        reservation=reservation,
        provider_name=provider_name,
        model_name=model_name,
        prompt_hash=prompt_hash,
        response_hash=_stable_hash(response),
        started_at=started_at,
        status="success",
        error={},
        usage=usage,
    )
    return response


async def _observe_terminal(
    *,
    observer: LlmCallObserver | None,
    phase: str,
    purpose: str,
    reservation: Any,
    provider_name: str,
    model_name: str,
    prompt_hash: str,
    response_hash: str | None,
    started_at: float,
    status: str,
    error: dict[str, str],
    usage: Any,
) -> None:
    if observer is None:
        return
    observation = LlmAttemptObservation(
        phase=_canonical_observation_phase(phase),
        purpose=purpose,
        reservation_id=reservation.reservation_id,
        provider_attempt=reservation.provider_attempt,
        provider=provider_name or "unknown",
        model_name=model_name or "unknown",
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        latency_ms=max((time.perf_counter() - started_at) * 1000, 0),
        status=status,
        error=error,
        usage={
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.visible_output_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "other_tokens": usage.other_tokens,
            "total_tokens": usage.total_tokens,
            "usage_status": usage.usage_status,
        },
    )
    try:
        recorded = await observer.on_terminal_attempt(observation)
    except Exception:
        observer.mark_partial("llm_call_observer_failed")
        return
    if not recorded:
        observer.mark_partial("llm_call_observer_failed")


def _canonical_observation_phase(phase: str) -> str:
    aliases = {
        "route_plan": "contract_planning",
        "contract_planning": "contract_planning",
        "query_rewrite": "retrieval_judge",
        "retrieval_judge": "retrieval_judge",
        "conflict_arbitration": "retrieval_judge",
        "claim_verifier": "retrieval_judge",
        "evidence_extract": "evidence_extract",
        "visual_extract": "visual_extract",
        "final_answer": "final_answer",
    }
    return aliases.get(phase, "retrieval_judge")


def _stable_hash(value: object) -> str:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        serialized = str(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _final_qualified_partial() -> FinalAnswerResult:
    """Return the stable non-LLM final fallback for an unavailable generation."""
    return FinalAnswerResult(
        response_status="qualified_partial",
        answer="Final generation was unavailable; evidence is returned as a qualified partial.",
        final_generation_count=0,
    )
