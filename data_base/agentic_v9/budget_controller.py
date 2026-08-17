"""Concurrency-safe pre-invoke provider budgets for Agentic v9."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from data_base.agentic_v9.phase_policy import (
    MAX_PROVIDER_CALLS_BY_PHASE,
    EffectivePhasePolicy,
    provider_reservation_tokens,
    resolve_phase_policy,
)
from data_base.agentic_v9.schemas import BudgetExceededError, BudgetReservation


def _non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _setup_reasoning_reserve(snapshot: Mapping[str, Any]) -> int | None:
    thinking = snapshot.get("thinking_mode", snapshot.get("thinking_enabled", False))
    if thinking is False:
        return 0
    for key in ("thinking_token_reserve", "thinking_budget"):
        value = snapshot.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _setup_output_ceiling(snapshot: Mapping[str, Any]) -> int:
    ceiling = _non_negative_int(
        snapshot.get("setup_max_output_tokens", snapshot.get("max_output_tokens", 0))
    )
    if ceiling < 1:
        raise ValueError("setup_snapshot requires max_output_tokens")
    return ceiling


def _setup_input_ceiling(snapshot: Mapping[str, Any], fallback: int) -> int:
    ceiling = _non_negative_int(
        snapshot.get("setup_max_input_tokens", snapshot.get("max_input_tokens", 0))
    )
    return ceiling if ceiling > 0 else max(fallback, 1)


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Flat run totals suitable for trace payloads and test assertions."""

    provider_attempt_count: int
    reserved_input_tokens: int
    reserved_visible_output_tokens: int
    reserved_reasoning_tokens: int
    reserved_tokens: int
    reconciled_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ReconciledUsage:
    """Normalized usage, without provider metadata, for one reservation."""

    input_tokens: int
    visible_output_tokens: int
    reasoning_tokens: int
    other_tokens: int
    total_tokens: int
    usage_status: str


class RunBudgetController:
    """Atomically protects provider and final-answer envelopes for one run."""

    def __init__(
        self,
        *,
        max_llm_calls: int,
        runtime_token_budget: int,
        setup_snapshot: Mapping[str, Any],
        final_input_tokens: int,
    ) -> None:
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must reserve one final provider call")
        if runtime_token_budget < 0:
            raise ValueError("runtime_token_budget must not be negative")

        self._max_llm_calls = max_llm_calls
        self._runtime_token_budget = runtime_token_budget
        self._setup_output_ceiling = _setup_output_ceiling(setup_snapshot)
        self._reasoning_reserve = _setup_reasoning_reserve(setup_snapshot)
        if self._reasoning_reserve is None:
            raise BudgetExceededError("thinking_reserve_unknown")
        self._setup_input_ceiling = _setup_input_ceiling(
            setup_snapshot, runtime_token_budget
        )
        self._lock = asyncio.Lock()
        self._reservations: dict[str, BudgetReservation] = {}
        self._reconciled: dict[str, ReconciledUsage] = {}
        self._final_reservation_id: str | None = None
        self._phase_attempt_counts: dict[str, int] = {}
        self._provider_attempt_count = 0

        final_output = (
            provider_reservation_tokens(
                "final_answer",
                setup_output_ceiling=self._setup_output_ceiling,
                setup_reasoning_reserve=self._reasoning_reserve,
            )
            - self._reasoning_reserve
        )
        self._protected_final_input_tokens = _non_negative_int(final_input_tokens)
        self._protected_final_output_tokens = final_output
        self._protected_final_reasoning_tokens = self._reasoning_reserve
        self._protected_final_tokens = (
            self._protected_final_input_tokens
            + self._protected_final_output_tokens
            + self._protected_final_reasoning_tokens
        )
        if self._protected_final_tokens > self._runtime_token_budget:
            raise BudgetExceededError("final_envelope_exceeds_runtime_token_budget")

    async def reserve_call(
        self,
        *,
        phase: str,
        purpose: str,
        estimated_input_tokens: int,
    ) -> BudgetReservation:
        """Reserve one provider attempt before it can be invoked."""
        del purpose  # Purpose is consumed by the caller's trace event in Task 3C.
        input_tokens = _non_negative_int(estimated_input_tokens)
        async with self._lock:
            self._ensure_phase_capacity(phase)
            if phase == "final_answer":
                return self._reserve_final(input_tokens)

            output_tokens = (
                provider_reservation_tokens(
                    phase,
                    setup_output_ceiling=self._setup_output_ceiling,
                    setup_reasoning_reserve=self._reasoning_reserve,
                )
                - self._reasoning_reserve
            )
            candidate_tokens = input_tokens + output_tokens + self._reasoning_reserve
            if len(self._reservations) + 1 >= self._max_llm_calls:
                raise BudgetExceededError("final_envelope_protected")
            if (
                self._reserved_tokens()
                + candidate_tokens
                + self._protected_final_tokens
                > (self._runtime_token_budget)
            ):
                raise BudgetExceededError("final_envelope_protected")
            reservation = BudgetReservation(
                reservation_id=str(uuid4()),
                phase=phase,
                estimated_input_tokens=input_tokens,
                reserved_output_tokens=output_tokens,
                reserved_reasoning_tokens=self._reasoning_reserve,
                provider_attempt=self._next_provider_attempt(),
            )
            self._reservations[reservation.reservation_id] = reservation
            self._phase_attempt_counts[phase] = (
                self._phase_attempt_counts.get(phase, 0) + 1
            )
            return reservation

    def _ensure_phase_capacity(self, phase: str) -> None:
        try:
            phase_limit = MAX_PROVIDER_CALLS_BY_PHASE[phase]
        except KeyError as error:
            raise ValueError(f"Unsupported Agentic v9 phase: {phase}") from error
        if self._phase_attempt_counts.get(phase, 0) >= phase_limit:
            raise BudgetExceededError("provider_phase_call_limit_exhausted")

    def _next_provider_attempt(self) -> int:
        self._provider_attempt_count += 1
        return self._provider_attempt_count

    def _reserve_final(self, input_tokens: int) -> BudgetReservation:
        if self._final_reservation_id is not None:
            raise BudgetExceededError("final_answer_already_reserved")
        if input_tokens > self._protected_final_input_tokens:
            raise BudgetExceededError("final_envelope_input_exceeded")
        reservation = BudgetReservation(
            reservation_id=str(uuid4()),
            phase="final_answer",
            estimated_input_tokens=input_tokens,
            reserved_output_tokens=self._protected_final_output_tokens,
            reserved_reasoning_tokens=self._protected_final_reasoning_tokens,
            provider_attempt=self._next_provider_attempt(),
        )
        self._reservations[reservation.reservation_id] = reservation
        self._final_reservation_id = reservation.reservation_id
        self._phase_attempt_counts["final_answer"] = (
            self._phase_attempt_counts.get("final_answer", 0) + 1
        )
        return reservation

    async def phase_policy(self, phase: str) -> EffectivePhasePolicy:
        """Resolve the provider scope from Setup ceilings and live run headroom."""
        async with self._lock:
            remaining_input_budget = max(
                1,
                self._runtime_token_budget - self._reserved_tokens(),
            )
            return resolve_phase_policy(
                phase,
                setup_output_ceiling=self._setup_output_ceiling,
                setup_input_ceiling=self._setup_input_ceiling,
                remaining_input_budget=remaining_input_budget,
            )

    async def reconcile_usage(
        self, reservation_id: str, usage: Mapping[str, Any] | None
    ) -> ReconciledUsage:
        """Record flat actual usage once; duplicate terminal events are inert."""
        async with self._lock:
            if reservation_id in self._reconciled:
                return self._reconciled[reservation_id]
            try:
                reservation = self._reservations[reservation_id]
            except KeyError as error:
                raise ValueError("unknown budget reservation") from error
            normalized = self._normalize_usage(reservation, usage or {})
            self._reconciled[reservation_id] = normalized
            return normalized

    def _normalize_usage(
        self, reservation: BudgetReservation, usage: Mapping[str, Any]
    ) -> ReconciledUsage:
        input_tokens = _non_negative_int(
            usage.get(
                "input_tokens",
                usage.get("prompt_tokens", usage.get("prompt_token_count", 0)),
            )
        )
        visible_output_tokens = _non_negative_int(
            usage.get(
                "output_tokens",
                usage.get("completion_tokens", usage.get("candidates_token_count", 0)),
            )
        )
        details = usage.get("output_token_details")
        detail_reasoning = (
            details.get("reasoning") if isinstance(details, Mapping) else 0
        )
        reasoning_tokens = _non_negative_int(
            usage.get(
                "reasoning_tokens", usage.get("thoughts_token_count", detail_reasoning)
            )
        )
        other_tokens = _non_negative_int(usage.get("other_tokens", 0))
        reported_total = usage.get("total_tokens", usage.get("total_token_count"))
        if reported_total is not None and not isinstance(reported_total, bool):
            return ReconciledUsage(
                input_tokens=input_tokens,
                visible_output_tokens=visible_output_tokens,
                reasoning_tokens=reasoning_tokens,
                other_tokens=other_tokens,
                total_tokens=_non_negative_int(reported_total),
                usage_status="measured",
            )

        known_total = (
            input_tokens + visible_output_tokens + reasoning_tokens + other_tokens
        )
        reserved_total = (
            reservation.estimated_input_tokens
            + reservation.reserved_output_tokens
            + reservation.reserved_reasoning_tokens
        )
        conservative_total = max(known_total, reserved_total)
        return ReconciledUsage(
            input_tokens=input_tokens,
            visible_output_tokens=visible_output_tokens,
            reasoning_tokens=reasoning_tokens,
            other_tokens=other_tokens + conservative_total - known_total,
            total_tokens=conservative_total,
            usage_status="estimated",
        )

    async def snapshot(self) -> BudgetSnapshot:
        """Return only flat reservation and reconciled token totals."""
        async with self._lock:
            reserved_input = sum(
                item.estimated_input_tokens for item in self._reservations.values()
            )
            reserved_output = sum(
                item.reserved_output_tokens for item in self._reservations.values()
            )
            reserved_reasoning = sum(
                item.reserved_reasoning_tokens for item in self._reservations.values()
            )
            if self._final_reservation_id is None:
                reserved_input += self._protected_final_input_tokens
                reserved_output += self._protected_final_output_tokens
                reserved_reasoning += self._protected_final_reasoning_tokens
            reconciled = sum(item.total_tokens for item in self._reconciled.values())
            return BudgetSnapshot(
                provider_attempt_count=len(self._reservations),
                reserved_input_tokens=reserved_input,
                reserved_visible_output_tokens=reserved_output,
                reserved_reasoning_tokens=reserved_reasoning,
                reserved_tokens=reserved_input + reserved_output + reserved_reasoning,
                reconciled_tokens=reconciled,
                total_tokens=reconciled,
            )

    async def reservations(self) -> tuple[BudgetReservation, ...]:
        """Return the immutable reservation ledger for a persisted v9 trace."""
        async with self._lock:
            return tuple(self._reservations.values())

    def _reserved_tokens(self) -> int:
        return sum(
            item.estimated_input_tokens
            + item.reserved_output_tokens
            + item.reserved_reasoning_tokens
            for item in self._reservations.values()
        )
