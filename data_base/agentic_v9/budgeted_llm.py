"""Single-attempt provider invocation behind the Agentic v9 budget gate."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from core.llm_factory import get_flat_llm_usage
from core.sensitive_data import (
    is_sensitive_credential_key,
    sanitize_json_object_text,
)
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
    prompt_hash: str | None
    prompt_preview: str | None
    full_prompt: str | None
    prompt_capture_status: str
    full_prompt_capture_status: str
    response_hash: str | None
    latency_ms: float
    status: str
    error: dict[str, str]
    usage: dict[str, int | str]


class LlmCallObserver(Protocol):
    """Best-effort sink for admitted provider terminal attempts."""

    async def on_terminal_attempt(self, observation: LlmAttemptObservation) -> bool:
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
    capture_policy: Mapping[str, Any] | None = None

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
            capture_policy=(
                self.capture_policy
                or getattr(self.observer, "prompt_capture_policy", None)
            ),
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
    capture_policy: Mapping[str, Any] | None = None,
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
    prompt_capture = _capture_prompt(messages, capture_policy)
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
            prompt_capture=prompt_capture,
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
            prompt_capture=prompt_capture,
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
    flat_usage = get_flat_llm_usage(response, include_provenance=True)
    provider_total_reported = bool(flat_usage.pop("provider_total_reported", False))
    known_tokens = (
        flat_usage.get("input_tokens", 0)
        + flat_usage.get("output_tokens", 0)
        + flat_usage.get("reasoning_tokens", 0)
    )
    flat_usage["other_tokens"] = max(
        flat_usage.get("total_tokens", known_tokens) - known_tokens,
        0,
    )
    if not provider_total_reported:
        flat_usage.pop("total_tokens", None)
    usage = await controller.reconcile_usage(reservation.reservation_id, flat_usage)
    await _observe_terminal(
        observer=observer,
        phase=phase,
        purpose=purpose,
        reservation=reservation,
        provider_name=provider_name,
        model_name=model_name,
        prompt_capture=prompt_capture,
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
    prompt_capture: "_PromptCapture",
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
        prompt_hash=prompt_capture.prompt_hash,
        prompt_preview=prompt_capture.prompt_preview,
        full_prompt=prompt_capture.full_prompt,
        prompt_capture_status=prompt_capture.prompt_capture_status,
        full_prompt_capture_status=prompt_capture.full_prompt_capture_status,
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
            "official_total_tokens": (
                usage.total_tokens if usage.usage_status == "measured" else None
            ),
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


@dataclass(frozen=True, slots=True)
class _PromptCapture:
    prompt_hash: str | None
    prompt_preview: str | None
    full_prompt: str | None
    prompt_capture_status: str
    full_prompt_capture_status: str


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|password|secret|token)\s*[:=]\s*[^\s,;\"}]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
)


def _sanitize_prompt_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        redacted = False
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_credential_key(key_text):
                sanitized[key_text] = "[REDACTED]"
                redacted = True
                continue
            sanitized_item, item_redacted = _sanitize_prompt_value(item)
            sanitized[key_text] = sanitized_item
            redacted = redacted or item_redacted
        return sanitized, redacted
    if isinstance(value, (list, tuple)):
        items = []
        redacted = False
        for item in value:
            sanitized_item, item_redacted = _sanitize_prompt_value(item)
            items.append(sanitized_item)
            redacted = redacted or item_redacted
        return items, redacted
    if isinstance(value, str):
        sanitized, structured_redacted = sanitize_json_object_text(value)
        sanitized = sanitized if sanitized is not None else value
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized, structured_redacted or sanitized != value
    return value, False


def _capture_prompt(
    messages: list[dict[str, Any]],
    policy: Mapping[str, Any] | None,
) -> _PromptCapture:
    frozen = dict(policy or {})
    capture_hash = bool(frozen.get("hash", True))
    capture_preview = bool(frozen.get("preview", True))
    capture_full = bool(frozen.get("full_prompt", False))
    preview_max_chars = frozen.get("preview_max_chars", 512)
    if (
        not isinstance(preview_max_chars, int)
        or isinstance(preview_max_chars, bool)
        or preview_max_chars < 1
    ):
        preview_max_chars = 512
    try:
        sanitized_value, structured_redacted = _sanitize_prompt_value(messages)
        sanitized = _stable_serialize(sanitized_value)
        canonical = sanitized
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        was_redacted = structured_redacted or sanitized != canonical
        prompt_hash = (
            hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
            if capture_hash
            else None
        )
        return _PromptCapture(
            prompt_hash=prompt_hash,
            prompt_preview=(sanitized[:preview_max_chars] if capture_preview else None),
            full_prompt=sanitized if capture_full else None,
            prompt_capture_status=(
                "redacted"
                if capture_preview and was_redacted
                else "captured"
                if capture_preview
                else "not_captured_at_execution"
            ),
            full_prompt_capture_status=(
                "redacted"
                if capture_full and was_redacted
                else "captured"
                if capture_full
                else "not_captured_at_execution"
            ),
        )
    except Exception:
        return _PromptCapture(
            prompt_hash=None,
            prompt_preview=None,
            full_prompt=None,
            prompt_capture_status="capture_failed",
            full_prompt_capture_status="capture_failed",
        )


def _stable_hash(value: object) -> str:
    serialized = _stable_serialize(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stable_serialize(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def _final_qualified_partial() -> FinalAnswerResult:
    """Return the stable non-LLM final fallback for an unavailable generation."""
    return FinalAnswerResult(
        response_status="qualified_partial",
        answer="Final generation was unavailable; evidence is returned as a qualified partial.",
        final_generation_count=0,
    )
