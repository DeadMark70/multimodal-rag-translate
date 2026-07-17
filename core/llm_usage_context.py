"""Task-local primitives for evaluation-only LLM usage accounting."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generator, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawLlmUsageEvent:
    """A provider-neutral usage observation before durable accounting."""

    usage_event_id: str
    scope_id: str
    campaign_id: str
    scope_type: str
    scope_key: str
    run_id: str | None
    provider_run_id: str | None
    phase: str
    purpose: str
    metric_name: str | None
    provider: str | None
    model_name: str | None
    raw_usage: dict[str, Any]
    latency_ms: float | None
    status: str
    error: dict[str, Any]
    created_at: datetime


class LlmUsageSink(Protocol):
    """Destination supplied by the evaluation accounting runtime."""

    async def record(self, event: RawLlmUsageEvent) -> None:
        """Persist one event."""


@dataclass
class LlmAccountingContext:
    """Accounting metadata isolated to the current async task."""

    scope_id: str
    campaign_id: str
    scope_type: str
    scope_key: str
    run_id: str | None
    metric_name: str | None
    sink: LlmUsageSink
    persistence_error_count: int = 0


_ACCOUNTING_CONTEXT: ContextVar[LlmAccountingContext | None] = ContextVar(
    "llm_accounting_context", default=None
)
_ACCOUNTING_PHASE: ContextVar[str] = ContextVar(
    "llm_accounting_phase", default="unclassified"
)


def current_llm_accounting_context() -> LlmAccountingContext | None:
    """Return the current task's accounting context, if evaluation enabled it."""
    return _ACCOUNTING_CONTEXT.get()


def current_llm_accounting_phase() -> str:
    """Return the current task's accounting phase."""
    return _ACCOUNTING_PHASE.get()


@contextmanager
def llm_accounting_scope(
    context: LlmAccountingContext,
) -> Generator[LlmAccountingContext, None, None]:
    """Set evaluation accounting metadata for the dynamic scope of a call."""
    token: Token[LlmAccountingContext | None] = _ACCOUNTING_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACCOUNTING_CONTEXT.reset(token)


@contextmanager
def llm_accounting_phase(phase: str) -> Generator[None, None, None]:
    """Set an accounting phase for nested LLM calls."""
    token: Token[str] = _ACCOUNTING_PHASE.set(phase)
    try:
        yield
    finally:
        _ACCOUNTING_PHASE.reset(token)


async def emit_direct_usage(
    *,
    purpose: str,
    provider: str | None,
    raw_usage: dict[str, Any],
    model_name: str | None = None,
    provider_run_id: str | None = None,
    latency_ms: float | None = None,
    status: str = "completed",
    error: dict[str, Any] | None = None,
    context: LlmAccountingContext | None = None,
    phase: str | None = None,
) -> None:
    """Record usage from a provider that does not expose LangChain callbacks.

    This intentionally does nothing outside an evaluation scope, preserving
    normal chat behaviour and preventing accidental accounting rows.
    """
    context = context or current_llm_accounting_context()
    if context is None:
        return

    event = RawLlmUsageEvent(
        usage_event_id=str(uuid4()),
        scope_id=context.scope_id,
        campaign_id=context.campaign_id,
        scope_type=context.scope_type,
        scope_key=context.scope_key,
        run_id=context.run_id,
        provider_run_id=provider_run_id,
        phase=phase or current_llm_accounting_phase(),
        purpose=purpose,
        metric_name=context.metric_name,
        provider=provider,
        model_name=model_name,
        raw_usage=dict(raw_usage),
        latency_ms=latency_ms,
        status=status,
        error=dict(error or {}),
        created_at=datetime.now(UTC),
    )
    try:
        await context.sink.record(event)
    except Exception as exc:  # Accounting must never affect the LLM result.
        context.persistence_error_count += 1
        logger.warning(
            "LLM usage persistence failed (scope_id=%s, error_type=%s)",
            context.scope_id,
            type(exc).__name__,
        )
