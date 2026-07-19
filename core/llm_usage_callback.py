"""LangChain callback which emits evaluation-scoped LLM usage events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from core.llm_usage_context import (
    LlmAccountingContext,
    current_llm_accounting_context,
    current_llm_accounting_phase,
    emit_direct_usage,
)


@dataclass(frozen=True)
class _StartState:
    context: LlmAccountingContext
    phase: str
    started_at: float


class EvaluationUsageCallback(AsyncCallbackHandler):
    """Capture usage while an evaluation accounting context is active."""

    def __init__(
        self,
        *,
        purpose: str,
        provider: str | None,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.purpose = purpose
        self.provider = provider
        # LangChain provider responses do not consistently expose the model
        # name. Keep the factory-resolved value as a deterministic accounting
        # identity for token attribution and trace diagnostics.
        self.model_name = model_name
        self._starts: dict[UUID, _StartState] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._remember_start(run_id)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._remember_start(run_id)

    def _remember_start(self, run_id: UUID) -> None:
        context = current_llm_accounting_context()
        if context is not None:
            self._starts[run_id] = _StartState(
                context=context,
                phase=current_llm_accounting_phase(),
                started_at=monotonic(),
            )

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        start = self._starts.pop(run_id, None)
        if start is None:
            return
        try:
            usage = _extract_usage(response)
            model_name = _extract_model_name(response)
            await self._emit(start, run_id, usage, model_name, "completed", {})
        finally:
            self._starts.pop(run_id, None)

    async def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        start = self._starts.pop(run_id, None)
        if start is None:
            return
        try:
            response = kwargs.get("response")
            usage = _extract_usage(response) if isinstance(response, LLMResult) else {}
            model_name = (
                _extract_model_name(response)
                if isinstance(response, LLMResult)
                else None
            )
            await self._emit(
                start,
                run_id,
                usage,
                model_name,
                "failed",
                _safe_error_details(error),
            )
        finally:
            self._starts.pop(run_id, None)

    async def _emit(
        self,
        start: _StartState,
        run_id: UUID,
        usage: dict[str, Any],
        model_name: str | None,
        status: str,
        error: dict[str, Any],
    ) -> None:
        # The context was snapshotted at start so task context changes between
        # streaming start and terminal chunk cannot misattribute the event.
        await emit_direct_usage(
            context=start.context,
            phase=start.phase,
            purpose=self.purpose,
            provider=self.provider,
            raw_usage=usage,
            model_name=model_name or self.model_name,
            provider_run_id=str(run_id),
            latency_ms=(monotonic() - start.started_at) * 1000,
            status=status,
            error=error,
        )


def _extract_usage(response: LLMResult) -> dict[str, Any]:
    """Prefer terminal message metadata, then provider ``llm_output``."""
    for generation_group in response.generations:
        for generation in generation_group:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict):
                return usage
    llm_output = response.llm_output or {}
    usage = llm_output.get("usage_metadata") or llm_output.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}


def _extract_model_name(response: LLMResult) -> str | None:
    llm_output = response.llm_output or {}
    for key in ("model_name", "model", "model_version"):
        value = llm_output.get(key)
        if isinstance(value, str) and value:
            return value

    for generation_group in response.generations:
        for generation in generation_group:
            for metadata in (
                getattr(generation, "generation_info", None),
                getattr(getattr(generation, "message", None), "response_metadata", None),
            ):
                if not isinstance(metadata, dict):
                    continue
                for key in ("model_name", "model", "model_version"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value:
                        return value
    return None


_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key)=)([^&#\s]+)"
)
_KEY_VALUE_RE = re.compile(
    r'''(?ix)
    (?P<prefix>
        ["']?(?:api[_-]?key|access[_-]?token|token|password|secret|x-goog-api-key)["']?
        \s*(?:=|:)\s*
    )
    (?P<quote>["']?)
    (?P<value>[^"'\s,;&}]+)
    '''
)


def _redact_sensitive_message(message: str) -> str:
    message = _BEARER_RE.sub("Bearer [REDACTED]", message)
    message = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", message)
    return _KEY_VALUE_RE.sub(
        lambda match: f'{match.group("prefix")}{match.group("quote")}[REDACTED]',
        message,
    )


def _safe_error_details(error: BaseException) -> dict[str, Any]:
    """Return useful provider diagnostics without leaking credentials."""
    message = str(error).strip()
    if message:
        message = _redact_sensitive_message(message)[:1000]

    details: dict[str, Any] = {"type": type(error).__name__}
    if message:
        details["message"] = message

    for attribute in ("status_code", "status", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            details["status_code"] = value
            break
    return details
