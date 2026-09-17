"""Task-local controls for evaluator calls; ordinary chat keeps its defaults."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


@dataclass
class EvaluationInference:
    service_tier: str = "standard"
    timeout_seconds: int = 180
    max_attempts: int = 5
    standard_fallback: bool = False
    limiter: Any = None
    slots: Any = None
    global_slots: Any = None
    on_retry: Callable[[int, BaseException], Awaitable[None]] | None = None


_INFERENCE: ContextVar[EvaluationInference | None] = ContextVar(
    "evaluation_inference", default=None
)


def scoring_config(config: Any) -> dict[str, Any]:
    """Freeze controls alongside durable work, including old campaign defaults."""
    return {
        name: getattr(config, name, default)
        for name, default in {
            "ragas_service_tier": "standard",
            "ragas_request_timeout_seconds": 900,
            "ragas_max_attempts": 5,
            "ragas_standard_fallback": False,
            "ragas_rpm_limit": 1000,
        }.items()
    }


@contextmanager
def evaluation_inference_scope(
    snapshot: Mapping[str, Any],
    *,
    limiter: Any,
    slots: asyncio.Semaphore | None = None,
    global_slots: asyncio.Semaphore | None = None,
    on_retry: Callable[[int, BaseException], Awaitable[None]] | None = None,
) -> Iterator[EvaluationInference]:
    tier = str(snapshot.get("ragas_service_tier") or "standard")
    settings = EvaluationInference(
        service_tier=tier,
        timeout_seconds=int(
            snapshot.get("ragas_request_timeout_seconds")
            or (900 if tier == "flex" else 180)
        ),
        max_attempts=int(snapshot.get("ragas_max_attempts") or 5),
        standard_fallback=bool(snapshot.get("ragas_standard_fallback", False)),
        limiter=limiter,
        slots=slots or asyncio.Semaphore(8),
        global_slots=global_slots or asyncio.Semaphore(8),
        on_retry=on_retry,
    )
    token = _INFERENCE.set(settings)
    try:
        yield settings
    finally:
        _INFERENCE.reset(token)


def metric_timeout_seconds() -> int:
    settings = _INFERENCE.get()
    if settings is None:
        return 360
    # A metric can contain multiple sequential generations and JSON repairs.
    # Transport calls have their own timeout; this outer bound must include them.
    return 12 * settings.max_attempts * (settings.timeout_seconds + 60)


class EvaluationGoogleChat(ChatGoogleGenerativeAI):
    """Count each actual request and retry only transient provider failures."""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        from core.llm_usage_context import emit_direct_usage
        from evaluation.error_policy import (
            classify_evaluation_error,
            retry_delay_seconds,
        )

        settings = _INFERENCE.get()
        if settings is None:
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        tier = settings.service_tier
        for attempt in range(1, settings.max_attempts + 2):
            try:
                async with settings.slots, settings.global_slots:
                    await settings.limiter.acquire()
                    from langchain_google_genai.chat_models import _response_to_result

                    request = self._prepare_request(
                        messages,
                        stop=stop,
                        **{
                            **kwargs,
                            "service_tier": tier,
                            "timeout": settings.timeout_seconds,
                            "max_retries": 1,
                        },
                    )
                    response = await self.async_client.models.generate_content(
                        **request
                    )
                    # Keep LangChain's cache_read=0 on measured responses.
                    # Removing it makes normal cache misses look uninstrumented.
                    result = _response_to_result(response)
                for generation in result.generations:
                    generation.message.response_metadata["requested_service_tier"] = (
                        tier
                    )
                    actual_tier = getattr(response, "service_tier", None)
                    if actual_tier:
                        generation.message.response_metadata["service_tier"] = getattr(
                            actual_tier, "value", actual_tier
                        )
                return result
            except Exception as exc:
                decision = classify_evaluation_error(exc)
                fallback = (
                    attempt == settings.max_attempts
                    and tier == "flex"
                    and settings.standard_fallback
                    and decision.error_type
                    in {"rate_limit", "server_error", "service_unavailable"}
                )
                if not decision.retryable or (
                    attempt >= settings.max_attempts and not fallback
                ):
                    raise
                # The final attempt is emitted by the LangChain callback. Earlier
                # failed requests need their own event so coverage stays honest.
                await emit_direct_usage(
                    purpose="evaluator",
                    provider="google",
                    model_name=self.model,
                    raw_usage={"requested_service_tier": tier},
                    status="failed",
                    error={"code": decision.error_type},
                )
                if settings.on_retry is not None:
                    await settings.on_retry(attempt, exc)
                if fallback:
                    tier = "standard"
                else:
                    await asyncio.sleep(
                        retry_delay_seconds(attempt, decision.retry_after_seconds)
                    )
        raise RuntimeError("Evaluator attempts exhausted")
