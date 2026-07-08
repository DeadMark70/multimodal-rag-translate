"""Best-effort recorder for evaluation run observability."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from types import TracebackType
from typing import Optional
from uuid import uuid4

from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.token_cost import normalize_llm_usage, price_llm_usage
from evaluation.trace_schemas import (
    EvaluationClaim,
    EvaluationContextPack,
    EvaluationHumanRating,
    EvaluationLlmCall,
    EvaluationRetrievalChunk,
    EvaluationRetrievalEvent,
    EvaluationRoutingDecision,
    EvaluationToolCall,
    EvaluationTraceEvent,
    TraceStageType,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _duration_ms(started_at: datetime, ended_at: datetime) -> float:
    return max((ended_at - started_at).total_seconds() * 1000, 0)


def _sanitize_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


class EvaluationSpan:
    """Async context manager representing one trace span."""

    def __init__(
        self,
        *,
        recorder: "EvaluationRunRecorder",
        stage_type: TraceStageType,
        stage_name: str,
        event_type: str,
        payload: Optional[dict] = None,
    ) -> None:
        self._recorder = recorder
        self.stage_type = stage_type
        self.stage_name = stage_name
        self.event_type = event_type
        self.payload = payload or {}
        self.start_event_id = str(uuid4())
        self.end_event_id: str | None = None
        self.event_id = self.start_event_id
        self.span_id = str(uuid4())
        self.parent_event_id: str | None = None
        self.parent_span_id: str | None = None
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.duration_ms: float | None = None
        self._stack_token: Token[tuple["EvaluationSpan", ...]] | None = None

    async def __aenter__(self) -> "EvaluationSpan":
        parent = self._recorder.current_span
        if parent is not None:
            self.parent_event_id = parent.start_event_id
            self.parent_span_id = parent.span_id
        self.started_at = _utc_now()
        stack = self._recorder._span_stack.get()
        self._stack_token = self._recorder._span_stack.set((*stack, self))
        try:
            await self._recorder._record_trace_event(
                span=self,
                event_id=self.start_event_id,
                status="running",
                ended_at=None,
                duration_ms=None,
                error={},
            )
        except Exception:
            if self._stack_token is not None:
                self._recorder._span_stack.reset(self._stack_token)
                self._stack_token = None
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        ended_at = _utc_now()
        self.ended_at = ended_at
        self.duration_ms = _duration_ms(self.started_at or ended_at, ended_at)
        self.end_event_id = str(uuid4())
        try:
            await self._recorder._record_trace_event(
                span=self,
                event_id=self.end_event_id,
                status="failed" if exc else "success",
                ended_at=ended_at,
                duration_ms=self.duration_ms,
                error=_sanitize_exception(exc) if exc else {},
            )
        finally:
            if self._stack_token is not None:
                self._recorder._span_stack.reset(self._stack_token)
                self._stack_token = None
        return False


class EvaluationRunRecorder:
    """Records trace spans and detail rows without breaking runtime by default."""

    def __init__(
        self,
        *,
        run_id: str,
        campaign_id: str,
        user_id: str,
        request_id: str | None = None,
        trace_repository: EvaluationObservabilityRepository | None = None,
        llm_call_repository: EvaluationObservabilityRepository | None = None,
        retrieval_event_repository: EvaluationObservabilityRepository | None = None,
        retrieval_chunk_repository: EvaluationObservabilityRepository | None = None,
        context_pack_repository: EvaluationObservabilityRepository | None = None,
        tool_call_repository: EvaluationObservabilityRepository | None = None,
        routing_decision_repository: EvaluationObservabilityRepository | None = None,
        claim_repository: EvaluationObservabilityRepository | None = None,
        human_rating_repository: EvaluationObservabilityRepository | None = None,
        strict: bool = False,
    ) -> None:
        repository = trace_repository or EvaluationObservabilityRepository()
        self.run_id = run_id
        self.campaign_id = campaign_id
        self.user_id = user_id
        self.request_id = request_id
        self.trace_repository = repository
        self.llm_call_repository = llm_call_repository or repository
        self.retrieval_event_repository = retrieval_event_repository or repository
        self.retrieval_chunk_repository = retrieval_chunk_repository or repository
        self.context_pack_repository = context_pack_repository or repository
        self.tool_call_repository = tool_call_repository or repository
        self.routing_decision_repository = routing_decision_repository or repository
        self.claim_repository = claim_repository or repository
        self.human_rating_repository = human_rating_repository or repository
        self.strict = strict
        self._sequence = 0
        self._span_stack: ContextVar[tuple[EvaluationSpan, ...]] = ContextVar(
            f"evaluation_span_stack_{id(self)}",
            default=(),
        )

    @property
    def current_span(self) -> EvaluationSpan | None:
        stack = self._span_stack.get()
        return stack[-1] if stack else None

    def start_span(
        self,
        *,
        stage_type: TraceStageType,
        stage_name: str,
        event_type: str = "span",
        payload: Optional[dict] = None,
    ) -> EvaluationSpan:
        return EvaluationSpan(
            recorder=self,
            stage_type=stage_type,
            stage_name=stage_name,
            event_type=event_type,
            payload=payload,
        )

    def span(
        self,
        *,
        stage_type: TraceStageType,
        stage_name: str,
        event_type: str = "span",
        payload: Optional[dict] = None,
    ) -> EvaluationSpan:
        return self.start_span(
            stage_type=stage_type,
            stage_name=stage_name,
            event_type=event_type,
            payload=payload,
        )

    async def _record_trace_event(
        self,
        *,
        span: EvaluationSpan,
        event_id: str,
        status: str,
        ended_at: datetime | None,
        duration_ms: float | None,
        error: dict,
    ) -> None:
        self._sequence += 1
        payload = dict(span.payload)
        if self.request_id:
            payload.setdefault("request_id", self.request_id)
        event = EvaluationTraceEvent(
            event_id=event_id,
            run_id=self.run_id,
            campaign_id=self.campaign_id,
            span_id=span.span_id,
            parent_event_id=span.parent_event_id,
            parent_span_id=span.parent_span_id,
            event_type=span.event_type,
            event_schema_version="1.0",
            sequence=self._sequence,
            stage_type=span.stage_type,
            stage_name=span.stage_name,
            started_at=span.started_at or _utc_now(),
            ended_at=ended_at,
            duration_ms=duration_ms,
            status=status,
            retry_count=0,
            payload=payload,
            error=error,
            created_at=_utc_now(),
        )
        await self._safe_record(self.trace_repository.record_trace_event, event)

    async def record_llm_call(self, call: EvaluationLlmCall) -> None:
        await self._safe_record(self.llm_call_repository.record_llm_call, call)

    async def record_llm_usage(
        self,
        *,
        purpose: str,
        provider: str | None = None,
        model_name: str | None = None,
        usage: object,
        latency_ms: float | None = None,
        status: str = "success",
        error: dict | None = None,
        prompt_hash: str | None = None,
        prompt_preview: str | None = None,
        response_hash: str | None = None,
        span_id: str | None = None,
        payload: dict | None = None,
        price_snapshot: dict | None = None,
        created_at: datetime | None = None,
    ) -> None:
        normalized_usage = normalize_llm_usage(usage)
        pricing = price_llm_usage(
            model_name=model_name,
            usage=normalized_usage,
            price_snapshot=price_snapshot,
        )
        enriched_payload = dict(payload or {})
        if pricing.get("price_snapshot_id"):
            enriched_payload["price_snapshot_id"] = pricing["price_snapshot_id"]
        if normalized_usage.get("reasoning_tokens"):
            enriched_payload["reasoning_tokens"] = normalized_usage["reasoning_tokens"]
        call = EvaluationLlmCall(
            llm_call_id=str(uuid4()),
            run_id=self.run_id,
            campaign_id=self.campaign_id,
            span_id=span_id if span_id is not None else (self.current_span.span_id if self.current_span else None),
            provider=provider,
            model_name=model_name,
            purpose=purpose,
            prompt_tokens=normalized_usage["prompt_tokens"],
            completion_tokens=normalized_usage["completion_tokens"],
            total_tokens=normalized_usage["total_tokens"],
            estimated_cost_usd=pricing["estimated_cost_usd"],
            estimated_cost_twd=pricing["estimated_cost_twd"],
            prompt_hash=prompt_hash,
            prompt_preview=prompt_preview,
            response_hash=response_hash,
            latency_ms=latency_ms,
            status=status,
            error=error or {},
            payload=enriched_payload,
            created_at=created_at or _utc_now(),
        )
        await self.record_llm_call(call)

    async def record_retrieval_event(self, event: EvaluationRetrievalEvent) -> None:
        await self._safe_record(self.retrieval_event_repository.record_retrieval_event, event)

    async def record_retrieval_chunk(self, chunk: EvaluationRetrievalChunk) -> None:
        await self._safe_record(self.retrieval_chunk_repository.record_retrieval_chunk, chunk)

    async def record_context_pack(self, pack: EvaluationContextPack) -> None:
        await self._safe_record(self.context_pack_repository.record_context_pack, pack)

    async def record_tool_call(self, call: EvaluationToolCall) -> None:
        await self._safe_record(self.tool_call_repository.record_tool_call, call)

    async def record_routing_decision(self, decision: EvaluationRoutingDecision) -> None:
        await self._safe_record(self.routing_decision_repository.record_routing_decision, decision)

    async def record_claim(self, claim: EvaluationClaim) -> None:
        await self._safe_record(self.claim_repository.record_claim, claim)

    async def record_human_rating(self, rating: EvaluationHumanRating) -> None:
        await self._safe_record(self.human_rating_repository.record_human_rating, rating)

    async def _safe_record(self, record_fn, payload) -> None:
        try:
            await record_fn(payload)
        except Exception:
            logger.warning("Failed to record evaluation observability event", exc_info=True)
            if self.strict:
                raise
