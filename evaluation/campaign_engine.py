"""Async campaign engine for evaluation benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, Optional
from uuid import uuid4

from core.errors import AppError, ErrorCode
from evaluation.campaign_schemas import (
    AblationCondition,
    CampaignMetricsResponse,
    CampaignConfig,
    CampaignCreateResponse,
    CampaignLifecycleStatus,
    CampaignResultStatus,
    CampaignResultsResponse,
    CampaignStatus,
)
from evaluation.db import (
    AgentTraceRepository,
    CampaignRepository,
    CampaignResultRepository,
)
from evaluation.job_schemas import (
    EvaluationAttempt,
    EvaluationJob,
    EvaluationJobItemSummary,
    EvaluationJobType,
    EvaluationRerunRequest,
    EvaluationWorkType,
    WorkItemSpec,
)
from evaluation.job_store import EvaluationJobStore
from evaluation.evidence import (
    build_gold_fact_attrition,
    content_hash,
    estimate_tokens,
    expected_evidence_matches_doc,
    text_mentions_fact,
)
from evaluation.observability import EvaluationRunRecorder
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.agentic_campaign_adapter import effective_agentic_execution_version
from evaluation.rag_modes import BenchmarkExecutionResult, run_campaign_case
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.retrieval_profiles import evaluation_failure_execution_profile
from evaluation.retry import RateBudget
from evaluation.schemas import TestCase
from evaluation.storage import list_test_cases
from evaluation.trace_schemas import (
    AgentTraceDetail,
    AgentTraceSummary,
    EvaluationClaim,
    EvaluationContextPack,
    EvaluationRetrievalChunk,
    EvaluationRetrievalEvent,
    EvaluationRoutingDecision,
    EvaluationToolCall,
    EvaluationTraceEvent,
)

logger = logging.getLogger(__name__)

CampaignRunner = Callable[..., Awaitable[BenchmarkExecutionResult]]
_TERMINAL_STATUSES = {
    CampaignLifecycleStatus.COMPLETED,
    CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
    CampaignLifecycleStatus.FAILED,
    CampaignLifecycleStatus.CANCELLED,
}
_LEGACY_RAGAS_METRIC = "legacy_campaign"


@dataclass(frozen=True)
class CampaignUnit:
    """One question-mode-run execution cell."""

    test_case: TestCase
    mode: str
    run_number: int
    repeat_number: int = 1
    condition_id: str | None = None
    condition_label: str | None = None
    ablation_flags: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    agentic_execution_version: Literal["v8", "v9"] = "v8"
    shadow_evaluation_policy: Literal["operational", "research"] | None = None


@dataclass(frozen=True)
class ExecutedCampaignUnit:
    """One executed unit plus immutable snapshot metadata."""

    unit: CampaignUnit
    payload: BenchmarkExecutionResult | Exception
    run_id: str
    request_id: str
    started_at: datetime
    completed_at: datetime
    total_latency_ms: float
    model_config: dict[str, Any]


def _unit_key(unit: CampaignUnit) -> tuple[str, str, int, str | None]:
    return (unit.test_case.id, unit.mode, unit.run_number, unit.condition_id)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _duration_ms(started_at: datetime, completed_at: datetime) -> float:
    return max((completed_at - started_at).total_seconds() * 1000, 0)


def _extract_total_tokens(token_usage: dict[str, Any]) -> int:
    raw_total = token_usage.get("total_tokens")
    if isinstance(raw_total, dict):
        total = 0
        for value in raw_total.values():
            try:
                total += int(value or 0)
            except (TypeError, ValueError):
                continue
        return total
    try:
        if raw_total is not None:
            return int(raw_total)
    except (TypeError, ValueError):
        pass

    total = 0
    for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens"):
        try:
            total += int(token_usage.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _build_question_snapshot(test_case: TestCase) -> dict[str, Any]:
    return {
        "id": test_case.id,
        "question": test_case.question,
        "ground_truth": test_case.ground_truth,
        "ground_truth_short": test_case.ground_truth_short,
        "key_points": list(test_case.key_points),
        "ragas_focus": list(test_case.ragas_focus),
        "category": test_case.category,
        "difficulty": test_case.difficulty,
        "question_version": test_case.question_version,
        "required_modalities": list(test_case.required_modalities),
        "atomic_facts": list(test_case.atomic_facts),
        "expected_evidence": list(test_case.expected_evidence),
        "source_docs": list(test_case.source_docs),
    }


def _build_system_version_snapshot(
    *,
    unit: CampaignUnit,
    payload: BenchmarkExecutionResult | Exception,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "mode": unit.mode,
        "run_number": unit.run_number,
        "repeat_number": unit.repeat_number,
        "agentic_execution_version": unit.agentic_execution_version,
    }
    if unit.condition_id:
        snapshot["condition_id"] = unit.condition_id
        snapshot["condition_label"] = unit.condition_label
        snapshot["ablation_flags"] = dict(unit.ablation_flags or {})
    if unit.budget:
        snapshot["budget"] = dict(unit.budget)
    if unit.shadow_evaluation_policy:
        snapshot["shadow_evaluation_policy"] = unit.shadow_evaluation_policy
    if isinstance(payload, BenchmarkExecutionResult):
        if payload.execution_profile:
            snapshot["execution_profile"] = payload.execution_profile
        if payload.context_policy_version:
            snapshot["context_policy_version"] = payload.context_policy_version
    return snapshot


def _build_derived_metrics(
    *,
    unit: CampaignUnit,
    payload: BenchmarkExecutionResult | Exception,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "repeat_number": unit.repeat_number,
        "agentic_execution_version": unit.agentic_execution_version,
    }
    if unit.condition_id:
        metrics["condition_id"] = unit.condition_id
        metrics["condition_label"] = unit.condition_label
        metrics["ablation_flags"] = dict(unit.ablation_flags or {})
    if isinstance(payload, Exception):
        metrics["response_status"] = "failed"
        return metrics
    if payload.execution_identity:
        metrics["execution_identity"] = payload.execution_identity
    if payload.shadow_evaluation_policy:
        metrics["shadow_evaluation_policy"] = payload.shadow_evaluation_policy
    metrics["response_status"] = payload.response_status or (
        "failed" if payload.error_message else "complete"
    )
    metrics.update(
        {
            "context_count": len(payload.contexts),
            "source_doc_count": len(payload.source_doc_ids),
            "expected_source_count": len(payload.expected_sources),
        }
    )
    trace_payload = payload.agent_trace or {}
    claims = trace_payload.get("claims") if isinstance(trace_payload, dict) else None
    if isinstance(claims, list):
        supported = sum(
            1 for claim in claims if _claim_support_status(claim) == "supported"
        )
        unsupported = sum(
            1
            for claim in claims
            if _claim_support_status(claim) in {"unsupported", "contradicted"}
        )
        total = len(claims)
        metrics.update(
            {
                "supported_claim_ratio": supported / total if total else 0,
                "unsupported_claim_ratio": unsupported / total if total else 0,
                "citation_precision": supported / total if total else 0,
                "evidence_coverage": supported / total if total else 0,
                "repair_count": sum(
                    1
                    for claim in claims
                    if isinstance(claim, dict) and claim.get("repair_action")
                ),
            }
        )
    if unit.test_case.atomic_facts:
        metrics["gold_fact_attrition"] = build_gold_fact_attrition(
            atomic_facts=list(unit.test_case.atomic_facts),
            expected_evidence=list(unit.test_case.expected_evidence),
            source_doc_ids=list(payload.source_doc_ids),
            contexts=list(payload.contexts),
            answer=payload.answer,
        )
    return metrics


def _final_answer_hash(answer: str | None) -> str | None:
    if not answer:
        return None
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _trace_payload(payload: BenchmarkExecutionResult | Exception) -> dict[str, Any]:
    if not isinstance(payload, BenchmarkExecutionResult):
        return {}
    return payload.agent_trace if isinstance(payload.agent_trace, dict) else {}


def _trace_event_status(value: Any) -> str:
    raw = str(value or "success").lower()
    if raw in {"completed", "ok", "success"}:
        return "success"
    if raw in {"failed", "error"}:
        return "failed"
    if raw in {"running", "skipped", "timeout", "partial"}:
        return raw
    return "success"


def _claim_support_status(claim: Any) -> str:
    if not isinstance(claim, dict):
        return "unsupported"
    raw = str(
        claim.get("support_status")
        or claim.get("status")
        or ("supported" if claim.get("supported") else "unsupported")
    ).lower()
    if raw in {"supported", "partially_supported", "unsupported", "contradicted"}:
        return raw
    return "unsupported"


def _claim_text(claim: dict[str, Any]) -> str:
    return str(
        claim.get("claim_text") or claim.get("claim") or claim.get("text") or ""
    ).strip()


def _enrich_agent_trace_payload(
    *,
    trace_payload: dict[str, Any],
    created_id: str,
    unit: CampaignUnit,
    payload: BenchmarkExecutionResult,
) -> dict[str, Any]:
    enriched = dict(trace_payload)
    enriched.setdefault("campaign_result_id", created_id)
    enriched.setdefault("question_id", payload.question_id or unit.test_case.id)
    enriched.setdefault("question", payload.question or unit.test_case.question)
    enriched.setdefault("mode", payload.mode or unit.mode)
    enriched.setdefault("run_number", unit.run_number)
    enriched.setdefault("repeat_number", unit.repeat_number)
    if unit.condition_id:
        enriched.setdefault("condition_id", unit.condition_id)
        enriched.setdefault("condition_label", unit.condition_label)
        enriched.setdefault("ablation_flags", dict(unit.ablation_flags or {}))
    enriched.setdefault(
        "trace_status", "failed" if payload.error_message else "completed"
    )
    enriched.setdefault("created_at", _utc_now().isoformat())
    return enriched


async def _record_unit_root_span(
    *,
    run_id: str,
    campaign_id: str,
    request_id: str,
    unit: CampaignUnit,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: float,
    failed: bool,
) -> str | None:
    repository = EvaluationObservabilityRepository()
    span_id = str(uuid4())
    created_at = _utc_now()
    payload = {
        "request_id": request_id,
        "question_id": unit.test_case.id,
        "mode": unit.mode,
        "run_number": unit.run_number,
        "repeat_number": unit.repeat_number,
    }
    if unit.condition_id:
        payload["condition_id"] = unit.condition_id
        payload["condition_label"] = unit.condition_label
    error = (
        {"type": "CampaignUnitFailed", "message": "Campaign unit failed"}
        if failed
        else {}
    )
    try:
        await repository.record_trace_events(
            [
                EvaluationTraceEvent(
                    event_id=str(uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=span_id,
                    parent_event_id=None,
                    parent_span_id=None,
                    event_type="campaign_unit_execution",
                    sequence=1,
                    stage_type="generation",
                    stage_name="campaign_unit_execution",
                    started_at=started_at,
                    ended_at=None,
                    duration_ms=None,
                    status="running",
                    payload=payload,
                    error={},
                    created_at=created_at,
                ),
                EvaluationTraceEvent(
                    event_id=str(uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=span_id,
                    parent_event_id=None,
                    parent_span_id=None,
                    event_type="campaign_unit_execution",
                    sequence=2,
                    stage_type="generation",
                    stage_name="campaign_unit_execution",
                    started_at=started_at,
                    ended_at=completed_at,
                    duration_ms=duration_ms,
                    status="failed" if failed else "success",
                    payload=payload,
                    error=error,
                    created_at=created_at,
                ),
            ]
        )
        return span_id
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to record campaign unit observability span",
            extra={
                "campaign_id": campaign_id,
                "run_id": run_id,
                "request_id": request_id,
                "question_id": unit.test_case.id,
                "mode": unit.mode,
            },
            exc_info=True,
        )
        return None


async def _record_unit_llm_usage(
    *,
    run_id: str,
    campaign_id: str,
    user_id: str,
    request_id: str,
    span_id: str | None,
    execution: ExecutedCampaignUnit,
) -> None:
    if not isinstance(execution.payload, BenchmarkExecutionResult):
        return
    if not execution.payload.token_usage:
        return

    model_name = execution.model_config.get("model_name")
    provider = execution.model_config.get("provider")
    if (
        provider is None
        and isinstance(model_name, str)
        and model_name.startswith("gemini")
    ):
        provider = "google"

    recorder = EvaluationRunRecorder(
        run_id=run_id,
        campaign_id=campaign_id,
        user_id=user_id,
        request_id=request_id,
    )
    await recorder.record_llm_usage(
        purpose="campaign_generation",
        provider=provider,
        model_name=str(model_name) if model_name else None,
        usage=execution.payload.token_usage,
        latency_ms=execution.payload.latency_ms,
        status="failed" if execution.payload.error_message else "success",
        error=(
            {"message": execution.payload.error_message}
            if execution.payload.error_message
            else None
        ),
        span_id=span_id,
        payload={
            "request_id": request_id,
            "question_id": execution.unit.test_case.id,
            "mode": execution.unit.mode,
            "run_number": execution.unit.run_number,
            "root_span_recorded": span_id is not None,
        },
        created_at=execution.completed_at,
    )


async def _record_unit_research_observability(
    *,
    run_id: str,
    campaign_id: str,
    user_id: str,
    request_id: str,
    root_span_id: str | None,
    execution: ExecutedCampaignUnit,
) -> None:
    if not isinstance(execution.payload, BenchmarkExecutionResult):
        return

    recorder = EvaluationRunRecorder(
        run_id=run_id,
        campaign_id=campaign_id,
        user_id=user_id,
        request_id=request_id,
    )
    created_at = execution.completed_at
    trace_payload = _trace_payload(execution.payload)
    classifier_decision = trace_payload.get("classifier_decision")
    if isinstance(classifier_decision, dict) or execution.unit.mode == "agentic":
        decision_payload = dict(classifier_decision or {})
        decision_payload.setdefault("router_version", "retrospective-v1")
        decision_payload.setdefault("router_type", "retrospective")
        decision_payload.setdefault("selected_mode", execution.unit.mode)
        decision_payload.setdefault(
            "selected_strategy_tier", trace_payload.get("strategy_tier")
        )
        decision_payload.setdefault("routing_reason", decision_payload.get("reason"))
        decision_payload.setdefault(
            "routing_features", decision_payload.get("features", {})
        )
        decision_payload.setdefault("fallback_used", False)
        decision_payload.setdefault("manual_override", False)
        decision_payload.setdefault("actual_router_execution_enabled", False)
        async with recorder.start_span(
            stage_type="routing",
            stage_name="retrospective_routing_analysis",
            event_type="routing_decision",
            payload={
                "request_id": request_id,
                "question_id": execution.unit.test_case.id,
                "selected_mode": execution.unit.mode,
                "router_version": decision_payload.get("router_version"),
            },
        ) as routing_span:
            await recorder.record_routing_decision(
                EvaluationRoutingDecision(
                    routing_decision_id=str(uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=routing_span.span_id,
                    selected_mode=execution.unit.mode,
                    analysis_type="retrospective",
                    confidence=decision_payload.get("confidence")
                    or trace_payload.get("semantic_gate_score"),
                    reason=decision_payload.get("routing_reason")
                    or decision_payload.get("reason"),
                    payload=decision_payload,
                    created_at=created_at,
                )
            )

    steps = trace_payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for index, tool_call in enumerate(step.get("tool_calls") or [], start=1):
                if not isinstance(tool_call, dict):
                    continue
                action = tool_call.get("action")
                tool_name = str(
                    tool_call.get("tool_name")
                    or tool_call.get("name")
                    or action
                    or step.get("step_type")
                    or "tool"
                )
                payload = {
                    "step_id": step.get("step_id"),
                    "step_type": step.get("step_type"),
                    "subtask_id": tool_call.get("subtask_id") or step.get("subtask_id"),
                    "tool_type": tool_call.get("tool_type")
                    or step.get("step_type")
                    or "tool",
                    "started_at": tool_call.get("started_at") or step.get("started_at"),
                    "ended_at": tool_call.get("ended_at") or step.get("completed_at"),
                    "duration_ms": tool_call.get("duration_ms")
                    or tool_call.get("latency_ms"),
                    "input_summary": tool_call.get("input_summary")
                    or tool_call.get("input_summary_json")
                    or {},
                    "output_summary": tool_call.get("output_summary")
                    or tool_call.get("output_summary_json")
                    or {},
                    "error": tool_call.get("error")
                    or tool_call.get("error_json")
                    or {},
                    "index": index,
                }
                await recorder.record_tool_call(
                    EvaluationToolCall(
                        tool_call_id=str(tool_call.get("tool_call_id") or uuid4()),
                        run_id=run_id,
                        campaign_id=campaign_id,
                        span_id=root_span_id,
                        tool_name=tool_name,
                        action=str(action) if action else None,
                        latency_ms=tool_call.get("latency_ms")
                        or tool_call.get("duration_ms"),
                        status=_trace_event_status(tool_call.get("status")),
                        payload=payload,
                        created_at=created_at,
                    )
                )

    retrieval_event_id = str(uuid4())
    expected_evidence = list(execution.unit.test_case.expected_evidence)
    expected_sources = list(
        execution.payload.expected_sources or execution.unit.test_case.source_docs
    )
    matched_expected = [
        item
        for item in expected_evidence
        if expected_evidence_matches_doc(
            doc_id=str(item.get("doc_id") or item.get("source_doc") or ""),
            expected_evidence=expected_evidence,
            expected_sources=list(execution.payload.source_doc_ids),
        )
    ]
    chunks: list[EvaluationRetrievalChunk] = []
    for index, context in enumerate(execution.payload.contexts, start=1):
        doc_id = (
            execution.payload.source_doc_ids[index - 1]
            if index - 1 < len(execution.payload.source_doc_ids)
            else None
        )
        chunk_id = f"{run_id}:chunk:{index}"
        expected_match = expected_evidence_matches_doc(
            doc_id=doc_id,
            expected_evidence=expected_evidence,
            expected_sources=expected_sources,
        )
        chunks.append(
            EvaluationRetrievalChunk(
                retrieval_chunk_id=str(uuid4()),
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=root_span_id,
                retrieval_event_id=retrieval_event_id,
                chunk_id=chunk_id,
                doc_id=doc_id,
                rank_before_rerank=index,
                rank_after_rerank=index,
                used_in_context=True,
                used_in_answer=expected_match
                or text_mentions_fact(execution.payload.answer, context),
                expected_evidence_match=expected_match,
                excerpt=context[:500],
                content_hash=content_hash(context),
                payload={"instrumentation_depth": "result_level"},
                created_at=created_at,
            )
        )
    hit_rate = (
        (len(matched_expected) / len(expected_evidence)) if expected_evidence else 0
    )
    await recorder.record_retrieval_event(
        EvaluationRetrievalEvent(
            retrieval_event_id=retrieval_event_id,
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=root_span_id,
            query=execution.unit.test_case.question,
            retriever_name=f"{execution.unit.mode}_result_level",
            top_k=len(execution.payload.contexts),
            result_count=len(execution.payload.contexts),
            latency_ms=execution.payload.latency_ms,
            payload={
                "query_type": "campaign_question",
                "retriever_type": execution.unit.mode,
                "top_k_requested": len(execution.payload.contexts),
                "top_k_returned": len(execution.payload.contexts),
                "filters": {},
                "candidate_count": len(execution.payload.contexts),
                "empty_retrieval": len(execution.payload.contexts) == 0,
                "retrieval_confidence": None,
                "required_doc_hit_rate": hit_rate,
                "expected_evidence_hit_rate": hit_rate,
                "instrumentation_depth": "result_level",
            },
            created_at=created_at,
        )
    )
    for chunk in chunks:
        await recorder.record_retrieval_chunk(chunk)

    selected_chunk_ids = [chunk.chunk_id for chunk in chunks]
    retrieved_but_not_packed = [
        {"doc_id": item.get("doc_id"), "evidence_id": item.get("evidence_id")}
        for item in expected_evidence
        if str(item.get("doc_id") or "")
        not in {chunk.doc_id for chunk in chunks if chunk.expected_evidence_match}
    ]
    await recorder.record_context_pack(
        EvaluationContextPack(
            context_pack_id=str(uuid4()),
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=root_span_id,
            input_chunk_count=len(chunks),
            packed_chunk_count=len(selected_chunk_ids),
            token_count=sum(
                estimate_tokens(context) for context in execution.payload.contexts
            ),
            retrieved_but_not_packed_evidence=retrieved_but_not_packed,
            payload={
                "selected_chunk_ids": selected_chunk_ids,
                "dropped_chunk_ids": [],
                "token_budget": execution.model_config.get("max_input_tokens"),
                "estimated_tokens": sum(
                    estimate_tokens(context) for context in execution.payload.contexts
                ),
                "packing_policy": "result_level_contexts",
                "drop_reasons": {},
                "instrumentation_depth": "result_level",
            },
            created_at=created_at,
        )
    )

    claims = trace_payload.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_text = _claim_text(claim)
            if not claim_text:
                continue
            await recorder.record_claim(
                EvaluationClaim(
                    claim_id=str(claim.get("claim_id") or uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=root_span_id,
                    claim_text=claim_text,
                    claim_type=claim.get("claim_type") or claim.get("type"),
                    support_status=_claim_support_status(claim),
                    evidence=claim.get("evidence") or claim.get("evidence_rows") or [],
                    unsupported_reason=claim.get("unsupported_reason")
                    or claim.get("reason"),
                    payload={
                        "support_score": claim.get("support_score"),
                        "evidence_chunk_ids": claim.get("evidence_chunk_ids") or [],
                        "contradicting_chunk_ids": claim.get("contradicting_chunk_ids")
                        or [],
                        "verifier_model": claim.get("verifier_model"),
                        "repair_action": claim.get("repair_action"),
                        "post_repair_status": claim.get("post_repair_status"),
                    },
                    created_at=created_at,
                )
            )


async def _cancel_and_drain_tasks(tasks: list[asyncio.Task]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class CampaignEngine:
    """Create, run, cancel, and inspect evaluation campaigns."""

    def __init__(
        self,
        campaign_repository: Optional[CampaignRepository] = None,
        result_repository: Optional[CampaignResultRepository] = None,
        trace_repository: Optional[AgentTraceRepository] = None,
        ragas_evaluator: Optional[RagasEvaluator] = None,
        runner: CampaignRunner = run_campaign_case,
        job_store: EvaluationJobStore | None = None,
        worker_notifier: Callable[[], None] | None = None,
        configure_worker: bool = True,
    ) -> None:
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._result_repository = result_repository or CampaignResultRepository()
        self._trace_repository = trace_repository or AgentTraceRepository()
        self._ragas_evaluator = ragas_evaluator or RagasEvaluator(
            result_repository=self._result_repository,
        )
        self._runner = runner
        self._job_store = job_store or EvaluationJobStore()
        worker = None
        self._worker = None
        self._worker_owned = False
        if worker_notifier is None:
            from evaluation.execution_worker import DatasetExecutionWorker
            from evaluation.accounting_store import EvaluationAccountingStore
            from evaluation.job_worker import configure_evaluation_job_worker
            from evaluation.job_worker import EvaluationJobWorker
            from evaluation.job_worker import get_evaluation_job_worker
            from evaluation.ragas_worker import RagasBatchWorker

            # The application singleton is reserved for the real process
            # engine.  Injected runners/evaluators (tests and embedded
            # callers) get an isolated worker so a lifespan-owned singleton
            # cannot process their ledger with stale handlers.
            use_process_worker = (
                runner is run_campaign_case
                and ragas_evaluator is None
                and job_store is None
            )
            accounting_store = EvaluationAccountingStore()
            worker = (
                get_evaluation_job_worker()
                if use_process_worker
                else EvaluationJobWorker(
                    store=self._job_store,
                    stop_when_idle=True,
                )
            )
            execution_handler = DatasetExecutionWorker(
                store=self._job_store,
                runner=runner,
                result_repository=self._result_repository,
                ragas_evaluator=self._ragas_evaluator,
                notify=worker.notify,
            ).execute
            ragas_batch_handler = RagasBatchWorker(
                store=self._job_store,
                evaluator=self._ragas_evaluator,
                campaign_repository=self._campaign_repository,
                accounting_store=accounting_store,
            ).execute
            if configure_worker:
                if use_process_worker:
                    configure_evaluation_job_worker(
                        execution_handler=execution_handler,
                        ragas_batch_handler=ragas_batch_handler,
                    )
                else:
                    worker.configure_handlers(
                        execution_handler=execution_handler,
                        ragas_batch_handler=ragas_batch_handler,
                    )
            worker_notifier = worker.notify
            self._worker = worker
            self._worker_owned = not use_process_worker
        self._worker_notifier = worker_notifier
        if getattr(self._job_store, "_on_job_created", None) is None:
            self._job_store._on_job_created = (
                worker.notify if worker is not None else worker_notifier
            )

    async def create_and_start(
        self,
        *,
        user_id: str,
        name: Optional[str],
        config: CampaignConfig,
    ) -> CampaignCreateResponse:
        if "router" in config.modes and not config.actual_router_execution_enabled:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="router mode is not implemented yet; use retrospective router analysis.",
                status_code=400,
            )
        resolved_cases = await self._resolve_test_cases(
            user_id=user_id, test_case_ids=config.test_case_ids
        )
        created = await self._campaign_repository.create(
            user_id=user_id, name=name, config=config
        )
        units = self._build_units(
            test_cases=resolved_cases,
            modes=config.modes,
            repeat_count=config.repeat_count,
            ablation_conditions=config.ablation_conditions,
            agentic_execution_version=config.agentic_execution_version,
            shadow_evaluation_policy=config.shadow_evaluation_policy,
        )
        await self._job_store.create_job_with_items(
            user_id=user_id,
            campaign_id=created.id,
            job_type=EvaluationJobType.INITIAL,
            selection={"campaign_id": created.id},
            config_snapshot=config.model_dump(mode="json", by_alias=True),
            items=[
                self._work_item_spec(
                    user_id=user_id, campaign_id=created.id, unit=unit, config=config
                )
                for unit in units
            ],
        )
        await self._start_worker_if_available()
        if self._worker_notifier is not None:
            self._worker_notifier()
        return CampaignCreateResponse(campaign_id=created.id, status=created.status)

    async def list_campaigns(self, *, user_id: str) -> list[CampaignStatus]:
        campaigns = await self._campaign_repository.list_by_user(user_id=user_id)
        return [
            await self._reconcile_read_status(user_id=user_id, campaign=campaign)
            for campaign in campaigns
        ]

    async def get_campaign(self, *, user_id: str, campaign_id: str) -> CampaignStatus:
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        return await self._reconcile_read_status(user_id=user_id, campaign=campaign)

    async def _reconcile_read_status(
        self, *, user_id: str, campaign: CampaignStatus
    ) -> CampaignStatus:
        """Repair a stale evaluating projection from the durable metric ledger.

        Worker callbacks are intentionally best-effort around event-loop
        shutdowns. A read must still converge once every durable RAGAS item is
        terminal, while unresolved work remains visibly evaluating.
        """
        if (
            campaign.status == CampaignLifecycleStatus.EVALUATING
            and campaign.evaluation_total_units > 0
        ):
            return await self._campaign_repository.derive_ragas_state(
                user_id=user_id,
                campaign_id=campaign.id,
            )
        return campaign

    async def get_results(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignResultsResponse:
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        return CampaignResultsResponse(campaign=campaign, results=results)

    async def get_metrics(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignMetricsResponse:
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._ragas_evaluator.get_metrics(
            user_id=user_id, campaign=campaign
        )

    async def list_traces(
        self, *, user_id: str, campaign_id: str
    ) -> list[AgentTraceSummary]:
        await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._trace_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )

    async def get_trace(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_result_id: str,
    ) -> AgentTraceDetail:
        await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        return await self._trace_repository.get_for_result(
            user_id=user_id,
            campaign_id=campaign_id,
            campaign_result_id=campaign_result_id,
        )

    async def cancel_campaign(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignStatus:
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.status in _TERMINAL_STATUSES:
            return campaign

        await self._campaign_repository.request_cancel(
            user_id=user_id, campaign_id=campaign_id
        )
        await self._job_store.cancel_campaign_jobs(
            user_id=user_id, campaign_id=campaign_id
        )
        return await self._campaign_repository.mark_cancelled(
            user_id=user_id, campaign_id=campaign_id
        )

    async def create_rerun(
        self,
        *,
        user_id: str,
        campaign_id: str,
        request: EvaluationRerunRequest,
    ) -> EvaluationJob:
        """Create one durable rerun job from immutable campaign work.

        Execution reruns reuse the original work snapshots, while metric-only
        reruns target the campaign's current successful official results.  The
        worker creates downstream RAGAS work after an execution rerun promotes
        a result, so a combined rerun never evaluates an uncommitted payload.
        """
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.status in {
            CampaignLifecycleStatus.RUNNING,
            CampaignLifecycleStatus.EVALUATING,
        }:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Campaign is already running",
                status_code=400,
            )

        configured_metrics = getattr(self._ragas_evaluator, "enabled_metrics", None)
        if configured_metrics is None:
            configured_metrics = (
                [_LEGACY_RAGAS_METRIC]
                if callable(getattr(self._ragas_evaluator, "evaluate_campaign", None))
                else ["faithfulness", "answer_correctness", "answer_relevancy"]
            )
        enabled_metrics = list(configured_metrics)
        if request.stages != "execution":
            unknown_metrics = [
                name for name in request.metric_names if name not in enabled_metrics
            ]
            if unknown_metrics:
                raise AppError(
                    code=ErrorCode.BAD_REQUEST,
                    message=f"Unknown RAGAS metrics: {', '.join(unknown_metrics)}",
                    status_code=400,
                )

        includes_execution = request.stages in {"execution", "execution_and_ragas"}
        if includes_execution:
            rows = await self._job_store.list_campaign_work_items(
                user_id=user_id,
                campaign_id=campaign_id,
                work_type=EvaluationWorkType.DATASET_EXECUTION,
            )
            selected_rows = self._select_rerun_work_rows(
                rows, request=request, kind="execution"
            )
            if not selected_rows:
                raise AppError(
                    code=ErrorCode.BAD_REQUEST,
                    message="No matching execution work is available for rerun",
                    status_code=400,
                )
            specs = [
                WorkItemSpec(
                    work_item_id=str(row["work_item_id"]),
                    work_type=EvaluationWorkType.DATASET_EXECUTION,
                    logical_key=str(row["logical_key"]),
                    input_snapshot=dict(row["input_snapshot"]),
                )
                for row in selected_rows
            ]
            downstream_question_ids = sorted(
                {
                    str(row["input_snapshot"].get("test_case", {}).get("id"))
                    for row in selected_rows
                    if row["input_snapshot"].get("test_case", {}).get("id")
                }
            )
            job = await self._job_store.create_job_with_items(
                user_id=user_id,
                campaign_id=campaign_id,
                job_type=EvaluationJobType.RERUN,
                selection=request.model_dump(mode="json"),
                config_snapshot={
                    "campaign_config": campaign.config.model_dump(
                        mode="json", by_alias=True
                    ),
                    "stages": request.stages,
                    "skip_ragas": request.stages == "execution",
                    "metric_names": list(request.metric_names),
                    "downstream_question_ids": downstream_question_ids,
                },
                items=specs,
            )
            await self._campaign_repository.mark_running(
                user_id=user_id,
                campaign_id=campaign_id,
            )
            await self._start_worker_if_available()
            if self._worker_notifier is not None:
                self._worker_notifier()
            return job

        metric_names = (
            list(request.metric_names) if request.metric_names else enabled_metrics
        )
        unknown_metrics = [name for name in metric_names if name not in enabled_metrics]
        if unknown_metrics:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=f"Unknown RAGAS metrics: {', '.join(unknown_metrics)}",
                status_code=400,
            )

        # Legacy campaigns created before the durable ledger have no attempt
        # provenance.  Backfill deterministic synthetic attempts so the
        # compatibility endpoint can still enqueue a metric-only rerun.
        await self._job_store.backfill_legacy_attempts()
        results = await self._result_repository.list_for_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
        )
        completed_results = [
            row
            for row in results
            if row.status == CampaignResultStatus.COMPLETED
            and row.source_attempt_id is not None
        ]
        metric_names_by_result: dict[str, list[str]] | None = None
        if request.scope == "selected":
            question_ids = set(request.question_ids)
            completed_results = [
                row for row in completed_results if row.question_id in question_ids
            ]
        elif request.scope == "failed_only":
            failed_rows = await self._job_store.list_campaign_work_items(
                user_id=user_id,
                campaign_id=campaign_id,
                work_type=EvaluationWorkType.RAGAS_METRIC,
            )
            failed_keys = {
                (
                    str(row["input_snapshot"].get("campaign_result_id") or ""),
                    str(row["input_snapshot"].get("metric_name") or ""),
                )
                for row in failed_rows
                if row["status"] in {"failed", "interrupted"}
            }
            completed_results = [
                row
                for row in completed_results
                if any(
                    row.id == result_id and metric in metric_names
                    for result_id, metric in failed_keys
                )
            ]
            failed_metric_names = {metric for _, metric in failed_keys}
            metric_names = [
                metric for metric in metric_names if metric in failed_metric_names
            ]
            metric_names_by_result = {}
            for result_id, metric in failed_keys:
                if metric in metric_names:
                    metric_names_by_result.setdefault(result_id, []).append(metric)

        if not completed_results or not metric_names:
            message = (
                "Requested question_ids have no completed raw results in this campaign"
                if request.scope == "selected"
                else "No matching completed results are available for RAGAS rerun"
            )
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=message,
                status_code=400,
            )

        existing_jobs = {
            job.job_id
            for job in await self._job_store.list_jobs(
                user_id=user_id, campaign_id=campaign_id
            )
        }
        selected_result_ids = [row.id for row in completed_results]
        created_count = await self._job_store.ensure_ragas_work(
            user_id=user_id,
            campaign_id=campaign_id,
            evaluator_model=str(getattr(self._ragas_evaluator, "evaluator_model", "")),
            evaluator_config={},
            enabled_metrics=metric_names,
            selected_result_ids=selected_result_ids,
            **(
                {"metric_names_by_result": metric_names_by_result}
                if metric_names_by_result is not None
                else {}
            ),
            force=True,
            ragas_batch_size=campaign.config.ragas_batch_size,
            ragas_parallel_batches=campaign.config.ragas_parallel_batches,
        )
        if not created_count:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="No RAGAS work was created for rerun",
                status_code=400,
            )
        await self._start_worker_if_available()
        if self._worker_notifier is not None:
            self._worker_notifier()
        jobs = await self._job_store.list_jobs(user_id=user_id, campaign_id=campaign_id)
        new_jobs = [job for job in jobs if job.job_id not in existing_jobs]
        if not new_jobs:
            raise AppError(
                code=ErrorCode.INTERNAL_ERROR,
                message="RAGAS rerun job was not persisted",
                status_code=500,
            )
        return max(new_jobs, key=lambda job: job.created_at)

    @staticmethod
    def _select_rerun_work_rows(
        rows: list[dict[str, Any]],
        *,
        request: EvaluationRerunRequest,
        kind: str,
    ) -> list[dict[str, Any]]:
        if request.scope == "all":
            return rows
        if request.scope == "failed_only":
            return [row for row in rows if row["status"] in {"failed", "interrupted"}]
        selected_ids = set(request.question_ids)
        selected: list[dict[str, Any]] = []
        for row in rows:
            snapshot = row["input_snapshot"]
            if kind == "execution":
                question_id = snapshot.get("test_case", {}).get("id")
            else:
                question_id = snapshot.get("result", {}).get("question_id")
            if question_id in selected_ids:
                selected.append(row)
        return selected

    async def list_jobs(self, *, user_id: str, campaign_id: str) -> list[EvaluationJob]:
        await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        return await self._job_store.list_jobs(user_id=user_id, campaign_id=campaign_id)

    async def get_job(self, *, user_id: str, job_id: str) -> EvaluationJob:
        return await self._job_store.get_job(user_id=user_id, job_id=job_id)

    async def list_job_items(
        self, *, user_id: str, job_id: str
    ) -> list[EvaluationJobItemSummary]:
        return await self._job_store.list_job_items(user_id=user_id, job_id=job_id)

    async def cancel_job(self, *, user_id: str, job_id: str) -> EvaluationJob:
        job = await self._job_store.get_job(user_id=user_id, job_id=job_id)
        campaign = None
        if job.campaign_id:
            campaign = await self._campaign_repository.get(
                user_id=user_id, campaign_id=job.campaign_id
            )
        work_types = await self._job_store.get_job_work_types(
            user_id=user_id, job_id=job_id
        )
        cancelled = await self._job_store.cancel_job(user_id=user_id, job_id=job_id)
        if campaign is not None:
            if (
                EvaluationWorkType.DATASET_EXECUTION in work_types
                or campaign.status == CampaignLifecycleStatus.RUNNING
            ):
                await self._campaign_repository.derive_execution_state(
                    user_id=user_id, campaign_id=campaign.id
                )
            if (
                EvaluationWorkType.RAGAS_METRIC in work_types
                or campaign.status == CampaignLifecycleStatus.EVALUATING
            ):
                await self._campaign_repository.derive_ragas_state(
                    user_id=user_id, campaign_id=campaign.id
                )
        return cancelled

    async def list_attempts(
        self, *, user_id: str, work_item_id: str
    ) -> list[EvaluationAttempt]:
        return await self._job_store.list_attempts(
            user_id=user_id, work_item_id=work_item_id
        )

    async def evaluate_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        question_ids: Optional[list[str]] = None,
    ) -> CampaignStatus:
        request = EvaluationRerunRequest(
            scope="selected" if question_ids else "all",
            stages="ragas",
            question_ids=list(question_ids or []),
        )
        await self.create_rerun(
            user_id=user_id,
            campaign_id=campaign_id,
            request=request,
        )
        campaign = await self.get_campaign(user_id=user_id, campaign_id=campaign_id)
        if campaign.status in _TERMINAL_STATUSES or request.question_ids:
            # The local worker may finish a tiny legacy-compatible rerun
            # before the HTTP handler serializes its response.  Preserve the
            # historical contract that POST /evaluate acknowledges the new
            # evaluation phase; subsequent polling reads the durable terminal
            # state from the database.
            results = await self._result_repository.list_for_campaign(
                user_id=user_id, campaign_id=campaign_id
            )
            selected_count = (
                len(
                    [
                        row
                        for row in results
                        if row.status == CampaignResultStatus.COMPLETED
                        and (
                            not request.question_ids
                            or row.question_id in request.question_ids
                        )
                    ]
                )
                if request.question_ids
                else campaign.evaluation_total_units
            )
            campaign = campaign.model_copy(
                update={
                    "status": CampaignLifecycleStatus.EVALUATING,
                    "phase": "evaluation",
                    "evaluation_completed_units": 0,
                    "evaluation_total_units": selected_count,
                }
            )
        return campaign

    async def recover_inflight_campaigns(self) -> None:
        """Recover non-terminal campaigns after process restart."""
        inflight = await self._campaign_repository.list_inflight()
        if not inflight:
            return

        drain_owned = (
            self._worker_owned
            and self._worker is not None
            and not self._worker.is_running
        )
        for user_id, campaign in inflight:
            try:
                await self._prepare_legacy_recovery(
                    user_id=user_id,
                    campaign=campaign,
                )
                await self.ensure_campaign_task(
                    user_id=user_id,
                    campaign_id=campaign.id,
                    campaign_snapshot=campaign,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to recover campaign %s for user %s: %s",
                    campaign.id,
                    user_id,
                    exc,
                    exc_info=True,
                )
                if isinstance(exc, AppError):
                    await self._campaign_repository.mark_failed(
                        user_id=user_id,
                        campaign_id=campaign.id,
                        error_message=str(exc),
                        phase="execution",
                    )
        if drain_owned and self._worker is not None:
            await self._worker.run_until_idle()

    async def ensure_campaign_task(
        self,
        *,
        user_id: str,
        campaign_id: str,
        campaign_snapshot: CampaignStatus | None = None,
    ) -> CampaignStatus:
        """Ensure one non-terminal campaign has a running task or terminal state."""
        campaign = campaign_snapshot or await self.get_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.status in _TERMINAL_STATUSES:
            return campaign

        self._worker_notifier()
        return campaign

    async def _start_worker_if_available(self) -> None:
        if self._worker is not None:
            await self._worker.start()

    async def _prepare_legacy_recovery(
        self, *, user_id: str, campaign: CampaignStatus
    ) -> None:
        """Bridge pre-ledger campaigns into the durable recovery path."""
        if campaign.cancel_requested:
            await self._campaign_repository.mark_cancelled(
                user_id=user_id, campaign_id=campaign.id
            )
            return

        durable_items = await self._job_store.list_campaign_work_items(
            user_id=user_id, campaign_id=campaign.id
        )
        if durable_items:
            # A normal ledger-backed campaign is recovered by the process
            # worker itself.  The compatibility bridge below is only for
            # campaigns that predate the ledger entirely.
            return

        await self._job_store.backfill_legacy_attempts()
        configured_metrics = getattr(self._ragas_evaluator, "enabled_metrics", None)
        if configured_metrics is None:
            configured_metrics = (
                [_LEGACY_RAGAS_METRIC]
                if callable(getattr(self._ragas_evaluator, "evaluate_campaign", None))
                else ["faithfulness", "answer_correctness", "answer_relevancy"]
            )
        metric_names = list(configured_metrics)
        if campaign.status == CampaignLifecycleStatus.EVALUATING:
            results = await self._result_repository.list_for_campaign(
                user_id=user_id, campaign_id=campaign.id
            )
            selected_ids = [
                row.id
                for row in results
                if row.status == CampaignResultStatus.COMPLETED
            ]
            created = await self._job_store.ensure_ragas_work(
                user_id=user_id,
                campaign_id=campaign.id,
                evaluator_model=str(
                    getattr(self._ragas_evaluator, "evaluator_model", "")
                ),
                evaluator_config={},
                enabled_metrics=metric_names,
                selected_result_ids=selected_ids,
                ragas_batch_size=campaign.config.ragas_batch_size,
                ragas_parallel_batches=campaign.config.ragas_parallel_batches,
            )
            if not created:
                await self._campaign_repository.mark_completed(
                    user_id=user_id,
                    campaign_id=campaign.id,
                    phase="evaluation",
                    completed_units=len(selected_ids),
                )
            return

        existing = await self._job_store.list_campaign_work_items(
            user_id=user_id,
            campaign_id=campaign.id,
            work_type=EvaluationWorkType.DATASET_EXECUTION,
        )
        test_cases = await self._resolve_test_cases(
            user_id=user_id, test_case_ids=campaign.config.test_case_ids
        )
        units = self._build_units(
            test_cases=test_cases,
            modes=campaign.config.modes,
            repeat_count=campaign.config.repeat_count,
            ablation_conditions=campaign.config.ablation_conditions,
            agentic_execution_version=campaign.config.agentic_execution_version,
            shadow_evaluation_policy=campaign.config.shadow_evaluation_policy,
        )
        existing_keys = {str(row.get("logical_key")) for row in existing}
        result_by_id = {
            row.id: row
            for row in await self._result_repository.list_for_campaign(
                user_id=user_id, campaign_id=campaign.id
            )
        }
        for row in existing:
            logical_key = str(row.get("logical_key") or "")
            if not logical_key.startswith("legacy:execution:"):
                continue
            legacy_result = result_by_id.get(logical_key.rsplit(":", 1)[-1])
            if legacy_result is not None:
                existing_keys.add(
                    f"execution:{legacy_result.question_id}:{legacy_result.mode}:"
                    f"{legacy_result.run_number}:none"
                )
        missing_specs = [
            spec
            for unit in units
            for spec in [
                self._work_item_spec(
                    user_id=user_id,
                    campaign_id=campaign.id,
                    unit=unit,
                    config=campaign.config,
                )
            ]
            if spec.logical_key not in existing_keys
        ]
        if missing_specs:
            await self._job_store.create_job_with_items(
                user_id=user_id,
                campaign_id=campaign.id,
                job_type=EvaluationJobType.INITIAL,
                selection={"campaign_id": campaign.id},
                config_snapshot=campaign.config.model_dump(mode="json", by_alias=True),
                items=missing_specs,
            )

    async def _run_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        config: CampaignConfig,
        test_cases: list[TestCase],
        units: list[CampaignUnit] | None = None,
        initial_completed_units: int = 0,
        total_units_override: int | None = None,
    ) -> None:
        batch_tasks: list[asyncio.Task] = []
        try:
            await self._campaign_repository.mark_running(
                user_id=user_id, campaign_id=campaign_id
            )
            pending_units = units
            if pending_units is None:
                pending_units = self._build_units(
                    test_cases=test_cases,
                    modes=config.modes,
                    repeat_count=config.repeat_count,
                    ablation_conditions=config.ablation_conditions,
                    agentic_execution_version=config.agentic_execution_version,
                    shadow_evaluation_policy=config.shadow_evaluation_policy,
                )
            rate_budget = RateBudget(rpm_limit=config.rpm_limit)
            completed_units = initial_completed_units
            total_units = total_units_override or (
                initial_completed_units + len(pending_units)
            )

            for offset in range(0, len(pending_units), config.batch_size):
                if await self._campaign_repository.is_cancel_requested(
                    user_id=user_id, campaign_id=campaign_id
                ):
                    await self._campaign_repository.mark_cancelled(
                        user_id=user_id, campaign_id=campaign_id
                    )
                    return

                batch = pending_units[offset : offset + config.batch_size]
                batch_tasks = [
                    asyncio.create_task(
                        self._execute_unit(
                            unit=unit,
                            user_id=user_id,
                            model_config=config.model_preset.model_dump(mode="json"),
                            rate_budget=rate_budget,
                            run_number=unit.run_number,
                            repeat_number=unit.repeat_number,
                            ablation_flags=unit.ablation_flags,
                            budget=unit.budget,
                        )
                    )
                    for unit in batch
                ]

                for completed_task in asyncio.as_completed(batch_tasks):
                    execution = await completed_task
                    unit = execution.unit
                    result = await self._persist_unit_result(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        execution=execution,
                    )
                    completed_units += 1
                    await self._campaign_repository.update_progress(
                        user_id=user_id,
                        campaign_id=campaign_id,
                        completed_units=completed_units,
                        current_question_id=unit.test_case.id,
                        current_mode=unit.mode,
                    )
                    logger.info(
                        "Campaign %s progress %s/%s latest_result=%s",
                        campaign_id,
                        completed_units,
                        total_units,
                        result.id,
                    )

            await self._run_ragas_evaluation(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                ragas_batch_size=config.ragas_batch_size,
                ragas_parallel_batches=config.ragas_parallel_batches,
                ragas_rpm_limit=config.ragas_rpm_limit,
            )
        except asyncio.CancelledError:
            await _cancel_and_drain_tasks(batch_tasks)
            if user_id and campaign_id:
                await self._campaign_repository.mark_cancelled(
                    user_id=user_id, campaign_id=campaign_id
                )
            raise
        except Exception as exc:  # noqa: BLE001
            await _cancel_and_drain_tasks(batch_tasks)
            logger.error("Campaign %s failed: %s", campaign_id, exc, exc_info=True)
            await self._campaign_repository.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message=str(exc),
                phase="execution",
            )

    async def _run_evaluation_only(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_total_units: int,
        ragas_batch_size: int,
        ragas_parallel_batches: int,
        ragas_rpm_limit: int,
        selected_result_ids: Optional[list[str]] = None,
    ) -> None:
        try:
            await self._evaluate_campaign_results(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                evaluation_total_units=evaluation_total_units,
                ragas_batch_size=ragas_batch_size,
                ragas_parallel_batches=ragas_parallel_batches,
                ragas_rpm_limit=ragas_rpm_limit,
                selected_result_ids=selected_result_ids,
            )
            await self._campaign_repository.mark_completed(
                user_id=user_id,
                campaign_id=campaign_id,
                phase="evaluation",
            )
        except asyncio.CancelledError:
            await self._campaign_repository.mark_cancelled(
                user_id=user_id, campaign_id=campaign_id
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Campaign %s evaluation rerun failed: %s",
                campaign_id,
                exc,
                exc_info=True,
            )
            await self._campaign_repository.mark_failed(
                user_id=user_id,
                campaign_id=campaign_id,
                error_message=str(exc),
                phase="evaluation",
            )

    async def _run_ragas_evaluation(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        ragas_batch_size: int,
        ragas_parallel_batches: int,
        ragas_rpm_limit: int,
    ) -> None:
        results = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed_results = [
            row for row in results if row.status == CampaignResultStatus.COMPLETED
        ]
        campaign = await self._campaign_repository.get(
            user_id=user_id, campaign_id=campaign_id
        )
        if campaign.config.shadow_evaluation_policy == "operational":
            completed_results = [
                row for row in completed_results if row.mode != "agentic-v9-shadow"
            ]
        if not completed_results:
            await self._campaign_repository.mark_completed(
                user_id=user_id, campaign_id=campaign_id
            )
            return

        await self._campaign_repository.mark_evaluating(
            user_id=user_id,
            campaign_id=campaign_id,
            evaluation_total_units=len(completed_results),
        )
        await self._evaluate_campaign_results(
            user_id=user_id,
            campaign_id=campaign_id,
            completed_units=completed_units,
            evaluation_total_units=len(completed_results),
            ragas_batch_size=ragas_batch_size,
            ragas_parallel_batches=ragas_parallel_batches,
            ragas_rpm_limit=ragas_rpm_limit,
        )
        await self._campaign_repository.mark_completed(
            user_id=user_id,
            campaign_id=campaign_id,
            phase="evaluation",
        )

    async def _evaluate_campaign_results(
        self,
        *,
        user_id: str,
        campaign_id: str,
        completed_units: int,
        evaluation_total_units: int,
        ragas_batch_size: int,
        ragas_parallel_batches: int,
        ragas_rpm_limit: int,
        selected_result_ids: Optional[list[str]] = None,
    ) -> None:
        async def on_progress(
            evaluation_completed_units: int,
            _evaluation_total_units: int,
            current_question_id: str | None,
            current_mode: str | None,
        ) -> None:
            await self._campaign_repository.update_progress(
                user_id=user_id,
                campaign_id=campaign_id,
                completed_units=completed_units,
                evaluation_completed_units=evaluation_completed_units,
                evaluation_total_units=evaluation_total_units,
                current_question_id=current_question_id,
                current_mode=current_mode,
            )

        await self._ragas_evaluator.evaluate_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
            ragas_batch_size=ragas_batch_size,
            ragas_parallel_batches=ragas_parallel_batches,
            ragas_rpm_limit=ragas_rpm_limit,
            selected_result_ids=selected_result_ids,
            on_progress=on_progress,
        )

    async def _execute_unit(
        self,
        *,
        unit: CampaignUnit,
        user_id: str,
        model_config: dict,
        rate_budget: RateBudget,
        run_number: int,
        repeat_number: int,
        ablation_flags: dict[str, Any] | None,
        budget: dict[str, Any] | None,
    ) -> ExecutedCampaignUnit:
        await rate_budget.acquire()
        run_id = str(uuid4())
        request_id = str(uuid4())
        started_at = _utc_now()
        runner_started_perf = time.perf_counter()
        try:
            payload = await self._runner(
                test_case=unit.test_case,
                user_id=user_id,
                mode=unit.mode,
                model_config=model_config,
                run_number=repeat_number,
                ablation_flags=ablation_flags,
                budget=budget,
                agentic_execution_version=unit.agentic_execution_version,
                shadow_evaluation_policy=unit.shadow_evaluation_policy,
            )
        except Exception as exc:  # noqa: BLE001
            payload = exc
        completed_at = _utc_now()
        total_latency_ms = max((time.perf_counter() - runner_started_perf) * 1000, 0)
        if total_latency_ms <= 0:
            total_latency_ms = _duration_ms(started_at, completed_at)
        return ExecutedCampaignUnit(
            unit=unit,
            payload=payload,
            run_id=run_id,
            request_id=request_id,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency_ms,
            model_config=dict(model_config),
        )

    async def _persist_unit_result(
        self,
        *,
        user_id: str,
        campaign_id: str,
        execution: ExecutedCampaignUnit,
    ):
        unit = execution.unit
        payload = execution.payload
        question_snapshot = _build_question_snapshot(unit.test_case)
        total_tokens = (
            _extract_total_tokens(payload.token_usage)
            if isinstance(payload, BenchmarkExecutionResult)
            else None
        )
        system_version_snapshot = _build_system_version_snapshot(
            unit=unit, payload=payload
        )
        derived_metrics = _build_derived_metrics(unit=unit, payload=payload)

        if isinstance(payload, Exception):
            created = await self._result_repository.create(
                result_id=execution.run_id,
                user_id=user_id,
                campaign_id=campaign_id,
                question_id=unit.test_case.id,
                question=unit.test_case.question,
                ground_truth=unit.test_case.ground_truth,
                ground_truth_short=unit.test_case.ground_truth_short,
                key_points=list(unit.test_case.key_points),
                ragas_focus=list(unit.test_case.ragas_focus),
                mode=unit.mode,
                execution_profile=evaluation_failure_execution_profile(
                    unit.mode,
                    payload,
                ),
                context_policy_version=None,
                run_number=unit.run_number,
                answer=f"ERROR: {payload}",
                contexts=[],
                source_doc_ids=[],
                expected_sources=list(unit.test_case.source_docs),
                latency_ms=0,
                token_usage={},
                category=unit.test_case.category,
                difficulty=unit.test_case.difficulty,
                status=CampaignResultStatus.FAILED,
                error_message=str(payload),
                question_version=unit.test_case.question_version,
                request_id=execution.request_id,
                started_at=execution.started_at.isoformat(),
                completed_at=execution.completed_at.isoformat(),
                total_latency_ms=execution.total_latency_ms,
                total_tokens=total_tokens,
                question_snapshot=question_snapshot,
                model_config_snapshot=execution.model_config,
                system_version_snapshot=system_version_snapshot,
                derived_metrics=derived_metrics,
            )
            span_id = await _record_unit_root_span(
                run_id=created.id,
                campaign_id=campaign_id,
                request_id=execution.request_id,
                unit=unit,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_ms=execution.total_latency_ms,
                failed=True,
            )
            await _record_unit_llm_usage(
                run_id=created.id,
                campaign_id=campaign_id,
                user_id=user_id,
                request_id=execution.request_id,
                span_id=span_id,
                execution=execution,
            )
            await _record_unit_research_observability(
                run_id=created.id,
                campaign_id=campaign_id,
                user_id=user_id,
                request_id=execution.request_id,
                root_span_id=span_id,
                execution=execution,
            )
            trace_payload = getattr(payload, "agent_trace", None)
            if trace_payload:
                await self._trace_repository.replace_for_result(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    campaign_result_id=created.id,
                    trace_payload=trace_payload,
                )
            return created

        created = await self._result_repository.create(
            result_id=execution.run_id,
            user_id=user_id,
            campaign_id=campaign_id,
            question_id=payload.question_id,
            question=payload.question,
            ground_truth=payload.ground_truth,
            ground_truth_short=payload.ground_truth_short,
            key_points=list(payload.key_points),
            ragas_focus=list(payload.ragas_focus),
            mode=payload.mode,
            execution_profile=payload.execution_profile,
            context_policy_version=payload.context_policy_version,
            run_number=unit.run_number,
            answer=payload.answer,
            contexts=payload.contexts,
            source_doc_ids=payload.source_doc_ids,
            expected_sources=payload.expected_sources,
            latency_ms=payload.latency_ms,
            token_usage=payload.token_usage,
            category=payload.category,
            difficulty=payload.difficulty,
            status=CampaignResultStatus.COMPLETED
            if not payload.error_message
            else CampaignResultStatus.FAILED,
            error_message=payload.error_message,
            question_version=unit.test_case.question_version,
            request_id=execution.request_id,
            started_at=execution.started_at.isoformat(),
            completed_at=execution.completed_at.isoformat(),
            total_latency_ms=execution.total_latency_ms,
            total_tokens=total_tokens,
            question_snapshot=question_snapshot,
            model_config_snapshot=execution.model_config,
            system_version_snapshot=system_version_snapshot,
            derived_metrics=derived_metrics,
            final_answer_hash=_final_answer_hash(payload.answer),
        )
        span_id = await _record_unit_root_span(
            run_id=created.id,
            campaign_id=campaign_id,
            request_id=execution.request_id,
            unit=unit,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_ms=execution.total_latency_ms,
            failed=payload.error_message is not None,
        )
        await _record_unit_llm_usage(
            run_id=created.id,
            campaign_id=campaign_id,
            user_id=user_id,
            request_id=execution.request_id,
            span_id=span_id,
            execution=execution,
        )
        await _record_unit_research_observability(
            run_id=created.id,
            campaign_id=campaign_id,
            user_id=user_id,
            request_id=execution.request_id,
            root_span_id=span_id,
            execution=execution,
        )
        if payload.agent_trace:
            await self._trace_repository.replace_for_result(
                user_id=user_id,
                campaign_id=campaign_id,
                campaign_result_id=created.id,
                trace_payload=_enrich_agent_trace_payload(
                    trace_payload=payload.agent_trace,
                    created_id=created.id,
                    unit=unit,
                    payload=payload,
                ),
            )
        return created

    async def _resolve_test_cases(
        self, *, user_id: str, test_case_ids: list[str]
    ) -> list[TestCase]:
        available = [
            TestCase.model_validate(item) for item in await list_test_cases(user_id)
        ]
        by_id = {item.id: item for item in available}
        selected: list[TestCase] = []
        missing: list[str] = []
        for test_case_id in test_case_ids:
            test_case = by_id.get(test_case_id)
            if test_case is None:
                missing.append(test_case_id)
            else:
                selected.append(test_case)

        if missing:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=f"Unknown test case ids: {', '.join(missing)}",
                status_code=400,
            )
        return selected

    @staticmethod
    def _build_units(
        *,
        test_cases: list[TestCase],
        modes: list[str],
        repeat_count: int,
        ablation_conditions: list[AblationCondition] | None = None,
        agentic_execution_version: Literal["v8", "v9"] = "v8",
        shadow_evaluation_policy: Literal["operational", "research"] | None = None,
    ) -> list[CampaignUnit]:
        units: list[CampaignUnit] = []
        if ablation_conditions:
            condition_count = len(ablation_conditions)
            for repeat_number in range(1, repeat_count + 1):
                for test_case in test_cases:
                    for condition_index, condition in enumerate(
                        ablation_conditions, start=1
                    ):
                        stored_run_number = (
                            (repeat_number - 1) * condition_count
                        ) + condition_index
                        units.append(
                            CampaignUnit(
                                test_case=test_case,
                                mode=condition.mode,
                                run_number=stored_run_number,
                                repeat_number=repeat_number,
                                condition_id=condition.condition_id,
                                condition_label=condition.label,
                                ablation_flags=dict(condition.ablation_flags),
                                budget=dict(condition.budget)
                                if condition.budget
                                else None,
                                agentic_execution_version=effective_agentic_execution_version(
                                    condition.mode, agentic_execution_version
                                ),
                                shadow_evaluation_policy=shadow_evaluation_policy,
                            )
                        )
            return units

        for run_number in range(1, repeat_count + 1):
            for test_case in test_cases:
                for mode in modes:
                    units.append(
                        CampaignUnit(
                            test_case=test_case,
                            mode=mode,
                            run_number=run_number,
                            repeat_number=run_number,
                            agentic_execution_version=effective_agentic_execution_version(
                                mode, agentic_execution_version
                            ),
                            shadow_evaluation_policy=shadow_evaluation_policy,
                        )
                    )
        return units

    @staticmethod
    def _work_item_spec(
        *, user_id: str, campaign_id: str, unit: CampaignUnit, config: CampaignConfig
    ) -> WorkItemSpec:
        condition_key = unit.condition_id or "none"
        return WorkItemSpec(
            work_type=EvaluationWorkType.DATASET_EXECUTION,
            logical_key=f"execution:{unit.test_case.id}:{unit.mode}:{unit.run_number}:{condition_key}",
            input_snapshot={
                "user_id": user_id,
                "campaign_id": campaign_id,
                "test_case": unit.test_case.model_dump(mode="json"),
                "mode": unit.mode,
                "run_number": unit.run_number,
                "repeat_number": unit.repeat_number,
                "condition_id": unit.condition_id,
                "condition_label": unit.condition_label,
                "ablation_flags": dict(unit.ablation_flags or {}),
                "budget": dict(unit.budget or {}),
                "agentic_execution_version": unit.agentic_execution_version,
                "shadow_evaluation_policy": unit.shadow_evaluation_policy,
                "model_config": config.model_preset.model_dump(mode="json"),
            },
        )


_campaign_engine: Optional[CampaignEngine] = None


def get_campaign_engine() -> CampaignEngine:
    global _campaign_engine
    if _campaign_engine is None:
        from evaluation.job_worker import get_evaluation_job_worker

        _campaign_engine = CampaignEngine(
            configure_worker=not get_evaluation_job_worker().is_configured
        )
    return _campaign_engine
