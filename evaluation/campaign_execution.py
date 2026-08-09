"""Campaign execution contracts and durable result projections."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4

from data_base import repository as document_repository
from evaluation.evidence import (
    build_gold_fact_attrition,
    content_hash,
    estimate_tokens,
    expected_evidence_matches_doc,
    text_mentions_fact,
)
from evaluation.observability import EvaluationRunRecorder
from evaluation.observability_storage import (
    EvaluationObservabilityRepository,
    safe_comparison_projection,
)
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem, TestCase
from evaluation.trace_schemas import (
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

_SAFE_FAILURE_CODES = frozenset({"EVALUATION_ANSWER_TOO_LARGE"})
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
    observability_partial: bool = False
    observability_partial_reasons: tuple[str, ...] = ()
    provider_name: str = "unknown"


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
        metrics["failure_diagnostics"] = _failure_diagnostics(
            unit=unit, payload=payload
        )
        return metrics
    if payload.execution_identity:
        metrics["execution_identity"] = payload.execution_identity
    if payload.shadow_evaluation_policy:
        metrics["shadow_evaluation_policy"] = payload.shadow_evaluation_policy
    metrics["response_status"] = payload.response_status or (
        "failed" if payload.error_message else "complete"
    )
    if payload.error_message:
        metrics["failure_diagnostics"] = _failure_diagnostics(
            unit=unit, payload=payload
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


def _safe_failure_message(raw: Any) -> str:
    """Never persist arbitrary provider or exception text as a run failure message."""
    if str(raw or "").strip() in _SAFE_FAILURE_CODES:
        return str(raw).strip()
    return "Provider error details were redacted."


def _failure_diagnostics(
    *, unit: CampaignUnit, payload: BenchmarkExecutionResult | Exception
) -> dict[str, Any]:
    trace_payload = getattr(payload, "agent_trace", None)
    trace = trace_payload if isinstance(trace_payload, dict) else {}
    raw_error = str(payload) if isinstance(payload, Exception) else payload.error_message
    retry_count = trace.get("retry_count")
    return {
        "error_code": (
            payload.__class__.__name__
            if isinstance(payload, Exception)
            else str(trace.get("error_code") or "RUN_FAILED")
        ),
        "safe_error_message": _safe_failure_message(raw_error),
        "last_completed_stage": (
            trace.get("last_completed_stage")
            if isinstance(trace.get("last_completed_stage"), str)
            else "campaign_unit_execution"
            if isinstance(payload, Exception)
            else None
        ),
        "provider_status": trace.get("provider_status")
        if isinstance(trace.get("provider_status"), str)
        else None,
        "retry_count": retry_count
        if isinstance(retry_count, int) and retry_count >= 0
        else 0,
        "timeout_state": trace.get("timeout_state")
        if isinstance(trace.get("timeout_state"), str)
        else None,
        "budget_state": trace.get("budget_state")
        if isinstance(trace.get("budget_state"), str)
        else None,
    }


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
    v9_payload = enriched.get("agentic_v9")
    if isinstance(v9_payload, dict):
        safe_v9_payload = dict(v9_payload)
        comparison = safe_v9_payload.get("comparison")
        if isinstance(comparison, dict):
            safe_v9_payload["comparison"] = safe_comparison_projection(
                comparison
            )
        else:
            safe_v9_payload.pop("comparison", None)
        enriched["agentic_v9"] = safe_v9_payload
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
    trace_payload = execution.payload.agent_trace
    v9_payload = (
        trace_payload.get("agentic_v9") if isinstance(trace_payload, dict) else None
    )
    if isinstance(v9_payload, dict) and "budget_reservations" in v9_payload:
        return
    if not execution.payload.token_usage:
        return

    model_name = execution.model_config.get("model_name")
    provider = execution.provider_name

    recorder = EvaluationRunRecorder(
        run_id=run_id,
        campaign_id=campaign_id,
        user_id=user_id,
        request_id=request_id,
        # Required-stage telemetry is part of the v9 completion contract.
        # Never silently convert a persistence outage into a successful run.
        strict=True,
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


def _v9_rerank_diagnostics_by_context(
    trace_payload: dict[str, Any],
) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    """Index selected rows by raw content and durable source-chunk identity."""
    v9_payload = trace_payload.get("agentic_v9")
    diagnostics = (
        v9_payload.get("retrieval_diagnostics")
        if isinstance(v9_payload, dict)
        else None
    )
    if not isinstance(diagnostics, list):
        return {"by_content": {}, "by_source_chunk": {}}

    indexed: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        "by_content": {},
        "by_source_chunk": {},
    }
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        selected_rows = diagnostic.get("selected")
        if not isinstance(selected_rows, list):
            continue
        task_metadata = {
            "reranker_status": str(
                diagnostic.get("status") or "not_instrumented"
            ),
            "reranker_fallback_reason": diagnostic.get("fallback_reason"),
            "retrieval_task_id": diagnostic.get("task_id"),
            "rerank_candidate_count": diagnostic.get("candidate_count"),
            "rerank_selected_count": diagnostic.get("selected_count"),
            "candidate_stage": _safe_candidate_stage_projection(
                diagnostic.get("candidate_diversification")
            ),
        }
        for selected_row in selected_rows:
            if not isinstance(selected_row, dict):
                continue
            doc_id = selected_row.get("doc_id")
            selected_content_hash = selected_row.get("content_hash")
            if doc_id in (None, "") or selected_content_hash in (None, ""):
                continue
            diagnostic_row = {
                **task_metadata,
                "chunk_id": selected_row.get("chunk_id"),
                "rank_before_rerank": selected_row.get("pre_rerank_rank"),
                "rank_after_rerank": selected_row.get("post_rerank_rank"),
                "rerank_score": selected_row.get("rerank_score"),
                "_consumed": False,
            }
            indexed["by_content"].setdefault(
                (str(doc_id), str(selected_content_hash)), []
            ).append(diagnostic_row)
            chunk_id = selected_row.get("chunk_id")
            if chunk_id not in (None, ""):
                indexed["by_source_chunk"].setdefault(
                    (str(doc_id), str(chunk_id)), []
                ).append(diagnostic_row)
    return indexed


def _safe_candidate_stage_projection(value: Any) -> dict[str, Any] | None:
    """Allowlist non-content candidate diagnostics for persistence and export."""
    if not isinstance(value, dict):
        return None
    policy = value.get("policy")
    if not isinstance(policy, str) or not policy:
        return None

    def ordered_ids(field: str) -> list[str]:
        raw_ids = value.get(field)
        if not isinstance(raw_ids, list):
            return []
        return list(
            dict.fromkeys(
                item for item in raw_ids if isinstance(item, str) and item
            )
        )

    return {
        "policy": policy,
        "enabled": bool(value.get("enabled")),
        "applied": bool(value.get("applied")),
        "retrieved_doc_ids": ordered_ids("retrieved_doc_ids"),
        "candidate_doc_ids": ordered_ids("candidate_doc_ids"),
        "represented_doc_ids_before_tail": ordered_ids(
            "represented_doc_ids_before_tail"
        ),
        "admitted_doc_ids": ordered_ids("admitted_doc_ids"),
    }


def _consume_v9_rerank_diagnostic(
    diagnostics_by_context: dict[
        str, dict[tuple[str, str], list[dict[str, Any]]]
    ],
    *,
    doc_id: str | None,
    selected_content_hash: str,
    source_chunk_id: str | None,
) -> dict[str, Any] | None:
    if doc_id in (None, ""):
        return None

    def consume_unique(
        candidates: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        available = [
            candidate
            for candidate in candidates or []
            if not candidate.get("_consumed")
        ]
        if len(available) != 1:
            return None
        selected = available[0]
        selected["_consumed"] = True
        return {
            key: value
            for key, value in selected.items()
            if key != "_consumed"
        }

    if source_chunk_id is not None:
        return consume_unique(
            diagnostics_by_context.get("by_source_chunk", {}).get(
                (str(doc_id), str(source_chunk_id))
            )
        )
    return consume_unique(
        diagnostics_by_context.get("by_content", {}).get(
            (str(doc_id), selected_content_hash)
        )
    )


async def _resolve_expected_source_document_ids(
    *,
    user_id: str,
    expected_sources: list[str],
) -> tuple[set[str], Literal["resolved", "identity_unresolved"]]:
    """Resolve expected-source metadata for evaluation observability only."""
    unique_references = list(
        dict.fromkeys(
            reference
            for reference in expected_sources
            if isinstance(reference, str) and reference
        )
    )
    if not unique_references:
        return set(), "resolved"

    try:
        resolved_references = await document_repository.resolve_document_references(
            user_id, unique_references
        )
    except Exception:
        logger.warning(
            "Unable to resolve expected-source identities for evaluation observability",
            exc_info=True,
        )
        return set(), "identity_unresolved"

    resolved_document_ids: set[str] = set()
    for reference in unique_references:
        candidates = resolved_references.get(reference, [])
        if len(candidates) != 1:
            return set(), "identity_unresolved"
        resolved_document_ids.add(candidates[0])
    return resolved_document_ids, "resolved"


def _graph_trace_outcome(
    graph_execution: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    execution_state = str(graph_execution.get("state") or "not_instrumented")
    if execution_state == "executed":
        return "success", {}
    if (
        graph_execution.get("policy") == "locator_fallback"
        and execution_state == "not_triggered"
        and not bool(graph_execution.get("attempted"))
    ):
        return "skipped", {}
    return "partial", {
        "reason": str(
            graph_execution.get("failure_reason")
            or "required_graph_not_satisfied"
        )
    }


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

    v9_payload = trace_payload.get("agentic_v9")
    query_contract = (
        v9_payload.get("query_contract") if isinstance(v9_payload, dict) else None
    )
    actual_route = (
        query_contract.get("route_decision")
        if isinstance(query_contract, dict)
        else None
    )
    if isinstance(actual_route, dict):
        async with recorder.start_span(
            stage_type="routing",
            stage_name="agentic_v9_actual_routing",
            event_type="routing_decision",
            payload={
                "request_id": request_id,
                "question_id": execution.unit.test_case.id,
                "selected_route": actual_route.get("selected_route"),
                "decision_source": actual_route.get("decision_source"),
            },
        ) as routing_span:
            await recorder.record_routing_decision(
                EvaluationRoutingDecision(
                    routing_decision_id=str(uuid4()),
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=routing_span.span_id,
                    selected_mode=execution.unit.mode,
                    analysis_type="actual",
                    decision_source=actual_route.get("decision_source"),
                    candidate_routes=list(
                        actual_route.get("candidate_routes") or []
                    ),
                    matched_rules=list(actual_route.get("matched_rules") or []),
                    fallback_reason=actual_route.get("fallback_reason"),
                    confidence=actual_route.get("confidence"),
                    reason=actual_route.get("route_reason"),
                    payload={
                        "selected_route": actual_route.get("selected_route"),
                        "decision_source": actual_route.get("decision_source"),
                        "candidate_routes": list(
                            actual_route.get("candidate_routes") or []
                        ),
                        "matched_rules": list(
                            actual_route.get("matched_rules") or []
                        ),
                        "planner_call_used": bool(
                            actual_route.get("planner_call_used")
                        ),
                        "fallback_reason": actual_route.get(
                            "fallback_reason"
                        ),
                    },
                    created_at=created_at,
                )
            )
    graph_execution = (
        v9_payload.get("graph_execution") if isinstance(v9_payload, dict) else None
    )
    if isinstance(graph_execution, dict) and graph_execution.get("policy") != "never":
        execution_state = str(graph_execution.get("state") or "not_instrumented")
        graph_route = str(
            graph_execution.get("route")
            or ("fallback" if execution_state != "executed" else "unknown")
        )
        latency_ms = graph_execution.get("latency_ms")
        async with recorder.start_span(
            stage_type="graph",
            stage_name="agentic_v9_graph_locator",
            event_type="graph_locator",
            payload={
                "request_id": request_id,
                "question_id": execution.unit.test_case.id,
                "policy": graph_execution.get("policy"),
                "execution_state": execution_state,
                "failure_reason": graph_execution.get("failure_reason"),
            },
        ) as graph_span:
            resolved_item_ids = list(graph_execution.get("resolved_item_ids") or [])
            scope_item_ids = list(
                graph_execution.get("scope_approved_item_ids") or []
            )
            graph_event_id = str(uuid4())
            await recorder.record_graph_event(
                EvaluationGraphEvent(
                    graph_event_id=graph_event_id,
                    run_id=run_id,
                    campaign_id=campaign_id,
                    span_id=graph_span.span_id,
                    graph_query=execution.unit.test_case.question,
                    graph_search_mode="generic",
                    graph_evidence_mode="locator_to_chunk",
                    graph_route=graph_route,
                    router_reason=graph_execution.get("failure_reason"),
                    graph_feature_flags={
                        "agentic_v9": True,
                        "graph_policy": graph_execution.get("policy"),
                        "execution_state": execution_state,
                        "attempted": bool(graph_execution.get("attempted")),
                        "fallback": graph_execution.get("fallback"),
                    },
                    matched_entity_ids=list(
                        graph_execution.get("candidate_item_ids") or []
                    ),
                    node_count=len(resolved_item_ids),
                    path_count=len(scope_item_ids),
                    graph_latency_ms=(
                        latency_ms
                        if isinstance(latency_ms, int) and latency_ms >= 0
                        else None
                    ),
                    graph_to_chunk_success_rate=(
                        1.0
                        if execution_state == "executed" and scope_item_ids
                        else 0.0
                        if bool(graph_execution.get("attempted"))
                        else None
                    ),
                    created_at=created_at,
                )
            )
            resolved_doc_ids = list(
                graph_execution.get("resolved_source_doc_ids") or []
            )
            resolved_chunk_ids = list(
                graph_execution.get("resolved_source_chunk_ids") or []
            )
            await recorder.record_graph_evidence_items(
                [
                    EvaluationGraphEvidenceItem(
                        graph_evidence_item_id=f"{graph_event_id}:{index}",
                        graph_event_id=graph_event_id,
                        source_doc_ids=[doc_id],
                        source_chunk_ids=(
                            [resolved_chunk_ids[index]]
                            if index < len(resolved_chunk_ids)
                            else []
                        ),
                        provenance_status="full",
                        used_as_locator=True,
                        packed_in_context=doc_id
                        in set(graph_execution.get("resolved_source_doc_ids") or []),
                        used_in_answer=False,
                        created_at=created_at,
                    )
                    for index, doc_id in enumerate(resolved_doc_ids)
                ]
            )
            outcome_status, outcome_error = _graph_trace_outcome(graph_execution)
            if outcome_status != "success":
                graph_span.set_outcome(
                    status=outcome_status,
                    error=outcome_error,
                )

    visual_execution = (
        v9_payload.get("visual_execution") if isinstance(v9_payload, dict) else None
    )
    if isinstance(visual_execution, dict) and visual_execution.get("required"):
        async with recorder.start_span(
            stage_type="visual",
            stage_name="agentic_v9_visual_extract",
            event_type="visual_extract",
            payload={
                "request_id": request_id,
                "question_id": execution.unit.test_case.id,
                "execution_state": visual_execution.get("state"),
                "failure_reason": visual_execution.get("failure_reason"),
                "selected_asset_count": visual_execution.get("selected_asset_count"),
                "dropped_asset_count": visual_execution.get("dropped_asset_count"),
                "evidence_packet_count": visual_execution.get("evidence_packet_count"),
            },
        ) as visual_span:
            # Entering the span records the durable stage event; the detailed
            # result lives in its payload so absent assets are never confused
            # with a measured zero successful extraction.
            if visual_execution.get("state") != "executed":
                visual_span.set_outcome(
                    status="partial",
                    error={
                        "reason": str(
                            visual_execution.get("failure_reason")
                            or "required_visual_not_satisfied"
                        )
                    },
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
    (
        resolved_expected_source_ids,
        expected_source_identity_state,
    ) = await _resolve_expected_source_document_ids(
        user_id=user_id,
        expected_sources=expected_sources,
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
    rerank_diagnostics_by_context = _v9_rerank_diagnostics_by_context(trace_payload)
    for index, context in enumerate(execution.payload.contexts, start=1):
        doc_id = (
            execution.payload.source_doc_ids[index - 1]
            if index - 1 < len(execution.payload.source_doc_ids)
            else None
        )
        durable_chunk_id = f"{run_id}:chunk:{index}"
        source_chunk_id = (
            execution.payload.source_chunk_ids[index - 1]
            if index - 1 < len(execution.payload.source_chunk_ids)
            else None
        )
        selected_content_hash = content_hash(context)
        rerank_diagnostic = _consume_v9_rerank_diagnostic(
            rerank_diagnostics_by_context,
            doc_id=doc_id,
            selected_content_hash=selected_content_hash,
            source_chunk_id=source_chunk_id,
        )
        expected_match = (
            expected_evidence_matches_doc(
                doc_id=doc_id,
                expected_evidence=expected_evidence,
                expected_sources=list(resolved_expected_source_ids),
            )
            if expected_source_identity_state == "resolved"
            else False
        )
        expected_evidence_match_status = (
            "identity_unresolved"
            if expected_source_identity_state == "identity_unresolved"
            else "matched"
            if expected_match
            else "not_matched"
        )
        chunks.append(
            EvaluationRetrievalChunk(
                retrieval_chunk_id=str(uuid4()),
                run_id=run_id,
                campaign_id=campaign_id,
                span_id=root_span_id,
                retrieval_event_id=retrieval_event_id,
                chunk_id=durable_chunk_id,
                doc_id=doc_id,
                rank_before_rerank=(
                    rerank_diagnostic.get("rank_before_rerank")
                    if rerank_diagnostic is not None
                    else None
                ),
                rank_after_rerank=(
                    rerank_diagnostic.get("rank_after_rerank")
                    if rerank_diagnostic is not None
                    else None
                ),
                rerank_score=(
                    rerank_diagnostic.get("rerank_score")
                    if rerank_diagnostic is not None
                    and rerank_diagnostic["reranker_status"] == "executed"
                    else None
                ),
                used_in_context=True,
                used_in_answer=expected_match
                or text_mentions_fact(execution.payload.answer, context),
                expected_evidence_match=expected_match,
                excerpt=context[:500],
                content_hash=selected_content_hash,
                payload={
                    "instrumentation_depth": "result_level",
                    "expected_evidence_match_status": expected_evidence_match_status,
                    "reranker_status": (
                        rerank_diagnostic["reranker_status"]
                        if rerank_diagnostic is not None
                        else "not_instrumented"
                    ),
                    "reranker_fallback_reason": (
                        rerank_diagnostic["reranker_fallback_reason"]
                        if rerank_diagnostic is not None
                        else None
                    ),
                    "retrieval_task_id": (
                        rerank_diagnostic["retrieval_task_id"]
                        if rerank_diagnostic is not None
                        else None
                    ),
                    "rerank_candidate_count": (
                        rerank_diagnostic["rerank_candidate_count"]
                        if rerank_diagnostic is not None
                        else None
                    ),
                    "rerank_selected_count": (
                        rerank_diagnostic["rerank_selected_count"]
                        if rerank_diagnostic is not None
                        else None
                    ),
                    **(
                        {"candidate_stage": rerank_diagnostic["candidate_stage"]}
                        if rerank_diagnostic is not None
                        and rerank_diagnostic["candidate_stage"] is not None
                        else {}
                    ),
                },
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
