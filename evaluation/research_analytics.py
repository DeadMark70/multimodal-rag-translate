"""Strict, version-2 research accounting aggregation.

This module deliberately reads the durable result, RAGAS and accounting stores;
it does not reuse the legacy analytics projection or RAGAS evaluator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from evaluation.accounting_schemas import (
    CampaignResearchSummaryResponse,
    CostSummary,
    EvaluationOverheadSummary,
    LatencySummary,
    MetricObservation,
    ModeResearchSummary,
    ResearchWarning,
    TokenBreakdown,
)
from evaluation.accounting_store import EvaluationAccountingStore
from evaluation.campaign_schemas import CampaignResultStatus
from evaluation.db import (
    CampaignRepository,
    CampaignResultRepository,
    RagasScoreRepository,
)

PRIMARY_QUALITY_METRICS = ("answer_correctness", "faithfulness", "answer_relevancy")
OPTIONAL_CONTEXT_METRICS = ("context_precision", "context_recall")


def nearest_rank(values: list[float], percentile: float) -> float | None:
    """Return an observed percentile value using deterministic nearest rank."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


class ResearchAnalyticsService:
    """Build a strict campaign summary from authoritative durable records."""

    def __init__(
        self,
        *,
        campaigns: CampaignRepository | None = None,
        results: CampaignResultRepository | None = None,
        ragas_scores: RagasScoreRepository | None = None,
        accounting: EvaluationAccountingStore | None = None,
    ) -> None:
        self._campaigns = campaigns or CampaignRepository()
        self._results = results or CampaignResultRepository()
        self._ragas_scores = ragas_scores or RagasScoreRepository()
        self._accounting = accounting or EvaluationAccountingStore()

    async def get_summary(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignResearchSummaryResponse:
        await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        all_results = await self._results.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed = [
            r for r in all_results if r.status == CampaignResultStatus.COMPLETED
        ]
        scores = await self._ragas_scores.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        scopes = await self._accounting.list_campaign_scopes(campaign_id)
        events = await self._accounting.list_campaign_events(campaign_id)
        events_by_scope: dict[str, list] = defaultdict(list)
        for event in events:
            events_by_scope[event.scope_id].append(event)
        requested_metrics = _requested_metrics(scopes)

        modes: list[ModeResearchSummary] = []
        warnings: list[ResearchWarning] = []
        for mode in sorted({str(result.mode) for result in completed}):
            included = [result for result in completed if str(result.mode) == mode]
            mode_results = [
                result for result in all_results if str(result.mode) == mode
            ]
            summary, mode_warnings = _mode_summary(
                included,
                mode_results,
                scores,
                scopes,
                events_by_scope,
                requested_metrics,
            )
            modes.append(summary)
            warnings.extend(
                ResearchWarning(code=code, message=message, mode=mode)
                for code, message in mode_warnings
            )

        official_scopes = _official_execution_scopes(completed, scopes)
        official_events = [
            event
            for scope in official_scopes
            for event in events_by_scope[scope.scope_id]
        ]
        quality = _quality_for_results(completed, scores, requested_metrics)
        tokens = _tokens(official_scopes, official_events)
        cost = _cost(
            official_events,
            operational_events=[
                e
                for s in scopes
                if s.scope_type == "execution_run"
                for e in events_by_scope[s.scope_id]
            ],
        )
        overhead_scopes = [s for s in scopes if s.scope_type == "ragas_batch"]
        overhead_events = [
            e for s in overhead_scopes for e in events_by_scope[s.scope_id]
        ]
        overhead_tokens = _tokens(
            overhead_scopes, overhead_events, legacy_status="partial"
        )
        overhead_cost = _cost(overhead_events, operational_events=overhead_events)
        overhead = EvaluationOverheadSummary(
            tokens=overhead_tokens,
            cost_usd=overhead_cost.operational_usd,
            pricing_status=overhead_cost.pricing_status,
            evaluator_models=sorted(
                {e.model_name for e in overhead_events if e.model_name}
            ),
            metric_names=sorted(
                {s.metric_name for s in overhead_scopes if s.metric_name}
            ),
            batch_count=len(overhead_scopes),
            retry_count=max(
                0, len(overhead_scopes) - len({s.scope_key for s in overhead_scopes})
            ),
        )
        latency = _latency(
            [r.total_latency_ms for r in completed if r.total_latency_ms is not None]
        )
        return CampaignResearchSummaryResponse(
            campaign_id=campaign_id,
            completed_run_count=len(completed),
            total_run_count=len(all_results),
            failed_run_count=sum(
                r.status == CampaignResultStatus.FAILED for r in all_results
            ),
            quality_status=_overall_quality_status(quality),
            token_accounting_status=tokens.accounting_status,
            pricing_status=cost.pricing_status,
            phase_attribution_status=tokens.phase_attribution_status,
            sample_count=len(completed),
            quality=quality,
            latency=latency,
            tokens=tokens,
            execution_cost=cost,
            modes=modes,
            evaluation_overhead=overhead,
            warnings=warnings,
        )


def _mode_summary(
    results,
    operational_results,
    scores,
    scopes,
    events_by_scope,
    requested_metrics,
):
    official = _official_execution_scopes(results, scopes)
    official_events = [
        event for scope in official for event in events_by_scope[scope.scope_id]
    ]
    operational_scopes = _execution_scopes_for_results(operational_results, scopes)
    operational = [
        event
        for scope in operational_scopes
        for event in events_by_scope[scope.scope_id]
    ]
    quality = _quality_for_results(results, scores, requested_metrics)
    tokens = _tokens(official, official_events)
    cost = _cost(official_events, operational_events=operational)
    reasons: list[str] = []
    warnings = []
    if tokens.accounting_status == "incomplete_legacy":
        reasons.append("legacy_accounting")
    elif tokens.accounting_status != "complete":
        reasons.append("incomplete_accounting")
    if cost.pricing_status != "complete":
        reasons.append("incomplete_pricing")
    if any(
        item.status != "complete" or item.valid_samples == 0
        for item in quality.values()
    ):
        reasons.append("incomplete_quality")
    evaluator_identities = _evaluator_identities_for_results(results, scores)
    if len(evaluator_identities) > 1:
        reasons.append("evaluator_metadata_mismatch")
        warnings.append(
            (
                "evaluator_metadata_mismatch",
                "Evaluator model, metric version, or signature differs across scores.",
            )
        )
    if any(scope.accounting_schema_version != "2" for scope in official):
        reasons.append("accounting_schema_version_mismatch")
    if len(results) < 5:
        warnings.append(
            ("low_sample_size", "Fewer than five official executions are included.")
        )
    return ModeResearchSummary(
        mode=str(results[0].mode),
        sample_count=len(results),
        comparable=not reasons,
        not_comparable_reasons=reasons,
        quality=quality,
        latency=_latency(
            [r.total_latency_ms for r in results if r.total_latency_ms is not None]
        ),
        tokens=tokens,
        execution_cost=cost,
    ), warnings


def _official_execution_scopes(results, scopes):
    by_attempt = {r.source_attempt_id: r.id for r in results if r.source_attempt_id}
    return [
        s
        for s in scopes
        if s.scope_type == "execution_run"
        and s.status == "completed"
        and s.accounting_schema_version == "2"
        and any(
            t.is_official
            and t.attempt_id in by_attempt
            and t.campaign_result_id in (None, by_attempt[t.attempt_id])
            for t in s.targets
        )
    ]


def _execution_scopes_for_results(results, scopes):
    result_ids = {result.id for result in results}
    attempts = {
        result.source_attempt_id for result in results if result.source_attempt_id
    }
    return [
        scope
        for scope in scopes
        if scope.scope_type == "execution_run"
        and (
            scope.run_id in result_ids
            or any(
                target.campaign_result_id in result_ids or target.attempt_id in attempts
                for target in scope.targets
            )
        )
    ]


def _requested_metrics(scopes) -> dict[str, set[str]]:
    requested: dict[str, set[str]] = defaultdict(set)
    for scope in scopes:
        if scope.scope_type == "ragas_batch" and scope.metric_name in (
            *PRIMARY_QUALITY_METRICS,
            *OPTIONAL_CONTEXT_METRICS,
        ):
            requested[scope.metric_name].add(scope.status)
    return dict(requested)


def _quality_for_results(results, scores, requested_work):
    result_ids = {r.id for r in results}
    attempts_by_result = {
        result.id: result.source_attempt_id
        for result in results
        if result.source_attempt_id
    }
    score_requested = {
        row["metric_name"] for row in scores if row["campaign_result_id"] in result_ids
    }
    requested = set(requested_work) | score_requested
    metric_names = (
        *PRIMARY_QUALITY_METRICS,
        *(m for m in OPTIONAL_CONTEXT_METRICS if m in requested),
    )
    output = {}
    for metric in metric_names:
        rows = [
            row
            for row in scores
            if row["metric_name"] == metric
            and row["campaign_result_id"] in result_ids
            and row.get("source_attempt_id")
            == attempts_by_result.get(row["campaign_result_id"])
        ]
        groups: dict[tuple[str, str, str], list] = defaultdict(list)
        for row in rows:
            groups[_evaluator_identity(row)].append(row)
        chosen = (
            min(groups, key=lambda identity: (-len(groups[identity]), identity))
            if groups
            else None
        )
        compatible = [
            row
            for row in rows
            if chosen is not None and _evaluator_identity(row) == chosen
        ]
        values = [
            float(row["metric_value"])
            for row in compatible
            if row.get("metric_value") is not None
        ]
        missing = len(results) - len({row["campaign_result_id"] for row in compatible})
        details = compatible[0].get("details", {}) if compatible else {}
        work_statuses = requested_work.get(metric, set())
        no_value_status = (
            "evaluating"
            if "running" in work_statuses
            else "failed"
            if work_statuses or metric in score_requested
            else "not_requested"
        )
        output[metric] = MetricObservation(
            value=mean(values) if values else None,
            status="complete"
            if values and not missing
            else "partial"
            if values
            else no_value_status,
            valid_samples=len(values),
            missing_samples=max(0, missing),
            failed_samples=max(
                0,
                len(results) - len(values)
                if no_value_status == "failed" and not values
                else 0,
            ),
            evaluator_model=details.get("evaluator_model") or details.get("model_name"),
            metric_version=details.get("metric_version"),
        )
    return output


def _evaluator_identity(row) -> tuple[str, str, str]:
    details = row.get("details") or {}
    return (
        str(details.get("evaluator_model") or details.get("model_name") or ""),
        str(details.get("metric_version") or ""),
        str(row.get("evaluation_signature") or ""),
    )


def _evaluator_identities_for_results(results, scores):
    attempts_by_result = {
        result.id: result.source_attempt_id
        for result in results
        if result.source_attempt_id
    }
    return {
        _evaluator_identity(row)
        for row in scores
        if row["campaign_result_id"] in attempts_by_result
        and row.get("source_attempt_id")
        == attempts_by_result[row["campaign_result_id"]]
        and row["metric_name"] in PRIMARY_QUALITY_METRICS
    }


def _latency(values):
    return LatencySummary(
        mean_ms=mean(values) if values else None,
        p50_ms=nearest_rank(values, 0.5),
        p95_ms=nearest_rank(values, 0.95),
        sample_count=len(values),
        low_sample_size=0 < len(values) < 5,
    )


def _tokens(scopes, events, legacy_status="incomplete_legacy"):
    if not scopes:
        return TokenBreakdown(
            accounting_status=legacy_status, phase_attribution_status="not_available"
        )
    complete = all(
        s.status == "completed"
        and s.observed_call_count == s.measured_call_count
        and s.missing_usage_call_count == 0
        for s in scopes
    ) and all(
        e.usage_status == "measured" and e.reconciliation_status == "balanced"
        for e in events
    )
    status = "complete" if complete else "partial"
    phase_complete = complete and all(e.phase != "unclassified" for e in events)
    if not events:
        return TokenBreakdown(
            accounting_status="partial", phase_attribution_status="not_available"
        )
    phase = defaultdict(int)
    for event in events:
        phase[event.phase] += (
            event.input_tokens
            + event.output_text_tokens
            + event.reasoning_tokens
            + event.other_tokens
        )
    values = dict(
        input_tokens=sum(e.input_tokens for e in events),
        output_text_tokens=sum(e.output_text_tokens for e in events),
        reasoning_tokens=sum(e.reasoning_tokens for e in events),
        other_tokens=sum(e.other_tokens for e in events),
        by_phase=dict(sorted(phase.items())),
        accounting_status=status,
        phase_attribution_status="complete" if phase_complete else "partial",
    )
    values["total_tokens"] = (
        sum(
            values[k]
            for k in (
                "input_tokens",
                "output_text_tokens",
                "reasoning_tokens",
                "other_tokens",
            )
        )
        if status == "complete"
        else None
    )
    return TokenBreakdown(**values)


def _cost(events, *, operational_events):
    def status(rows):
        if not rows:
            return "unknown"
        return (
            "complete"
            if all(
                e.pricing_status == "priced" and e.estimated_cost_usd is not None
                for e in rows
            )
            else "partial"
        )

    pricing = status(operational_events)
    priced = sum(
        e.pricing_status == "priced" and e.estimated_cost_usd is not None
        for e in operational_events
    )
    return CostSummary(
        benchmark_usd=sum(e.estimated_cost_usd or 0 for e in events)
        if status(events) == "complete"
        else None,
        operational_usd=sum(e.estimated_cost_usd or 0 for e in operational_events)
        if pricing == "complete"
        else None,
        pricing_status=pricing,
        priced_call_count=priced,
        unpriced_call_count=len(operational_events) - priced,
    )


def _overall_quality_status(quality):
    statuses = {item.status for item in quality.values()}
    if statuses == {"complete"}:
        return "complete"
    if "evaluating" in statuses:
        return "evaluating"
    if statuses <= {"not_requested"}:
        return "not_requested"
    if statuses <= {"failed", "not_requested"}:
        return "failed"
    return "partial"
