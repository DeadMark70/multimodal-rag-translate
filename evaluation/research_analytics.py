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
from evaluation.campaign_schemas import (
    AgentBehaviorResponse,
    AgentBehaviorRow,
    CampaignResultStatus,
    QuestionComparisonRow,
    QuestionModeComparison,
    ResearchQuestionComparisonResponse,
)
from evaluation.db import (
    AgentTraceRepository,
    CampaignRepository,
    CampaignResultRepository,
    RagasScoreRepository,
)
from evaluation.job_store import (
    EVALUATOR_COMPATIBILITY_SIGNATURE_VERSION,
    build_legacy_evaluator_compatibility_signature,
    build_evaluator_compatibility_signature,
)

PRIMARY_QUALITY_METRICS = ("answer_correctness", "faithfulness", "answer_relevancy")
OPTIONAL_CONTEXT_METRICS = ("context_precision", "context_recall")


def _attach_work_metadata(scores, work_metadata):
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for metadata in work_metadata:
        result_id = metadata.get("campaign_result_id")
        metric_name = metadata.get("metric_name")
        if result_id and metric_name:
            by_key[(str(result_id), str(metric_name))].append(metadata)
    return [
        {
            **row,
            "_work_metadata": by_key.get(
                (str(row["campaign_result_id"]), str(row["metric_name"])), []
            ),
        }
        for row in scores
    ]


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
        traces: AgentTraceRepository | None = None,
    ) -> None:
        self._campaigns = campaigns or CampaignRepository()
        self._results = results or CampaignResultRepository()
        self._ragas_scores = ragas_scores or RagasScoreRepository()
        self._accounting = accounting or EvaluationAccountingStore()
        self._traces = traces or AgentTraceRepository()

    async def get_summary(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignResearchSummaryResponse:
        campaign = await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        all_results = await self._results.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed = [
            r for r in all_results if r.status == CampaignResultStatus.COMPLETED
        ]
        scores = await self._ragas_scores.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        work_metadata = await self._ragas_scores.list_work_metadata_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        scores = _attach_work_metadata(scores, work_metadata)
        scopes = await self._accounting.list_campaign_scopes(campaign_id)
        events = await self._accounting.list_campaign_events(campaign_id)
        events_by_scope: dict[str, list] = defaultdict(list)
        for event in events:
            events_by_scope[event.scope_id].append(event)
        requested_metrics = _requested_metrics(scopes)
        execution_scope_modes = {
            scope.scope_id: _execution_scope_mode(scope, all_results)
            for scope in scopes
            if scope.scope_type == "execution_run"
        }
        has_unattributed_execution_scopes = any(
            mode is None for mode in execution_scope_modes.values()
        )
        canonical_identities = _canonical_identities_by_metric(completed, scores)

        modes: list[ModeResearchSummary] = []
        warnings: list[ResearchWarning] = []
        for mode in sorted({str(result.mode) for result in completed}):
            included = [result for result in completed if str(result.mode) == mode]
            summary, mode_warnings = _mode_summary(
                included,
                scores,
                scopes,
                events_by_scope,
                requested_metrics,
                execution_scope_modes,
                has_unattributed_execution_scopes,
                canonical_identities,
                all_results,
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
        quality = _quality_for_results(
            completed,
            scores,
            requested_metrics,
            scopes,
            canonical_identities,
            all_results,
        )
        tokens = _tokens(official_scopes, official_events)
        if has_unattributed_execution_scopes:
            tokens = _partial_for_missing_mode_attribution(tokens)
        warnings.extend(
            ResearchWarning(code=code, message=message)
            for code, message in _token_warning_tuples(tokens)
        )
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
        retry_count = (
            None
            if any(scope.retry_count is None for scope in overhead_scopes)
            else sum(scope.retry_count or 0 for scope in overhead_scopes)
        )
        if retry_count is None:
            warnings.append(
                ResearchWarning(
                    code="unknown_ragas_retry_count",
                    message=(
                        "RAGAS retry counts are unavailable for one or more historical "
                        "accounting scopes."
                    ),
                )
            )
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
            retry_count=retry_count,
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

    async def get_run_token_breakdown(self, *, campaign_id: str, run_id: str) -> TokenBreakdown:
        """Return strict accounting for one selected execution run."""
        scopes = await self._accounting.list_campaign_scopes(campaign_id)
        events = await self._accounting.list_campaign_events(campaign_id)
        run_scopes = [
            scope
            for scope in scopes
            if scope.scope_type == "execution_run"
            and scope.run_id == run_id
            and scope.accounting_schema_version == "2"
            and any(target.is_official for target in scope.targets)
        ]
        if not run_scopes:
            return TokenBreakdown(
                total_tokens=None,
                accounting_status="incomplete_legacy",
                phase_attribution_status="not_available",
            )
        events_by_scope: dict[str, list] = defaultdict(list)
        for event in events:
            if event.scope_id in {scope.scope_id for scope in run_scopes}:
                events_by_scope[event.scope_id].append(event)
        return _tokens(
            run_scopes,
            [event for rows in events_by_scope.values() for event in rows],
        )

    async def get_question_comparison(
        self, *, user_id: str, campaign_id: str
    ) -> ResearchQuestionComparisonResponse:
        """Return measured question/mode comparisons with strict null semantics."""
        campaign = await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        all_results = await self._results.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed = [
            result
            for result in all_results
            if result.status == CampaignResultStatus.COMPLETED
        ]
        scores = await self._ragas_scores.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        work_metadata = await self._ragas_scores.list_work_metadata_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        scores = _attach_work_metadata(scores, work_metadata)
        canonical_identities = _canonical_identities_by_metric(completed, scores)
        result_by_id = {str(result.id): result for result in completed}
        attempts_by_result = {
            str(result.id): result.source_attempt_id
            for result in completed
            if result.source_attempt_id
        }
        scopes = await self._accounting.list_campaign_scopes(campaign_id)
        events = await self._accounting.list_campaign_events(campaign_id)
        events_by_scope: dict[str, list] = defaultdict(list)
        for event in events:
            events_by_scope[event.scope_id].append(event)

        score_map: dict[str, dict[str, float]] = defaultdict(dict)
        for score in scores:
            result_id = str(score.get("campaign_result_id"))
            metric_name = str(score.get("metric_name"))
            if (
                result_id not in attempts_by_result
                or score.get("source_attempt_id") != attempts_by_result[result_id]
                or metric_name not in canonical_identities
                or _evaluator_identity(score, result_by_id.get(result_id))
                != canonical_identities[metric_name]
            ):
                continue
            value = score.get("metric_value")
            if isinstance(value, (int, float)):
                score_map[result_id][metric_name] = float(value)

        official_scopes = _official_execution_scopes(completed, scopes)
        tokens_by_result: dict[str, TokenBreakdown] = {}
        for scope in official_scopes:
            result_ids = {
                target.campaign_result_id
                for target in scope.targets
                if target.is_official and target.campaign_result_id
            }
            if scope.run_id:
                result_ids.add(scope.run_id)
            if not result_ids:
                continue
            breakdown = _tokens([scope], events_by_scope[scope.scope_id])
            for result_id in result_ids:
                tokens_by_result[str(result_id)] = breakdown

        results_by_question: dict[str, list] = defaultdict(list)
        for result in completed:
            results_by_question[str(result.question_id)].append(result)

        configured_modes = list(
            getattr(getattr(campaign, "config", None), "modes", None) or []
        )
        configured_modes.extend(
            str(result.mode) for result in all_results if str(result.mode) not in configured_modes
        )

        rows: list[QuestionComparisonRow] = []
        warnings: list[str] = []
        for question_id, question_results in sorted(results_by_question.items()):
            by_mode_results: dict[str, list] = defaultdict(list)
            for result in question_results:
                by_mode_results[str(result.mode)].append(result)

            mode_rows: list[QuestionModeComparison] = []
            mode_quality: dict[str, dict[str, float | None]] = {}
            mode_tokens: dict[str, float | None] = {}
            mode_accounting: dict[str, str] = {}
            modes_for_question = sorted(
                {str(mode) for mode in configured_modes}
                | set(by_mode_results.keys())
            )
            for mode in modes_for_question:
                mode_results = by_mode_results.get(mode, [])
                mode_quality[mode] = {}
                for metric in PRIMARY_QUALITY_METRICS:
                    values = [
                        score_map.get(str(result.id), {}).get(metric)
                        for result in mode_results
                    ]
                    present = [value for value in values if value is not None]
                    mode_quality[mode][metric] = (
                        mean(present) if present and len(present) == len(values) else None
                    )
                latency_values = [
                    result.total_latency_ms
                    if result.total_latency_ms is not None
                    else result.latency_ms
                    for result in mode_results
                ]
                token_values = [
                    tokens_by_result.get(str(result.id)) for result in mode_results
                ]
                complete_tokens = [
                    item.total_tokens
                    for item in token_values
                    if item is not None
                    and item.accounting_status == "complete"
                    and item.total_tokens is not None
                ]
                if mode_results and len(complete_tokens) == len(mode_results):
                    mode_tokens[mode] = mean(complete_tokens)
                    accounting_status = "complete"
                elif any(item is not None for item in token_values):
                    mode_tokens[mode] = None
                    accounting_status = "partial"
                else:
                    mode_tokens[mode] = None
                    accounting_status = "not_available"
                mode_accounting[mode] = accounting_status
                quality_values = list(mode_quality[mode].values())
                quality_status = (
                    "complete"
                    if all(value is not None for value in quality_values)
                    else "partial"
                    if any(value is not None for value in quality_values)
                    else "not_available"
                )
                mode_rows.append(
                    QuestionModeComparison(
                        mode=mode,
                        sample_count=len(mode_results),
                        answer_correctness=mode_quality[mode]["answer_correctness"],
                        faithfulness=mode_quality[mode]["faithfulness"],
                        answer_relevancy=mode_quality[mode]["answer_relevancy"],
                        mean_latency_ms=mean(latency_values)
                        if latency_values
                        else None,
                        mean_tokens=mode_tokens[mode],
                        quality_status=quality_status,
                        accounting_status=accounting_status,
                    )
                )

            best_mode = _best_quality_mode(mode_rows)
            baseline = next(
                (row for row in mode_rows if row.mode == "naive"), None
            )
            target = next(
                (
                    row
                    for row in mode_rows
                    if row.mode == best_mode and row.mode != "naive"
                ),
                None,
            )
            comparability_reason: str | None = None
            if baseline is None:
                comparability_reason = "baseline_missing"
            elif target is None:
                comparability_reason = "comparison_mode_missing"
            elif baseline.quality_status != "complete" or (
                target is not None and target.quality_status != "complete"
            ):
                comparability_reason = "incomplete_quality"
            elif baseline.accounting_status != "complete" or (
                target is not None and target.accounting_status != "complete"
            ):
                comparability_reason = "incomplete_accounting"

            delta_correctness = (
                target.answer_correctness - baseline.answer_correctness
                if target
                and baseline
                and target.answer_correctness is not None
                and baseline.answer_correctness is not None
                else None
            )
            delta_faithfulness = (
                target.faithfulness - baseline.faithfulness
                if target
                and baseline
                and target.faithfulness is not None
                and baseline.faithfulness is not None
                else None
            )
            delta_latency = (
                target.mean_latency_ms - baseline.mean_latency_ms
                if target
                and baseline
                and target.mean_latency_ms is not None
                and baseline.mean_latency_ms is not None
                else None
            )
            delta_tokens = (
                target.mean_tokens - baseline.mean_tokens
                if target
                and baseline
                and target.mean_tokens is not None
                and baseline.mean_tokens is not None
                else None
            )
            ecr = (
                1000 * delta_correctness / delta_tokens
                if delta_correctness is not None
                and delta_tokens is not None
                and delta_tokens > 0
                else None
            )

            first = question_results[0]
            snapshot = first.question_snapshot if isinstance(first.question_snapshot, dict) else {}
            derived = [
                getattr(result, "derived_metrics", {})
                for result in question_results
                if isinstance(getattr(result, "derived_metrics", {}), dict)
            ]
            evidence_values = [
                item.get("evidence_coverage")
                for item in derived
                if isinstance(item.get("evidence_coverage"), (int, float))
            ]
            unsupported_values = [
                item.get("unsupported_claim_ratio")
                for item in derived
                if isinstance(item.get("unsupported_claim_ratio"), (int, float))
            ]
            rows.append(
                QuestionComparisonRow(
                    question_id=question_id,
                    category=getattr(first, "category", None),
                    difficulty=getattr(first, "difficulty", None),
                    required_modalities=(
                        list(snapshot["required_modalities"])
                        if isinstance(snapshot.get("required_modalities"), list)
                        else None
                    ),
                    by_mode=mode_rows,
                    delta_correctness=delta_correctness
                    if comparability_reason not in {"incomplete_quality", "comparison_mode_missing"}
                    else None,
                    delta_faithfulness=delta_faithfulness
                    if comparability_reason not in {"incomplete_quality", "comparison_mode_missing"}
                    else None,
                    delta_latency_ms=delta_latency
                    if comparability_reason not in {"incomplete_quality", "comparison_mode_missing"}
                    else None,
                    delta_tokens=delta_tokens
                    if comparability_reason is None
                    else None,
                    ecr_correctness=ecr if comparability_reason is None else None,
                    best_quality_mode=best_mode,
                    evidence_coverage=(
                        mean(evidence_values)
                        if evidence_values and len(evidence_values) == len(question_results)
                        else None
                    ),
                    unsupported_claim_ratio=(
                        mean(unsupported_values)
                        if unsupported_values
                        and len(unsupported_values) == len(question_results)
                        else None
                    ),
                    comparability_reason=comparability_reason,
                )
            )
            if comparability_reason:
                warnings.append(f"{question_id}: {comparability_reason}")

        return ResearchQuestionComparisonResponse(
            campaign_id=campaign_id,
            analysis_unit="question",
            sample_count=len(completed),
            independent_question_count=len(rows),
            repeat_count=max(
                (int(getattr(result, "repeat_number", 1) or 1) for result in completed),
                default=0,
            ),
            sample_note=(
                f"n = {len(completed)} execution samples = {len(rows)} questions."
            ),
            warnings=warnings,
            rows=rows,
            summaries={row.question_id: row.model_dump(mode="json") for row in rows},
        )

    async def get_agent_behavior(
        self, *, user_id: str, campaign_id: str
    ) -> AgentBehaviorResponse:
        """Return trace-backed behavior rows for every persisted campaign run."""
        await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        results = await self._results.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed = [
            result for result in results if result.status == CampaignResultStatus.COMPLETED
        ]
        traces = await self._traces.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        traces_by_result = {trace.campaign_result_id: trace for trace in traces}
        scores = await self._ragas_scores.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        work_metadata = await self._ragas_scores.list_work_metadata_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        scores = _attach_work_metadata(scores, work_metadata)
        canonical_identities = _canonical_identities_by_metric(completed, scores)
        result_by_id = {str(result.id): result for result in completed}
        attempts_by_result = {
            str(result.id): result.source_attempt_id
            for result in completed
            if result.source_attempt_id
        }
        score_map: dict[str, dict[str, float]] = defaultdict(dict)
        for score in scores:
            result_id = str(score.get("campaign_result_id"))
            metric_name = str(score.get("metric_name"))
            if (
                result_id not in attempts_by_result
                or score.get("source_attempt_id") != attempts_by_result[result_id]
                or metric_name not in canonical_identities
                or _evaluator_identity(score, result_by_id.get(result_id))
                != canonical_identities[metric_name]
            ):
                continue
            value = score.get("metric_value")
            if isinstance(value, (int, float)):
                score_map[result_id][metric_name] = float(value)
        scopes = await self._accounting.list_campaign_scopes(campaign_id)
        events = await self._accounting.list_campaign_events(campaign_id)
        events_by_scope: dict[str, list] = defaultdict(list)
        for event in events:
            events_by_scope[event.scope_id].append(event)
        rows: list[AgentBehaviorRow] = []
        for result in results:
            trace = traces_by_result.get(result.id)
            metrics = result.derived_metrics or {}
            run_scopes = [
                scope
                for scope in scopes
                if scope.scope_type == "execution_run"
                and scope.accounting_schema_version == "2"
                and any(
                    target.is_official
                    and (
                        scope.run_id == result.id
                        or target.campaign_result_id == result.id
                    )
                    for target in scope.targets
                )
            ]
            token_breakdown = _tokens(
                run_scopes,
                [
                    event
                    for scope in run_scopes
                    for event in events_by_scope[scope.scope_id]
                ],
            )
            token_status = (
                "not_available"
                if token_breakdown.accounting_status == "incomplete_legacy"
                else token_breakdown.accounting_status
            )
            quality_scores = score_map.get(str(result.id), {})
            rows.append(
                AgentBehaviorRow(
                    run_id=result.id,
                    campaign_id=result.campaign_id,
                    question_id=result.question_id,
                    mode=result.mode,
                    repeat_number=result.repeat_number,
                    trace_status=(
                        trace.trace_status
                        if trace
                        else "not_applicable"
                        if result.mode != "agentic"
                        else "not_instrumented"
                    ),
                    accounting_status=token_status,
                    subtasks=trace.subtask_count if trace else None,
                    tool_calls=trace.tool_call_count if trace else None,
                    visual_calls=trace.visual_tool_call_count if trace else None,
                    graph_calls=trace.graph_tool_call_count if trace else None,
                    drilldown_depth=trace.drilldown_depth if trace else None,
                    correctness=quality_scores.get("answer_correctness"),
                    faithfulness=quality_scores.get("faithfulness"),
                    unsupported_claim_ratio=_optional_metric(
                        metrics.get("unsupported_claim_ratio")
                    ),
                    supported_claim_ratio=_optional_metric(
                        metrics.get("supported_claim_ratio")
                    ),
                    total_tokens=token_breakdown.total_tokens,
                )
            )
        return AgentBehaviorResponse(
            campaign_id=campaign_id,
            analysis_unit="execution",
            sample_count=len(rows),
            independent_question_count=len({row.question_id for row in rows}),
            repeat_count=max((row.repeat_number for row in rows), default=0),
            sample_note="Trace-backed per-run behavior; missing traces remain N/A.",
            rows=rows,
            summaries={},
        )


def _mode_summary(
    results,
    scores,
    scopes,
    events_by_scope,
    requested_metrics,
    execution_scope_modes,
    has_unattributed_execution_scopes,
    canonical_identities,
    campaign_results,
):
    official = _official_execution_scopes(results, scopes)
    official_events = [
        event for scope in official for event in events_by_scope[scope.scope_id]
    ]
    mode = str(results[0].mode)
    operational_scopes = [
        scope for scope in scopes if execution_scope_modes.get(scope.scope_id) == mode
    ]
    operational = [
        event
        for scope in operational_scopes
        for event in events_by_scope[scope.scope_id]
    ]
    quality = _quality_for_results(
        results,
        scores,
        requested_metrics,
        scopes,
        canonical_identities,
        campaign_results,
    )
    tokens = _tokens(official, official_events)
    if has_unattributed_execution_scopes:
        tokens = _partial_for_missing_mode_attribution(tokens)
    cost = _cost(official_events, operational_events=operational)
    if has_unattributed_execution_scopes:
        cost = cost.model_copy(
            update={"operational_usd": None, "pricing_status": "partial"}
        )
    reasons: list[str] = []
    warnings = []
    if has_unattributed_execution_scopes:
        warnings.append(
            (
                "missing_mode_attribution",
                "An execution scope has no durable mode and cannot be attributed exactly.",
            )
        )
    if tokens.accounting_status == "incomplete_legacy":
        reasons.append("legacy_accounting")
    elif tokens.accounting_status != "complete":
        reasons.append("incomplete_accounting")
    # Token-only evaluations do not require a monetary price list. Pricing is
    # still returned as an independent optional status, but unknown/partial
    # USD accounting must not make otherwise valid mode results incomparable.
    if any(
        item.status != "complete" or item.valid_samples == 0
        for item in quality.values()
    ):
        reasons.append("incomplete_quality")
    if _has_noncanonical_current_scores(results, scores, canonical_identities):
        reasons.append("evaluator_metadata_mismatch")
        warnings.append(
            (
                "evaluator_metadata_mismatch",
                "Evaluator model, metric version, or signature differs across scores.",
            )
        )
    if any(scope.accounting_schema_version != "2" for scope in official):
        reasons.append("accounting_schema_version_mismatch")
    warnings.extend(_token_warning_tuples(tokens))
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


def _token_warning_tuples(tokens: TokenBreakdown) -> list[tuple[str, str]]:
    """Return stable, non-sensitive reasons for incomplete token accounting."""
    warnings: list[tuple[str, str]] = []
    if tokens.observed_call_count == 0 and tokens.accounting_status != "incomplete_legacy":
        warnings.append(
            (
                "no_usage_events",
                "No durable LLM usage events were recorded for this accounting scope.",
            )
        )
    if tokens.missing_usage_call_count:
        warnings.append(
            (
                "missing_usage",
                f"{tokens.missing_usage_call_count} LLM call(s) did not report token usage.",
            )
        )
    if tokens.unbalanced_call_count:
        warnings.append(
            (
                "unbalanced_usage",
                f"{tokens.unbalanced_call_count} measured LLM call(s) could not be reconciled to a complete total.",
            )
        )
    if tokens.unclassified_phase_call_count:
        warnings.append(
            (
                "unclassified_phase",
                f"{tokens.unclassified_phase_call_count} LLM call(s) have no explicit accounting phase.",
            )
        )
    return warnings


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


def _execution_scope_mode(scope, results) -> str | None:
    durable_modes = {str(target.mode) for target in scope.targets if target.mode}
    if len(durable_modes) == 1:
        return next(iter(durable_modes))
    if durable_modes:
        return None

    matching_modes = {
        str(result.mode)
        for result in results
        if scope.run_id == result.id
        or any(
            target.campaign_result_id == result.id
            or (
                result.source_attempt_id
                and target.attempt_id == result.source_attempt_id
            )
            for target in scope.targets
        )
    }
    return next(iter(matching_modes)) if len(matching_modes) == 1 else None


def _requested_metrics(scopes) -> dict[str, set[str]]:
    requested: dict[str, set[str]] = defaultdict(set)
    for scope in scopes:
        if scope.scope_type == "ragas_batch" and scope.metric_name in (
            *PRIMARY_QUALITY_METRICS,
            *OPTIONAL_CONTEXT_METRICS,
        ):
            requested[scope.metric_name].add(scope.status)
    return dict(requested)


def _quality_for_results(
    results,
    scores,
    requested_work,
    scopes,
    canonical_identities,
    campaign_results,
):
    result_ids = {r.id for r in results}
    results_by_id = {r.id: r for r in results}
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
        chosen = canonical_identities.get(metric)
        compatible = [
            row
            for row in rows
            if chosen is not None
            and _evaluator_identity(row, results_by_id.get(row["campaign_result_id"]))
            == chosen
        ]
        values_by_result = {
            row["campaign_result_id"]: float(row["metric_value"])
            for row in compatible
            if row.get("metric_value") is not None
        }
        work_states = _ragas_work_states_by_result(
            results,
            scopes,
            metric,
            campaign_results,
        )
        classifications = {
            result.id: _quality_sample_state(
                result.id,
                values_by_result,
                work_states,
            )
            for result in results
        }
        values = [
            values_by_result[result_id]
            for result_id, state in classifications.items()
            if state == "valid"
        ]
        valid_samples = sum(state == "valid" for state in classifications.values())
        failed_samples = sum(state == "failed" for state in classifications.values())
        missing_samples = sum(state == "missing" for state in classifications.values())
        details = compatible[0].get("details", {}) if compatible else {}
        states = set(classifications.values())
        status = (
            "complete"
            if valid_samples == len(results)
            else "partial"
            if valid_samples
            else "evaluating"
            if "evaluating" in states
            else "failed"
            if "failed" in states
            else "not_requested"
            if metric not in requested_work and metric not in score_requested
            else "partial"
        )
        output[metric] = MetricObservation(
            value=mean(values) if values else None,
            status=status,
            valid_samples=valid_samples,
            missing_samples=missing_samples,
            failed_samples=failed_samples,
            evaluator_model=details.get("evaluator_model") or details.get("model_name"),
            metric_version=details.get("metric_version"),
        )
    return output


def _evaluator_identity(row, result=None) -> tuple[str, str, str]:
    details = row.get("details") or {}
    compatibility_signature = details.get("compatibility_signature")
    signature_version = details.get("compatibility_signature_version")
    model = str(details.get("evaluator_model") or details.get("model_name") or "")
    metric_version = str(details.get("metric_version") or "")
    if signature_version != EVALUATOR_COMPATIBILITY_SIGNATURE_VERSION:
        legacy_identity = _legacy_context_policy_identity(
            row=row,
            result=result,
            evaluator_model=model,
            metric_version=metric_version,
        )
        if legacy_identity is not None:
            return legacy_identity
    if not compatibility_signature:
        compatibility_signature = row.get("evaluation_signature") or ""
    return (
        model,
        metric_version,
        str(compatibility_signature),
    )


def _legacy_context_policy_identity(
    *,
    row,
    result,
    evaluator_model: str,
    metric_version: str,
) -> tuple[str, str, str] | None:
    """Normalize a v1 score only when its durable work item verifies the hash."""
    if result is None or not getattr(result, "context_policy_version", None):
        return None
    details = row.get("details") or {}
    stored_signature = details.get("compatibility_signature")
    if not stored_signature:
        return None
    for metadata in row.get("_work_metadata", []):
        evaluator_config = metadata.get("evaluator_config")
        metadata_model = metadata.get("evaluator_model") or evaluator_model
        metadata_metric_version = metadata.get("metric_version") or metric_version
        if (
            metadata.get("compatibility_signature") != stored_signature
            or not metadata_model
            or not metadata_metric_version
            or not isinstance(evaluator_config, dict)
            or not evaluator_model
            or not metric_version
            or str(metadata_model) != evaluator_model
            or str(metadata_metric_version) != metric_version
        ):
            continue
        expected_signature = build_legacy_evaluator_compatibility_signature(
            evaluator_model=str(metadata_model),
            evaluator_config=evaluator_config,
            metric_name=str(row["metric_name"]),
            metric_version=str(metadata_metric_version),
            context_policy_version=result.context_policy_version,
            context_metrics_enabled=str(row["metric_name"]).startswith("context_"),
        )
        if expected_signature == stored_signature:
            normalized_signature = build_evaluator_compatibility_signature(
                evaluator_model=str(metadata_model),
                evaluator_config=evaluator_config,
                metric_name=str(row["metric_name"]),
                metric_version=str(metadata_metric_version),
                context_metrics_enabled=str(row["metric_name"]).startswith("context_"),
            )
            return (
                str(metadata_model),
                str(metadata_metric_version),
                normalized_signature,
            )
    return None


def _canonical_identities_by_metric(results, scores) -> dict[str, tuple[str, str, str]]:
    attempts_by_result = {
        result.id: result.source_attempt_id
        for result in results
        if result.source_attempt_id
    }
    grouped: dict[str, dict[tuple[str, str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    results_by_id = {result.id: result for result in results}
    for row in scores:
        result_id = row["campaign_result_id"]
        if (
            result_id in attempts_by_result
            and row.get("source_attempt_id") == attempts_by_result[result_id]
        ):
            grouped[row["metric_name"]][
                _evaluator_identity(row, results_by_id.get(result_id))
            ].add(result_id)
    return {
        metric: min(
            identities,
            key=lambda identity: (-len(identities[identity]), identity),
        )
        for metric, identities in grouped.items()
        if identities
    }


def _has_noncanonical_current_scores(results, scores, canonical_identities) -> bool:
    attempts_by_result = {
        result.id: result.source_attempt_id
        for result in results
        if result.source_attempt_id
    }
    results_by_id = {result.id: result for result in results}
    return any(
        row["campaign_result_id"] in attempts_by_result
        and row.get("source_attempt_id")
        == attempts_by_result[row["campaign_result_id"]]
        and row["metric_name"] in canonical_identities
        and _evaluator_identity(
            row, results_by_id.get(row["campaign_result_id"])
        )
        != canonical_identities[row["metric_name"]]
        for row in scores
    )


def _ragas_work_states_by_result(
    results,
    scopes,
    metric_name,
    campaign_results,
) -> dict[str, str]:
    campaign_results_by_attempt = {
        result.source_attempt_id: result.id
        for result in campaign_results
        if result.source_attempt_id
    }
    result_ids = {result.id for result in results}
    campaign_result_ids = {result.id for result in campaign_results}
    states: dict[str, str] = {}
    fallback_statuses: list[str] = []
    for scope in scopes:
        if scope.scope_type != "ragas_batch" or scope.metric_name != metric_name:
            continue
        has_campaign_target = False
        for target in scope.targets:
            result_id = target.campaign_result_id or campaign_results_by_attempt.get(
                target.attempt_id
            )
            if result_id not in campaign_result_ids:
                continue
            has_campaign_target = True
            if not target.is_official and result_id in result_ids:
                _set_ragas_work_state(states, result_id, scope.status)
        if not has_campaign_target:
            fallback_statuses.append(scope.status)
    if fallback_statuses:
        for result_id in result_ids:
            if result_id not in states:
                for scope_status in fallback_statuses:
                    _set_ragas_work_state(states, result_id, scope_status)
    return states


def _set_ragas_work_state(states, result_id, scope_status) -> None:
    state = (
        "failed"
        if scope_status in {"failed", "interrupted", "cancelled"}
        else "evaluating"
        if scope_status == "running"
        else None
    )
    if state is None:
        return
    current = states.get(result_id)
    if current != "failed" and (state == "failed" or current is None):
        states[result_id] = state


def _quality_sample_state(result_id, values_by_result, work_states) -> str:
    if work_states.get(result_id) == "failed":
        return "failed"
    if result_id in values_by_result:
        return "valid"
    if work_states.get(result_id) == "evaluating":
        return "evaluating"
    return "missing"


def _latency(values):
    return LatencySummary(
        mean_ms=mean(values) if values else None,
        p50_ms=nearest_rank(values, 0.5),
        p95_ms=nearest_rank(values, 0.95),
        sample_count=len(values),
        low_sample_size=0 < len(values) < 5,
    )


def _best_quality_mode(rows: list[QuestionModeComparison]) -> str | None:
    """Choose a deterministic quality winner without using mode ordering."""
    candidates = [
        row
        for row in rows
        if row.quality_status == "complete"
        and row.answer_correctness is not None
        and row.faithfulness is not None
    ]
    if not candidates:
        return None
    winner = sorted(
        candidates,
        key=lambda row: (
            -float(row.answer_correctness or 0),
            -float(row.faithfulness),
            row.mean_tokens if row.mean_tokens is not None else float("inf"),
            str(row.mode),
        ),
    )[0]
    return str(winner.mode)


def _tokens(scopes, events, legacy_status="incomplete_legacy"):
    if not scopes:
        return TokenBreakdown(
            accounting_status=legacy_status, phase_attribution_status="not_available"
        )
    observed_call_count = len(events)
    measured_call_count = sum(event.usage_status == "measured" for event in events)
    missing_usage_call_count = sum(event.usage_status == "missing" for event in events)
    unbalanced_call_count = sum(
        event.usage_status == "measured"
        and event.reconciliation_status != "balanced"
        for event in events
    )
    unclassified_phase_call_count = sum(
        event.phase == "unclassified" for event in events
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
    measured = [
        event
        for event in events
        if event.usage_status == "measured"
        and event.reconciliation_status == "balanced"
    ]
    if not measured:
        return TokenBreakdown(
            input_tokens=None,
            output_text_tokens=None,
            reasoning_tokens=None,
            other_tokens=None,
            total_tokens=None,
            by_phase={},
            observed_call_count=observed_call_count,
            measured_call_count=measured_call_count,
            missing_usage_call_count=missing_usage_call_count,
            unbalanced_call_count=unbalanced_call_count,
            unclassified_phase_call_count=unclassified_phase_call_count,
            accounting_status="partial",
            phase_attribution_status="not_available",
        )
    phase_complete = complete and all(e.phase != "unclassified" for e in measured)
    phase = defaultdict(int)
    for event in measured:
        phase[event.phase] += (
            event.input_tokens
            + event.output_text_tokens
            + event.reasoning_tokens
            + event.other_tokens
        )
    values = dict(
        input_tokens=sum(e.input_tokens for e in measured),
        output_text_tokens=sum(e.output_text_tokens for e in measured),
        reasoning_tokens=sum(e.reasoning_tokens for e in measured),
        other_tokens=sum(e.other_tokens for e in measured),
        by_phase=dict(sorted(phase.items())),
        observed_call_count=observed_call_count,
        measured_call_count=measured_call_count,
        missing_usage_call_count=missing_usage_call_count,
        unbalanced_call_count=unbalanced_call_count,
        unclassified_phase_call_count=unclassified_phase_call_count,
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


def _partial_for_missing_mode_attribution(tokens: TokenBreakdown) -> TokenBreakdown:
    if tokens.accounting_status == "incomplete_legacy":
        return tokens
    return tokens.model_copy(
        update={
            "accounting_status": "partial",
            "phase_attribution_status": "partial",
            "total_tokens": None,
        }
    )


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


def _optional_metric(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _derived_correctness(metrics: dict) -> float | None:
    unsupported = _optional_metric(metrics.get("unsupported_claim_ratio"))
    return None if unsupported is None else max(0.0, min(1.0, 1.0 - unsupported))
