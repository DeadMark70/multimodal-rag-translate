"""Strict, version-2 research accounting aggregation.

This module deliberately reads the durable result, RAGAS and accounting stores;
it does not reuse the legacy analytics projection or RAGAS evaluator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from data_base.agentic_v9.schemas import (
    ATOMIC_SLOT_MATCHING_EXPERIMENTAL,
    BudgetReservation,
    ConflictCandidate,
    EvidencePacket,
    FinalClaim,
    QueryContract,
    SlotResolution,
    SufficiencyReport,
    V9ExecutionMetrics,
)
from data_base.agentic_v9.repair import RepairPlan
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
    LegacyAgentBehaviorMetrics,
    QuestionComparisonRow,
    QuestionModeComparison,
    ResearchQuestionComparisonResponse,
    V9AgentBehaviorMetrics,
    V9ContextPack,
    V9EvidencePacket,
    V9ExecutionObservability,
    V9SlotResolution,
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
from evaluation.observability_storage import (
    EvaluationObservabilityRepository,
    redact_sensitive_value,
    safe_plain_text_excerpt,
)
from evaluation.analytics import reconcile_official_tokens
from evaluation.trace_schemas import (
    EvaluationRunObservabilityDetail,
    EvaluationRunSummary,
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
        observability: EvaluationObservabilityRepository | None = None,
    ) -> None:
        self._campaigns = campaigns or CampaignRepository()
        self._results = results or CampaignResultRepository()
        self._ragas_scores = ragas_scores or RagasScoreRepository()
        self._accounting = accounting or EvaluationAccountingStore()
        self._traces = traces or AgentTraceRepository()
        self._observability = observability or EvaluationObservabilityRepository()

    async def _list_for_campaign_research(self, *, user_id: str, campaign_id: str):
        """Prefer the bounded projection, retaining injected legacy doubles."""
        list_research = getattr(self._results, "list_for_campaign_research", None)
        if list_research is not None:
            return await list_research(user_id=user_id, campaign_id=campaign_id)
        return await self._results.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )

    async def _list_llm_calls_for_campaign(self, campaign_id: str):
        loader = getattr(self._observability, "list_llm_calls_for_campaign", None)
        return await loader(campaign_id) if loader is not None else {}

    async def get_summary(
        self, *, user_id: str, campaign_id: str
    ) -> CampaignResearchSummaryResponse:
        await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        all_results = await self._list_for_campaign_research(
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
        llm_calls_by_run = await self._list_llm_calls_for_campaign(campaign_id)
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
                llm_calls_by_run,
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
        tokens = _reconcile_v9_result_tokens(
            tokens=tokens,
            scopes=official_scopes,
            events=official_events,
            results=completed,
            llm_calls_by_run=llm_calls_by_run,
        )
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

    async def get_run_token_breakdown(
        self,
        *,
        campaign_id: str,
        run_id: str,
        agentic_execution_version: str = "v8",
        observability_partial_reasons: list[str] | None = None,
    ) -> TokenBreakdown:
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
        tokens = _tokens(
            run_scopes,
            [event for rows in events_by_scope.values() for event in rows],
        )
        if agentic_execution_version != "v9":
            return tokens
        return _tokens(
            run_scopes,
            [event for rows in events_by_scope.values() for event in rows],
            provider_attempts=(
                await self._observability.list_llm_calls_for_run(run_id)
            ),
            runtime_total_tokens=tokens.total_tokens,
            observability_partial_reasons=observability_partial_reasons,
        )

    async def get_run_observability(
        self,
        *,
        user_id: str,
        campaign_id: str,
        run_id: str,
    ) -> EvaluationRunObservabilityDetail:
        """Return the safe canonical projection for one owned campaign run."""
        await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        result = await self._results.get(
            user_id=user_id, campaign_id=campaign_id, result_id=run_id
        )
        derived_metrics = (
            result.derived_metrics if isinstance(result.derived_metrics, dict) else {}
        )
        token_breakdown = await self.get_run_token_breakdown(
            campaign_id=campaign_id,
            run_id=run_id,
            agentic_execution_version=result.agentic_execution_version,
            observability_partial_reasons=derived_metrics.get(
                "observability_partial_reasons", []
            ),
        )
        trace_events = [
            item.model_copy(update={"payload": {}, "error": {}})
            for item in await self._observability.list_trace_events_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        llm_calls = [
            item.model_copy(
                update={
                    "prompt_preview": safe_plain_text_excerpt(item.prompt_preview),
                    "payload": {},
                    "error": {},
                }
            )
            for item in await self._observability.list_llm_calls_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        retrieval_events = [
            item
            for item in await self._observability.list_retrieval_events_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        retrieval_chunks = [
            item.model_copy(
                update={
                    "excerpt": safe_plain_text_excerpt(item.excerpt),
                    "payload": {},
                }
            )
            for item in await self._observability.list_retrieval_chunks_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        graph_events = [
            item
            for item in await self._observability.list_graph_events_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        graph_evidence_items = [
            item
            for item in await self._observability.list_graph_evidence_items_for_run(run_id)
            if any(event.graph_event_id == item.graph_event_id for event in graph_events)
        ]
        context_packs = [
            item
            for item in await self._observability.list_context_packs_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        tool_calls = [
            item
            for item in await self._observability.list_tool_calls_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        routing_decisions = [
            item
            for item in await self._observability.list_routing_decisions_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        claims = [
            item.model_copy(
                update={
                    "claim_text": safe_plain_text_excerpt(item.claim_text),
                    "evidence": [],
                    "payload": {},
                }
            )
            for item in await self._observability.list_claims_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        human_ratings = [
            item
            for item in await self._observability.list_human_ratings_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        graph_observability_status = "not_instrumented"
        if graph_events:
            graph_observability_status = "recorded"
            if any(
                event.graph_route.lower() in {"skip", "fallback"}
                or "fallback=" in (event.router_reason or "").lower()
                or "fallback" in event.graph_route.lower()
                for event in graph_events
            ):
                graph_observability_status = "fallback"
        else:
            for event in retrieval_events:
                payload = event.payload
                if not isinstance(payload, dict):
                    continue
                fallback_reason = payload.get("graph_fallback_reason") or payload.get(
                    "fallback_reason"
                )
                if payload.get("graph_fallback_used") or fallback_reason:
                    graph_observability_status = "fallback"
                    break
        return EvaluationRunObservabilityDetail(
            run_id=run_id,
            campaign_id=campaign_id,
            trace_events=trace_events,
            llm_calls=llm_calls,
            retrieval_events=retrieval_events,
            retrieval_chunks=retrieval_chunks,
            graph_events=graph_events,
            graph_evidence_items=graph_evidence_items,
            graph_observability_status=graph_observability_status,
            context_packs=context_packs,
            tool_calls=tool_calls,
            routing_decisions=routing_decisions,
            claims=claims,
            human_ratings=human_ratings,
            agentic_v9=await self._v9_observability_for_result(
                result=result, campaign_id=campaign_id, run_id=run_id
            ),
            accounting_diagnostics=token_breakdown,
            evidence_coverage=(
                derived_metrics.get("gold_fact_attrition")
                if isinstance(derived_metrics.get("gold_fact_attrition"), list)
                else None
            ),
            evidence_coverage_status=(
                "complete"
                if isinstance(derived_metrics.get("gold_fact_attrition"), list)
                else "not_instrumented"
            ),
            run_summary=EvaluationRunSummary(
                run_id=run_id,
                campaign_id=campaign_id,
                question_id=result.question_id,
                mode=result.mode,
                repeat_number=result.repeat_number,
                answer_preview=result.answer[:500] if result.answer else None,
                latency_ms=(
                    result.total_latency_ms
                    if result.total_latency_ms is not None
                    else result.latency_ms
                ),
                total_tokens=token_breakdown.total_tokens,
                accounting_status=(
                    token_breakdown.accounting_status
                    if token_breakdown.accounting_status != "incomplete_legacy"
                    else "not_available"
                ),
                created_at=result.created_at,
            ),
        )

    async def _v9_observability_for_result(
        self, *, result, campaign_id: str, run_id: str
    ) -> V9ExecutionObservability | None:
        """Project the selected result's owned v9 attempt without a second result query."""
        attempt_id = result.source_attempt_id
        if not attempt_id:
            return None
        materialization = await self._observability.get_v9_attempt_materialization(
            attempt_id
        )
        if (
            materialization is None
            or materialization.campaign_id != campaign_id
            or materialization.run_id != run_id
        ):
            return None
        payload = materialization.trace_payload
        try:
            evidence = []
            for item in await self._observability.list_evidence_packets_for_attempt(
                attempt_id
            ):
                packet = EvidencePacket.model_validate(item.packet)
                evidence.append(
                    V9EvidencePacket(
                        evidence_id=item.evidence_id,
                        packet=packet.model_copy(
                            update={
                                "statement": safe_plain_text_excerpt(packet.statement)
                            }
                        ),
                    )
                )
            slots = [
                V9SlotResolution(
                    slot_id=item.slot_id,
                    resolution_stage=item.resolution_stage,
                    resolution=SlotResolution.model_validate(item.resolution),
                )
                for item in await self._observability.list_slot_resolutions_for_attempt(
                    attempt_id
                )
            ]
            return V9ExecutionObservability(
                schema_version=materialization.schema_version,
                contract=(
                    QueryContract.model_validate(payload["query_contract"])
                    if payload.get("query_contract")
                    else None
                ),
                slot_resolutions=slots,
                evidence_packets=evidence,
                sufficiency=(
                    SufficiencyReport.model_validate(payload["sufficiency"])
                    if payload.get("sufficiency")
                    else None
                ),
                context_pack=(
                    V9ContextPack.model_validate(payload["context_pack"])
                    if payload.get("context_pack")
                    else None
                ),
                budget=[
                    BudgetReservation.model_validate(item)
                    for item in payload.get("budget_reservations", [])
                ],
                repairs=[
                    RepairPlan.model_validate(item) for item in payload.get("repairs", [])
                ],
                conflicts=[
                    ConflictCandidate.model_validate(item)
                    for item in payload.get("conflicts", [])
                ],
                final_claims=[
                    FinalClaim.model_validate(item)
                    for item in payload.get("final_claims", [])
                ],
                metrics=V9ExecutionMetrics.model_validate(payload.get("metrics", {})),
                comparison=(
                    redact_sensitive_value(payload["comparison"])
                    if isinstance(payload.get("comparison"), dict)
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    async def get_question_comparison(
        self, *, user_id: str, campaign_id: str
    ) -> ResearchQuestionComparisonResponse:
        """Return measured question/mode comparisons with strict null semantics."""
        campaign = await self._campaigns.get(user_id=user_id, campaign_id=campaign_id)
        all_results = await self._list_for_campaign_research(
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
        llm_calls_by_run = await self._list_llm_calls_for_campaign(campaign_id)

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
                result = result_by_id.get(str(result_id))
                tokens_by_result[str(result_id)] = (
                    _reconcile_v9_result_tokens(
                        tokens=breakdown,
                        scopes=[scope],
                        events=events_by_scope[scope.scope_id],
                        results=[result] if result is not None else [],
                        llm_calls_by_run=llm_calls_by_run,
                    )
                )

        results_by_question: dict[str, list] = defaultdict(list)
        for result in completed:
            results_by_question[str(result.question_id)].append(result)

        configured_modes = list(
            getattr(getattr(campaign, "config", None), "modes", None) or []
        )
        configured_modes.extend(
            str(result.mode)
            for result in all_results
            if str(result.mode) not in configured_modes
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
                {str(mode) for mode in configured_modes} | set(by_mode_results.keys())
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
                        mean(present)
                        if present and len(present) == len(values)
                        else None
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
                    and item.phase_attribution_status == "complete"
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
            baseline = next((row for row in mode_rows if row.mode == "naive"), None)
            target = next(
                (
                    row
                    for row in mode_rows
                    if row.mode == "agentic" and row.sample_count > 0
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
                        list(first.required_modalities)
                        if isinstance(getattr(first, "required_modalities", None), list)
                        else None
                    ),
                    by_mode=mode_rows,
                    delta_correctness=delta_correctness
                    if comparability_reason
                    not in {"incomplete_quality", "comparison_mode_missing"}
                    else None,
                    delta_faithfulness=delta_faithfulness
                    if comparability_reason
                    not in {"incomplete_quality", "comparison_mode_missing"}
                    else None,
                    delta_latency_ms=delta_latency
                    if comparability_reason
                    not in {"incomplete_quality", "comparison_mode_missing"}
                    else None,
                    delta_tokens=delta_tokens if comparability_reason is None else None,
                    ecr_correctness=ecr if comparability_reason is None else None,
                    best_quality_mode=best_mode,
                    evidence_coverage=(
                        mean(evidence_values)
                        if evidence_values
                        and len(evidence_values) == len(question_results)
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
        results = await self._list_for_campaign_research(
            user_id=user_id, campaign_id=campaign_id
        )
        completed = [
            result
            for result in results
            if result.status == CampaignResultStatus.COMPLETED
        ]
        traces = await self._traces.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        traces_by_result = {trace.campaign_result_id: trace for trace in traces}
        v9_materializations = (
            await self._observability.list_v9_attempt_materializations_for_campaign(
                campaign_id
            )
        )
        v9_counts = await self._observability.list_v9_behavior_counts_for_campaign(
            campaign_id
        )
        graph_events_by_run = await self._observability.list_graph_events_for_campaign(
            campaign_id
        )
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
        llm_calls_by_run = await self._list_llm_calls_for_campaign(campaign_id)
        rows: list[AgentBehaviorRow] = []
        for result in results:
            trace = traces_by_result.get(result.id)
            materialization = v9_materializations.get(str(result.id))
            is_v9 = bool(
                materialization
                or (trace is not None and trace.agentic_execution_version == "v9")
                or str(result.mode) in {"agentic-v9", "v9", "agentic-v9-shadow"}
            )
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
            token_breakdown = _reconcile_v9_result_tokens(
                tokens=token_breakdown,
                scopes=run_scopes,
                events=[
                    event
                    for scope in run_scopes
                    for event in events_by_scope[scope.scope_id]
                ],
                results=[result],
                llm_calls_by_run=llm_calls_by_run,
            )
            token_status = (
                "not_available"
                if token_breakdown.accounting_status == "incomplete_legacy"
                else "partial"
                if token_breakdown.phase_attribution_status != "complete"
                else token_breakdown.accounting_status
            )
            quality_scores = score_map.get(str(result.id), {})
            behavior_schema = "v9" if is_v9 else "v8" if trace else "not_applicable"
            trace_status = _agent_behavior_trace_status(
                result=result, trace=trace, is_v9=is_v9
            )
            legacy = (
                LegacyAgentBehaviorMetrics(
                    subtasks=trace.subtask_count,
                    tool_calls=trace.tool_call_count,
                    visual_calls=trace.visual_tool_call_count,
                    graph_calls=trace.graph_tool_call_count,
                    drilldown_depth=trace.drilldown_depth,
                )
                if trace is not None and not is_v9
                else None
            )
            v9 = (
                _v9_behavior_metrics(
                    trace_payload=materialization.trace_payload,
                    counts=v9_counts.get(str(result.id), {}),
                    graph_events=graph_events_by_run.get(str(result.id), []),
                )
                if is_v9 and materialization is not None
                else None
            )
            rows.append(
                AgentBehaviorRow(
                    run_id=result.id,
                    campaign_id=result.campaign_id,
                    question_id=result.question_id,
                    mode=result.mode,
                    repeat_number=result.repeat_number,
                    behavior_schema=behavior_schema,
                    trace_status=trace_status,
                    failure_reason=(
                        _safe_agent_behavior_failure_reason(result.error_message)
                        if result.status == CampaignResultStatus.FAILED
                        else None
                    ),
                    accounting_status=token_status,
                    subtasks=legacy.subtasks if legacy else None,
                    tool_calls=legacy.tool_calls if legacy else None,
                    visual_calls=legacy.visual_calls if legacy else None,
                    graph_calls=legacy.graph_calls if legacy else None,
                    drilldown_depth=legacy.drilldown_depth if legacy else None,
                    correctness=quality_scores.get("answer_correctness"),
                    faithfulness=quality_scores.get("faithfulness"),
                    unsupported_claim_ratio=_optional_metric(
                        metrics.get("unsupported_claim_ratio")
                    ),
                    supported_claim_ratio=_optional_metric(
                        metrics.get("supported_claim_ratio")
                    ),
                    total_tokens=token_breakdown.total_tokens,
                    legacy=legacy,
                    v9=v9,
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


def _agent_behavior_trace_status(*, result, trace, is_v9: bool) -> str:
    if result.status == CampaignResultStatus.FAILED:
        return "failed"
    if trace is not None:
        return trace.trace_status
    return "not_instrumented" if is_v9 else "not_applicable"


def _safe_agent_behavior_failure_reason(value: object) -> str:
    text = str(value or "").strip()
    return text or "failure_reason_not_recorded"


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _v9_behavior_metrics(
    *, trace_payload: dict, counts: dict, graph_events: list
) -> V9AgentBehaviorMetrics:
    contract = trace_payload.get("query_contract") or {}
    metrics = trace_payload.get("metrics") or {}
    sufficiency = trace_payload.get("sufficiency") or {}
    context_pack = trace_payload.get("context_pack") or {}
    graph_policy = contract.get("graph_policy")
    visual_requested = contract.get("visual_requested")
    visual_required = contract.get("visual_required")
    graph_execution_payload = trace_payload.get("graph_execution") or {}
    visual_execution_payload = trace_payload.get("visual_execution") or {}
    graph_execution = graph_execution_payload.get("state")
    if graph_execution not in {
        "not_requested",
        "not_triggered",
        "executed",
        "failed",
        "required_but_not_satisfied",
        "not_instrumented",
    }:
        graph_execution = (
            "not_requested"
            if graph_policy in (None, "never")
            else "executed"
            if graph_events
            else "required_but_not_satisfied"
            if graph_policy == "required_locator"
            else "not_triggered"
        )
    visual_execution = visual_execution_payload.get("state")
    if visual_execution not in {
        "not_requested",
        "not_triggered",
        "executed",
        "failed",
        "required_but_not_satisfied",
        "attempted_without_evidence",
        "not_instrumented",
    }:
        visual_execution = (
            "required_but_not_satisfied"
            if visual_required
            else "not_triggered"
            if visual_requested
            else "not_requested"
        )
    contract_version = str(contract.get("contract_version") or "1")
    experimental_slots = contract_version == "2"
    return V9AgentBehaviorMetrics(
        route=contract.get("route"),
        contract_version=contract_version,
        slot_plan_status=contract.get("slot_plan_status"),
        slot_semantics=(
            "heuristic_experimental" if experimental_slots else "legacy_generic"
        ),
        atomic_completeness=None,
        atomic_completeness_reason=(
            ATOMIC_SLOT_MATCHING_EXPERIMENTAL if experimental_slots else None
        ),
        graph_policy=graph_policy,
        visual_requested=(
            visual_requested if isinstance(visual_requested, bool) else None
        ),
        visual_required=visual_required if isinstance(visual_required, bool) else None,
        evidence_extraction_required=contract.get("evidence_extraction_required")
        if isinstance(contract.get("evidence_extraction_required"), bool)
        else None,
        retrieval_query_count=_optional_nonnegative_int(
            metrics.get("retrieval_query_count")
        ),
        provider_attempt_count=_optional_nonnegative_int(
            metrics.get("provider_attempt_count")
        ),
        final_generation_count=_optional_nonnegative_int(
            metrics.get("final_generation_count")
        ),
        evidence_packet_count=_optional_nonnegative_int(
            counts.get("evidence_packet_count")
        ),
        packed_evidence_count=_optional_nonnegative_int(
            len(context_pack.get("packed_evidence_ids", []))
        ),
        slot_resolution_count=_optional_nonnegative_int(
            counts.get("slot_resolution_count")
        ),
        required_slot_count=_optional_nonnegative_int(
            len(contract.get("required_slots", []))
        ),
        supported_slot_count=_optional_nonnegative_int(
            len(sufficiency.get("supported_slot_ids", []))
        ),
        repair_count=_optional_nonnegative_int(len(trace_payload.get("repairs", []))),
        final_claim_count=_optional_nonnegative_int(
            len(trace_payload.get("final_claims", []))
        ),
        reserved_tokens=sum(
            int(item.get("reserved_tokens", 0) or 0)
            for item in trace_payload.get("budget_reservations", [])
            if isinstance(item, dict)
        ),
        reconciled_tokens=_optional_nonnegative_int(metrics.get("reconciled_tokens")),
        graph_execution=graph_execution,
        visual_execution=visual_execution,
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
    llm_calls_by_run,
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
    tokens = _reconcile_v9_result_tokens(
        tokens=tokens,
        scopes=official,
        events=official_events,
        results=results,
        llm_calls_by_run=llm_calls_by_run,
    )
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
    elif (
        tokens.accounting_status != "complete"
        or tokens.phase_attribution_status != "complete"
    ):
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
    if (
        tokens.observed_call_count == 0
        and tokens.accounting_status != "incomplete_legacy"
    ):
        warnings.append(
            (
                "no_usage_events",
                "No durable LLM usage events were recorded for this accounting scope.",
            )
        )
    if tokens.missing_usage_call_count:
        dimensions = "; ".join(
            _format_usage_gap_dimension(label, values)
            for label, values in (
                ("phase", tokens.missing_usage_by_phase),
                ("purpose", tokens.missing_usage_by_purpose),
                ("provider", tokens.missing_usage_by_provider),
            )
            if values
        )
        warnings.append(
            (
                "missing_usage",
                f"{tokens.missing_usage_call_count} LLM call(s) did not report token usage. {dimensions}",
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


def _format_usage_gap_dimension(label: str, values: dict[str, int]) -> str:
    if not values:
        return ""
    detail = ", ".join(f"{key}={count}" for key, count in values.items())
    return f"by {label}: {detail}"


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
        and _evaluator_identity(row, results_by_id.get(row["campaign_result_id"]))
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


def _reconcile_v9_result_tokens(
    *,
    tokens: TokenBreakdown,
    scopes,
    events,
    results,
    llm_calls_by_run,
) -> TokenBreakdown:
    v9_results = [
        result
        for result in results
        if getattr(result, "agentic_execution_version", "v8") == "v9"
    ]
    if not v9_results:
        return tokens
    reconciled_runs = []
    for result in v9_results:
        run_id = str(result.id)
        run_scopes = [scope for scope in scopes if str(scope.run_id) == run_id]
        run_scope_ids = {scope.scope_id for scope in run_scopes}
        run_events = [event for event in events if event.scope_id in run_scope_ids]
        run_tokens = _tokens(run_scopes, run_events)
        partial_reasons = (
            (getattr(result, "derived_metrics", {}) or {}).get(
                "observability_partial_reasons", []
            )
        )
        reconciled_runs.append(
            _tokens(
                run_scopes,
                run_events,
                provider_attempts=llm_calls_by_run.get(run_id, []),
                runtime_total_tokens=run_tokens.total_tokens,
                observability_partial_reasons=partial_reasons,
            )
        )

    reasons = sorted(
        {
            reason
            for run_tokens in reconciled_runs
            for reason in run_tokens.phase_attribution_reasons
        }
    )
    phase: defaultdict[str, int] = defaultdict(int)
    for run_tokens in reconciled_runs:
        for name, count in run_tokens.by_phase.items():
            phase[name] += count
    all_complete = all(
        run_tokens.accounting_status == "complete"
        and run_tokens.phase_attribution_status == "complete"
        for run_tokens in reconciled_runs
    )
    return tokens.model_copy(
        update={
            "accounting_status": "complete"
            if tokens.accounting_status == "complete" and all_complete
            else "partial",
            "by_phase": dict(sorted(phase.items())),
            "phase_attribution_status": "complete"
            if tokens.phase_attribution_status == "complete" and all_complete
            else "partial",
            "phase_attribution_reasons": reasons,
        }
    )


def _tokens(
    scopes,
    events,
    legacy_status="incomplete_legacy",
    *,
    provider_attempts=None,
    runtime_total_tokens=None,
    observability_partial_reasons=None,
):
    if not scopes:
        return TokenBreakdown(
            accounting_status=legacy_status, phase_attribution_status="not_available"
        )
    observed_call_count = len(events)
    measured_call_count = sum(event.usage_status == "measured" for event in events)
    missing_usage_call_count = sum(event.usage_status == "missing" for event in events)
    unbalanced_call_count = sum(
        event.usage_status == "measured" and event.reconciliation_status != "balanced"
        for event in events
    )
    unclassified_phase_call_count = sum(
        event.phase == "unclassified" for event in events
    )
    missing_usage_by_phase = _usage_gap_dimension(events, "phase")
    missing_usage_by_purpose = _usage_gap_dimension(events, "purpose")
    missing_usage_by_provider = _usage_gap_dimension(events, "provider")
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
            missing_usage_by_phase=missing_usage_by_phase,
            missing_usage_by_purpose=missing_usage_by_purpose,
            missing_usage_by_provider=missing_usage_by_provider,
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
        missing_usage_by_phase=missing_usage_by_phase,
        missing_usage_by_purpose=missing_usage_by_purpose,
        missing_usage_by_provider=missing_usage_by_provider,
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
    if provider_attempts is not None:
        reconciliation = reconcile_official_tokens(
            runtime_total_tokens=runtime_total_tokens,
            calls=list(provider_attempts),
            observability_partial_reasons=observability_partial_reasons or [],
        )
        values["by_phase"] = reconciliation.by_phase
        values["phase_attribution_status"] = (
            "complete" if reconciliation.status == "complete" else "partial"
        )
        values["phase_attribution_reasons"] = list(reconciliation.reasons)
        if reconciliation.status != "complete":
            values["accounting_status"] = "partial"
    return TokenBreakdown(**values)


def _usage_gap_dimension(events, attribute: str) -> dict[str, int]:
    gaps: defaultdict[str, int] = defaultdict(int)
    for event in events:
        if event.usage_status != "missing":
            continue
        value = getattr(event, attribute, None)
        label = str(value or "unknown")
        gaps[label] += 1
    return dict(sorted(gaps.items()))


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
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
        else None
    )


def _derived_correctness(metrics: dict) -> float | None:
    unsupported = _optional_metric(metrics.get("unsupported_claim_ratio"))
    return None if unsupported is None else max(0.0, min(1.0, 1.0 - unsupported))
