"""Research analytics queries for persisted evaluation runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any, Literal
from uuid import uuid4
from pathlib import Path

from pydantic import BaseModel

from core.errors import AppError, ErrorCode
from evaluation.campaign_schemas import (
    AblationResponse,
    CampaignAnalyticsDashboardResponse,
    CampaignLifecycleStatus,
    CampaignErrorsResponse,
    CampaignOverviewResponse,
    CampaignPreflightIssue,
    CampaignPreflightQuestion,
    CampaignPreflightRequest,
    CampaignPreflightResponse,
    CostLatencyResponse,
    EvaluationRunListItem,
    EvaluationRunListResponse,
    ExportCampaignRequest,
    ExportCampaignResponse,
    HumanEvalQueueItem,
    HumanEvalQueueResponse,
    HumanRatingRequest,
    HumanRatingResponse,
    HumanVsAutoResponse,
    ModeComparisonResponse,
    QuestionComparisonResponse,
    RepeatStabilitySummary,
    RouterAnalysisResponse,
    RunClaimsResponse,
    RunContextResponse,
    RunDetailResponse,
    RunDiffResponse,
    RunLlmCallsResponse,
    RunMetricsResponse,
    RunRetrievalResponse,
    RunToolsResponse,
    RunTraceResponse,
    V9ContextPack,
    V9EvidencePacket,
    V9ExecutionObservability,
    V9SlotResolution,
    SanitizedErrorRow,
)
from evaluation.db import (
    CampaignRepository,
    CampaignResultRepository,
    connect_db,
    init_db,
)
from evaluation.rag_modes import RAG_MODES
from evaluation.observability_storage import (
    EvaluationObservabilityRepository,
    redact_sensitive_value,
    safe_plain_text_excerpt,
)
from data_base.agentic_v9.budget_feasibility import (
    FeasibilityStatus,
    validate_post_contract_feasibility,
    validate_pre_route_feasibility,
)
from evaluation.agentic_v9_admission import build_v9_admission_contract
from data_base.agentic_v9.repair import RepairPlan
from data_base.agentic_v9.schemas import QueryContract
from data_base.agentic_v9.schemas import (
    BudgetReservation,
    ConflictCandidate,
    EvidencePacket,
    FinalClaim,
    SlotResolution,
    SufficiencyReport,
    V9ExecutionMetrics,
)
from evaluation.trace_schemas import EvaluationHumanRating


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"api[_-]?key\s*=\s*\S+", re.IGNORECASE),
]

_GRAPH_ABLATION_METRICS = (
    "graph_node_hit_at_k",
    "graph_edge_hit_at_k",
    "graph_doc_hit_rate",
    "graph_evidence_chunk_hit_rate",
    "graph_to_chunk_success_rate",
    "graph_context_noise_ratio",
    "unsupported_graph_claim_rate",
    "router_skip_graph_accuracy",
    "router_use_graph_accuracy",
)
_V9_GOLDEN_DATASET = Path(__file__).resolve().parent / "golden" / "agentic_v9_questions_v2.json"


def _dump(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _load_v9_golden_routes() -> dict[str, str]:
    """Read the immutable route authority once per preflight request."""
    try:
        payload = json.loads(_V9_GOLDEN_DATASET.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    questions = payload.get("questions") if isinstance(payload, dict) else []
    return {
        str(question["id"]): str(question["expected_route"])
        for question in questions
        if isinstance(question, dict)
        and isinstance(question.get("id"), str)
        and isinstance(question.get("expected_route"), str)
    }


def _cost_rollup(values: list[float | None]) -> tuple[float | None, int, int, str]:
    if not values:
        return None, 0, 0, "unknown"
    priced = [float(value) for value in values if value is not None]
    unpriced_count = len(values) - len(priced)
    if unpriced_count:
        return None, len(priced), unpriced_count, "partial" if priced else "unknown"
    return sum(priced), len(priced), 0, "complete"


def _average(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _tool_matches(call: dict[str, Any], tool_type: Literal["graph", "visual"]) -> bool:
    payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
    marker = tool_type.lower()
    fields = (
        payload.get("tool_type"),
        payload.get("step_type"),
        call.get("tool_name"),
        call.get("action"),
    )
    return any(marker in str(value or "").lower() for value in fields)


def _normalized_answer(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _hash_rater_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _repeat_number(result: Any) -> int:
    repeat_number = result.derived_metrics.get("repeat_number")
    if isinstance(repeat_number, int) and repeat_number >= 1:
        return repeat_number
    return int(result.run_number)


def _sample_note(sample_count: int, question_count: int, repeat_count: int) -> str:
    return (
        f"n = {sample_count} execution samples = "
        f"{question_count} questions x up to {repeat_count} repeats, "
        "not independent questions."
    )


def _sanitize_error_message(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "Error details unavailable."
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    lowered = text.lower()
    if "\n" in text or "traceback" in lowered or "stack trace" in lowered:
        return "Provider error details were redacted."
    if len(text) > 200:
        return f"{text[:197]}..."
    return text


def _redact_question_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(snapshot)
    for key in ("ground_truth", "ground_truth_short", "source_docs", "atomic_facts", "expected_evidence"):
        redacted.pop(key, None)
    return redacted


def _redact_export_run(row: dict[str, Any], request: ExportCampaignRequest) -> dict[str, Any]:
    redacted = dict(row)
    if not request.include_answers:
        for key in ("answer", "ground_truth", "ground_truth_short", "key_points", "final_answer_hash"):
            redacted.pop(key, None)
    if not request.include_retrieved_excerpts:
        redacted["contexts"] = []
        redacted["source_doc_ids"] = []
        redacted["expected_sources"] = []
    if isinstance(redacted.get("question_snapshot"), dict):
        redacted["question_snapshot"] = _redact_question_snapshot(redacted["question_snapshot"])
    return redacted


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        rank_value = (position + 1 + end) / 2
        for _, original_index in ordered[position:end]:
            ranks[original_index] = rank_value
        position = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sum_x = sum((x - mean_x) ** 2 for x in xs)
    sum_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(sum_x * sum_y)
    if denominator <= 0:
        return None
    return numerator / denominator


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rank(xs), _rank(ys))


def _inter_rater_agreement(values_by_run: dict[str, list[EvaluationHumanRating]]) -> float | None:
    agreement_scores: list[float] = []
    for ratings in values_by_run.values():
        if len(ratings) < 2:
            continue
        correctness_values = [rating.correctness_score for rating in ratings]
        faithfulness_values = [rating.faithfulness_score for rating in ratings]
        correctness_range = max(correctness_values) - min(correctness_values)
        faithfulness_range = max(faithfulness_values) - min(faithfulness_values)
        agreement_scores.append(1 - ((correctness_range + faithfulness_range) / 2))
    if not agreement_scores:
        return None
    return max(min(mean(agreement_scores), 1.0), 0.0)


@dataclass
class _CampaignAnalyticsContext:
    campaign: Any
    campaign_id: str
    results: list[Any]
    llm_calls_by_run: dict[str, list[Any]]
    overview: CampaignOverviewResponse | None = None


class EvaluationAnalyticsService:
    """Read-only analytics API over campaign result and observability tables."""

    def __init__(
        self,
        *,
        campaign_repository: CampaignRepository | None = None,
        result_repository: CampaignResultRepository | None = None,
        observability_repository: EvaluationObservabilityRepository | None = None,
    ) -> None:
        self._campaign_repository = campaign_repository or CampaignRepository()
        self._result_repository = result_repository or CampaignResultRepository()
        self._observability_repository = observability_repository or EvaluationObservabilityRepository()
        self._terminal_context_cache: dict[
            str, tuple[str, _CampaignAnalyticsContext]
        ] = {}

    async def campaign_overview(self, *, user_id: str, campaign_id: str) -> CampaignOverviewResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        return context.overview or self._build_campaign_overview(context)

    async def campaign_preflight(
        self, *, user_id: str, request: CampaignPreflightRequest
    ) -> CampaignPreflightResponse:
        """Check the two v9 feasibility stages without reserving or invoking work."""
        from evaluation.storage import list_test_cases

        test_cases = {
            str(case.get("id")): case
            for case in await list_test_cases(user_id)
            if isinstance(case, dict) and case.get("id")
        }
        golden = _load_v9_golden_routes()
        setup_snapshot = request.model_preset.model_dump(mode="json")
        questions: list[CampaignPreflightQuestion] = []
        for question_id in request.test_case_ids:
            expected_route = golden.get(question_id)
            issues: list[CampaignPreflightIssue] = []
            test_case = test_cases.get(question_id)
            if test_case is None or expected_route is None:
                issues.append(
                    CampaignPreflightIssue(
                        stage="post_contract",
                        reason="golden_expected_route_unavailable",
                    )
                )
            else:
                question = str(test_case.get("question") or "")
                source_docs = [
                    str(value)
                    for value in test_case.get("source_docs", [])
                    if isinstance(value, str) and value
                ]
                pre_route = validate_pre_route_feasibility(
                    setup_snapshot=setup_snapshot,
                    remaining_token_budget=request.runtime_token_budget,
                    remaining_llm_calls=request.max_llm_calls,
                )
                if pre_route.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE:
                    issues.append(CampaignPreflightIssue(stage="pre_route", reason=pre_route.reason or "configuration_incompatible"))
                admission = await build_v9_admission_contract(
                    question=question,
                    user_id=user_id,
                    source_references=source_docs,
                )
                contract = admission.contract
                expected_route = contract.route
                if admission.source_scope.rejected_source_names:
                    issues.append(
                        CampaignPreflightIssue(
                            stage="post_contract",
                            reason="source_reference_unresolved",
                        )
                    )
                post_contract = validate_post_contract_feasibility(
                    contract=contract,
                    setup_snapshot=setup_snapshot,
                    remaining_token_budget=request.runtime_token_budget,
                    remaining_llm_calls=request.max_llm_calls,
                )
                if post_contract.status is FeasibilityStatus.CONFIGURATION_INCOMPATIBLE:
                    issues.append(CampaignPreflightIssue(stage="post_contract", reason=post_contract.reason or "configuration_incompatible"))
            questions.append(
                CampaignPreflightQuestion(
                    question_id=question_id,
                    expected_route=expected_route,
                    status="configuration_incompatible" if issues else "feasible",
                    issues=issues,
                )
            )
        return CampaignPreflightResponse(questions=questions)

    async def _load_campaign_context(self, *, user_id: str, campaign_id: str) -> _CampaignAnalyticsContext:
        campaign = await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        campaign_status = getattr(campaign, "status", None)
        updated_at = getattr(campaign, "updated_at", None)
        cache_marker = updated_at.isoformat() if updated_at is not None else ""
        is_terminal = campaign_status in {
            CampaignLifecycleStatus.COMPLETED,
            CampaignLifecycleStatus.COMPLETED_WITH_ERRORS,
            CampaignLifecycleStatus.FAILED,
            CampaignLifecycleStatus.CANCELLED,
        } and bool(cache_marker)
        if is_terminal:
            cached = self._terminal_context_cache.get(campaign_id)
            if cached and cached[0] == cache_marker:
                return cached[1]
        list_analytics_results = getattr(self._result_repository, "list_for_campaign_analytics", None)
        if list_analytics_results is not None:
            results = await list_analytics_results(user_id=user_id, campaign_id=campaign_id)
        else:
            # Keep lightweight test doubles and older repository adapters
            # compatible while production uses the projection query above.
            results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        llm_calls_by_run = await self._observability_repository.list_llm_calls_for_campaign(campaign_id)
        context = _CampaignAnalyticsContext(
            campaign=campaign,
            campaign_id=campaign_id,
            results=results,
            llm_calls_by_run=llm_calls_by_run,
        )
        context.overview = self._build_campaign_overview(context)
        if is_terminal:
            self._terminal_context_cache[campaign_id] = (cache_marker, context)
        return context

    def _build_campaign_overview(self, context: _CampaignAnalyticsContext) -> CampaignOverviewResponse:
        all_llm_calls = [
            call
            for calls in context.llm_calls_by_run.values()
            for call in calls
            if call.campaign_id == context.campaign_id
        ]
        results = context.results
        sample_count = len(results)
        independent_question_count = len({item.question_id for item in results})
        repeat_count = max((_repeat_number(item) for item in results), default=0)
        mode_counts = Counter(item.mode for item in results)
        latencies = [item.total_latency_ms if item.total_latency_ms is not None else item.latency_ms for item in results]
        total_cost_usd, priced_usd_count, unpriced_usd_count, cost_status = _cost_rollup(
            [call.estimated_cost_usd for call in all_llm_calls]
        )
        total_cost_twd, _, _, _ = _cost_rollup([call.estimated_cost_twd for call in all_llm_calls])
        return CampaignOverviewResponse(
            campaign_id=context.campaign.id,
            sample_count=sample_count,
            independent_question_count=independent_question_count,
            repeat_count=repeat_count,
            sample_note=_sample_note(sample_count, independent_question_count, repeat_count),
            mode_counts=dict(mode_counts),
            total_tokens=sum(item.total_tokens or 0 for item in results),
            total_cost_usd=total_cost_usd,
            total_cost_twd=total_cost_twd,
            cost_status=cost_status,
            priced_call_count=priced_usd_count,
            unpriced_call_count=unpriced_usd_count,
            avg_latency_ms=_average(latencies),
        )

    async def mode_comparison(self, *, user_id: str, campaign_id: str) -> ModeComparisonResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        return self._build_mode_comparison(context)

    def _build_mode_comparison(self, context: _CampaignAnalyticsContext) -> ModeComparisonResponse:
        overview = context.overview or self._build_campaign_overview(context)
        results = context.results
        by_mode: dict[str, list[Any]] = defaultdict(list)
        for result in results:
            by_mode[str(result.mode)].append(result)
        return ModeComparisonResponse(
            campaign_id=context.campaign_id,
            analysis_unit="execution",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            summaries={
                mode: {
                    "sample_count": len(items),
                    "total_tokens_mean": _average([item.total_tokens for item in items]),
                    "latency_ms_mean": _average(
                        [item.total_latency_ms if item.total_latency_ms is not None else item.latency_ms for item in items]
                    ),
                    "unsupported_claim_ratio_mean": _average(
                        [item.derived_metrics.get("unsupported_claim_ratio") for item in items if isinstance(item.derived_metrics.get("unsupported_claim_ratio"), (int, float))]
                    ),
                    "evidence_coverage_mean": _average(
                        [item.derived_metrics.get("evidence_coverage") for item in items if isinstance(item.derived_metrics.get("evidence_coverage"), (int, float))]
                    ),
                }
                for mode, items in by_mode.items()
            },
        )

    async def question_comparison(self, *, user_id: str, campaign_id: str) -> QuestionComparisonResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        return self._build_question_comparison(context)

    def _build_question_comparison(self, context: _CampaignAnalyticsContext) -> QuestionComparisonResponse:
        overview = context.overview or self._build_campaign_overview(context)
        results = context.results
        by_question: dict[str, list[Any]] = defaultdict(list)
        for result in results:
            by_question[result.question_id].append(result)
        return QuestionComparisonResponse(
            campaign_id=context.campaign_id,
            analysis_unit="question",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            summaries={
                question_id: {
                    "execution_sample_count": len(items),
                    "modes": sorted({str(item.mode) for item in items}),
                    "total_tokens_mean": _average([item.total_tokens for item in items]),
                }
                for question_id, items in by_question.items()
            },
        )

    async def cost_latency(self, *, user_id: str, campaign_id: str) -> CostLatencyResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        return self._build_cost_latency(context)

    def _build_cost_latency(self, context: _CampaignAnalyticsContext) -> CostLatencyResponse:
        overview = context.overview or self._build_campaign_overview(context)
        warnings = []
        if overview.cost_status != "complete":
            warnings.append("Some LLM calls have unknown price estimates; cost totals are omitted.")
        return CostLatencyResponse(
            campaign_id=context.campaign_id,
            analysis_unit="execution",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            warnings=warnings,
            summaries={
                "total_tokens": overview.total_tokens,
                "total_cost_usd": overview.total_cost_usd,
                "total_cost_twd": overview.total_cost_twd,
                "cost_status": overview.cost_status,
                "priced_call_count": overview.priced_call_count,
                "unpriced_call_count": overview.unpriced_call_count,
                "avg_latency_ms": overview.avg_latency_ms,
            },
        )

    async def router_analysis(self, *, user_id: str, campaign_id: str) -> RouterAnalysisResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        decisions = await self._routing_decisions_for_context(context)
        return self._build_router_analysis(context, decisions)

    async def _routing_decisions_for_context(self, context: _CampaignAnalyticsContext) -> list[dict[str, Any]]:
        list_for_campaign = getattr(
            self._observability_repository,
            "list_routing_decisions_for_campaign",
            None,
        )
        if list_for_campaign is not None:
            grouped = await list_for_campaign(context.campaign_id)
            return [
                {
                    **_dump(item),
                    "question_id": result.question_id,
                    "repeat_number": _repeat_number(result),
                    "run_id": result.id,
                }
                for result in context.results
                for item in grouped.get(result.id, [])
                if item.campaign_id == context.campaign_id
            ]

        # Compatibility fallback for older injected repository doubles only.
        # The production repository always exposes the bulk method above.
        decisions: list[dict[str, Any]] = []
        for result in context.results:
            decisions.extend(
                {
                    **_dump(item),
                    "question_id": result.question_id,
                    "repeat_number": _repeat_number(result),
                    "run_id": result.id,
                }
                for item in await self._observability_repository.list_routing_decisions_for_run(result.id)
                if item.campaign_id == context.campaign_id
            )
        return decisions

    def _build_router_analysis(
        self,
        context: _CampaignAnalyticsContext,
        decisions: list[dict[str, Any]],
    ) -> RouterAnalysisResponse:
        overview = context.overview or self._build_campaign_overview(context)
        return RouterAnalysisResponse(
            campaign_id=context.campaign_id,
            analysis_unit="execution",
            analysis_type="retrospective",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            rows=decisions,
            summaries={"decision_count": len(decisions)},
        )

    async def ablation(self, *, user_id: str, campaign_id: str) -> AblationResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        return self._build_ablation(context)

    def _build_ablation(self, context: _CampaignAnalyticsContext) -> AblationResponse:
        overview = context.overview or self._build_campaign_overview(context)
        results = context.results
        by_condition: dict[str, int] = Counter(
            str(result.derived_metrics.get("condition_id") or result.execution_profile or "default")
            for result in results
        )
        condition_labels: dict[str, str] = {}
        family_conditions: dict[str, dict[str, int]] = {}
        family_results: dict[str, list[Any]] = defaultdict(list)
        for result in results:
            condition_id = result.derived_metrics.get("condition_id")
            label = result.derived_metrics.get("condition_label")
            if isinstance(condition_id, str) and isinstance(label, str):
                condition_labels[condition_id] = label
            flags = result.derived_metrics.get("ablation_flags")
            family = flags.get("ablation_family") if isinstance(flags, dict) else None
            if not isinstance(family, str):
                family = RAG_MODES.get(result.mode, {}).get("ablation_family", "compatibility")
            family_conditions.setdefault(family, {})[str(condition_id or result.mode)] = (
                family_conditions.setdefault(family, {}).get(str(condition_id or result.mode), 0) + 1
            )
            family_results[family].append(result)
        family_metrics = {
            family: {
                metric: _average(
                    [
                        value
                        for item in items
                        if isinstance(
                            value := item.derived_metrics.get(metric),
                            (int, float),
                        )
                    ]
                )
                for metric in _GRAPH_ABLATION_METRICS
            }
            for family, items in family_results.items()
            if family.startswith("graph_")
        }
        return AblationResponse(
            campaign_id=context.campaign_id,
            analysis_unit="execution",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            summaries={
                "condition_counts": dict(by_condition),
                "condition_labels": condition_labels,
                "conditions_by_ablation_family": family_conditions,
                "graph_metrics_by_ablation_family": family_metrics,
            },
        )

    async def human_eval_queue(self, *, user_id: str, campaign_id: str) -> HumanEvalQueueResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        ratings_by_run = await self._observability_repository.list_human_ratings_for_campaign(campaign_id)
        return self._build_human_eval_queue(context=context, ratings_by_run=ratings_by_run, user_id=user_id)

    def _build_human_eval_queue(
        self,
        *,
        context: _CampaignAnalyticsContext,
        ratings_by_run: dict[str, list[EvaluationHumanRating]],
        user_id: str,
    ) -> HumanEvalQueueResponse:
        rater_hash = _hash_rater_id(user_id)
        rows: list[HumanEvalQueueItem] = []
        for result in context.results:
            ratings = [rating for rating in ratings_by_run.get(result.id, []) if rating.campaign_id == context.campaign_id]
            rows.append(
                HumanEvalQueueItem(
                    run_id=result.id,
                    campaign_id=context.campaign_id,
                    question_id=result.question_id,
                    question=result.question,
                    mode=result.mode,
                    run_number=result.run_number,
                    repeat_number=_repeat_number(result),
                    answer_preview=(
                        getattr(result, "answer_preview", None)
                        if getattr(result, "answer_preview", None) is not None
                        else str(getattr(result, "answer", ""))[:240]
                    ),
                    existing_rating_count=len(ratings),
                    already_rated_by_current_user=any(rating.rater_id_hash == rater_hash for rating in ratings),
                )
            )
        return HumanEvalQueueResponse(campaign_id=context.campaign_id, rows=rows)

    async def create_human_rating(
        self,
        *,
        user_id: str,
        run_id: str,
        request: HumanRatingRequest,
    ) -> HumanRatingResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        rating = EvaluationHumanRating(
            human_rating_id=str(uuid4()),
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=None,
            rater_id_hash=_hash_rater_id(user_id),
            rubric_version=request.rubric_version,
            correctness_score=request.correctness_score,
            faithfulness_score=request.faithfulness_score,
            completeness_score=request.completeness_score,
            citation_quality_score=request.citation_quality_score,
            usefulness_score=request.usefulness_score,
            comments=request.comments,
            is_blinded=request.is_blinded,
            shown_mode_label=request.shown_mode_label,
            payload={},
            created_at=await self._current_db_time(),
        )
        await self._observability_repository.record_human_rating(rating)
        return HumanRatingResponse(
            human_rating_id=rating.human_rating_id,
            run_id=rating.run_id,
            campaign_id=rating.campaign_id,
            rater_id_hash=rating.rater_id_hash,
            rubric_version=rating.rubric_version,
            correctness_score=rating.correctness_score,
            faithfulness_score=rating.faithfulness_score,
            completeness_score=rating.completeness_score,
            citation_quality_score=rating.citation_quality_score,
            usefulness_score=rating.usefulness_score,
            comments=rating.comments,
            is_blinded=rating.is_blinded,
            shown_mode_label=rating.shown_mode_label,
            created_at=rating.created_at,
        )

    async def human_vs_auto(self, *, user_id: str, campaign_id: str) -> HumanVsAutoResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        ragas_by_run = await self._ragas_metrics_for_campaign(user_id=user_id, campaign_id=campaign_id)
        ratings_by_run = await self._observability_repository.list_human_ratings_for_campaign(campaign_id)
        return self._build_human_vs_auto(context=context, ragas_by_run=ragas_by_run, ratings_by_run=ratings_by_run)

    def _build_human_vs_auto(
        self,
        *,
        context: _CampaignAnalyticsContext,
        ragas_by_run: dict[str, dict[str, float]],
        ratings_by_run: dict[str, list[EvaluationHumanRating]],
    ) -> HumanVsAutoResponse:
        rows: list[dict[str, Any]] = []
        human_scores: list[float] = []
        auto_scores: list[float] = []
        paired_ratings_by_run: dict[str, list[EvaluationHumanRating]] = {}
        for result in context.results:
            ratings = [
                rating
                for rating in ratings_by_run.get(result.id, [])
                if rating.campaign_id == context.campaign_id
            ]
            if not ratings:
                continue
            paired_ratings_by_run[result.id] = ratings
            human_correctness_mean = mean(rating.correctness_score for rating in ratings)
            human_faithfulness_mean = mean(rating.faithfulness_score for rating in ratings)
            ragas_metrics = ragas_by_run.get(result.id, {})
            auto_correctness = ragas_metrics.get("answer_correctness")
            auto_faithfulness = ragas_metrics.get("faithfulness")
            row = {
                "run_id": result.id,
                "question_id": result.question_id,
                "mode": result.mode,
                "rating_count": len(ratings),
                "human_correctness_mean": human_correctness_mean,
                "human_faithfulness_mean": human_faithfulness_mean,
                "ragas_answer_correctness": auto_correctness,
                "ragas_faithfulness": auto_faithfulness,
            }
            rows.append(row)
            if isinstance(auto_correctness, (int, float)) and isinstance(auto_faithfulness, (int, float)):
                human_scores.append((human_correctness_mean + human_faithfulness_mean) / 2)
                auto_scores.append((float(auto_correctness) + float(auto_faithfulness)) / 2)
        sample_count = len(rows)
        return HumanVsAutoResponse(
            campaign_id=context.campaign_id,
            analysis_unit="execution",
            sample_count=sample_count,
            independent_question_count=len({row["question_id"] for row in rows}),
            repeat_count=max((self._paired_repeat_number(result, rows) for result in context.results), default=0),
            sample_note=_sample_note(sample_count, len({row["question_id"] for row in rows}), max((_repeat_number(result) for result in context.results if result.id in paired_ratings_by_run), default=0)) if sample_count else "No paired human/auto samples yet.",
            warnings=["Correlation summaries require at least 2 paired samples."] if sample_count < 2 else [],
            rows=rows,
            summaries={
                "human_rating_count": sum(len(items) for items in paired_ratings_by_run.values()),
                "paired_sample_count": sample_count,
                "human_correctness_mean": _average([row["human_correctness_mean"] for row in rows]),
                "human_faithfulness_mean": _average([row["human_faithfulness_mean"] for row in rows]),
                "ragas_human_pearson_r": _pearson(human_scores, auto_scores),
                "ragas_human_spearman_r": _spearman(human_scores, auto_scores),
                "inter_rater_agreement": _inter_rater_agreement(paired_ratings_by_run),
            },
        )

    async def campaign_errors(self, *, user_id: str, campaign_id: str) -> CampaignErrorsResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        trace_events_by_run = await self._observability_repository.list_trace_events_for_campaign(campaign_id)
        return self._build_campaign_errors(
            context=context,
            trace_events_by_run=trace_events_by_run,
            llm_calls_by_run=context.llm_calls_by_run,
        )

    def _build_campaign_errors(
        self,
        *,
        context: _CampaignAnalyticsContext,
        trace_events_by_run: dict[str, list[Any]],
        llm_calls_by_run: dict[str, list[Any]],
    ) -> CampaignErrorsResponse:
        rows: list[SanitizedErrorRow] = []
        for result in context.results:
            if result.error_message:
                rows.append(
                    SanitizedErrorRow(
                        run_id=result.id,
                        campaign_id=context.campaign_id,
                        stage_name="campaign_run",
                        code="RUN_FAILED",
                        message=_sanitize_error_message(result.error_message),
                        source="run",
                        created_at=result.created_at,
                    )
                )
            for item in trace_events_by_run.get(result.id, []):
                if item.campaign_id != context.campaign_id or (not item.error and item.status != "failed"):
                    continue
                rows.append(
                    SanitizedErrorRow(
                        run_id=result.id,
                        campaign_id=context.campaign_id,
                        stage_name=item.stage_name,
                        code=str(item.error.get("code") or item.error.get("type") or "TRACE_FAILED"),
                        message=_sanitize_error_message(item.error.get("message")),
                        source="trace",
                        created_at=item.created_at,
                    )
                )
            for call in llm_calls_by_run.get(result.id, []):
                if call.campaign_id != context.campaign_id or (not call.error and call.status != "failed"):
                    continue
                rows.append(
                    SanitizedErrorRow(
                        run_id=result.id,
                        campaign_id=context.campaign_id,
                        stage_name=str(call.purpose or "llm_call"),
                        code=str(call.error.get("code") or "LLM_CALL_FAILED"),
                        message=_sanitize_error_message(call.error.get("message")),
                        source="llm_call",
                        created_at=call.created_at,
                    )
                )
        rows.sort(key=lambda item: (item.created_at, item.run_id, item.stage_name))
        return CampaignErrorsResponse(campaign_id=context.campaign_id, rows=rows)

    async def export_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        request: ExportCampaignRequest,
    ) -> ExportCampaignResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        # Export is intentionally the full-detail path. Analytics/dashboard
        # reads use the bounded projection, while explicit export retains its
        # existing redaction-aware complete result contract.
        full_results = await self._result_repository.list_for_campaign(
            user_id=user_id,
            campaign_id=campaign_id,
        )
        context = replace(context, results=full_results)
        trace_events_by_run = await self._observability_repository.list_trace_events_for_campaign(campaign_id)
        retrieval_chunks_by_run = await self._observability_repository.list_retrieval_chunks_for_campaign(campaign_id)
        claims_by_run = await self._observability_repository.list_claims_for_campaign(campaign_id)
        list_graph_events = getattr(
            self._observability_repository,
            "list_graph_events_for_campaign",
            None,
        )
        list_graph_evidence_items = getattr(
            self._observability_repository,
            "list_graph_evidence_items_for_campaign",
            None,
        )
        graph_events_by_run = (
            await list_graph_events(campaign_id) if list_graph_events is not None else {}
        )
        graph_evidence_items_by_run = (
            await list_graph_evidence_items(campaign_id)
            if list_graph_evidence_items is not None
            else {}
        )
        runs: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []
        llm_calls: list[dict[str, Any]] = []
        retrieval_summary: list[dict[str, Any]] = []
        claim_summary: list[dict[str, Any]] = []
        for result in context.results:
            row = result.model_dump(mode="json")
            runs.append(_redact_export_run(row, request))

            for event in trace_events_by_run.get(result.id, []):
                if event.campaign_id != campaign_id:
                    continue
                event_row = redact_sensitive_value(_dump(event))
                if not request.include_raw_trace_payloads:
                    event_row["payload"] = {}
                trace_events.append(event_row)

            for call in context.llm_calls_by_run.get(result.id, []):
                if call.campaign_id != campaign_id:
                    continue
                call_row = redact_sensitive_value(_dump(call))
                if not request.include_prompt_previews:
                    call_row["prompt_preview"] = None
                payload = dict(call_row.get("payload") or {})
                full_prompt = payload.get("full_prompt")
                if full_prompt is not None and not request.include_full_prompts:
                    payload.pop("full_prompt", None)
                call_row["payload"] = payload
                llm_calls.append(call_row)

            chunks = [
                redact_sensitive_value(_dump(chunk))
                for chunk in retrieval_chunks_by_run.get(result.id, [])
                if chunk.campaign_id == campaign_id
            ]
            if not request.include_retrieved_excerpts:
                for chunk in chunks:
                    chunk["excerpt"] = None
            retrieval_summary.append(
                {
                    "run_id": result.id,
                    "chunk_count": len(chunks),
                    "chunks": chunks,
                    "graph_event_count": len(graph_events_by_run.get(result.id, [])),
                    "graph_events": [
                        _dump(event)
                        for event in graph_events_by_run.get(result.id, [])
                        if event.campaign_id == campaign_id
                    ],
                    "graph_evidence_item_count": len(graph_evidence_items_by_run.get(result.id, [])),
                    "graph_evidence_items": [
                        _dump(item)
                        for item in graph_evidence_items_by_run.get(result.id, [])
                    ],
                }
            )
            claim_summary.append(
                {
                    "run_id": result.id,
                    "claims": [
                        _dump(claim)
                        for claim in claims_by_run.get(result.id, [])
                        if claim.campaign_id == campaign_id
                    ],
                }
            )
        overview = context.overview or self._build_campaign_overview(context)
        errors = self._build_campaign_errors(
            context=context,
            trace_events_by_run=trace_events_by_run,
            llm_calls_by_run=context.llm_calls_by_run,
        )
        return ExportCampaignResponse(
            campaign=context.campaign.model_dump(mode="json"),
            redaction=request.model_dump(mode="json"),
            runs=runs,
            metrics={
                "overview": overview.model_dump(mode="json"),
                "errors": errors.model_dump(mode="json"),
            },
            trace_events=trace_events,
            llm_calls=llm_calls,
            retrieval_summary=retrieval_summary,
            claim_summary=claim_summary,
        )

    async def repeat_stability(self, *, user_id: str, campaign_id: str) -> RepeatStabilitySummary:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        overview = context.overview or self._build_campaign_overview(context)
        results = context.results
        by_unit: dict[str, list[Any]] = defaultdict(list)
        for result in results:
            key = f"{result.question_id}:{result.mode}"
            by_unit[key].append(result)
        return RepeatStabilitySummary(
            campaign_id=context.campaign_id,
            analysis_unit="execution",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            summaries={
                key: {
                    "repeat_count": len(items),
                    "total_tokens_min": min((item.total_tokens or 0 for item in items), default=0),
                    "total_tokens_max": max((item.total_tokens or 0 for item in items), default=0),
                }
                for key, items in by_unit.items()
            },
        )

    async def analytics_dashboard(self, *, user_id: str, campaign_id: str) -> CampaignAnalyticsDashboardResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        routing_decisions = await self._routing_decisions_for_context(context)
        trace_events_by_run = await self._observability_repository.list_trace_events_for_campaign(campaign_id)
        ratings_by_run = await self._observability_repository.list_human_ratings_for_campaign(campaign_id)
        ragas_by_run = await self._ragas_metrics_for_campaign(user_id=user_id, campaign_id=campaign_id)
        overview = context.overview or self._build_campaign_overview(context)
        return CampaignAnalyticsDashboardResponse(
            campaign_id=campaign_id,
            overview=overview,
            runs=self._build_campaign_runs(context),
            mode_comparison=self._build_mode_comparison(context),
            question_comparison=self._build_question_comparison(context),
            cost_latency=self._build_cost_latency(context),
            router_analysis=self._build_router_analysis(context, routing_decisions),
            ablation=self._build_ablation(context),
            human_vs_auto=self._build_human_vs_auto(
                context=context,
                ragas_by_run=ragas_by_run,
                ratings_by_run=ratings_by_run,
            ),
            human_queue=self._build_human_eval_queue(
                context=context,
                ratings_by_run=ratings_by_run,
                user_id=user_id,
            ),
            errors=self._build_campaign_errors(
                context=context,
                trace_events_by_run=trace_events_by_run,
                llm_calls_by_run=context.llm_calls_by_run,
            ),
        )

    async def list_campaign_runs(self, *, user_id: str, campaign_id: str) -> EvaluationRunListResponse:
        context = await self._load_campaign_context(user_id=user_id, campaign_id=campaign_id)
        return self._build_campaign_runs(context)

    def _build_campaign_runs(self, context: _CampaignAnalyticsContext) -> EvaluationRunListResponse:
        return EvaluationRunListResponse(
            campaign_id=context.campaign_id,
            runs=[
                EvaluationRunListItem(
                    run_id=item.id,
                    campaign_id=item.campaign_id,
                    question_id=item.question_id,
                    question=item.question,
                    mode=item.mode,
                    run_number=item.run_number,
                    repeat_number=_repeat_number(item),
                    condition_id=item.condition_id,
                    execution_profile=item.execution_profile,
                    agentic_execution_version=item.agentic_execution_version,
                    response_status=item.response_status,
                    status=item.status,
                    total_tokens=item.total_tokens or 0,
                    total_latency_ms=item.total_latency_ms,
                    created_at=item.created_at,
                )
                for item in context.results
            ],
        )

    async def run_trace(self, *, user_id: str, run_id: str) -> RunTraceResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        return RunTraceResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            trace_events=await self._dump_owned_trace_events(run_id=run_id, campaign_id=campaign_id),
            routing_decisions=[
                _dump(item)
                for item in await self._observability_repository.list_routing_decisions_for_run(run_id)
                if item.campaign_id == campaign_id
            ],
        )

    async def run_detail(self, *, user_id: str, run_id: str) -> RunDetailResponse:
        """Return the legacy detail fields plus the optional versioned v9 envelope."""
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        trace, retrieval, context, tools, claims = await asyncio.gather(
            self.run_trace(user_id=user_id, run_id=run_id),
            self.run_retrieval(user_id=user_id, run_id=run_id),
            self.run_context(user_id=user_id, run_id=run_id),
            self.run_tools(user_id=user_id, run_id=run_id),
            self.run_claims(user_id=user_id, run_id=run_id),
        )
        llm = await self.run_llm_calls(user_id=user_id, run_id=run_id)
        return RunDetailResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            trace_events=trace.trace_events,
            llm_calls=llm.llm_calls,
            retrieval_events=retrieval.retrieval_events,
            retrieval_chunks=retrieval.retrieval_chunks,
            context_packs=context.context_packs,
            tool_calls=tools.tool_calls,
            routing_decisions=trace.routing_decisions,
            claims=claims.claims,
            agentic_v9=await self._v9_observability_for_run(
                user_id=user_id, campaign_id=campaign_id, run_id=run_id
            ),
        )

    async def _v9_observability_for_run(
        self, *, user_id: str, campaign_id: str, run_id: str
    ) -> V9ExecutionObservability | None:
        await init_db()
        async with connect_db() as connection:
            row = await (
                await connection.execute(
                    "SELECT source_attempt_id FROM campaign_results WHERE id = ? AND campaign_id = ? AND user_id = ?",
                    (run_id, campaign_id, user_id),
                )
            ).fetchone()
        attempt_id = str(row["source_attempt_id"]) if row and row["source_attempt_id"] else None
        if not attempt_id:
            return None
        materialization = await self._observability_repository.get_v9_attempt_materialization(attempt_id)
        if materialization is None or materialization.campaign_id != campaign_id:
            return None
        payload = materialization.trace_payload
        try:
            evidence = []
            for item in await self._observability_repository.list_evidence_packets_for_attempt(
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
                for item in await self._observability_repository.list_slot_resolutions_for_attempt(attempt_id)
            ]
            return V9ExecutionObservability(
                schema_version=materialization.schema_version,
                contract=QueryContract.model_validate(payload["query_contract"]) if payload.get("query_contract") else None,
                slot_resolutions=slots,
                evidence_packets=evidence,
                sufficiency=SufficiencyReport.model_validate(payload["sufficiency"]) if payload.get("sufficiency") else None,
                context_pack=V9ContextPack.model_validate(payload["context_pack"]) if payload.get("context_pack") else None,
                budget=[BudgetReservation.model_validate(item) for item in payload.get("budget_reservations", [])],
                repairs=[RepairPlan.model_validate(item) for item in payload.get("repairs", [])],
                conflicts=[ConflictCandidate.model_validate(item) for item in payload.get("conflicts", [])],
                final_claims=[FinalClaim.model_validate(item) for item in payload.get("final_claims", [])],
                metrics=V9ExecutionMetrics.model_validate(payload.get("metrics", {})),
            )
        except (KeyError, TypeError, ValueError):
            # Older/partial materializations are intentionally represented as N/A.
            return None

    async def run_retrieval(self, *, user_id: str, run_id: str) -> RunRetrievalResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        return RunRetrievalResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            retrieval_events=[
                _dump(item)
                for item in await self._observability_repository.list_retrieval_events_for_run(run_id)
                if item.campaign_id == campaign_id
            ],
            retrieval_chunks=[
                _dump(item)
                for item in await self._observability_repository.list_retrieval_chunks_for_run(run_id)
                if item.campaign_id == campaign_id
            ],
        )

    async def run_context(self, *, user_id: str, run_id: str) -> RunContextResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        return RunContextResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            context_packs=[
                _dump(item)
                for item in await self._observability_repository.list_context_packs_for_run(run_id)
                if item.campaign_id == campaign_id
            ],
        )

    async def run_llm_calls(self, *, user_id: str, run_id: str) -> RunLlmCallsResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        return RunLlmCallsResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            llm_calls=[
                _dump(item)
                for item in await self._observability_repository.list_llm_calls_for_run(run_id)
                if item.campaign_id == campaign_id
            ],
        )

    async def run_tools(
        self,
        *,
        user_id: str,
        run_id: str,
        tool_type: Literal["all", "graph", "visual"] = "all",
    ) -> RunToolsResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        tool_calls = [
            _dump(item)
            for item in await self._observability_repository.list_tool_calls_for_run(run_id)
            if item.campaign_id == campaign_id
        ]
        if tool_type != "all":
            tool_calls = [item for item in tool_calls if _tool_matches(item, tool_type)]
        return RunToolsResponse(run_id=run_id, campaign_id=campaign_id, tool_calls=tool_calls)

    async def run_claims(self, *, user_id: str, run_id: str) -> RunClaimsResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        return RunClaimsResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            claims=[
                _dump(item)
                for item in await self._observability_repository.list_claims_for_run(run_id)
                if item.campaign_id == campaign_id
            ],
        )

    async def run_metrics(self, *, user_id: str, run_id: str) -> RunMetricsResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        result = await self._result_repository.get(user_id=user_id, campaign_id=campaign_id, result_id=run_id)
        return RunMetricsResponse(
            run_id=run_id,
            campaign_id=campaign_id,
            derived_metrics=result.derived_metrics,
            token_usage=result.token_usage,
            total_tokens=result.total_tokens or 0,
            latency_ms=result.latency_ms,
            total_latency_ms=result.total_latency_ms,
        )

    async def run_diff(self, *, user_id: str, run_id: str, baseline_run_id: str) -> RunDiffResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        baseline_campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=baseline_run_id)
        result = await self._result_repository.get(user_id=user_id, campaign_id=campaign_id, result_id=run_id)
        baseline = await self._result_repository.get(user_id=user_id, campaign_id=baseline_campaign_id, result_id=baseline_run_id)
        if campaign_id != baseline_campaign_id or result.question_id != baseline.question_id:
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message="Run diff requires runs from the same campaign and question.",
                status_code=400,
                details={
                    "run_campaign_id": campaign_id,
                    "baseline_campaign_id": baseline_campaign_id,
                    "run_question_id": result.question_id,
                    "baseline_question_id": baseline.question_id,
                },
            )
        answer_changed, answer_change_status = self._answer_change(
            result.answer,
            result.final_answer_hash,
            baseline.answer,
            baseline.final_answer_hash,
        )
        return RunDiffResponse(
            run_id=run_id,
            baseline_run_id=baseline_run_id,
            campaign_id=campaign_id,
            baseline_campaign_id=baseline_campaign_id,
            token_delta=(result.total_tokens or 0) - (baseline.total_tokens or 0),
            latency_delta_ms=(
                (result.total_latency_ms if result.total_latency_ms is not None else result.latency_ms)
                - (baseline.total_latency_ms if baseline.total_latency_ms is not None else baseline.latency_ms)
            ),
            comparable=True,
            comparison_scope="same_run" if run_id == baseline_run_id else "same_campaign_question",
            answer_changed=answer_changed,
            answer_change_status=answer_change_status,
            derived_metric_delta=self._numeric_metric_delta(result.derived_metrics, baseline.derived_metrics),
        )

    async def _campaign_id_for_owned_run(self, *, user_id: str, run_id: str) -> str:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                "SELECT campaign_id FROM campaign_results WHERE id = ? AND user_id = ?",
                (run_id, user_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise AppError(code=ErrorCode.NOT_FOUND, message="Evaluation run not found", status_code=404)
        return str(row["campaign_id"])

    async def _dump_owned_trace_events(self, *, run_id: str, campaign_id: str) -> list[dict[str, Any]]:
        return [
            _dump(item)
            for item in await self._observability_repository.list_trace_events_for_run(run_id)
            if item.campaign_id == campaign_id
        ]

    async def _ragas_metrics_for_campaign(self, *, user_id: str, campaign_id: str) -> dict[str, dict[str, float]]:
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute(
                """
                SELECT campaign_result_id, metric_name, metric_value
                FROM ragas_scores
                WHERE campaign_id = ? AND user_id = ?
                """,
                (campaign_id, user_id),
            )
            rows = await cursor.fetchall()
        metrics: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            metrics[str(row["campaign_result_id"])][str(row["metric_name"])] = float(row["metric_value"])
        return metrics

    async def _current_db_time(self):
        await init_db()
        async with connect_db() as connection:
            cursor = await connection.execute("SELECT CURRENT_TIMESTAMP AS now")
            row = await cursor.fetchone()
        from datetime import datetime, timezone
        raw = row["now"]
        return datetime.fromisoformat(str(raw).replace(" ", "T") + "+00:00").astimezone(timezone.utc)

    def _paired_repeat_number(self, result: Any, rows: list[dict[str, Any]]) -> int:
        if any(row["run_id"] == result.id for row in rows):
            return _repeat_number(result)
        return 0

    def _numeric_metric_delta(self, metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, float]:
        delta: dict[str, float] = {}
        for key, value in metrics.items():
            baseline_value = baseline_metrics.get(key)
            if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                delta[key] = float(value) - float(baseline_value)
        return delta

    def _answer_change(
        self,
        answer: str,
        answer_hash: str | None,
        baseline_answer: str,
        baseline_answer_hash: str | None,
    ) -> tuple[bool, str]:
        if answer_hash and baseline_answer_hash:
            changed = answer_hash != baseline_answer_hash
            return changed, "changed" if changed else "unchanged"
        if answer or baseline_answer:
            changed = _normalized_answer(answer) != _normalized_answer(baseline_answer)
            return changed, "changed" if changed else "unchanged"
        return False, "unknown"
