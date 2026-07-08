"""Research analytics queries for persisted evaluation runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel

from core.errors import AppError, ErrorCode
from evaluation.campaign_schemas import (
    AblationResponse,
    CampaignOverviewResponse,
    CostLatencyResponse,
    EvaluationRunListItem,
    EvaluationRunListResponse,
    HumanVsAutoResponse,
    ModeComparisonResponse,
    QuestionComparisonResponse,
    RepeatStabilitySummary,
    RouterAnalysisResponse,
    RunClaimsResponse,
    RunContextResponse,
    RunDiffResponse,
    RunLlmCallsResponse,
    RunMetricsResponse,
    RunRetrievalResponse,
    RunToolsResponse,
    RunTraceResponse,
)
from evaluation.db import CampaignRepository, CampaignResultRepository, connect_db, init_db
from evaluation.observability_storage import EvaluationObservabilityRepository


def _dump(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json")


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

    async def campaign_overview(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> CampaignOverviewResponse:
        campaign = await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        llm_calls_by_run = [await self._observability_repository.list_llm_calls_for_run(item.id) for item in results]
        all_llm_calls = [
            call
            for calls in llm_calls_by_run
            for call in calls
            if call.campaign_id == campaign_id
        ]
        sample_count = len(results)
        independent_question_count = len({item.question_id for item in results})
        repeat_count = max((item.run_number for item in results), default=0)
        mode_counts = Counter(item.mode for item in results)
        latencies = [
            item.total_latency_ms if item.total_latency_ms is not None else item.latency_ms
            for item in results
        ]
        total_cost_usd, priced_usd_count, unpriced_usd_count, cost_status = _cost_rollup(
            [call.estimated_cost_usd for call in all_llm_calls]
        )
        total_cost_twd, _, _, _ = _cost_rollup([call.estimated_cost_twd for call in all_llm_calls])
        return CampaignOverviewResponse(
            campaign_id=campaign.id,
            sample_count=sample_count,
            independent_question_count=independent_question_count,
            repeat_count=repeat_count,
            sample_note=(
                f"n = {sample_count} execution samples = "
                f"{independent_question_count} questions x up to {repeat_count} repeats, "
                "not independent questions."
            ),
            mode_counts=dict(mode_counts),
            total_tokens=sum(item.total_tokens or 0 for item in results),
            total_cost_usd=total_cost_usd,
            total_cost_twd=total_cost_twd,
            cost_status=cost_status,
            priced_call_count=priced_usd_count,
            unpriced_call_count=unpriced_usd_count,
            avg_latency_ms=_average(latencies),
        )

    async def mode_comparison(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> ModeComparisonResponse:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        by_mode: dict[str, list[Any]] = {}
        for result in results:
            by_mode.setdefault(str(result.mode), []).append(result)
        return ModeComparisonResponse(
            campaign_id=campaign_id,
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
                        [
                            item.total_latency_ms if item.total_latency_ms is not None else item.latency_ms
                            for item in items
                        ]
                    ),
                    "unsupported_claim_ratio_mean": _average(
                        [
                            item.derived_metrics.get("unsupported_claim_ratio")
                            for item in items
                            if isinstance(item.derived_metrics.get("unsupported_claim_ratio"), (int, float))
                        ]
                    ),
                    "evidence_coverage_mean": _average(
                        [
                            item.derived_metrics.get("evidence_coverage")
                            for item in items
                            if isinstance(item.derived_metrics.get("evidence_coverage"), (int, float))
                        ]
                    ),
                }
                for mode, items in by_mode.items()
            },
        )

    async def question_comparison(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> QuestionComparisonResponse:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        by_question: dict[str, list[Any]] = {}
        for result in results:
            by_question.setdefault(result.question_id, []).append(result)
        return QuestionComparisonResponse(
            campaign_id=campaign_id,
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

    async def cost_latency(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> CostLatencyResponse:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        warnings = []
        if overview.cost_status != "complete":
            warnings.append("Some LLM calls have unknown price estimates; cost totals are omitted.")
        return CostLatencyResponse(
            campaign_id=campaign_id,
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

    async def router_analysis(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> RouterAnalysisResponse:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        decisions = []
        for result in results:
            decisions.extend(
                _dump(item)
                for item in await self._observability_repository.list_routing_decisions_for_run(result.id)
                if item.campaign_id == campaign_id
            )
        return RouterAnalysisResponse(
            campaign_id=campaign_id,
            analysis_unit="execution",
            analysis_type="retrospective",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            rows=decisions,
            summaries={"decision_count": len(decisions)},
        )

    async def ablation(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> AblationResponse:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        by_condition: dict[str, int] = Counter(
            str(result.derived_metrics.get("condition_id") or result.execution_profile or "default")
            for result in results
        )
        return AblationResponse(
            campaign_id=campaign_id,
            analysis_unit="execution",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            summaries={"condition_counts": dict(by_condition)},
        )

    async def human_vs_auto(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> HumanVsAutoResponse:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        ratings = []
        for result in results:
            ratings.extend(
                _dump(item)
                for item in await self._observability_repository.list_human_ratings_for_run(result.id)
                if item.campaign_id == campaign_id
            )
        return HumanVsAutoResponse(
            campaign_id=campaign_id,
            analysis_unit="execution",
            sample_count=overview.sample_count,
            independent_question_count=overview.independent_question_count,
            repeat_count=overview.repeat_count,
            sample_note=overview.sample_note,
            warnings=["Correlation summaries require human ratings."] if not ratings else [],
            rows=ratings,
            summaries={"human_rating_count": len(ratings)},
        )

    async def repeat_stability(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> RepeatStabilitySummary:
        overview = await self.campaign_overview(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        by_unit: dict[str, list[Any]] = {}
        for result in results:
            key = f"{result.question_id}:{result.mode}"
            by_unit.setdefault(key, []).append(result)
        return RepeatStabilitySummary(
            campaign_id=campaign_id,
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

    async def list_campaign_runs(
        self,
        *,
        user_id: str,
        campaign_id: str,
    ) -> EvaluationRunListResponse:
        await self._campaign_repository.get(user_id=user_id, campaign_id=campaign_id)
        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        return EvaluationRunListResponse(
            campaign_id=campaign_id,
            runs=[
                EvaluationRunListItem(
                    run_id=item.id,
                    campaign_id=item.campaign_id,
                    question_id=item.question_id,
                    question=item.question,
                    mode=item.mode,
                    run_number=item.run_number,
                    status=item.status,
                    total_tokens=item.total_tokens or 0,
                    total_latency_ms=item.total_latency_ms,
                    created_at=item.created_at,
                )
                for item in results
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

    async def run_diff(
        self,
        *,
        user_id: str,
        run_id: str,
        baseline_run_id: str,
    ) -> RunDiffResponse:
        campaign_id = await self._campaign_id_for_owned_run(user_id=user_id, run_id=run_id)
        baseline_campaign_id = await self._campaign_id_for_owned_run(
            user_id=user_id,
            run_id=baseline_run_id,
        )
        result = await self._result_repository.get(user_id=user_id, campaign_id=campaign_id, result_id=run_id)
        baseline = await self._result_repository.get(
            user_id=user_id,
            campaign_id=baseline_campaign_id,
            result_id=baseline_run_id,
        )
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
        answer_changed, answer_change_status = self._answer_change(result.answer, result.final_answer_hash, baseline.answer, baseline.final_answer_hash)
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
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Evaluation run not found",
                status_code=404,
            )
        return str(row["campaign_id"])

    async def _dump_owned_trace_events(self, *, run_id: str, campaign_id: str) -> list[dict[str, Any]]:
        return [
            _dump(item)
            for item in await self._observability_repository.list_trace_events_for_run(run_id)
            if item.campaign_id == campaign_id
        ]

    def _numeric_metric_delta(
        self,
        metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
    ) -> dict[str, float]:
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
