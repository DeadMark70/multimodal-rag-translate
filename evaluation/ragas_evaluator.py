"""RAGAS evaluation workflow for persisted evaluation campaign results."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Awaitable, Callable, Optional

from core.providers import get_llm

from evaluation.campaign_schemas import (
    CampaignMetricRow,
    CampaignMetricsResponse,
    CampaignResultStatus,
    CampaignStatus,
    DeltaGroupSummary,
    DeltaModeSummary,
    GroupMetricsSummary,
    MetricAggregate,
    ModeMetricsSummary,
)
from evaluation.db import CampaignResultRepository, RagasScoreRepository
from evaluation.retry import RateBudget, run_with_retry

PRIMARY_RAGAS_METRICS = ["faithfulness", "answer_correctness", "answer_relevancy"]
CONTEXT_RAGAS_METRICS = ["context_precision", "context_recall"]
ECR_TOKEN_FLOOR = 200

logger = logging.getLogger(__name__)

EvaluationProgressCallback = Callable[
    [int, int, str | None, str | None], Awaitable[None]
]


def _clean_metric(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return numeric


def _metric_aggregate(values: list[float]) -> MetricAggregate:
    if not values:
        return MetricAggregate()
    return MetricAggregate(
        mean=sum(values) / len(values),
        max=max(values),
        stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _metric_mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ecr_note_for_delta_tokens(delta_total_tokens: float) -> str | None:
    if delta_total_tokens <= 0:
        return "non_positive_marginal_cost"
    if 0 < delta_total_tokens < ECR_TOKEN_FLOOR:
        return "marginal_cost_too_small"
    return None


def _classify_metric_error(exc: Exception) -> str:
    winerror = getattr(exc, "winerror", None)
    if winerror == 995:
        return "io_aborted"
    message = str(exc).lower()
    if "different loop" in message or "attached to a different loop" in message:
        return "loop_conflict"
    if (
        "winerror 995" in message
        or "aborted i/o" in message
        or "i/o operation has been aborted" in message
    ):
        return "io_aborted"
    if "resource exhausted" in message or "rate limit" in message or "429" in message:
        return "rate_limit"
    return "unknown"


def _ecr_direction(*, delta: float | None, note: str | None) -> str:
    if note:
        return "neutral"
    if delta is None:
        return "neutral"
    if delta > 0:
        return "positive"
    return "negative"


class RagasEvaluator:
    """Evaluate campaign results with RAGAS and aggregate chart-ready metrics."""

    def __init__(
        self,
        result_repository: Optional[CampaignResultRepository] = None,
        score_repository: Optional[RagasScoreRepository] = None,
        *,
        evaluator_model: Optional[str] = None,
        batch_size: int = 4,
        parallel_batches: int = 1,
        rpm_limit: int = 240,
        enable_context_metrics: Optional[bool] = None,
    ) -> None:
        self._result_repository = result_repository or CampaignResultRepository()
        self._score_repository = score_repository or RagasScoreRepository()
        self._evaluator_model = evaluator_model or os.getenv(
            "EVALUATION_EVALUATOR_MODEL",
            "gemini-3.1-flash-lite-preview",
        )
        self._batch_size = batch_size
        self._parallel_batches = parallel_batches
        self._rpm_limit = rpm_limit
        if enable_context_metrics is None:
            enable_context_metrics = os.getenv(
                "ENABLE_RAGAS_CONTEXT_METRICS", "false"
            ).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self._enable_context_metrics = enable_context_metrics

    @property
    def evaluator_model(self) -> str:
        return self._evaluator_model

    @property
    def enabled_metrics(self) -> list[str]:
        metrics = list(PRIMARY_RAGAS_METRICS)
        if self._enable_context_metrics:
            metrics.extend(CONTEXT_RAGAS_METRICS)
        return metrics

    async def evaluate_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        ragas_batch_size: Optional[int] = None,
        ragas_parallel_batches: Optional[int] = None,
        ragas_rpm_limit: Optional[int] = None,
        selected_result_ids: Optional[list[str]] = None,
        on_progress: Optional[EvaluationProgressCallback] = None,
    ) -> str:
        rows = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed_rows = [
            row for row in rows if row.status == CampaignResultStatus.COMPLETED
        ]
        selected_result_id_set = {
            result_id for result_id in (selected_result_ids or []) if result_id
        }
        if selected_result_id_set:
            completed_rows = [
                row for row in completed_rows if row.id in selected_result_id_set
            ]

        if not completed_rows:
            if selected_result_id_set:
                await self._score_repository.replace_for_campaign_subset(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    selected_result_ids=list(selected_result_id_set),
                    score_rows=[],
                )
            else:
                await self._score_repository.replace_for_campaign(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    score_rows=[],
                )
            return self._evaluator_model

        try:
            ragas_dependencies = await self._load_ragas_dependencies()
        except ModuleNotFoundError as exc:
            missing_name = getattr(exc, "name", None) or "unknown"
            logger.warning(
                "RAGAS evaluation dependencies missing (%s); skipping evaluation for campaign %s.",
                missing_name,
                campaign_id,
            )
            if selected_result_id_set:
                await self._score_repository.replace_for_campaign_subset(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    selected_result_ids=list(selected_result_id_set),
                    score_rows=[],
                )
            else:
                await self._score_repository.replace_for_campaign(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    score_rows=[],
                )
            return self._evaluator_model

        evaluator_llm = ragas_dependencies["LangchainLLMWrapper"](
            get_llm("evaluator", model_name=self._evaluator_model)
        )
        await ragas_dependencies["initialize_embeddings"]()
        evaluator_embeddings = ragas_dependencies["LangchainEmbeddingsWrapper"](
            ragas_dependencies["get_embeddings"]()
        )

        score_rows: list[dict[str, Any]] = []
        total_rows = len(completed_rows)
        completed_count = 0
        effective_batch_size = max(1, min(8, ragas_batch_size or self._batch_size))
        effective_parallel_batches = max(
            1, min(8, ragas_parallel_batches or self._parallel_batches)
        )
        effective_rpm_limit = max(1, min(1000, ragas_rpm_limit or self._rpm_limit))
        rate_budget = RateBudget(rpm_limit=effective_rpm_limit)

        batches = [
            completed_rows[offset : offset + effective_batch_size]
            for offset in range(0, total_rows, effective_batch_size)
        ]
        semaphore = asyncio.Semaphore(effective_parallel_batches)

        async def evaluate_one_batch(
            batch_rows: list[Any],
        ) -> tuple[list[Any], list[dict[str, Any]]]:
            async with semaphore:
                batch_scores = await self._evaluate_batch(
                    batch_rows=batch_rows,
                    ragas_dependencies=ragas_dependencies,
                    evaluator_llm=evaluator_llm,
                    evaluator_embeddings=evaluator_embeddings,
                    rate_budget=rate_budget,
                )
                return batch_rows, batch_scores

        tasks = [
            asyncio.create_task(evaluate_one_batch(batch_rows))
            for batch_rows in batches
        ]
        for done in asyncio.as_completed(tasks):
            batch_rows, batch_scores = await done
            score_rows.extend(batch_scores)
            completed_count += len(batch_rows)
            if on_progress and batch_rows:
                last_row = batch_rows[-1]
                await on_progress(
                    completed_count, total_rows, last_row.question_id, last_row.mode
                )

        if selected_result_id_set:
            await self._score_repository.replace_for_campaign_subset(
                user_id=user_id,
                campaign_id=campaign_id,
                selected_result_ids=[row.id for row in completed_rows],
                score_rows=score_rows,
            )
        else:
            await self._score_repository.replace_for_campaign(
                user_id=user_id,
                campaign_id=campaign_id,
                score_rows=score_rows,
            )
        return self._evaluator_model

    async def get_metrics(
        self,
        *,
        user_id: str,
        campaign: CampaignStatus,
    ) -> CampaignMetricsResponse:
        score_rows = await self._score_repository.list_for_campaign(
            user_id=user_id,
            campaign_id=campaign.id,
        )
        if not score_rows:
            return CampaignMetricsResponse(
                campaign=campaign,
                evaluator_model=self._evaluator_model,
                available_metrics=self.enabled_metrics,
                summary_by_mode={},
                summary_by_category={},
                summary_by_focus={},
                delta_by_category={},
                delta_by_difficulty={},
                delta_by_question={},
                evaluation_warnings={
                    "total_metric_rows": 0,
                    "invalid_metric_rows": 0,
                    "invalid_ratio": 0,
                    "invalid_by_metric": {},
                },
                rows=[],
            )

        results = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign.id
        )
        score_map: dict[str, dict[str, float]] = defaultdict(dict)
        invalid_map: dict[str, dict[str, bool]] = defaultdict(dict)
        invalid_reasons_map: dict[str, dict[str, str]] = defaultdict(dict)
        reference_sources: dict[str, str] = {}
        context_policy_versions: dict[str, str] = {}
        evaluator_model = self._evaluator_model
        invalid_rows = 0
        invalid_by_metric: dict[str, int] = defaultdict(int)

        for score_row in score_rows:
            campaign_result_id = score_row["campaign_result_id"]
            metric_name = str(score_row["metric_name"])
            score_map[campaign_result_id][metric_name] = _clean_metric(
                score_row["metric_value"]
            )
            details = score_row.get("details", {})
            if isinstance(details, dict):
                invalid_metric = bool(details.get("invalid_metric"))
                invalid_map[campaign_result_id][metric_name] = invalid_metric
                if invalid_metric:
                    invalid_rows += 1
                    invalid_by_metric[metric_name] += 1
                    reason = str(
                        details.get("error_type") or details.get("error") or "unknown"
                    )
                    invalid_reasons_map[campaign_result_id][metric_name] = reason
                if details.get("evaluator_model"):
                    evaluator_model = str(details["evaluator_model"])
                if details.get("reference_source"):
                    reference_sources[campaign_result_id] = str(
                        details["reference_source"]
                    )
                if details.get("context_policy_version"):
                    context_policy_versions[campaign_result_id] = str(
                        details["context_policy_version"]
                    )

        chart_rows: list[CampaignMetricRow] = []
        for result in results:
            if result.status != CampaignResultStatus.COMPLETED:
                continue
            metrics = score_map.get(result.id)
            if not metrics:
                continue
            metric_values = {
                metric_name: _clean_metric(metrics.get(metric_name))
                for metric_name in self.enabled_metrics
                if metric_name in metrics
            }
            chart_rows.append(
                CampaignMetricRow(
                    campaign_result_id=result.id,
                    question_id=result.question_id,
                    question=result.question,
                    mode=result.mode,
                    run_number=result.run_number,
                    category=result.category,
                    difficulty=result.difficulty,
                    ragas_focus=list(result.ragas_focus),
                    reference_source=reference_sources.get(result.id),
                    context_policy_version=(
                        context_policy_versions.get(result.id)
                        or result.context_policy_version
                    ),
                    total_tokens=int(result.token_usage.get("total_tokens", 0)),
                    metric_values=metric_values,
                    invalid_metrics=invalid_map.get(result.id, {}),
                    invalid_reasons=invalid_reasons_map.get(result.id, {}),
                    faithfulness=_clean_metric(metric_values.get("faithfulness")),
                    answer_correctness=_clean_metric(
                        metric_values.get("answer_correctness")
                    ),
                )
            )

        total_metric_rows = len(score_rows)
        invalid_ratio = invalid_rows / total_metric_rows if total_metric_rows > 0 else 0
        summary_by_mode = self._summaries_by_mode(chart_rows)
        return CampaignMetricsResponse(
            campaign=campaign,
            evaluator_model=evaluator_model,
            available_metrics=self.enabled_metrics,
            summary_by_mode=summary_by_mode,
            summary_by_category=self._group_summaries(
                chart_rows,
                lambda row: [row.category] if row.category else [],
            ),
            summary_by_focus=self._group_summaries(
                chart_rows,
                lambda row: row.ragas_focus,
            ),
            delta_by_category=self._delta_groups(
                chart_rows,
                lambda row: [row.category] if row.category else [],
            ),
            delta_by_difficulty=self._delta_groups(
                chart_rows,
                lambda row: [row.difficulty] if row.difficulty else [],
            ),
            delta_by_question=self._delta_groups(
                chart_rows,
                lambda row: [row.question_id] if row.question_id else [],
            ),
            evaluation_warnings={
                "total_metric_rows": total_metric_rows,
                "invalid_metric_rows": invalid_rows,
                "invalid_ratio": invalid_ratio,
                "invalid_by_metric": dict(sorted(invalid_by_metric.items())),
            },
            rows=chart_rows,
        )

    async def _load_ragas_dependencies(self) -> dict[str, Any]:
        # Lazy import avoids sandbox-time network fetches during module import.
        from data_base.vector_store_manager import get_embeddings, initialize_embeddings
        from datasets import Dataset
        from ragas import aevaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        return {
            "Dataset": Dataset,
            "aevaluate": aevaluate,
            "RunConfig": RunConfig,
            "LangchainLLMWrapper": LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
            "initialize_embeddings": initialize_embeddings,
            "get_embeddings": get_embeddings,
            "metrics": {
                "faithfulness": faithfulness,
                "answer_correctness": answer_correctness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            },
        }

    async def _evaluate_batch(
        self,
        *,
        batch_rows: list[Any],
        ragas_dependencies: dict[str, Any],
        evaluator_llm: Any,
        evaluator_embeddings: Any,
        rate_budget: Optional[RateBudget] = None,
    ) -> list[dict[str, Any]]:
        dataset = ragas_dependencies["Dataset"].from_dict(
            {
                "question": [row.question for row in batch_rows],
                "answer": [row.answer for row in batch_rows],
                "contexts": [row.contexts for row in batch_rows],
                "ground_truth": [
                    self._effective_ground_truth(row) for row in batch_rows
                ],
            }
        )

        metric_scores: dict[str, list[float]] = {}
        metric_errors: dict[str, dict[str, str]] = {}
        for metric_name in self.enabled_metrics:
            metric_template = ragas_dependencies["metrics"][metric_name]

            try:
                if rate_budget is not None:
                    await rate_budget.acquire()
                metric_scores[metric_name] = await run_with_retry(
                    lambda: self._evaluate_metric_async(
                        dataset=dataset,
                        metric_name=metric_name,
                        metric_template=metric_template,
                        ragas_dependencies=ragas_dependencies,
                        evaluator_llm=evaluator_llm,
                        evaluator_embeddings=evaluator_embeddings,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                metric_scores[metric_name] = [0.0] * len(batch_rows)
                metric_errors[metric_name] = {
                    "message": str(exc),
                    "error_type": _classify_metric_error(exc),
                }

        return self._score_rows_from_batch(
            batch_rows,
            metric_scores,
            metric_errors,
        )

    def _effective_ground_truth(self, row: Any) -> str:
        short = getattr(row, "ground_truth_short", None)
        if isinstance(short, str) and short.strip():
            return short
        return str(getattr(row, "ground_truth", "") or "")

    def _reference_source(self, row: Any) -> str:
        short = getattr(row, "ground_truth_short", None)
        if isinstance(short, str) and short.strip():
            return "ground_truth_short"
        return "ground_truth_fallback_long"

    async def _evaluate_metric_async(
        self,
        *,
        dataset: Any,
        metric_name: str,
        metric_template: Any,
        ragas_dependencies: dict[str, Any],
        evaluator_llm: Any,
        evaluator_embeddings: Any,
    ) -> list[float]:
        metric = copy.deepcopy(metric_template)
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

        result = await ragas_dependencies["aevaluate"](
            dataset=dataset,
            metrics=[metric],
            llm=evaluator_llm,
            raise_exceptions=True,
            run_config=ragas_dependencies["RunConfig"](timeout=360),
        )
        values = result[metric_name]
        if not isinstance(values, list):
            values = [values]
        return [_clean_metric(value) for value in values]

    def _score_rows_from_batch(
        self,
        batch_rows: list[Any],
        metric_scores: dict[str, list[float]],
        metric_errors: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        score_rows: list[dict[str, Any]] = []
        for index, row in enumerate(batch_rows):
            reference_source = self._reference_source(row)
            for metric_name in self.enabled_metrics:
                scores = metric_scores.get(metric_name, [])
                metric_value = _clean_metric(
                    scores[index] if index < len(scores) else 0.0
                )
                details = {
                    "evaluator_model": self._evaluator_model,
                    "question_id": row.question_id,
                    "reference_source": reference_source,
                    "context_policy_version": getattr(
                        row, "context_policy_version", None
                    ),
                    "invalid_metric": False,
                }
                if metric_name in metric_errors:
                    details["error"] = metric_errors[metric_name].get("message", "")
                    details["error_type"] = metric_errors[metric_name].get(
                        "error_type", "unknown"
                    )
                    details["invalid_metric"] = True
                score_rows.append(
                    {
                        "campaign_result_id": row.id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "details": details,
                    }
                )
        return score_rows

    def _valid_metric_values(
        self,
        rows: list[CampaignMetricRow],
        metric_name: str,
    ) -> list[float]:
        values: list[float] = []
        for row in rows:
            if row.invalid_metrics.get(metric_name):
                continue
            if metric_name not in row.metric_values:
                continue
            values.append(_clean_metric(row.metric_values.get(metric_name)))
        return values

    def _group_summaries(
        self,
        rows: list[CampaignMetricRow],
        key_getter: Callable[[CampaignMetricRow], list[str]],
    ) -> dict[str, GroupMetricsSummary]:
        grouped: dict[str, list[CampaignMetricRow]] = defaultdict(list)
        for row in rows:
            for key in key_getter(row):
                if key:
                    grouped[key].append(row)

        summaries: dict[str, GroupMetricsSummary] = {}
        for group_key, group_rows in grouped.items():
            summaries[group_key] = GroupMetricsSummary(
                group_key=group_key,
                sample_count=len(group_rows),
                metric_summaries={
                    metric_name: _metric_aggregate(
                        self._valid_metric_values(group_rows, metric_name)
                    )
                    for metric_name in self.enabled_metrics
                },
                total_tokens=_metric_aggregate(
                    [float(row.total_tokens) for row in group_rows]
                ),
            )
        return summaries

    def _delta_groups(
        self,
        rows: list[CampaignMetricRow],
        key_getter: Callable[[CampaignMetricRow], list[str]],
    ) -> dict[str, DeltaGroupSummary]:
        grouped: dict[str, list[CampaignMetricRow]] = defaultdict(list)
        for row in rows:
            for key in key_getter(row):
                if key:
                    grouped[key].append(row)

        output: dict[str, DeltaGroupSummary] = {}
        for group_key, group_rows in grouped.items():
            by_mode_rows: dict[str, list[CampaignMetricRow]] = defaultdict(list)
            for row in group_rows:
                by_mode_rows[row.mode].append(row)

            naive_rows = by_mode_rows.get("naive", [])
            has_naive_baseline = bool(naive_rows)
            naive_correctness_values = self._valid_metric_values(
                naive_rows,
                "answer_correctness",
            )
            naive_faithfulness_values = self._valid_metric_values(
                naive_rows,
                "faithfulness",
            )
            naive_correctness = _metric_mean_or_none(naive_correctness_values)
            naive_faithfulness = _metric_mean_or_none(naive_faithfulness_values)
            naive_tokens = (
                _metric_aggregate([float(row.total_tokens) for row in naive_rows]).mean
                if naive_rows
                else None
            )

            by_mode: dict[str, DeltaModeSummary] = {}
            for mode, mode_rows in by_mode_rows.items():
                answer_correctness_values = self._valid_metric_values(
                    mode_rows,
                    "answer_correctness",
                )
                faithfulness_values = self._valid_metric_values(
                    mode_rows,
                    "faithfulness",
                )
                answer_correctness_mean = _metric_aggregate(
                    answer_correctness_values
                ).mean
                faithfulness_mean = _metric_aggregate(faithfulness_values).mean
                total_tokens_mean = _metric_aggregate(
                    [float(row.total_tokens) for row in mode_rows]
                ).mean

                if mode == "naive":
                    by_mode[mode] = DeltaModeSummary(
                        mode=mode,
                        sample_count=len(mode_rows),
                        answer_correctness_mean=answer_correctness_mean,
                        faithfulness_mean=faithfulness_mean,
                        total_tokens_mean=total_tokens_mean,
                        delta_answer_correctness=0,
                        delta_faithfulness=0,
                        delta_total_tokens=0,
                        ecr=0,
                        ecr_note=None,
                        ecr_faithfulness=0,
                        ecr_faithfulness_note=None,
                        ecr_direction_correctness="neutral",
                        ecr_direction_faithfulness="neutral",
                    )
                    continue

                if not has_naive_baseline or naive_tokens is None:
                    by_mode[mode] = DeltaModeSummary(
                        mode=mode,
                        sample_count=len(mode_rows),
                        answer_correctness_mean=answer_correctness_mean,
                        faithfulness_mean=faithfulness_mean,
                        total_tokens_mean=total_tokens_mean,
                        delta_answer_correctness=None,
                        delta_faithfulness=None,
                        delta_total_tokens=None,
                        ecr=None,
                        ecr_note="baseline_missing",
                        ecr_faithfulness=None,
                        ecr_faithfulness_note="baseline_missing",
                        ecr_direction_correctness="neutral",
                        ecr_direction_faithfulness="neutral",
                    )
                    continue

                if (
                    not naive_correctness_values
                    or not naive_faithfulness_values
                    or not answer_correctness_values
                    or not faithfulness_values
                ):
                    by_mode[mode] = DeltaModeSummary(
                        mode=mode,
                        sample_count=len(mode_rows),
                        answer_correctness_mean=answer_correctness_mean,
                        faithfulness_mean=faithfulness_mean,
                        total_tokens_mean=total_tokens_mean,
                        delta_answer_correctness=None,
                        delta_faithfulness=None,
                        delta_total_tokens=total_tokens_mean - naive_tokens,
                        ecr=None,
                        ecr_note="insufficient_valid_samples",
                        ecr_faithfulness=None,
                        ecr_faithfulness_note="insufficient_valid_samples",
                        ecr_direction_correctness="neutral",
                        ecr_direction_faithfulness="neutral",
                    )
                    continue

                delta_answer_correctness = answer_correctness_mean - naive_correctness
                delta_faithfulness = faithfulness_mean - naive_faithfulness
                delta_total_tokens = total_tokens_mean - naive_tokens
                ecr_note = _ecr_note_for_delta_tokens(delta_total_tokens)
                ecr_faithfulness_note = ecr_note

                by_mode[mode] = DeltaModeSummary(
                    mode=mode,
                    sample_count=len(mode_rows),
                    answer_correctness_mean=answer_correctness_mean,
                    faithfulness_mean=faithfulness_mean,
                    total_tokens_mean=total_tokens_mean,
                    delta_answer_correctness=delta_answer_correctness,
                    delta_faithfulness=delta_faithfulness,
                    delta_total_tokens=delta_total_tokens,
                    ecr=(
                        None
                        if ecr_note
                        else 1000 * delta_answer_correctness / delta_total_tokens
                    ),
                    ecr_note=ecr_note,
                    ecr_faithfulness=(
                        None
                        if ecr_faithfulness_note
                        else 1000 * delta_faithfulness / delta_total_tokens
                    ),
                    ecr_faithfulness_note=ecr_faithfulness_note,
                    ecr_direction_correctness=_ecr_direction(
                        delta=delta_answer_correctness,
                        note=ecr_note,
                    ),
                    ecr_direction_faithfulness=_ecr_direction(
                        delta=delta_faithfulness,
                        note=ecr_faithfulness_note,
                    ),
                )

            output[group_key] = DeltaGroupSummary(group_key=group_key, by_mode=by_mode)

        return output

    def _summaries_by_mode(
        self,
        rows: list[CampaignMetricRow],
    ) -> dict[str, ModeMetricsSummary]:
        grouped: dict[str, list[CampaignMetricRow]] = defaultdict(list)
        for row in rows:
            grouped[row.mode].append(row)

        summaries: dict[str, ModeMetricsSummary] = {}
        naive_summary: ModeMetricsSummary | None = None
        for mode, mode_rows in grouped.items():
            metric_summaries = {
                metric_name: _metric_aggregate(
                    self._valid_metric_values(mode_rows, metric_name)
                )
                for metric_name in self.enabled_metrics
            }
            summary = ModeMetricsSummary(
                mode=mode,
                sample_count=len(mode_rows),
                metric_summaries=metric_summaries,
                faithfulness=metric_summaries.get("faithfulness", MetricAggregate()),
                answer_correctness=metric_summaries.get(
                    "answer_correctness", MetricAggregate()
                ),
                total_tokens=_metric_aggregate(
                    [float(row.total_tokens) for row in mode_rows]
                ),
            )
            summaries[mode] = summary
            if mode == "naive":
                naive_summary = summary

        if naive_summary is None:
            for summary in summaries.values():
                summary.ecr = None
                summary.ecr_note = "baseline_missing"
                summary.ecr_faithfulness = None
                summary.ecr_faithfulness_note = "baseline_missing"
                summary.ecr_direction_correctness = "neutral"
                summary.ecr_direction_faithfulness = "neutral"
            return summaries

        naive_rows = grouped.get("naive", [])
        naive_correctness_values = self._valid_metric_values(
            naive_rows,
            "answer_correctness",
        )
        naive_faithfulness_values = self._valid_metric_values(
            naive_rows,
            "faithfulness",
        )
        naive_correctness = naive_summary.answer_correctness.mean
        naive_faithfulness = naive_summary.faithfulness.mean
        naive_tokens = naive_summary.total_tokens.mean
        for mode, summary in summaries.items():
            if mode == "naive":
                summary.delta_answer_correctness = 0
                summary.delta_faithfulness = 0
                summary.delta_total_tokens = 0
                summary.ecr = 0
                summary.ecr_note = None
                summary.ecr_faithfulness = 0
                summary.ecr_faithfulness_note = None
                summary.ecr_direction_correctness = "neutral"
                summary.ecr_direction_faithfulness = "neutral"
                continue
            mode_rows = grouped.get(mode, [])
            mode_correctness_values = self._valid_metric_values(
                mode_rows,
                "answer_correctness",
            )
            mode_faithfulness_values = self._valid_metric_values(
                mode_rows,
                "faithfulness",
            )
            if (
                not naive_correctness_values
                or not naive_faithfulness_values
                or not mode_correctness_values
                or not mode_faithfulness_values
            ):
                summary.delta_answer_correctness = None
                summary.delta_faithfulness = None
                summary.delta_total_tokens = summary.total_tokens.mean - naive_tokens
                summary.ecr = None
                summary.ecr_note = "insufficient_valid_samples"
                summary.ecr_faithfulness = None
                summary.ecr_faithfulness_note = "insufficient_valid_samples"
                summary.ecr_direction_correctness = "neutral"
                summary.ecr_direction_faithfulness = "neutral"
                continue
            summary.delta_answer_correctness = (
                summary.answer_correctness.mean - naive_correctness
            )
            summary.delta_faithfulness = summary.faithfulness.mean - naive_faithfulness
            summary.delta_total_tokens = summary.total_tokens.mean - naive_tokens
            summary.ecr_note = _ecr_note_for_delta_tokens(summary.delta_total_tokens)
            summary.ecr = (
                None
                if summary.ecr_note
                else 1000
                * summary.delta_answer_correctness
                / summary.delta_total_tokens
            )
            summary.ecr_faithfulness_note = summary.ecr_note
            summary.ecr_faithfulness = (
                None
                if summary.ecr_faithfulness_note
                else 1000 * summary.delta_faithfulness / summary.delta_total_tokens
            )
            summary.ecr_direction_correctness = _ecr_direction(
                delta=summary.delta_answer_correctness,
                note=summary.ecr_note,
            )
            summary.ecr_direction_faithfulness = _ecr_direction(
                delta=summary.delta_faithfulness,
                note=summary.ecr_faithfulness_note,
            )

        return summaries
