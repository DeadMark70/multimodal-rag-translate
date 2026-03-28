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
    GroupMetricsSummary,
    MetricAggregate,
    ModeMetricsSummary,
)
from evaluation.db import CampaignResultRepository, RagasScoreRepository
from evaluation.retry import run_with_retry

PRIMARY_RAGAS_METRICS = ["faithfulness", "answer_correctness", "answer_relevancy"]
CONTEXT_RAGAS_METRICS = ["context_precision", "context_recall"]

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


class RagasEvaluator:
    """Evaluate campaign results with RAGAS and aggregate chart-ready metrics."""

    def __init__(
        self,
        result_repository: Optional[CampaignResultRepository] = None,
        score_repository: Optional[RagasScoreRepository] = None,
        *,
        evaluator_model: Optional[str] = None,
        batch_size: int = 4,
        enable_context_metrics: Optional[bool] = None,
    ) -> None:
        self._result_repository = result_repository or CampaignResultRepository()
        self._score_repository = score_repository or RagasScoreRepository()
        self._evaluator_model = evaluator_model or os.getenv(
            "EVALUATION_EVALUATOR_MODEL",
            "gemini-2.5-flash",
        )
        self._batch_size = batch_size
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
        on_progress: Optional[EvaluationProgressCallback] = None,
    ) -> str:
        rows = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign_id
        )
        completed_rows = [
            row for row in rows if row.status == CampaignResultStatus.COMPLETED
        ]

        if not completed_rows:
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

        for offset in range(0, total_rows, self._batch_size):
            batch_rows = completed_rows[offset : offset + self._batch_size]
            batch_scores = await self._evaluate_batch(
                batch_rows=batch_rows,
                ragas_dependencies=ragas_dependencies,
                evaluator_llm=evaluator_llm,
                evaluator_embeddings=evaluator_embeddings,
            )
            score_rows.extend(batch_scores)
            completed_count += len(batch_rows)
            if on_progress and batch_rows:
                last_row = batch_rows[-1]
                await on_progress(
                    completed_count, total_rows, last_row.question_id, last_row.mode
                )

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
                rows=[],
            )

        results = await self._result_repository.list_for_campaign(
            user_id=user_id, campaign_id=campaign.id
        )
        score_map: dict[str, dict[str, float]] = defaultdict(dict)
        reference_sources: dict[str, str] = {}
        evaluator_model = self._evaluator_model

        for score_row in score_rows:
            campaign_result_id = score_row["campaign_result_id"]
            score_map[campaign_result_id][score_row["metric_name"]] = _clean_metric(
                score_row["metric_value"]
            )
            details = score_row.get("details", {})
            if isinstance(details, dict):
                if details.get("evaluator_model"):
                    evaluator_model = str(details["evaluator_model"])
                if details.get("reference_source"):
                    reference_sources[campaign_result_id] = str(
                        details["reference_source"]
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
                    total_tokens=int(result.token_usage.get("total_tokens", 0)),
                    metric_values=metric_values,
                    faithfulness=_clean_metric(metric_values.get("faithfulness")),
                    answer_correctness=_clean_metric(
                        metric_values.get("answer_correctness")
                    ),
                )
            )

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
            rows=chart_rows,
        )

    async def _load_ragas_dependencies(self) -> dict[str, Any]:
        # Lazy import avoids sandbox-time network fetches during module import.
        from data_base.vector_store_manager import get_embeddings, initialize_embeddings
        from datasets import Dataset
        from ragas import evaluate
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
            "evaluate": evaluate,
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
        metric_errors: dict[str, str] = {}
        for metric_name in self.enabled_metrics:
            metric_template = ragas_dependencies["metrics"][metric_name]

            async def run_metric() -> list[float]:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self._evaluate_metric_sync(
                        dataset=dataset,
                        metric_name=metric_name,
                        metric_template=metric_template,
                        ragas_dependencies=ragas_dependencies,
                        evaluator_llm=evaluator_llm,
                        evaluator_embeddings=evaluator_embeddings,
                    ),
                )

            try:
                metric_scores[metric_name] = await run_with_retry(run_metric)
            except Exception as exc:  # noqa: BLE001
                metric_scores[metric_name] = [0.0] * len(batch_rows)
                metric_errors[metric_name] = str(exc)

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

    def _evaluate_metric_sync(
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

        result = ragas_dependencies["evaluate"](
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
        metric_errors: dict[str, str],
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
                }
                if metric_name in metric_errors:
                    details["error"] = metric_errors[metric_name]
                score_rows.append(
                    {
                        "campaign_result_id": row.id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "details": details,
                    }
                )
        return score_rows

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
                        [row.metric_values.get(metric_name, 0.0) for row in group_rows]
                    )
                    for metric_name in self.enabled_metrics
                },
                total_tokens=_metric_aggregate(
                    [float(row.total_tokens) for row in group_rows]
                ),
            )
        return summaries

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
                    [row.metric_values.get(metric_name, 0.0) for row in mode_rows]
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
            return summaries

        naive_correctness = naive_summary.answer_correctness.mean
        naive_tokens = naive_summary.total_tokens.mean
        for mode, summary in summaries.items():
            if mode == "naive":
                summary.delta_answer_correctness = 0
                summary.delta_total_tokens = 0
                summary.ecr = 0
                summary.ecr_note = None
                continue
            summary.delta_answer_correctness = (
                summary.answer_correctness.mean - naive_correctness
            )
            summary.delta_total_tokens = summary.total_tokens.mean - naive_tokens
            if summary.delta_total_tokens <= 0:
                summary.ecr = None
                summary.ecr_note = "non_positive_marginal_cost"
                continue
            summary.ecr = (
                1000 * summary.delta_answer_correctness / summary.delta_total_tokens
            )
            summary.ecr_note = None

        return summaries



