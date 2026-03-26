"""RAGAS evaluation workflow for persisted evaluation campaign results."""

from __future__ import annotations

import asyncio
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
    MetricAggregate,
    ModeMetricsSummary,
)
from evaluation.db import CampaignResultRepository, RagasScoreRepository
from evaluation.retry import run_with_retry

EvaluationProgressCallback = Callable[[int, int, str | None, str | None], Awaitable[None]]


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
    ) -> None:
        self._result_repository = result_repository or CampaignResultRepository()
        self._score_repository = score_repository or RagasScoreRepository()
        self._evaluator_model = evaluator_model or os.getenv(
            "EVALUATION_EVALUATOR_MODEL",
            "gemini-2.5-pro",
        )
        self._batch_size = batch_size

    @property
    def evaluator_model(self) -> str:
        return self._evaluator_model

    async def evaluate_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        on_progress: Optional[EvaluationProgressCallback] = None,
    ) -> str:
        rows = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign_id)
        completed_rows = [row for row in rows if row.status == CampaignResultStatus.COMPLETED]

        if not completed_rows:
            await self._score_repository.replace_for_campaign(
                user_id=user_id,
                campaign_id=campaign_id,
                score_rows=[],
            )
            return self._evaluator_model

        ragas_dependencies = await self._load_ragas_dependencies()
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
                await on_progress(completed_count, total_rows, last_row.question_id, last_row.mode)

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
                summary_by_mode={},
                rows=[],
            )

        results = await self._result_repository.list_for_campaign(user_id=user_id, campaign_id=campaign.id)
        score_map: dict[str, dict[str, float]] = defaultdict(dict)
        evaluator_model = self._evaluator_model

        for score_row in score_rows:
            score_map[score_row["campaign_result_id"]][score_row["metric_name"]] = _clean_metric(
                score_row["metric_value"]
            )
            details = score_row.get("details", {})
            if isinstance(details, dict) and details.get("evaluator_model"):
                evaluator_model = str(details["evaluator_model"])

        chart_rows: list[CampaignMetricRow] = []
        for result in results:
            if result.status != CampaignResultStatus.COMPLETED:
                continue
            metrics = score_map.get(result.id)
            if not metrics:
                continue
            chart_rows.append(
                CampaignMetricRow(
                    campaign_result_id=result.id,
                    question_id=result.question_id,
                    question=result.question,
                    mode=result.mode,
                    run_number=result.run_number,
                    category=result.category,
                    difficulty=result.difficulty,
                    total_tokens=int(result.token_usage.get("total_tokens", 0)),
                    faithfulness=_clean_metric(metrics.get("faithfulness")),
                    answer_correctness=_clean_metric(metrics.get("answer_correctness")),
                )
            )

        summary_by_mode = self._summaries_by_mode(chart_rows)
        return CampaignMetricsResponse(
            campaign=campaign,
            evaluator_model=evaluator_model,
            summary_by_mode=summary_by_mode,
            rows=chart_rows,
        )

    async def _load_ragas_dependencies(self) -> dict[str, Any]:
        # Lazy import avoids sandbox-time network fetches during module import.
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_correctness, faithfulness
        from ragas.run_config import RunConfig

        from data_base.vector_store_manager import get_embeddings, initialize_embeddings

        return {
            "Dataset": Dataset,
            "evaluate": evaluate,
            "faithfulness": faithfulness,
            "answer_correctness": answer_correctness,
            "RunConfig": RunConfig,
            "LangchainLLMWrapper": LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
            "initialize_embeddings": initialize_embeddings,
            "get_embeddings": get_embeddings,
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
                "ground_truth": [row.ground_truth for row in batch_rows],
            }
        )

        async def run_batch() -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._evaluate_sync(
                    dataset=dataset,
                    ragas_dependencies=ragas_dependencies,
                    evaluator_llm=evaluator_llm,
                    evaluator_embeddings=evaluator_embeddings,
                ),
            )

        try:
            result = await run_with_retry(run_batch)
            faithfulness_scores = result["faithfulness"]
            correctness_scores = result["answer_correctness"]
            if not isinstance(faithfulness_scores, list):
                faithfulness_scores = [faithfulness_scores]
            if not isinstance(correctness_scores, list):
                correctness_scores = [correctness_scores]
        except Exception as exc:  # noqa: BLE001
            faithfulness_scores = [0.0] * len(batch_rows)
            correctness_scores = [0.0] * len(batch_rows)
            return self._score_rows_from_batch(
                batch_rows,
                faithfulness_scores,
                correctness_scores,
                error=str(exc),
            )

        return self._score_rows_from_batch(
            batch_rows,
            faithfulness_scores,
            correctness_scores,
        )

    def _evaluate_sync(
        self,
        *,
        dataset: Any,
        ragas_dependencies: dict[str, Any],
        evaluator_llm: Any,
        evaluator_embeddings: Any,
    ) -> Any:
        faithfulness_metric = ragas_dependencies["faithfulness"]
        answer_correctness_metric = ragas_dependencies["answer_correctness"]
        faithfulness_metric.llm = evaluator_llm
        answer_correctness_metric.llm = evaluator_llm
        answer_correctness_metric.embeddings = evaluator_embeddings

        return ragas_dependencies["evaluate"](
            dataset=dataset,
            metrics=[faithfulness_metric, answer_correctness_metric],
            llm=evaluator_llm,
            raise_exceptions=True,
            run_config=ragas_dependencies["RunConfig"](timeout=360),
        )

    def _score_rows_from_batch(
        self,
        batch_rows: list[Any],
        faithfulness_scores: list[Any],
        correctness_scores: list[Any],
        *,
        error: str | None = None,
    ) -> list[dict[str, Any]]:
        score_rows: list[dict[str, Any]] = []
        for index, row in enumerate(batch_rows):
            faithfulness_score = _clean_metric(
                faithfulness_scores[index] if index < len(faithfulness_scores) else 0.0
            )
            correctness_score = _clean_metric(
                correctness_scores[index] if index < len(correctness_scores) else 0.0
            )
            details = {
                "evaluator_model": self._evaluator_model,
                "question_id": row.question_id,
            }
            if error:
                details["error"] = error
            score_rows.append(
                {
                    "campaign_result_id": row.id,
                    "metric_name": "faithfulness",
                    "metric_value": faithfulness_score,
                    "details": details,
                }
            )
            score_rows.append(
                {
                    "campaign_result_id": row.id,
                    "metric_name": "answer_correctness",
                    "metric_value": correctness_score,
                    "details": details,
                }
            )
        return score_rows

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
            summary = ModeMetricsSummary(
                mode=mode,
                sample_count=len(mode_rows),
                faithfulness=_metric_aggregate([row.faithfulness for row in mode_rows]),
                answer_correctness=_metric_aggregate(
                    [row.answer_correctness for row in mode_rows]
                ),
                total_tokens=_metric_aggregate([float(row.total_tokens) for row in mode_rows]),
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
            summary.delta_answer_correctness = summary.answer_correctness.mean - naive_correctness
            summary.delta_total_tokens = summary.total_tokens.mean - naive_tokens
            if summary.delta_total_tokens <= 0:
                summary.ecr = None
                summary.ecr_note = "non_positive_marginal_cost"
                continue
            summary.ecr = 1000 * summary.delta_answer_correctness / summary.delta_total_tokens
            summary.ecr_note = None

        return summaries
