from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evaluation.campaign_schemas import (
    CampaignConfig,
    CampaignLifecycleStatus,
    CampaignResult,
    CampaignResultStatus,
    CampaignStatus,
)
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.schemas import ModelConfig


class FakeResultRepository:
    def __init__(self, results: list[CampaignResult]) -> None:
        self._results = results

    async def list_for_campaign(self, *, user_id: str, campaign_id: str) -> list[CampaignResult]:
        return list(self._results)


class FakeScoreRepository:
    def __init__(self, scores: list[dict]) -> None:
        self._scores = scores

    async def list_for_campaign(self, *, user_id: str, campaign_id: str) -> list[dict]:
        return list(self._scores)

    async def replace_for_campaign(self, *, user_id: str, campaign_id: str, score_rows: list[dict]) -> None:
        self._scores = list(score_rows)


def _campaign_status() -> CampaignStatus:
    return CampaignStatus(
        id="cmp-1",
        name="Metrics",
        status=CampaignLifecycleStatus.COMPLETED,
        phase="evaluation",
        config=CampaignConfig(
            test_case_ids=["Q1"],
            modes=["naive", "advanced"],
            model_preset=ModelConfig(
                id="cfg-1",
                name="Balanced",
                model_name="gemini-2.5-flash",
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_input_tokens=8192,
                max_output_tokens=2048,
                thinking_mode=False,
                thinking_budget=8192,
            ),
            repeat_count=1,
            batch_size=1,
            rpm_limit=60,
        ),
        completed_units=2,
        total_units=2,
        evaluation_completed_units=2,
        evaluation_total_units=2,
        cancel_requested=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _result(result_id: str, mode: str, total_tokens: int, *, status: CampaignResultStatus = CampaignResultStatus.COMPLETED) -> CampaignResult:
    return CampaignResult(
        id=result_id,
        campaign_id="cmp-1",
        question_id=f"{mode}-{result_id}",
        question="Question",
        ground_truth="Ground truth",
        mode=mode,
        run_number=1,
        answer="Answer",
        contexts=["ctx"],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=10,
        token_usage={"total_tokens": total_tokens},
        status=status,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_ragas_evaluator_aggregates_marginal_ecr_and_skips_failed_rows():
    results = [
        _result("r1", "naive", 100),
        _result("r2", "advanced", 160),
        _result("r3", "graph", 90),
        _result("r4", "agentic", 200, status=CampaignResultStatus.FAILED),
    ]
    scores = [
        {"campaign_result_id": "r1", "metric_name": "faithfulness", "metric_value": 0.4, "details": {"evaluator_model": "fake"}},
        {"campaign_result_id": "r1", "metric_name": "answer_correctness", "metric_value": 0.5, "details": {"evaluator_model": "fake"}},
        {"campaign_result_id": "r2", "metric_name": "faithfulness", "metric_value": 0.8, "details": {"evaluator_model": "fake"}},
        {"campaign_result_id": "r2", "metric_name": "answer_correctness", "metric_value": 0.9, "details": {"evaluator_model": "fake"}},
        {"campaign_result_id": "r3", "metric_name": "faithfulness", "metric_value": 0.7, "details": {"evaluator_model": "fake"}},
        {"campaign_result_id": "r3", "metric_name": "answer_correctness", "metric_value": 0.75, "details": {"evaluator_model": "fake"}},
    ]

    evaluator = RagasEvaluator(
        result_repository=FakeResultRepository(results),
        score_repository=FakeScoreRepository(scores),
        evaluator_model="fake",
    )

    response = await evaluator.get_metrics(user_id="user-a", campaign=_campaign_status())

    assert response.evaluator_model == "fake"
    assert len(response.rows) == 3
    assert "agentic" not in response.summary_by_mode
    assert response.summary_by_mode["naive"].ecr == 0
    assert response.summary_by_mode["advanced"].delta_total_tokens == 60
    assert response.summary_by_mode["advanced"].ecr == pytest.approx(6.6666666667)
    assert response.summary_by_mode["graph"].ecr is None
    assert response.summary_by_mode["graph"].ecr_note == "non_positive_marginal_cost"
