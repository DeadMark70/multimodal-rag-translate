from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from evaluation.campaign_schemas import (
    CampaignConfig,
    CampaignLifecycleStatus,
    CampaignResult,
    CampaignResultStatus,
    CampaignStatus,
)
from evaluation.ragas_evaluator import PRIMARY_RAGAS_METRICS, RagasEvaluator
from evaluation.schemas import ModelConfig

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "bergen"


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


class _FakeDataset:
    payload: dict | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> dict:
        cls.payload = payload
        return payload


def _campaign_status(*, modes: list[str] | None = None) -> CampaignStatus:
    return CampaignStatus(
        id="cmp-1",
        name="Metrics",
        status=CampaignLifecycleStatus.COMPLETED,
        phase="evaluation",
        config=CampaignConfig(
            test_case_ids=["Q1"],
            modes=modes or ["naive", "advanced", "graph"],
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
        completed_units=3,
        total_units=3,
        evaluation_completed_units=3,
        evaluation_total_units=3,
        cancel_requested=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _result(
    result_id: str,
    mode: str,
    total_tokens: int,
    *,
    category: str = "綜合比較題",
    ragas_focus: list[str] | None = None,
    ground_truth_short: str | None = None,
    status: CampaignResultStatus = CampaignResultStatus.COMPLETED,
) -> CampaignResult:
    return CampaignResult(
        id=result_id,
        campaign_id="cmp-1",
        question_id=f"{mode}-{result_id}",
        question="Question",
        ground_truth="Long ground truth",
        ground_truth_short=ground_truth_short,
        key_points=["kp-1"],
        ragas_focus=ragas_focus or ["answer_correctness"],
        mode=mode,
        run_number=1,
        answer="Answer",
        contexts=["ctx"],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=10,
        token_usage={"total_tokens": total_tokens},
        category=category,
        difficulty="medium",
        status=status,
        created_at=datetime.now(timezone.utc),
    )


def _fixture_json(filename: str) -> dict:
    return json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_ragas_evaluator_aggregates_summaries_reference_sources_and_groupings():
    results = [
        _result("r1", "naive", 100, category="綜合比較題", ragas_focus=["faithfulness"], ground_truth_short="Short GT"),
        _result("r2", "advanced", 160, category="視覺驗證題", ragas_focus=["answer_correctness"]),
        _result("r3", "graph", 90, category="視覺驗證題", ragas_focus=["answer_relevancy"]),
        _result("r4", "agentic", 200, status=CampaignResultStatus.FAILED),
    ]
    scores = [
        {"campaign_result_id": "r1", "metric_name": "faithfulness", "metric_value": 0.4, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_short"}},
        {"campaign_result_id": "r1", "metric_name": "answer_correctness", "metric_value": 0.5, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_short"}},
        {"campaign_result_id": "r1", "metric_name": "answer_relevancy", "metric_value": 0.55, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_short"}},
        {"campaign_result_id": "r2", "metric_name": "faithfulness", "metric_value": 0.8, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_fallback_long"}},
        {"campaign_result_id": "r2", "metric_name": "answer_correctness", "metric_value": 0.9, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_fallback_long"}},
        {"campaign_result_id": "r2", "metric_name": "answer_relevancy", "metric_value": 0.88, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_fallback_long"}},
        {"campaign_result_id": "r3", "metric_name": "faithfulness", "metric_value": 0.7, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_fallback_long"}},
        {"campaign_result_id": "r3", "metric_name": "answer_correctness", "metric_value": 0.75, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_fallback_long"}},
        {"campaign_result_id": "r3", "metric_name": "answer_relevancy", "metric_value": 0.77, "details": {"evaluator_model": "fake", "reference_source": "ground_truth_fallback_long"}},
    ]

    evaluator = RagasEvaluator(
        result_repository=FakeResultRepository(results),
        score_repository=FakeScoreRepository(scores),
        evaluator_model="fake",
    )

    response = await evaluator.get_metrics(user_id="user-a", campaign=_campaign_status())

    assert response.evaluator_model == "fake"
    assert response.available_metrics == PRIMARY_RAGAS_METRICS
    assert len(response.rows) == 3
    assert "agentic" not in response.summary_by_mode
    assert response.rows[0].reference_source == "ground_truth_short"
    assert response.summary_by_mode["naive"].ecr == 0
    assert response.summary_by_mode["advanced"].delta_total_tokens == 60
    assert response.summary_by_mode["advanced"].ecr == pytest.approx(6.6666666667)
    assert response.summary_by_mode["graph"].ecr is None
    assert response.summary_by_mode["graph"].ecr_note == "non_positive_marginal_cost"
    assert response.summary_by_category["視覺驗證題"].sample_count == 2
    assert response.summary_by_focus["answer_relevancy"].sample_count == 1


def test_benchmark_results_ragas_fixture_has_aligned_columns():
    fixture = _fixture_json("benchmark_results_ragas.json")

    row_count = len(fixture["questions"])
    assert row_count > 0
    assert len(fixture["answers"]) == row_count
    assert len(fixture["contexts"]) == row_count
    assert len(fixture["ground_truths"]) == row_count
    assert len(fixture["metadata"]["modes"]) == row_count
    assert len(fixture["metadata"]["question_ids"]) == row_count
    assert len(fixture["metadata"]["categories"]) == row_count


@pytest.mark.asyncio
async def test_evaluate_batch_uses_short_ground_truth_and_keeps_metric_failures_non_blocking():
    evaluator = RagasEvaluator(
        result_repository=FakeResultRepository([]),
        score_repository=FakeScoreRepository([]),
        evaluator_model="fake-evaluator",
    )
    batch_rows = [
        SimpleNamespace(
            id="r1",
            question_id="Q1",
            question="Question 1",
            answer="Answer 1",
            contexts=["ctx-1"],
            ground_truth="Long ground truth",
            ground_truth_short="Short ground truth",
        )
    ]

    async def passthrough(invocation):
        return await invocation()

    def fake_metric_sync(*, metric_name: str, **_kwargs):
        if metric_name == "answer_relevancy":
            raise RuntimeError("metric broke")
        return [0.42 if metric_name == "faithfulness" else 0.84]

    with patch("evaluation.ragas_evaluator.run_with_retry", new=passthrough), patch.object(
        evaluator,
        "_evaluate_metric_sync",
        side_effect=fake_metric_sync,
    ):
        score_rows = await evaluator._evaluate_batch(
            batch_rows=batch_rows,
            ragas_dependencies={
                "Dataset": _FakeDataset,
                "metrics": {metric_name: object() for metric_name in PRIMARY_RAGAS_METRICS},
            },
            evaluator_llm=object(),
            evaluator_embeddings=object(),
        )

    assert _FakeDataset.payload is not None
    assert _FakeDataset.payload["ground_truth"] == ["Short ground truth"]
    assert len(score_rows) == len(PRIMARY_RAGAS_METRICS)
    assert {row["metric_name"] for row in score_rows} == set(PRIMARY_RAGAS_METRICS)
    answer_relevancy_row = next(row for row in score_rows if row["metric_name"] == "answer_relevancy")
    assert answer_relevancy_row["metric_value"] == 0.0
    assert answer_relevancy_row["details"]["reference_source"] == "ground_truth_short"
    assert "metric broke" in answer_relevancy_row["details"]["error"]


@pytest.mark.asyncio
async def test_evaluate_campaign_reports_progress_when_batches_fall_back():
    result = CampaignResult(
        id="r1",
        campaign_id="cmp-1",
        question_id="Q1",
        question="Question 1",
        ground_truth="Ground truth 1",
        mode="naive",
        run_number=1,
        answer="Answer 1",
        contexts=["ctx-1"],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=10,
        token_usage={"total_tokens": 100},
        status=CampaignResultStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
    )
    score_repository = FakeScoreRepository([])
    evaluator = RagasEvaluator(
        result_repository=FakeResultRepository([result]),
        score_repository=score_repository,
        evaluator_model="fake-evaluator",
    )
    progress_updates: list[tuple[int, int, str | None, str | None]] = []

    with patch.object(
        evaluator,
        "_load_ragas_dependencies",
        new=AsyncMock(
            return_value={
                "LangchainLLMWrapper": lambda llm: llm,
                "initialize_embeddings": AsyncMock(),
                "LangchainEmbeddingsWrapper": lambda embeddings: embeddings,
                "get_embeddings": lambda: object(),
            }
        ),
    ), patch.object(
        evaluator,
        "_evaluate_batch",
        new=AsyncMock(
            return_value=[
                {
                    "campaign_result_id": "r1",
                    "metric_name": metric_name,
                    "metric_value": 0.0,
                    "details": {
                        "evaluator_model": "fake-evaluator",
                        "question_id": "Q1",
                        "reference_source": "ground_truth_fallback_long",
                        "error": "ragas down",
                    },
                }
                for metric_name in PRIMARY_RAGAS_METRICS
            ]
        ),
    ):
        model_name = await evaluator.evaluate_campaign(
            user_id="user-a",
            campaign_id="cmp-1",
            on_progress=lambda completed, total, question_id, mode: _record_progress(
                progress_updates,
                completed,
                total,
                question_id,
                mode,
            ),
        )

    assert model_name == "fake-evaluator"
    assert progress_updates == [(1, 1, "Q1", "naive")]
    assert len(score_repository._scores) == len(PRIMARY_RAGAS_METRICS)
    assert all(score["metric_value"] == 0.0 for score in score_repository._scores)
    assert all(score["details"]["error"] == "ragas down" for score in score_repository._scores)


async def _record_progress(
    sink: list[tuple[int, int, str | None, str | None]],
    completed: int,
    total: int,
    question_id: str | None,
    mode: str | None,
) -> None:
    sink.append((completed, total, question_id, mode))


@pytest.mark.asyncio
async def test_evaluate_campaign_skips_when_ragas_dependency_missing() -> None:
    result = CampaignResult(
        id="r1",
        campaign_id="cmp-1",
        question_id="Q1",
        question="Question 1",
        ground_truth="Ground truth 1",
        mode="naive",
        run_number=1,
        answer="Answer 1",
        contexts=["ctx-1"],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=10,
        token_usage={"total_tokens": 100},
        status=CampaignResultStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
    )
    score_repository = FakeScoreRepository(
        [
            {
                "campaign_result_id": "legacy",
                "metric_name": "faithfulness",
                "metric_value": 0.9,
                "details": {},
            }
        ]
    )
    evaluator = RagasEvaluator(
        result_repository=FakeResultRepository([result]),
        score_repository=score_repository,
        evaluator_model="fake-evaluator",
    )

    with patch.object(
        evaluator,
        "_load_ragas_dependencies",
        new=AsyncMock(side_effect=ModuleNotFoundError("No module named 'datasets'")),
    ):
        model_name = await evaluator.evaluate_campaign(
            user_id="user-a",
            campaign_id="cmp-1",
        )

    assert model_name == "fake-evaluator"
    assert score_repository._scores == []
