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
from evaluation.ragas_evaluator import RagasEvaluator
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


def _campaign_status(*, modes: list[str] | None = None) -> CampaignStatus:
    return CampaignStatus(
        id="cmp-1",
        name="Metrics",
        status=CampaignLifecycleStatus.COMPLETED,
        phase="evaluation",
        config=CampaignConfig(
            test_case_ids=["Q1"],
            modes=modes or ["naive", "advanced"],
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


def _fixture_json(filename: str) -> dict:
    return json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))


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
async def test_ragas_evaluator_matches_bergen_fixture_summary():
    full_results = _fixture_json("full_results.json")
    evaluation_results = _fixture_json("evaluation_results.json")

    results: list[CampaignResult] = []
    scores: list[dict] = []

    detail_by_mode_question = {
        mode: {
            detail["question"]: detail
            for detail in payload["details"]
        }
        for mode, payload in evaluation_results["results_by_mode"].items()
    }

    for index, raw in enumerate(full_results["results"], start=1):
        result_id = f"fixture-{index}"
        results.append(
            CampaignResult(
                id=result_id,
                campaign_id="cmp-1",
                question_id=raw["question_id"],
                question=raw["question"],
                ground_truth=raw["ground_truth"],
                mode=raw["mode"],
                run_number=1,
                answer=raw["answer"],
                contexts=raw["contexts"],
                source_doc_ids=raw["source_doc_ids"],
                expected_sources=raw["expected_sources"],
                category=raw.get("category"),
                difficulty=raw.get("difficulty"),
                latency_ms=float(raw["latency_ms"]),
                token_usage={
                    "input_tokens": raw["token_usage"].get("input_tokens", 0),
                    "output_tokens": raw["token_usage"].get("output_tokens", 0),
                    "total_tokens": raw["token_usage"].get("total_tokens", 0),
                },
                status=CampaignResultStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
            )
        )
        detail = detail_by_mode_question[raw["mode"]][raw["question"]]
        scores.extend(
            [
                {
                    "campaign_result_id": result_id,
                    "metric_name": "faithfulness",
                    "metric_value": detail["faithfulness"],
                    "details": {"evaluator_model": evaluation_results["evaluator_model"]},
                },
                {
                    "campaign_result_id": result_id,
                    "metric_name": "answer_correctness",
                    "metric_value": detail["answer_correctness"],
                    "details": {"evaluator_model": evaluation_results["evaluator_model"]},
                },
            ]
        )

    evaluator = RagasEvaluator(
        result_repository=FakeResultRepository(results),
        score_repository=FakeScoreRepository(scores),
        evaluator_model=evaluation_results["evaluator_model"],
    )

    response = await evaluator.get_metrics(
        user_id="user-a",
        campaign=_campaign_status(modes=list(evaluation_results["results_by_mode"].keys())),
    )

    for mode, expected in evaluation_results["results_by_mode"].items():
        summary = response.summary_by_mode[mode]
        expected_faithfulness = sum(
            detail["faithfulness"] for detail in expected["details"]
        ) / len(expected["details"])
        expected_correctness = sum(
            detail["answer_correctness"] for detail in expected["details"]
        ) / len(expected["details"])
        assert summary.sample_count == expected["num_questions"]
        assert summary.faithfulness.mean == pytest.approx(expected_faithfulness, abs=1e-4)
        assert summary.answer_correctness.mean == pytest.approx(
            expected_correctness,
            abs=1e-4,
        )


class _FakeDataset:
    @staticmethod
    def from_dict(payload: dict) -> dict:
        return payload


@pytest.mark.asyncio
async def test_evaluate_batch_falls_back_to_zero_scores_on_ragas_failure():
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
            ground_truth="Ground truth 1",
        )
    ]

    with patch("evaluation.ragas_evaluator.run_with_retry", new=AsyncMock(side_effect=RuntimeError("ragas down"))):
        score_rows = await evaluator._evaluate_batch(
            batch_rows=batch_rows,
            ragas_dependencies={
                "Dataset": _FakeDataset,
            },
            evaluator_llm=object(),
            evaluator_embeddings=object(),
        )

    assert len(score_rows) == 2
    assert all(row["metric_value"] == 0.0 for row in score_rows)
    assert all(row["details"]["evaluator_model"] == "fake-evaluator" for row in score_rows)
    assert all("ragas down" in row["details"]["error"] for row in score_rows)


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
                    "metric_name": "faithfulness",
                    "metric_value": 0.0,
                    "details": {
                        "evaluator_model": "fake-evaluator",
                        "question_id": "Q1",
                        "error": "ragas down",
                    },
                },
                {
                    "campaign_result_id": "r1",
                    "metric_name": "answer_correctness",
                    "metric_value": 0.0,
                    "details": {
                        "evaluator_model": "fake-evaluator",
                        "question_id": "Q1",
                        "error": "ragas down",
                    },
                },
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
    assert len(score_repository._scores) == 2
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
