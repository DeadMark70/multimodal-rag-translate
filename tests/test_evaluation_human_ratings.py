from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation import db as evaluation_db
from evaluation.campaign_engine import CampaignEngine
from evaluation.rag_modes import BenchmarkExecutionResult
from main import app


class FakeRagasEvaluator:
    async def evaluate_campaign(self, *, on_progress=None, **kwargs) -> str:
        if on_progress:
            await on_progress(1, 1, "Q-HUMAN", "agentic")
        return "fake-ragas"


@contextmanager
def _build_client(user_id: str, upload_root: Path, db_path: Path, engine: CampaignEngine):
    process_worker = Mock(is_configured=False)
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.storage.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("evaluation.db.EVALUATION_DB_PATH", db_path),
        patch("evaluation.campaign_engine.get_campaign_engine", return_value=engine),
        patch("evaluation.job_worker.get_evaluation_job_worker", return_value=process_worker),
        patch("evaluation.router.get_campaign_engine", return_value=engine),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: user_id
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def _wait_for_completed(client: TestClient, campaign_id: str) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        response = client.get("/api/evaluation/campaigns")
        assert response.status_code == 200
        current = next(item for item in response.json() if item["id"] == campaign_id)
        if current["status"] == "completed":
            return
        time.sleep(0.05)
    raise AssertionError(f"campaign {campaign_id} did not complete")


async def _insert_ragas_scores(*, db_path: Path, campaign_id: str, run_id: str, user_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with patch.object(evaluation_db, "EVALUATION_DB_PATH", db_path):
        await evaluation_db.init_db()
        async with evaluation_db.connect_db() as connection:
            await connection.execute(
                """
                INSERT INTO ragas_scores (
                    id, campaign_id, campaign_result_id, user_id, metric_name, metric_value, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{run_id}-correctness",
                    campaign_id,
                    run_id,
                    user_id,
                    "answer_correctness",
                    0.8,
                    json.dumps({"evaluator_model": "fake-ragas"}),
                    now,
                ),
            )
            await connection.execute(
                """
                INSERT INTO ragas_scores (
                    id, campaign_id, campaign_result_id, user_id, metric_name, metric_value, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{run_id}-faithfulness",
                    campaign_id,
                    run_id,
                    user_id,
                    "faithfulness",
                    0.6,
                    json.dumps({"evaluator_model": "fake-ragas"}),
                    now,
                ),
            )
            await connection.commit()


def _campaign_payload() -> dict:
    return {
        "name": "Human calibration",
        "test_case_ids": ["Q-HUMAN"],
        "modes": ["agentic"],
        "model_config": {
            "id": "cfg-1",
            "name": "Balanced",
            "model_name": "gemini-2.5-flash",
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "thinking_mode": False,
            "thinking_budget": 8192,
        },
        "repeat_count": 1,
        "batch_size": 1,
        "rpm_limit": 60,
    }


def _make_workspace_paths(prefix: str) -> tuple[Path, Path]:
    root = Path.cwd() / "output" / "test_tmp" / f"{prefix}_{uuid4().hex}"
    return root / "uploads", root / "evaluation.db"


def test_human_eval_queue_and_calibration_handle_empty_and_insufficient_samples() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="A grounded answer",
            contexts=["ctx-1"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 18},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("human")

    with _build_client("user-a", upload_root, db_path, engine) as client:
        created_case = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-HUMAN",
                "question": "What supports the answer?",
                "ground_truth": "Evidence-backed answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200

        created = client.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client, campaign_id)

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results").json()["results"]
        run_id = results[0]["id"]
        asyncio.run(_insert_ragas_scores(db_path=db_path, campaign_id=campaign_id, run_id=run_id, user_id="user-a"))

        queue_response = client.get(f"/api/evaluation/campaigns/{campaign_id}/human-eval-queue")
        assert queue_response.status_code == 200
        queue_body = queue_response.json()
        assert queue_body["campaign_id"] == campaign_id
        assert queue_body["rows"][0]["run_id"] == run_id
        assert queue_body["rows"][0]["repeat_number"] == 1
        assert queue_body["rows"][0]["already_rated_by_current_user"] is False

        empty_calibration = client.get(f"/api/evaluation/campaigns/{campaign_id}/human-vs-auto")
        assert empty_calibration.status_code == 200
        assert empty_calibration.json()["sample_count"] == 0
        assert empty_calibration.json()["rows"] == []

        create_rating = client.post(
            f"/api/evaluation/runs/{run_id}/human-ratings",
            json={
                "rubric_version": "v1",
                "correctness_score": 0.9,
                "faithfulness_score": 0.8,
                "completeness_score": 0.75,
                "citation_quality_score": 0.85,
                "usefulness_score": 0.8,
                "comments": "grounded",
            },
        )
        assert create_rating.status_code == 200
        rating_body = create_rating.json()
        assert rating_body["run_id"] == run_id
        assert rating_body["rater_id_hash"] != "user-a"
        assert rating_body["is_blinded"] is True
        assert rating_body["shown_mode_label"] is False

        queue_after_rating = client.get(f"/api/evaluation/campaigns/{campaign_id}/human-eval-queue")
        assert queue_after_rating.status_code == 200
        assert queue_after_rating.json()["rows"][0]["already_rated_by_current_user"] is True

        calibration = client.get(f"/api/evaluation/campaigns/{campaign_id}/human-vs-auto")
        assert calibration.status_code == 200
        calibration_body = calibration.json()
        assert calibration_body["sample_count"] == 1
        assert calibration_body["summaries"]["human_correctness_mean"] == 0.9
        assert calibration_body["summaries"]["human_faithfulness_mean"] == 0.8
        assert calibration_body["summaries"]["ragas_human_pearson_r"] is None
        assert calibration_body["summaries"]["ragas_human_spearman_r"] is None


def test_user_cannot_submit_human_rating_to_another_users_run() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="A grounded answer",
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 12},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("human")

    with _build_client("user-a", upload_root, db_path, engine) as client_a:
        created_case = client_a.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-HUMAN",
                "question": "What supports the answer?",
                "ground_truth": "Evidence-backed answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        created = client_a.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client_a, campaign_id)
        run_id = client_a.get(f"/api/evaluation/campaigns/{campaign_id}/results").json()["results"][0]["id"]

    with _build_client("user-b", upload_root, db_path, engine) as client_b:
        denied = client_b.post(
            f"/api/evaluation/runs/{run_id}/human-ratings",
            json={
                "rubric_version": "v1",
                "correctness_score": 0.9,
                "faithfulness_score": 0.8,
                "completeness_score": 0.75,
                "citation_quality_score": 0.85,
                "usefulness_score": 0.8,
            },
        )
        assert denied.status_code == 404
