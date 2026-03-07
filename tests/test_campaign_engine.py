"""Backend tests for evaluation phase 2 campaigns."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from google.api_core import exceptions as google_exceptions

from core.auth import get_current_user_id
from evaluation.campaign_engine import CampaignEngine
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.retry import run_with_retry
from main import app


def _make_upload_root() -> Path:
    root = Path("output") / "test_tmp" / f"campaign_api_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_db_path() -> Path:
    root = Path("output") / "test_tmp" / f"campaign_db_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root / "evaluation.db"


@contextmanager
def _build_client(user_id: str, upload_root: Path, db_path: Path, engine: CampaignEngine):
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.storage.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("evaluation.db.EVALUATION_DB_PATH", db_path),
        patch("evaluation.router.get_campaign_engine", return_value=engine),
    ):
        app.dependency_overrides[get_current_user_id] = lambda: user_id
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def _create_test_case(client: TestClient, test_case_id: str = "Q1") -> None:
    response = client.post(
        "/api/evaluation/test-cases",
        json={
            "id": test_case_id,
            "question": "What is the answer?",
            "ground_truth": "42",
            "category": "smoke",
            "difficulty": "easy",
            "source_docs": [],
            "requires_multi_doc_reasoning": False,
        },
    )
    assert response.status_code == 200


def _wait_for_terminal_status(client: TestClient, campaign_id: str, timeout_seconds: float = 3.0) -> dict:
    deadline = time.time() + timeout_seconds
    latest = {}
    while time.time() < deadline:
        response = client.get("/api/evaluation/campaigns")
        assert response.status_code == 200
        campaigns = response.json()
        latest = next(item for item in campaigns if item["id"] == campaign_id)
        if latest["status"] in {"completed", "failed", "cancelled"}:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"Campaign {campaign_id} did not reach terminal state: {latest}")


def test_campaign_api_runs_and_streams_results() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="generated answer",
            contexts=["ctx-1"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=12.5,
            token_usage={"total_tokens": 123},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client)
        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Smoke",
                "test_case_ids": ["Q1"],
                "modes": ["naive"],
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
                "model_config_id": "cfg-1",
                "repeat_count": 1,
                "batch_size": 1,
                "rpm_limit": 60,
            },
        )
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]

        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"
        assert terminal["completed_units"] == 1

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        assert results.status_code == 200
        body = results.json()
        assert body["campaign"]["id"] == campaign_id
        assert len(body["results"]) == 1
        assert body["results"][0]["answer"] == "generated answer"

        with client.stream("GET", f"/api/evaluation/campaigns/{campaign_id}/stream") as stream_response:
            stream_body = "".join(stream_response.iter_text())
        assert "event: campaign_snapshot" in stream_body
        assert "event: campaign_completed" in stream_body


def test_campaign_cancel_marks_campaign_cancelled() -> None:
    async def slow_runner(**kwargs) -> BenchmarkExecutionResult:
        await asyncio.sleep(0.5)
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="late answer",
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=10,
            token_usage={},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=slow_runner)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client)
        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Cancelable",
                "test_case_ids": ["Q1"],
                "modes": ["naive"],
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
            },
        )
        campaign_id = created.json()["campaign_id"]

        cancelled = client.post(f"/api/evaluation/campaigns/{campaign_id}/cancel")
        assert cancelled.status_code == 200

        terminal = _wait_for_terminal_status(client, campaign_id, timeout_seconds=5.0)
        assert terminal["status"] == "cancelled"


def test_run_with_retry_retries_resource_exhausted() -> None:
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise google_exceptions.ResourceExhausted("rate limited")
        return "ok"

    result = asyncio.run(run_with_retry(flaky))
    assert result == "ok"
    assert attempts["count"] == 3

