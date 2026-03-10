"""Backend tests for evaluation phase 2 campaigns."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from google.api_core import exceptions as google_exceptions

from core.auth import get_current_user_id
from evaluation.campaign_engine import CampaignEngine
from evaluation.campaign_schemas import CampaignMetricsResponse, MetricAggregate, ModeMetricsSummary
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


class FakeRagasEvaluator:
    def __init__(self, progress_total: int = 1) -> None:
        self.evaluate_calls: list[str] = []
        self.progress_total = progress_total

    async def evaluate_campaign(self, *, user_id: str, campaign_id: str, on_progress=None) -> str:
        self.evaluate_calls.append(f"{user_id}:{campaign_id}")
        if on_progress:
            await on_progress(self.progress_total, self.progress_total, "Q1", "naive")
        return "fake-evaluator"

    async def get_metrics(self, *, user_id: str, campaign):
        return CampaignMetricsResponse(
            campaign=campaign,
            evaluator_model="fake-evaluator",
            summary_by_mode={
                "naive": ModeMetricsSummary(
                    mode="naive",
                    sample_count=1,
                    faithfulness=MetricAggregate(mean=0.5, max=0.5, stddev=0),
                    answer_correctness=MetricAggregate(mean=0.5, max=0.5, stddev=0),
                    total_tokens=MetricAggregate(mean=123, max=123, stddev=0),
                    delta_answer_correctness=0,
                    delta_total_tokens=0,
                    ecr=0,
                    ecr_note=None,
                )
            },
            rows=[],
        )


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

    fake_ragas = FakeRagasEvaluator()
    engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)
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
        assert terminal["phase"] == "evaluation"
        assert terminal["completed_units"] == 1
        assert terminal["evaluation_completed_units"] == 1
        assert terminal["evaluation_total_units"] == 1

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        assert results.status_code == 200
        body = results.json()
        assert body["campaign"]["id"] == campaign_id
        assert len(body["results"]) == 1
        assert body["results"][0]["answer"] == "generated answer"

        metrics = client.get(f"/api/evaluation/campaigns/{campaign_id}/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["evaluator_model"] == "fake-evaluator"

        with client.stream("GET", f"/api/evaluation/campaigns/{campaign_id}/stream") as stream_response:
            stream_body = "".join(stream_response.iter_text())
        assert "event: campaign_snapshot" in stream_body
        assert "event: campaign_completed" in stream_body
        assert fake_ragas.evaluate_calls


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

    engine = CampaignEngine(runner=slow_runner, ragas_evaluator=FakeRagasEvaluator())
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


def test_campaign_manual_evaluate_reruns_ragas() -> None:
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
            latency_ms=10,
            token_usage={"total_tokens": 50},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    fake_ragas = FakeRagasEvaluator()
    engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client)
        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Manual evaluate",
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
        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"

        rerun = client.post(f"/api/evaluation/campaigns/{campaign_id}/evaluate")
        assert rerun.status_code == 200
        assert rerun.json()["status"] == "evaluating"

        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"
        assert len(fake_ragas.evaluate_calls) == 2


def test_campaign_sqlite_concurrent_writes_complete_without_loss() -> None:
    async def concurrent_runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        mode = kwargs["mode"]
        await asyncio.sleep(0.02)
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=mode,
            answer=f"{test_case.id}-{mode}-answer",
            contexts=[f"context-{test_case.id}-{mode}"],
            source_doc_ids=[f"doc-{test_case.id}"],
            expected_sources=list(test_case.source_docs),
            latency_ms=20,
            token_usage={"total_tokens": 10},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    test_case_ids = [f"Q{i}" for i in range(1, 5)]
    modes = ["naive", "advanced", "graph", "agentic"]
    repeat_count = 2
    expected_total = len(test_case_ids) * len(modes) * repeat_count
    engine = CampaignEngine(
        runner=concurrent_runner,
        ragas_evaluator=FakeRagasEvaluator(progress_total=expected_total),
    )
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        for test_case_id in test_case_ids:
            _create_test_case(client, test_case_id)

        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Concurrent stress",
                "test_case_ids": test_case_ids,
                "modes": modes,
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
                "repeat_count": repeat_count,
                "batch_size": 4,
                "rpm_limit": 600,
            },
        )
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]

        terminal = _wait_for_terminal_status(client, campaign_id, timeout_seconds=8.0)
        assert terminal["status"] == "completed"
        assert terminal["phase"] == "evaluation"
        assert terminal["completed_units"] == expected_total
        assert terminal["total_units"] == expected_total
        assert terminal["evaluation_completed_units"] == expected_total
        assert terminal["evaluation_total_units"] == expected_total

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        assert results.status_code == 200
        body = results.json()
        assert len(body["results"]) == expected_total
        assert all(item["status"] == "completed" for item in body["results"])

        unique_result_keys = {
            (item["question_id"], item["mode"], item["run_number"])
            for item in body["results"]
        }
        assert len(unique_result_keys) == expected_total

    with sqlite3.connect(db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode;").fetchone()
        assert journal_mode is not None
        assert str(journal_mode[0]).lower() == "wal"


def test_agent_trace_api_persists_and_reads_trace_payload() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        run_number = kwargs["run_number"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="agentic answer",
            contexts=["ctx-1", "ctx-2"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=33,
            token_usage={"total_tokens": 120},
            category=test_case.category,
            difficulty=test_case.difficulty,
            execution_profile="agentic_eval_v1",
            agent_trace={
                "trace_id": f"trace-{run_number}",
                "question_id": test_case.id,
                "question": test_case.question,
                "mode": "agentic",
                "execution_profile": "agentic_eval_v1",
                "run_number": run_number,
                "trace_status": "completed",
                "summary": "trace summary",
                "step_count": 2,
                "tool_call_count": 1,
                "total_tokens": 120,
                "created_at": "2026-03-08T00:00:00+00:00",
                "steps": [
                    {
                        "step_id": "planning-1",
                        "phase": "planning",
                        "step_type": "plan_generation",
                        "title": "Generate research plan",
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:00+00:00",
                        "completed_at": "2026-03-08T00:00:00+00:00",
                        "input_preview": test_case.question,
                        "output_preview": "2 tasks",
                        "raw_text": "1. sub task",
                        "tool_calls": [],
                        "token_usage": {"total_tokens": 20},
                        "metadata": {},
                    },
                    {
                        "step_id": "execution-2",
                        "phase": "execution",
                        "step_type": "sub_task_execution",
                        "title": "Step 1",
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:01+00:00",
                        "completed_at": "2026-03-08T00:00:01+00:00",
                        "input_preview": "sub task 1",
                        "output_preview": "answer preview",
                        "raw_text": "raw thought",
                        "tool_calls": [
                            {
                                "index": 0,
                                "action": "VERIFY_IMAGE",
                                "status": "completed",
                                "payload": {"path": "img.png"},
                                "result_preview": "verified",
                            }
                        ],
                        "token_usage": {"total_tokens": 100},
                        "metadata": {},
                    },
                ],
            },
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client)
        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Agent trace",
                "test_case_ids": ["Q1"],
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
            },
        )
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]

        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        assert results.status_code == 200
        result_rows = results.json()["results"]
        assert len(result_rows) == 1
        assert result_rows[0]["has_trace"] is True
        assert result_rows[0]["execution_profile"] == "agentic_eval_v1"
        result_id = result_rows[0]["id"]

        traces = client.get(f"/api/evaluation/campaigns/{campaign_id}/traces")
        assert traces.status_code == 200
        trace_rows = traces.json()
        assert len(trace_rows) == 1
        assert trace_rows[0]["campaign_result_id"] == result_id
        assert trace_rows[0]["execution_profile"] == "agentic_eval_v1"
        assert trace_rows[0]["tool_call_count"] == 1

        detail = client.get(f"/api/evaluation/campaigns/{campaign_id}/results/{result_id}/trace")
        assert detail.status_code == 200
        trace_detail = detail.json()
        assert trace_detail["summary"] == "trace summary"
        assert trace_detail["execution_profile"] == "agentic_eval_v1"
        assert trace_detail["steps"][0]["phase"] == "planning"
        assert trace_detail["steps"][1]["tool_calls"][0]["action"] == "VERIFY_IMAGE"
