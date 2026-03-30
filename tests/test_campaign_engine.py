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
from langchain_core.documents import Document
import pytest

from core.auth import get_current_user_id
from evaluation.campaign_engine import CampaignEngine
from evaluation.campaign_schemas import CampaignMetricsResponse, MetricAggregate, ModeMetricsSummary
from evaluation.rag_modes import BenchmarkExecutionResult, run_campaign_case
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.retry import run_with_retry
from data_base.RAG_QA_service import RAGResult
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
            "ground_truth_short": "short 42",
            "key_points": ["point-1"],
            "ragas_focus": ["answer_correctness"],
            "category": "smoke",
            "difficulty": "easy",
            "source_docs": [],
            "requires_multi_doc_reasoning": False,
        },
    )
    assert response.status_code == 200


def _campaign_payload(
    *,
    name: str,
    test_case_ids: list[str],
    modes: list[str],
    repeat_count: int = 1,
    batch_size: int = 1,
    rpm_limit: int = 60,
) -> dict:
    return {
        "name": name,
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
        "batch_size": batch_size,
        "rpm_limit": rpm_limit,
    }


def _fake_ragas_dependencies() -> dict:
    return {
        "LangchainLLMWrapper": lambda llm: llm,
        "initialize_embeddings": AsyncMock(),
        "LangchainEmbeddingsWrapper": lambda embeddings: embeddings,
        "get_embeddings": lambda: object(),
        "aevaluate": AsyncMock(),
    }


class FakeRagasEvaluator:
    def __init__(self, progress_total: int = 1) -> None:
        self.evaluate_calls: list[str] = []
        self.selected_result_ids_calls: list[list[str] | None] = []
        self.progress_total = progress_total

    async def evaluate_campaign(
        self,
        *,
        user_id: str,
        campaign_id: str,
        ragas_batch_size: int | None = None,
        ragas_parallel_batches: int | None = None,
        ragas_rpm_limit: int | None = None,
        selected_result_ids: list[str] | None = None,
        on_progress=None,
    ) -> str:
        self.evaluate_calls.append(f"{user_id}:{campaign_id}")
        self.selected_result_ids_calls.append(selected_result_ids)
        if on_progress:
            await on_progress(self.progress_total, self.progress_total, "Q1", "naive")
        return "fake-evaluator"

    async def get_metrics(self, *, user_id: str, campaign):
        return CampaignMetricsResponse(
            campaign=campaign,
            evaluator_model="fake-evaluator",
            available_metrics=["faithfulness", "answer_correctness", "answer_relevancy"],
            summary_by_mode={
                "naive": ModeMetricsSummary(
                    mode="naive",
                    sample_count=1,
                    metric_summaries={
                        "faithfulness": MetricAggregate(mean=0.5, max=0.5, stddev=0),
                        "answer_correctness": MetricAggregate(mean=0.5, max=0.5, stddev=0),
                        "answer_relevancy": MetricAggregate(mean=0.45, max=0.45, stddev=0),
                    },
                    faithfulness=MetricAggregate(mean=0.5, max=0.5, stddev=0),
                    answer_correctness=MetricAggregate(mean=0.5, max=0.5, stddev=0),
                    total_tokens=MetricAggregate(mean=123, max=123, stddev=0),
                    delta_answer_correctness=0,
                    delta_total_tokens=0,
                    ecr=0,
                    ecr_note=None,
                )
            },
            summary_by_category={},
            summary_by_focus={},
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


@pytest.mark.asyncio
async def test_run_with_retry_retries_resource_exhausted() -> None:
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise google_exceptions.ResourceExhausted("rate limited")
        return "ok"

    result = await run_with_retry(flaky)
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


def test_campaign_manual_evaluate_can_rerun_selected_questions_only() -> None:
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

    fake_ragas = FakeRagasEvaluator(progress_total=2)
    engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client, "Q1")
        _create_test_case(client, "Q2")
        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Manual evaluate subset",
                "test_case_ids": ["Q1", "Q2"],
                "modes": ["naive", "advanced"],
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
                "batch_size": 2,
                "rpm_limit": 60,
            },
        )
        campaign_id = created.json()["campaign_id"]
        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"

        results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        assert results.status_code == 200
        q2_result_ids = [
            row["id"]
            for row in results.json()["results"]
            if row["status"] == "completed" and row["question_id"] == "Q2"
        ]
        assert len(q2_result_ids) == 2

        rerun = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/evaluate",
            json={"question_ids": ["Q2"]},
        )
        assert rerun.status_code == 200
        assert rerun.json()["status"] == "evaluating"
        assert rerun.json()["evaluation_total_units"] == 2

        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"
        assert sorted(fake_ragas.selected_result_ids_calls[-1] or []) == sorted(
            q2_result_ids
        )


def test_campaign_manual_evaluate_selected_questions_without_completed_rows_returns_400() -> None:
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

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client, "Q1")
        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Manual evaluate subset invalid",
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

        rerun = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/evaluate",
            json={"question_ids": ["Q404"]},
        )
        assert rerun.status_code == 400
        assert (
            rerun.json()["error"]["message"]
            == "Requested question_ids have no completed raw results in this campaign"
        )


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
            execution_profile="agentic_eval_v4",
            agent_trace={
                "trace_id": f"trace-{run_number}",
                "question_id": test_case.id,
                "question": test_case.question,
                "mode": "agentic",
                "execution_profile": "agentic_eval_v4",
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
        assert result_rows[0]["execution_profile"] == "agentic_eval_v4"
        result_id = result_rows[0]["id"]

        traces = client.get(f"/api/evaluation/campaigns/{campaign_id}/traces")
        assert traces.status_code == 200
        trace_rows = traces.json()
        assert len(trace_rows) == 1
        assert trace_rows[0]["campaign_result_id"] == result_id
        assert trace_rows[0]["execution_profile"] == "agentic_eval_v4"
        assert trace_rows[0]["tool_call_count"] == 1

        detail = client.get(f"/api/evaluation/campaigns/{campaign_id}/results/{result_id}/trace")
        assert detail.status_code == 200
        trace_detail = detail.json()
        assert trace_detail["summary"] == "trace summary"
        assert trace_detail["execution_profile"] == "agentic_eval_v4"
        assert trace_detail["steps"][0]["phase"] == "planning"
        assert trace_detail["steps"][1]["tool_calls"][0]["action"] == "VERIFY_IMAGE"


def test_campaign_integration_uses_real_runner_and_real_ragas_persistence() -> None:
    ragas = RagasEvaluator(batch_size=2)
    engine = CampaignEngine(runner=run_campaign_case, ragas_evaluator=ragas)
    upload_root = _make_upload_root()
    db_path = _make_db_path()
    rag_calls: list[dict] = []

    async def fake_rag_answer_question(*, question: str, user_id: str, return_docs: bool, **kwargs) -> RAGResult:
        assert user_id == "user-a"
        assert return_docs is True
        if kwargs.get("enable_graph_rag"):
            mode = "graph"
        elif kwargs.get("enable_reranking"):
            mode = "advanced"
        else:
            mode = "naive"
        rag_calls.append({"question": question, "mode": mode, **kwargs})
        return RAGResult(
            answer=f"{mode} answer",
            source_doc_ids=[f"{mode}-doc"],
            documents=[Document(page_content=f"{mode} context", metadata={"doc_id": f"{mode}-doc"})],
            usage={"total_tokens": {"naive": 20, "advanced": 40, "graph": 60}[mode]},
        )

    async def fake_evaluate_batch(*, batch_rows, **_kwargs):
        score_rows: list[dict] = []
        for row in batch_rows:
            faithfulness = {
                "naive": 0.45,
                "advanced": 0.7,
                "graph": 0.8,
                "agentic": 0.9,
            }[row.mode]
            correctness = {
                "naive": 0.5,
                "advanced": 0.72,
                "graph": 0.83,
                "agentic": 0.91,
            }[row.mode]
            for metric_name, metric_value in (
                ("faithfulness", faithfulness),
                ("answer_correctness", correctness),
            ):
                score_rows.append(
                    {
                        "campaign_result_id": row.id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "details": {
                            "evaluator_model": "fake-ragas",
                            "question_id": row.question_id,
                        },
                    }
                )
        return score_rows

    agentic_result = RAGResult(
        answer="agentic answer",
        source_doc_ids=["agentic-doc"],
        documents=[Document(page_content="agentic context", metadata={"doc_id": "agentic-doc"})],
        usage={"total_tokens": 120},
        thought_process="trace summary",
        tool_calls=[],
        agent_trace={
            "trace_id": "trace-1",
            "question_id": "Q1",
            "question": "What is the answer?",
            "mode": "agentic",
            "execution_profile": "agentic_eval_v4",
            "run_number": 1,
            "trace_status": "completed",
            "summary": "trace summary",
            "step_count": 1,
            "tool_call_count": 0,
            "total_tokens": 120,
            "created_at": "2026-03-21T00:00:00+00:00",
            "steps": [],
        },
    )

    with (
        patch("evaluation.rag_modes.rag_answer_question", new=AsyncMock(side_effect=fake_rag_answer_question)),
        patch("evaluation.rag_modes.AgenticEvaluationService") as mock_service_cls,
        patch.object(ragas, "_load_ragas_dependencies", new=AsyncMock(return_value=_fake_ragas_dependencies())),
        patch.object(ragas, "_evaluate_batch", new=AsyncMock(side_effect=fake_evaluate_batch)),
    ):
        mock_service = mock_service_cls.return_value
        mock_service.run_case = AsyncMock(return_value=agentic_result)

        with _build_client("user-a", upload_root, db_path, engine) as client:
            _create_test_case(client)
            created = client.post(
                "/api/evaluation/campaigns",
                json=_campaign_payload(
                    name="Real runner integration",
                    test_case_ids=["Q1"],
                    modes=["naive", "advanced", "graph", "agentic"],
                    batch_size=2,
                ),
            )
            assert created.status_code == 200
            campaign_id = created.json()["campaign_id"]

            terminal = _wait_for_terminal_status(client, campaign_id, timeout_seconds=8.0)
            assert terminal["status"] == "completed"
            assert terminal["phase"] == "evaluation"
            assert terminal["completed_units"] == 4
            assert terminal["total_units"] == 4
            assert terminal["evaluation_completed_units"] == 4
            assert terminal["evaluation_total_units"] == 4

            results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
            assert results.status_code == 200
            result_rows = results.json()["results"]
            assert len(result_rows) == 4
            assert {row["mode"] for row in result_rows} == {"naive", "advanced", "graph", "agentic"}
            assert all(row["status"] == "completed" for row in result_rows)
            agentic_row = next(row for row in result_rows if row["mode"] == "agentic")
            assert agentic_row["execution_profile"] == "agentic_eval_v4"
            assert agentic_row["has_trace"] is True

            metrics = client.get(f"/api/evaluation/campaigns/{campaign_id}/metrics")
            assert metrics.status_code == 200
            metrics_body = metrics.json()
            assert metrics_body["evaluator_model"] == "fake-ragas"
            assert set(metrics_body["summary_by_mode"].keys()) == {"naive", "advanced", "graph", "agentic"}
            assert len(metrics_body["rows"]) == 4

            traces = client.get(f"/api/evaluation/campaigns/{campaign_id}/traces")
            assert traces.status_code == 200
            trace_rows = traces.json()
            assert len(trace_rows) == 1
            assert trace_rows[0]["execution_profile"] == "agentic_eval_v4"

            trace_detail = client.get(
                f"/api/evaluation/campaigns/{campaign_id}/results/{agentic_row['id']}/trace"
            )
            assert trace_detail.status_code == 200
            assert trace_detail.json()["execution_profile"] == "agentic_eval_v4"

    assert len(rag_calls) == 3
    graph_call = next(call for call in rag_calls if call["mode"] == "graph")
    assert graph_call["enable_graph_rag"] is True
    assert graph_call["graph_search_mode"] == "generic"


def test_campaign_integration_keeps_running_when_one_mode_fails() -> None:
    ragas = RagasEvaluator(batch_size=2)
    engine = CampaignEngine(runner=run_campaign_case, ragas_evaluator=ragas)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    async def flaky_rag_answer_question(*, return_docs: bool, **kwargs) -> RAGResult:
        assert return_docs is True
        if kwargs.get("enable_graph_rag"):
            raise RuntimeError("graph retrieval blew up")
        mode = "advanced" if kwargs.get("enable_reranking") else "naive"
        return RAGResult(
            answer=f"{mode} answer",
            source_doc_ids=[f"{mode}-doc"],
            documents=[Document(page_content=f"{mode} context", metadata={"doc_id": f"{mode}-doc"})],
            usage={"total_tokens": {"naive": 20, "advanced": 40}[mode]},
        )

    async def fake_evaluate_batch(*, batch_rows, **_kwargs):
        score_rows: list[dict] = []
        for row in batch_rows:
            for metric_name, metric_value in (
                ("faithfulness", 0.6),
                ("answer_correctness", 0.7),
            ):
                score_rows.append(
                    {
                        "campaign_result_id": row.id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "details": {
                            "evaluator_model": "fake-ragas",
                            "question_id": row.question_id,
                        },
                    }
                )
        return score_rows

    agentic_result = RAGResult(
        answer="agentic answer",
        source_doc_ids=["agentic-doc"],
        documents=[Document(page_content="agentic context", metadata={"doc_id": "agentic-doc"})],
        usage={"total_tokens": 120},
        agent_trace={
            "trace_id": "trace-2",
            "question_id": "Q1",
            "question": "What is the answer?",
            "mode": "agentic",
            "execution_profile": "agentic_eval_v4",
            "run_number": 1,
            "trace_status": "completed",
            "summary": "trace summary",
            "step_count": 1,
            "tool_call_count": 0,
            "total_tokens": 120,
            "created_at": "2026-03-21T00:00:00+00:00",
            "steps": [],
        },
    )

    with (
        patch("evaluation.rag_modes.rag_answer_question", new=AsyncMock(side_effect=flaky_rag_answer_question)),
        patch("evaluation.rag_modes.AgenticEvaluationService") as mock_service_cls,
        patch.object(ragas, "_load_ragas_dependencies", new=AsyncMock(return_value=_fake_ragas_dependencies())),
        patch.object(ragas, "_evaluate_batch", new=AsyncMock(side_effect=fake_evaluate_batch)),
    ):
        mock_service = mock_service_cls.return_value
        mock_service.run_case = AsyncMock(return_value=agentic_result)

        with _build_client("user-a", upload_root, db_path, engine) as client:
            _create_test_case(client)
            created = client.post(
                "/api/evaluation/campaigns",
                json=_campaign_payload(
                    name="Partial failure integration",
                    test_case_ids=["Q1"],
                    modes=["naive", "graph", "agentic"],
                    batch_size=2,
                ),
            )
            assert created.status_code == 200
            campaign_id = created.json()["campaign_id"]

            terminal = _wait_for_terminal_status(client, campaign_id, timeout_seconds=8.0)
            assert terminal["status"] == "completed"
            assert terminal["completed_units"] == 3
            assert terminal["total_units"] == 3
            assert terminal["evaluation_completed_units"] == 2
            assert terminal["evaluation_total_units"] == 2

            results = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
            assert results.status_code == 200
            result_rows = results.json()["results"]
            assert len(result_rows) == 3
            failed_row = next(row for row in result_rows if row["mode"] == "graph")
            assert failed_row["status"] == "failed"
            assert "graph retrieval blew up" in failed_row["error_message"]

            metrics = client.get(f"/api/evaluation/campaigns/{campaign_id}/metrics")
            assert metrics.status_code == 200
            metrics_body = metrics.json()
            assert set(metrics_body["summary_by_mode"].keys()) == {"naive", "agentic"}
            assert len(metrics_body["rows"]) == 2



