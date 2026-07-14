"""Backend tests for evaluation phase 2 campaigns."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from google.api_core import exceptions as google_exceptions
from langchain_core.documents import Document
import pytest

from core.auth import get_current_user_id
from evaluation.agentic_evaluation_service import AGENTIC_EVAL_PROFILE
from evaluation.campaign_engine import CampaignEngine
from evaluation.campaign_schemas import (
    CampaignConfig,
    CampaignLifecycleStatus,
    CampaignMetricsResponse,
    CampaignResultStatus,
    MetricAggregate,
    ModeMetricsSummary,
)
from evaluation.db import CampaignRepository, CampaignResultRepository
from evaluation.job_store import EvaluationJobStore
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult, CONTEXT_POLICY_VERSION, run_campaign_case
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.retry import run_with_retry
from evaluation.schemas import ModelConfig, TestCase
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


def _model_preset() -> ModelConfig:
    return ModelConfig(
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
    )


def _campaign_config_for_test_case_ids(
    test_case_ids: list[str],
    *,
    modes: list[str],
    repeat_count: int = 1,
    batch_size: int = 1,
    rpm_limit: int = 60,
) -> CampaignConfig:
    return CampaignConfig(
        test_case_ids=test_case_ids,
        modes=modes,
        model_preset=_model_preset(),
        model_config_id="cfg-1",
        repeat_count=repeat_count,
        batch_size=batch_size,
        rpm_limit=rpm_limit,
    )


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


async def _wait_for_terminal_campaign(
    repository: CampaignRepository,
    *,
    user_id: str,
    campaign_id: str,
    timeout_seconds: float = 3.0,
) -> tuple[CampaignLifecycleStatus, object]:
    deadline = time.time() + timeout_seconds
    latest = await repository.get(user_id=user_id, campaign_id=campaign_id)
    while time.time() < deadline:
        latest = await repository.get(user_id=user_id, campaign_id=campaign_id)
        if latest.status in {
            CampaignLifecycleStatus.COMPLETED,
            CampaignLifecycleStatus.FAILED,
            CampaignLifecycleStatus.CANCELLED,
        }:
            return latest.status, latest
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Campaign {campaign_id} did not reach terminal state in time: {latest.status}"
    )


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


def test_campaign_run_persists_snapshot_and_minimal_root_span() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            ground_truth_short=test_case.ground_truth_short,
            key_points=list(test_case.key_points),
            ragas_focus=list(test_case.ragas_focus),
            mode=kwargs["mode"],
            answer="snapshot answer",
            contexts=["ctx-1"],
            source_doc_ids=["doc-1"],
            expected_sources=list(test_case.source_docs),
            latency_ms=42.5,
            token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            category=test_case.category,
            difficulty=test_case.difficulty,
            context_policy_version=CONTEXT_POLICY_VERSION,
        )

    fake_ragas = FakeRagasEvaluator()
    engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-SNAPSHOT",
                "question": "Which evidence supports the answer?",
                "ground_truth": "Grounded answer",
                "ground_truth_short": "Grounded",
                "key_points": ["point-1"],
                "ragas_focus": ["faithfulness"],
                "category": "research",
                "difficulty": "very_hard",
                "source_docs": ["paper-a.pdf"],
                "requires_multi_doc_reasoning": False,
                "question_version": "v2.0.0",
                "required_modalities": ["text", "table"],
                "atomic_facts": [{"atomic_fact_id": "F1", "fact_text": "Fact"}],
                "expected_evidence": [{"evidence_id": "E1", "doc_id": "paper-a.pdf"}],
            },
        )
        created = client.post(
            "/api/evaluation/campaigns",
            json=_campaign_payload(
                name="Snapshot",
                test_case_ids=["Q-SNAPSHOT"],
                modes=["naive"],
            ),
        )
        campaign_id = created.json()["campaign_id"]

        terminal = _wait_for_terminal_status(client, campaign_id)
        assert terminal["status"] == "completed"

        results_response = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        result = results_response.json()["results"][0]
        assert result["question_version"] == "v2.0.0"
        assert result["request_id"]
        assert result["started_at"]
        assert result["completed_at"]
        assert result["latency_ms"] == 42.5
        assert result["total_latency_ms"] >= 0
        assert result["total_tokens"] == 15
        assert result["question_snapshot"]["required_modalities"] == ["text", "table"]
        assert result["question_snapshot"]["expected_evidence"][0]["evidence_id"] == "E1"
        assert result["model_config_snapshot"]["model_name"] == "gemini-2.5-flash"
        assert result["system_version_snapshot"]["context_policy_version"] == CONTEXT_POLICY_VERSION
        assert result["final_answer_hash"]

        observability = client.get(
            f"/api/evaluation/campaigns/{campaign_id}/runs/{result['id']}/observability"
        )
        assert observability.status_code == 200
        trace_events = observability.json()["trace_events"]
        assert [event["status"] for event in trace_events] == ["running", "success"]
        assert {event["stage_name"] for event in trace_events} == {"campaign_unit_execution"}
        llm_calls = observability.json()["llm_calls"]
        assert len(llm_calls) == 1
        assert llm_calls[0]["purpose"] == "campaign_generation"
        assert llm_calls[0]["model_name"] == "gemini-2.5-flash"
        assert llm_calls[0]["prompt_tokens"] == 10
        assert llm_calls[0]["completion_tokens"] == 5
        assert llm_calls[0]["total_tokens"] == 15
        assert llm_calls[0]["span_id"] == trace_events[0]["span_id"]

        wrong_run = client.get(
            f"/api/evaluation/campaigns/{campaign_id}/runs/not-this-campaign-run/observability"
        )
        assert wrong_run.status_code == 404


def test_campaign_rejects_router_mode_without_feature_flag() -> None:
    fake_ragas = FakeRagasEvaluator()
    engine = CampaignEngine(ragas_evaluator=fake_ragas)
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        _create_test_case(client, "Q-ROUTER")
        response = client.post(
            "/api/evaluation/campaigns",
            json=_campaign_payload(
                name="Router guard",
                test_case_ids=["Q-ROUTER"],
                modes=["router"],
            ),
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "router mode is not implemented yet; use retrospective router analysis."
    )


@pytest.mark.asyncio
async def test_durable_campaign_creation_enqueues_units_without_process_local_task() -> None:
    worker = Mock()
    store = AsyncMock(spec=EvaluationJobStore)
    store.create_job_with_items = AsyncMock()
    engine = CampaignEngine(job_store=store, worker_notifier=worker.notify)
    config = _campaign_config_for_test_case_ids(["Q1", "Q2"], modes=["naive"], repeat_count=2)
    resolved_cases = [
        TestCase(
            id=question_id,
            question="Question",
            ground_truth="Answer",
            source_docs=[],
            requires_multi_doc_reasoning=False,
        )
        for question_id in config.test_case_ids
    ]
    with patch.object(engine, "_resolve_test_cases", new=AsyncMock(return_value=resolved_cases)):
        response = await engine.create_and_start(user_id="user-a", name="durable", config=config)

    assert response.status == CampaignLifecycleStatus.PENDING
    assert len(store.create_job_with_items.await_args.kwargs["items"]) == 4
    worker.notify.assert_called_once_with()


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


def test_cancel_campaign_without_active_task_marks_cancelled() -> None:
    engine = CampaignEngine(ragas_evaluator=FakeRagasEvaluator())
    upload_root = _make_upload_root()
    db_path = _make_db_path()

    with _build_client("user-a", upload_root, db_path, engine) as client:
        campaign_repo = CampaignRepository()
        created = asyncio.run(
            campaign_repo.create(
                user_id="user-a",
                name="No task cancel",
                config=_campaign_config_for_test_case_ids(["Q1"], modes=["naive"]),
            )
        )
        cancelled = client.post(f"/api/evaluation/campaigns/{created.id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_recover_inflight_running_campaign_resumes_remaining_units() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer=f"resumed-{test_case.id}",
            contexts=["ctx-1"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=7,
            token_usage={"total_tokens": 33},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    db_path = _make_db_path()
    fake_ragas = FakeRagasEvaluator(progress_total=2)
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Recover execution",
            config=_campaign_config_for_test_case_ids(["Q1", "Q2"], modes=["naive"]),
        )
        await result_repo.create(
            user_id="user-a",
            campaign_id=campaign.id,
            question_id="Q1",
            question="What is the answer?",
            ground_truth="42",
            ground_truth_short="short 42",
            key_points=["point-1"],
            ragas_focus=["answer_correctness"],
            mode="naive",
            execution_profile=None,
            context_policy_version=None,
            run_number=1,
            answer="seeded-Q1",
            contexts=["ctx-q1"],
            source_doc_ids=["doc-q1"],
            expected_sources=[],
            latency_ms=5,
            token_usage={"total_tokens": 21},
            category="smoke",
            difficulty="easy",
            status=CampaignResultStatus.COMPLETED,
            error_message=None,
        )
        await campaign_repo.mark_running(user_id="user-a", campaign_id=campaign.id)
        await campaign_repo.update_progress(
            user_id="user-a",
            campaign_id=campaign.id,
            completed_units=1,
            current_question_id="Q1",
            current_mode="naive",
        )

        engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)
        seeded_cases = [
            TestCase.model_validate(
                {
                    "id": "Q1",
                    "question": "What is the answer?",
                    "ground_truth": "42",
                    "ground_truth_short": "short 42",
                    "key_points": ["point-1"],
                    "ragas_focus": ["answer_correctness"],
                    "category": "smoke",
                    "difficulty": "easy",
                    "source_docs": [],
                    "requires_multi_doc_reasoning": False,
                }
            ),
            TestCase.model_validate(
                {
                    "id": "Q2",
                    "question": "What is the answer?",
                    "ground_truth": "42",
                    "ground_truth_short": "short 42",
                    "key_points": ["point-1"],
                    "ragas_focus": ["answer_correctness"],
                    "category": "smoke",
                    "difficulty": "easy",
                    "source_docs": [],
                    "requires_multi_doc_reasoning": False,
                }
            ),
        ]
        with patch.object(engine, "_resolve_test_cases", new=AsyncMock(return_value=seeded_cases)):
            await engine.recover_inflight_campaigns()
            status, latest = await _wait_for_terminal_campaign(
                campaign_repo,
                user_id="user-a",
                campaign_id=campaign.id,
                timeout_seconds=5.0,
            )
            assert status == CampaignLifecycleStatus.COMPLETED
            assert latest.completed_units == 2
            assert latest.evaluation_total_units == 2
            results = await result_repo.list_for_campaign(
                user_id="user-a", campaign_id=campaign.id
            )
            keys = {(row.question_id, row.mode, row.run_number) for row in results}
            assert len(keys) == 2


@pytest.mark.asyncio
async def test_completed_run_persists_snapshots_and_root_observability_span() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        await asyncio.sleep(0.01)
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            ground_truth_short=test_case.ground_truth_short,
            key_points=list(test_case.key_points),
            ragas_focus=list(test_case.ragas_focus),
            mode=kwargs["mode"],
            answer="Snapshot answer",
            contexts=["ctx-1", "ctx-2"],
            source_doc_ids=["doc-a"],
            expected_sources=list(test_case.source_docs),
            latency_ms=18,
            token_usage={"total_tokens": 77, "input_tokens": 55, "output_tokens": 22},
            category=test_case.category,
            difficulty=test_case.difficulty,
            execution_profile="snapshot-profile",
            context_policy_version="ctx-policy-v1",
        )

    db_path = _make_db_path()
    fake_ragas = FakeRagasEvaluator(progress_total=1)
    rich_case = TestCase.model_validate(
        {
            "id": "Q-SNAPSHOT",
            "question": "What changed in the system?",
            "ground_truth": "The system persisted snapshots.",
            "ground_truth_short": "Snapshots persisted",
            "key_points": ["point-1", "point-2"],
            "ragas_focus": ["answer_correctness", "faithfulness"],
            "category": "regression",
            "difficulty": "medium",
            "question_version": "v2.1.0",
            "required_modalities": ["text", "table"],
            "atomic_facts": [{"atomic_fact_id": "Q-SNAPSHOT-F1", "text": "fact"}],
            "expected_evidence": [{"evidence_id": "Q-SNAPSHOT-E1", "doc_id": "doc-a"}],
            "source_docs": ["doc-a", "doc-b"],
            "requires_multi_doc_reasoning": True,
        }
    )

    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        observability_repo = EvaluationObservabilityRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Snapshot persistence",
            config=_campaign_config_for_test_case_ids(["Q-SNAPSHOT"], modes=["naive"]),
        )
        engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)

        await engine._run_campaign(
            user_id="user-a",
            campaign_id=campaign.id,
            config=campaign.config,
            test_cases=[rich_case],
        )

        latest = await campaign_repo.get(user_id="user-a", campaign_id=campaign.id)
        assert latest.status == CampaignLifecycleStatus.COMPLETED
        assert latest.phase == "evaluation"

        results = await result_repo.list_for_campaign(user_id="user-a", campaign_id=campaign.id)
        assert len(results) == 1
        result = results[0]
        assert result.status == CampaignResultStatus.COMPLETED
        assert result.question_version == "v2.1.0"
        assert result.request_id
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at
        assert result.total_latency_ms is not None
        assert result.total_latency_ms >= 0
        assert result.total_tokens == 77
        assert result.question_snapshot == {
            "id": "Q-SNAPSHOT",
            "question": "What changed in the system?",
            "ground_truth": "The system persisted snapshots.",
            "ground_truth_short": "Snapshots persisted",
            "key_points": ["point-1", "point-2"],
            "ragas_focus": ["answer_correctness", "faithfulness"],
            "category": "regression",
            "difficulty": "medium",
            "question_version": "v2.1.0",
            "required_modalities": ["text", "table"],
            "atomic_facts": [{"atomic_fact_id": "Q-SNAPSHOT-F1", "text": "fact"}],
            "expected_evidence": [{"evidence_id": "Q-SNAPSHOT-E1", "doc_id": "doc-a"}],
            "source_docs": ["doc-a", "doc-b"],
        }
        assert result.model_config_snapshot == campaign.config.model_preset.model_dump(mode="json")
        assert result.system_version_snapshot["execution_profile"] == "snapshot-profile"
        assert result.system_version_snapshot["context_policy_version"] == "ctx-policy-v1"
        assert isinstance(result.derived_metrics, dict)
        assert result.final_answer_hash

        trace_events = await observability_repo.list_trace_events_for_run(result.id)
        assert [event.status for event in trace_events] == ["running", "success"]
        assert all(event.stage_name == "campaign_unit_execution" for event in trace_events)
        assert all(event.parent_event_id is None for event in trace_events)
        assert all(event.parent_span_id is None for event in trace_events)
        assert trace_events[0].payload["request_id"] == result.request_id
        assert trace_events[1].duration_ms == pytest.approx(result.total_latency_ms, rel=0.2, abs=20)

        llm_calls = await observability_repo.list_llm_calls_for_run(result.id)
        assert len(llm_calls) == 1
        assert llm_calls[0].purpose == "campaign_generation"
        assert llm_calls[0].model_name == "gemini-2.5-flash"
        assert llm_calls[0].prompt_tokens == 55
        assert llm_calls[0].completion_tokens == 22
        assert llm_calls[0].total_tokens == 77
        assert llm_calls[0].span_id == trace_events[0].span_id


@pytest.mark.asyncio
async def test_agentic_trace_persists_routing_and_tool_observability() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="agentic answer",
            contexts=["ctx"],
            source_doc_ids=["doc-a"],
            expected_sources=["doc-a"],
            latency_ms=9,
            token_usage={"total_tokens": 12},
            category=test_case.category,
            difficulty=test_case.difficulty,
            execution_profile="agentic-eval",
            agent_trace={
                "classifier_decision": {
                    "router_version": "semantic-v1",
                    "router_type": "semantic_gate",
                    "selected_strategy_tier": "tier_3",
                    "complexity_score": 4,
                    "modality_score": 2,
                    "multi_doc_score": 3,
                    "conflict_score": 1,
                    "exact_value_score": 0,
                    "hallucination_risk_score": 2,
                    "retrieval_uncertainty_score": 1,
                    "routing_reason": "needs visual verification",
                    "routing_features": {"has_figure": True},
                },
                "strategy_tier": "tier_3",
                "route_profile": "visual_verify",
                "steps": [
                    {
                        "step_id": "s1",
                        "phase": "execution",
                        "step_type": "visual",
                        "title": "Verify figure",
                        "status": "completed",
                        "tool_calls": [
                            {
                                "tool_name": "visual_verifier",
                                "tool_type": "visual",
                                "action": "VERIFY_IMAGE",
                                "status": "completed",
                                "subtask_id": "1",
                                "input_summary": {"image": "figure-1"},
                                "output_summary": {"finding": "supported"},
                            }
                        ],
                    }
                ],
            },
        )

    db_path = _make_db_path()
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        observability_repo = EvaluationObservabilityRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Routing observability",
            config=_campaign_config_for_test_case_ids(["Q-ROUTE"], modes=["agentic"]),
        )
        engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
        await engine._run_campaign(
            user_id="user-a",
            campaign_id=campaign.id,
            config=campaign.config,
            test_cases=[
                TestCase(
                    id="Q-ROUTE",
                    question="What does the figure show?",
                    ground_truth="Evidence",
                    category="visual",
                    difficulty="hard",
                )
            ],
        )

        result = (await result_repo.list_for_campaign(user_id="user-a", campaign_id=campaign.id))[0]
        trace_events = await observability_repo.list_trace_events_for_run(result.id)
        routing_events = [event for event in trace_events if event.stage_type == "routing"]
        assert [event.status for event in routing_events] == ["running", "success"]

        decisions = await observability_repo.list_routing_decisions_for_run(result.id)
        assert len(decisions) == 1
        assert decisions[0].selected_mode == "agentic"
        assert decisions[0].payload["router_version"] == "semantic-v1"
        assert decisions[0].payload["selected_strategy_tier"] == "tier_3"
        assert decisions[0].payload["routing_features"] == {"has_figure": True}
        assert decisions[0].payload["actual_router_execution_enabled"] is False

        tool_calls = await observability_repo.list_tool_calls_for_run(result.id)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "visual_verifier"
        assert tool_calls[0].action == "VERIFY_IMAGE"
        assert tool_calls[0].payload["tool_type"] == "visual"
        assert tool_calls[0].payload["subtask_id"] == "1"
        assert tool_calls[0].payload["input_summary"] == {"image": "figure-1"}


@pytest.mark.asyncio
async def test_campaign_result_records_retrieval_context_and_evidence_flow() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="Fact A is supported by paper A.",
            contexts=["Fact A appears in paper A.", "Distractor text"],
            source_doc_ids=["paper-a.pdf", "paper-b.pdf"],
            expected_sources=["paper-a.pdf"],
            latency_ms=11,
            token_usage={"total_tokens": 20},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    db_path = _make_db_path()
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        observability_repo = EvaluationObservabilityRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Retrieval observability",
            config=_campaign_config_for_test_case_ids(["Q-EVIDENCE"], modes=["naive"]),
        )
        engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
        await engine._run_campaign(
            user_id="user-a",
            campaign_id=campaign.id,
            config=campaign.config,
            test_cases=[
                TestCase(
                    id="Q-EVIDENCE",
                    question="Where is Fact A?",
                    ground_truth="paper A",
                    category="evidence",
                    difficulty="medium",
                    atomic_facts=[{"atomic_fact_id": "F1", "text": "Fact A"}],
                    expected_evidence=[
                        {"evidence_id": "E1", "doc_id": "paper-a.pdf", "atomic_fact_id": "F1"}
                    ],
                    source_docs=["paper-a.pdf"],
                )
            ],
        )

        result = (await result_repo.list_for_campaign(user_id="user-a", campaign_id=campaign.id))[0]
        retrieval_events = await observability_repo.list_retrieval_events_for_run(result.id)
        assert len(retrieval_events) == 1
        assert retrieval_events[0].query == "Where is Fact A?"
        assert retrieval_events[0].result_count == 2
        assert retrieval_events[0].payload["instrumentation_depth"] == "result_level"
        assert retrieval_events[0].payload["expected_evidence_hit_rate"] == 1.0

        chunks = await observability_repo.list_retrieval_chunks_for_run(result.id)
        assert len(chunks) == 2
        assert chunks[0].doc_id == "paper-a.pdf"
        assert chunks[0].rank_before_rerank == 1
        assert chunks[0].rank_after_rerank == 1
        assert chunks[0].used_in_context is True
        assert chunks[0].used_in_answer is True
        assert chunks[0].expected_evidence_match is True
        assert chunks[1].expected_evidence_match is False

        context_packs = await observability_repo.list_context_packs_for_run(result.id)
        assert len(context_packs) == 1
        assert context_packs[0].input_chunk_count == 2
        assert context_packs[0].packed_chunk_count == 2
        assert context_packs[0].payload["selected_chunk_ids"] == [chunk.chunk_id for chunk in chunks]
        assert context_packs[0].payload["packing_policy"] == "result_level_contexts"
        assert context_packs[0].retrieved_but_not_packed_evidence == []
        assert result.derived_metrics["gold_fact_attrition"][0]["retrieved"] is True
        assert result.derived_metrics["gold_fact_attrition"][0]["packed"] is True


@pytest.mark.asyncio
async def test_campaign_result_persists_claim_rows_and_derived_claim_metrics() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="One supported claim. One weak claim.",
            contexts=["evidence"],
            source_doc_ids=["doc-a"],
            expected_sources=["doc-a"],
            latency_ms=7,
            token_usage={"total_tokens": 9},
            category=test_case.category,
            difficulty=test_case.difficulty,
            agent_trace={
                "claims": [
                    {
                        "claim_text": "Supported claim",
                        "claim_type": "answer",
                        "support_status": "supported",
                        "support_score": 0.9,
                        "evidence": [{"chunk_id": "doc-a:1"}],
                    },
                    {
                        "claim_text": "Weak claim",
                        "claim_type": "answer",
                        "support_status": "unsupported",
                        "unsupported_reason": "No evidence found",
                    },
                ]
            },
        )

    db_path = _make_db_path()
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        observability_repo = EvaluationObservabilityRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Claim observability",
            config=_campaign_config_for_test_case_ids(["Q-CLAIMS"], modes=["agentic"]),
        )
        engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
        await engine._run_campaign(
            user_id="user-a",
            campaign_id=campaign.id,
            config=campaign.config,
            test_cases=[
                TestCase(
                    id="Q-CLAIMS",
                    question="Which claims are supported?",
                    ground_truth="One supported claim",
                    category="claims",
                    difficulty="medium",
                )
            ],
        )

        result = (await result_repo.list_for_campaign(user_id="user-a", campaign_id=campaign.id))[0]
        claims = await observability_repo.list_claims_for_run(result.id)
        assert [claim.support_status for claim in claims] == ["supported", "unsupported"]
        assert claims[0].evidence == [{"chunk_id": "doc-a:1"}]
        assert claims[0].payload["support_score"] == 0.9
        assert claims[1].unsupported_reason == "No evidence found"
        assert result.derived_metrics["supported_claim_ratio"] == pytest.approx(0.5)
        assert result.derived_metrics["unsupported_claim_ratio"] == pytest.approx(0.5)
        assert result.derived_metrics["repair_count"] == 0


@pytest.mark.asyncio
async def test_campaign_failure_cancels_and_drains_pending_batch_tasks() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        if test_case.id == "Q-SLOW":
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="fast",
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=1,
            token_usage={"total_tokens": 1},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    db_path = _make_db_path()
    fake_ragas = FakeRagasEvaluator(progress_total=2)
    cases = [
        TestCase(
            id="Q-FAST",
            question="Fast?",
            ground_truth="yes",
            category="stress",
            difficulty="easy",
        ),
        TestCase(
            id="Q-SLOW",
            question="Slow?",
            ground_truth="yes",
            category="stress",
            difficulty="easy",
        ),
    ]

    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Failure drains batch",
            config=_campaign_config_for_test_case_ids(
                ["Q-FAST", "Q-SLOW"],
                modes=["naive"],
                batch_size=2,
            ),
        )
        engine = CampaignEngine(runner=runner, ragas_evaluator=fake_ragas)
        with patch.object(
            engine._campaign_repository,
            "update_progress",
            side_effect=RuntimeError("progress write failed"),
        ):
            await engine._run_campaign(
                user_id="user-a",
                campaign_id=campaign.id,
                config=campaign.config,
                test_cases=cases,
            )

        assert started.is_set()
        assert cancelled.is_set()
        latest = await campaign_repo.get(user_id="user-a", campaign_id=campaign.id)
        assert latest.status == CampaignLifecycleStatus.FAILED


@pytest.mark.asyncio
async def test_recover_inflight_evaluating_campaign_reruns_full_ragas() -> None:
    db_path = _make_db_path()
    fake_ragas = FakeRagasEvaluator(progress_total=1)
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Recover evaluating",
            config=_campaign_config_for_test_case_ids(["Q1"], modes=["naive"]),
        )
        seeded_result = await result_repo.create(
            user_id="user-a",
            campaign_id=campaign.id,
            question_id="Q1",
            question="What is the answer?",
            ground_truth="42",
            ground_truth_short="short 42",
            key_points=["point-1"],
            ragas_focus=["answer_correctness"],
            mode="naive",
            execution_profile=None,
            context_policy_version=None,
            run_number=1,
            answer="seeded-Q1",
            contexts=["ctx-q1"],
            source_doc_ids=["doc-q1"],
            expected_sources=[],
            latency_ms=5,
            token_usage={"total_tokens": 21},
            category="smoke",
            difficulty="easy",
            status=CampaignResultStatus.COMPLETED,
            error_message=None,
        )
        await campaign_repo.mark_evaluating(
            user_id="user-a",
            campaign_id=campaign.id,
            evaluation_total_units=1,
        )

        engine = CampaignEngine(ragas_evaluator=fake_ragas)
        await engine.recover_inflight_campaigns()
        status, _latest = await _wait_for_terminal_campaign(
            campaign_repo,
            user_id="user-a",
            campaign_id=campaign.id,
            timeout_seconds=5.0,
        )
        assert status == CampaignLifecycleStatus.COMPLETED
        assert fake_ragas.selected_result_ids_calls
        assert fake_ragas.selected_result_ids_calls[-1] == [seeded_result.id]


@pytest.mark.asyncio
async def test_recover_inflight_cancel_requested_campaign_marks_cancelled() -> None:
    db_path = _make_db_path()
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Recover cancel",
            config=_campaign_config_for_test_case_ids(["Q1"], modes=["naive"]),
        )
        await campaign_repo.request_cancel(user_id="user-a", campaign_id=campaign.id)

        engine = CampaignEngine(ragas_evaluator=FakeRagasEvaluator())
        await engine.recover_inflight_campaigns()
        latest = await campaign_repo.get(user_id="user-a", campaign_id=campaign.id)
        assert latest.status == CampaignLifecycleStatus.CANCELLED


@pytest.mark.asyncio
async def test_recover_inflight_missing_test_cases_marks_failed() -> None:
    db_path = _make_db_path()
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Recover missing",
            config=_campaign_config_for_test_case_ids(["Q404"], modes=["naive"]),
        )
        await campaign_repo.mark_running(user_id="user-a", campaign_id=campaign.id)

        engine = CampaignEngine(ragas_evaluator=FakeRagasEvaluator())
        await engine.recover_inflight_campaigns()
        latest = await campaign_repo.get(user_id="user-a", campaign_id=campaign.id)
        assert latest.status == CampaignLifecycleStatus.FAILED
        assert latest.error_message is not None
        assert "Unknown test case ids" in latest.error_message


@pytest.mark.asyncio
async def test_campaign_result_create_is_idempotent_on_reinsert() -> None:
    db_path = _make_db_path()
    with patch("evaluation.db.EVALUATION_DB_PATH", db_path):
        campaign_repo = CampaignRepository()
        result_repo = CampaignResultRepository()
        campaign = await campaign_repo.create(
            user_id="user-a",
            name="Idempotent result",
            config=_campaign_config_for_test_case_ids(["Q1"], modes=["naive"]),
        )
        first = await result_repo.create(
            user_id="user-a",
            campaign_id=campaign.id,
            question_id="Q1",
            question="What is the answer?",
            ground_truth="42",
            ground_truth_short="short 42",
            key_points=["point-1"],
            ragas_focus=["answer_correctness"],
            mode="naive",
            execution_profile=None,
            context_policy_version=None,
            run_number=1,
            answer="first-answer",
            contexts=["ctx-q1"],
            source_doc_ids=["doc-q1"],
            expected_sources=[],
            latency_ms=5,
            token_usage={"total_tokens": 21},
            category="smoke",
            difficulty="easy",
            status=CampaignResultStatus.COMPLETED,
            error_message=None,
        )
        second = await result_repo.create(
            user_id="user-a",
            campaign_id=campaign.id,
            question_id="Q1",
            question="What is the answer?",
            ground_truth="42",
            ground_truth_short="short 42",
            key_points=["point-1"],
            ragas_focus=["answer_correctness"],
            mode="naive",
            execution_profile=None,
            context_policy_version=None,
            run_number=1,
            answer="second-answer-ignored",
            contexts=["ctx-q1"],
            source_doc_ids=["doc-q1"],
            expected_sources=[],
            latency_ms=5,
            token_usage={"total_tokens": 21},
            category="smoke",
            difficulty="easy",
            status=CampaignResultStatus.COMPLETED,
            error_message=None,
        )

        assert first.id == second.id
        results = await result_repo.list_for_campaign(user_id="user-a", campaign_id=campaign.id)
        assert len(results) == 1


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
            execution_profile=AGENTIC_EVAL_PROFILE,
            context_policy_version=CONTEXT_POLICY_VERSION,
            agent_trace={
                "trace_id": f"trace-{run_number}",
                "question_id": test_case.id,
                "question": test_case.question,
                "mode": "agentic",
                "execution_profile": AGENTIC_EVAL_PROFILE,
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
        assert result_rows[0]["execution_profile"] == AGENTIC_EVAL_PROFILE
        result_id = result_rows[0]["id"]

        traces = client.get(f"/api/evaluation/campaigns/{campaign_id}/traces")
        assert traces.status_code == 200
        trace_rows = traces.json()
        assert len(trace_rows) == 1
        assert trace_rows[0]["campaign_result_id"] == result_id
        assert trace_rows[0]["execution_profile"] == AGENTIC_EVAL_PROFILE
        assert trace_rows[0]["tool_call_count"] == 1

        detail = client.get(f"/api/evaluation/campaigns/{campaign_id}/results/{result_id}/trace")
        assert detail.status_code == 200
        trace_detail = detail.json()
        assert trace_detail["summary"] == "trace summary"
        assert trace_detail["execution_profile"] == AGENTIC_EVAL_PROFILE
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
            "execution_profile": AGENTIC_EVAL_PROFILE,
            "context_policy_version": CONTEXT_POLICY_VERSION,
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
            assert agentic_row["execution_profile"] == AGENTIC_EVAL_PROFILE
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
            assert trace_rows[0]["execution_profile"] == AGENTIC_EVAL_PROFILE

            trace_detail = client.get(
                f"/api/evaluation/campaigns/{campaign_id}/results/{agentic_row['id']}/trace"
            )
            assert trace_detail.status_code == 200
            assert trace_detail.json()["execution_profile"] == AGENTIC_EVAL_PROFILE

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
            "execution_profile": AGENTIC_EVAL_PROFILE,
            "context_policy_version": CONTEXT_POLICY_VERSION,
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






def test_campaign_runtime_model_config_preserves_thinking_level_for_level_models() -> None:
    from evaluation.rag_modes import _runtime_overrides

    overrides = _runtime_overrides(
        {
            "model_name": "gemini-3.0-flash",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
            "thinking_mode": True,
            "thinking_budget": 8192,
            "thinking_level": "high",
        }
    )

    assert overrides["thinking_level"] == "high"
    assert "thinking_budget" not in overrides


def test_campaign_runtime_model_config_preserves_thinking_budget_for_budget_models() -> None:
    from evaluation.rag_modes import _runtime_overrides

    overrides = _runtime_overrides(
        {
            "model_name": "gemini-2.5-flash",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
            "thinking_mode": True,
            "thinking_budget": 2048,
            "thinking_level": "high",
        }
    )

    assert overrides["thinking_budget"] == 2048
    assert "thinking_level" not in overrides
