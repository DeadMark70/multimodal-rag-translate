"""API tests for evaluation phase 1 endpoints."""

from __future__ import annotations

import json
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation import db as evaluation_db
from evaluation.campaign_schemas import CampaignResultStatus
from evaluation.campaign_schemas import CampaignLifecycleStatus
from evaluation.campaign_engine import CampaignEngine
from evaluation.db import CampaignResultRepository
from evaluation.job_schemas import EvaluationJob, EvaluationJobType, EvaluationWorkType
from evaluation.job_store import EvaluationJobStore
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.schemas import AvailableModel
from evaluation.trace_schemas import EvaluationTraceEvent
from main import app


@contextmanager
def _build_client(user_id: str, upload_root: Path, with_auth: bool = True):
    """Build test client with auth override and startup stubs."""
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.storage.BASE_UPLOAD_FOLDER", str(upload_root)),
    ):
        if with_auth:
            app.dependency_overrides[get_current_user_id] = lambda: user_id
        with TestClient(app) as client:
            yield client
    app.dependency_overrides = {}


def _load_golden_dataset() -> dict:
    dataset_path = Path(__file__).resolve().parents[1] / "bergen" / "golden_dataset.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _make_upload_root() -> Path:
    root = Path("output") / "test_tmp" / f"evaluation_api_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


async def _seed_campaign(campaign_id: str, user_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await evaluation_db.init_db()
    async with evaluation_db.connect_db() as connection:
        await connection.execute(
            """
            INSERT INTO campaigns (
                id, user_id, name, status, phase, config_json, completed_units, total_units,
                evaluation_completed_units, evaluation_total_units, current_question_id,
                current_mode, error_message, cancel_requested, created_at, started_at,
                completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
            """,
            (
                campaign_id,
                user_id,
                "Observability API test",
                "completed",
                "evaluation",
                json.dumps(
                    {
                        "test_case_ids": ["TC-1"],
                        "modes": ["advanced"],
                        "model_config": {
                            "id": "preset",
                            "name": "Preset",
                            "model_name": "gemini-2.5-flash",
                            "temperature": 0.2,
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_input_tokens": 8192,
                            "max_output_tokens": 2048,
                            "thinking_mode": False,
                            "thinking_budget": None,
                            "thinking_level": None,
                            "thinking_include_thoughts": False,
                        },
                        "repeat_count": 1,
                        "batch_size": 1,
                        "rpm_limit": 60,
                        "ragas_batch_size": 8,
                        "ragas_parallel_batches": 8,
                        "ragas_rpm_limit": 1000,
                    },
                    ensure_ascii=False,
                ),
                now,
                now,
            ),
        )
        await connection.commit()


async def _seed_trace_event(campaign_id: str, run_id: str) -> None:
    repository = EvaluationObservabilityRepository()
    now = datetime.now(timezone.utc)
    await repository.record_trace_event(
        EvaluationTraceEvent(
            event_id=f"{run_id}-event-1",
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=f"{run_id}-span-1",
            parent_event_id=None,
            parent_span_id=None,
            event_type="span",
            event_schema_version="1.0",
            sequence=1,
            stage_type="retrieval",
            stage_name="retrieve",
            started_at=now,
            ended_at=None,
            duration_ms=None,
            status="running",
            retry_count=0,
            payload={"query_hash": "hash-1"},
            error={},
            created_at=now,
        )
    )


async def _seed_campaign_result(campaign_id: str, run_id: str, user_id: str) -> None:
    await CampaignResultRepository().create(
        result_id=run_id,
        user_id=user_id,
        campaign_id=campaign_id,
        question_id="TC-1",
        question="What is the test question?",
        ground_truth="Ground truth answer",
        ground_truth_short=None,
        key_points=[],
        ragas_focus=[],
        mode="advanced",
        execution_profile=None,
        context_policy_version=None,
        run_number=1,
        answer="Answer",
        contexts=[],
        source_doc_ids=[],
        expected_sources=[],
        latency_ms=1,
        token_usage={"total_tokens": 1},
        category=None,
        difficulty=None,
        status=CampaignResultStatus.COMPLETED,
        derived_metrics={"repeat_number": 1},
    )


def test_test_case_crud_and_import() -> None:
    upload_root = _make_upload_root()
    payload = {
        "id": "TC-1",
        "question": "What is the test question?",
        "ground_truth": "Ground truth answer",
        "ground_truth_short": "Short answer",
        "key_points": ["point-1", "point-2"],
        "ragas_focus": ["answer_correctness", "faithfulness"],
        "category": "basic",
        "difficulty": "easy",
        "source_docs": ["doc-a.pdf"],
        "requires_multi_doc_reasoning": False,
    }

    with _build_client("user-a", upload_root) as client:
        response = client.get("/api/evaluation/test-cases")
        assert response.status_code == 200
        assert response.json() == []

        created = client.post("/api/evaluation/test-cases", json=payload)
        assert created.status_code == 200
        created_body = created.json()
        assert created_body["id"] == "TC-1"
        assert created_body["ground_truth_short"] == "Short answer"
        assert created_body["key_points"] == ["point-1", "point-2"]
        assert created_body["ragas_focus"] == ["answer_correctness", "faithfulness"]

        listed = client.get("/api/evaluation/test-cases")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        updated = client.put(
            "/api/evaluation/test-cases/TC-1",
            json={**payload, "question": "Updated question", "ground_truth_short": "Updated short answer"},
        )
        assert updated.status_code == 200
        assert updated.json()["question"] == "Updated question"
        assert updated.json()["ground_truth_short"] == "Updated short answer"

        deleted = client.delete("/api/evaluation/test-cases/TC-1")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_id"] == "TC-1"
        assert deleted.json()["total"] == 0

        imported = client.post(
            "/api/evaluation/test-cases",
            json=_load_golden_dataset(),
        )
        assert imported.status_code == 200
        assert imported.json()["imported"] == 8
        assert imported.json()["total"] == 8

        listed_after_import = client.get("/api/evaluation/test-cases")
        assert listed_after_import.status_code == 200
        assert len(listed_after_import.json()) == 8


def test_test_case_put_preserves_omitted_research_metadata() -> None:
    upload_root = _make_upload_root()
    payload = {
        "id": "TC-META",
        "question": "What evidence supports the result?",
        "ground_truth": "Ground truth answer",
        "question_version": "v2.0.0",
        "difficulty": "Very Hard",
        "source_docs": [],
        "requires_multi_doc_reasoning": False,
        "required_modalities": ["text", "table"],
        "atomic_facts": [
            {
                "atomic_fact_id": "TC-META-F1",
                "fact_text": "The reported value is 0.9079.",
            }
        ],
        "expected_evidence": [
            {
                "evidence_id": "TC-META-E1",
                "doc_id": "paper-a.pdf",
                "page": 5,
                "modality": "table",
            }
        ],
    }

    legacy_update_payload = {
        "id": "TC-META",
        "question": "Updated question",
        "ground_truth": "Updated ground truth answer",
        "difficulty": "hard",
        "source_docs": [],
        "requires_multi_doc_reasoning": False,
    }

    with _build_client("user-a", upload_root) as client:
        created = client.post("/api/evaluation/test-cases", json=payload)
        assert created.status_code == 200
        assert created.json()["difficulty"] == "very-hard"

        updated = client.put(
            "/api/evaluation/test-cases/TC-META",
            json=legacy_update_payload,
        )
        assert updated.status_code == 200
        updated_body = updated.json()
        assert updated_body["question"] == "Updated question"
        assert updated_body["question_version"] == "v2.0.0"
        assert updated_body["required_modalities"] == ["text", "table"]
        assert updated_body["atomic_facts"][0]["atomic_fact_id"] == "TC-META-F1"
        assert updated_body["expected_evidence"][0]["evidence_id"] == "TC-META-E1"


def test_run_observability_endpoint_returns_only_owned_campaign_rows(tmp_path) -> None:
    upload_root = _make_upload_root()
    db_path = tmp_path / "evaluation.db"

    with patch.object(evaluation_db, "EVALUATION_DB_PATH", db_path):
        asyncio.run(_seed_campaign("campaign-owned", "user-a"))
        asyncio.run(_seed_campaign("campaign-other", "user-a"))
        asyncio.run(_seed_campaign_result("campaign-owned", "run-1", "user-a"))
        asyncio.run(_seed_trace_event("campaign-owned", "run-1"))

        with _build_client("user-a", upload_root) as client:
            response = client.get("/api/evaluation/campaigns/campaign-owned/runs/run-1/observability")
            assert response.status_code == 200
            body = response.json()
            assert body["campaign_id"] == "campaign-owned"
            assert body["run_id"] == "run-1"
            assert len(body["trace_events"]) == 1
            assert body["trace_events"][0]["stage_name"] == "retrieve"

            cross_campaign = client.get("/api/evaluation/campaigns/campaign-other/runs/run-1/observability")
            assert cross_campaign.status_code == 404


def test_model_config_crud_and_validation() -> None:
    upload_root = _make_upload_root()
    payload = {
        "name": "Balanced",
        "model_name": "gemini-2.5-flash",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
        "thinking_mode": True,
        "thinking_budget": 4096,
    }

    with _build_client("user-a", upload_root) as client:
        initial = client.get("/api/evaluation/model-configs")
        assert initial.status_code == 200
        assert initial.json() == []

        created = client.post("/api/evaluation/model-configs", json=payload)
        assert created.status_code == 200
        config_id = created.json()["id"]

        listed = client.get("/api/evaluation/model-configs")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        replaced = client.put(
            f"/api/evaluation/model-configs/{config_id}",
            json={**payload, "name": "Fast", "temperature": 0.2},
        )
        assert replaced.status_code == 200
        assert replaced.json()["name"] == "Fast"
        assert replaced.json()["temperature"] == 0.2

        invalid = client.post(
            "/api/evaluation/model-configs",
            json={**payload, "name": "Invalid", "top_p": 1.5},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"

        deleted = client.delete(f"/api/evaluation/model-configs/{config_id}")
        assert deleted.status_code == 200
        assert deleted.json()["total"] == 0


def test_model_config_level_model_normalizes_thinking_fields() -> None:
    upload_root = _make_upload_root()
    payload = {
        "name": "Gemini 3 High",
        "model_name": "gemini-3.0-flash",
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
        "thinking_mode": True,
        "thinking_budget": 4096,
        "thinking_level": "high",
        "thinking_include_thoughts": False,
    }

    with _build_client("user-a", upload_root) as client:
        created = client.post("/api/evaluation/model-configs", json=payload)
        assert created.status_code == 200
        body = created.json()
        assert body["thinking_level"] == "high"
        assert body["thinking_budget"] is None

        listed = client.get("/api/evaluation/model-configs")
        assert listed.status_code == 200
        assert listed.json()[0]["thinking_level"] == "high"
        assert listed.json()[0]["thinking_budget"] is None

def test_models_endpoint_uses_dynamic_discovery() -> None:
    upload_root = _make_upload_root()
    mocked = [
        AvailableModel(
            name="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            description="Fast model",
            input_token_limit=1048576,
            output_token_limit=8192,
            supported_actions=["generateContent"],
        )
    ]

    with _build_client("user-a", upload_root) as client, patch(
        "evaluation.router.list_available_models",
        new=AsyncMock(return_value=mocked),
    ) as mocked_discovery:
        response = client.get("/api/evaluation/models?force_refresh=true")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["name"] == "gemini-2.5-flash"
        mocked_discovery.assert_awaited_once_with(force_refresh=True)


def test_models_endpoint_requires_authentication() -> None:
    upload_root = _make_upload_root()
    with _build_client("user-a", upload_root, with_auth=False) as client:
        response = client.get("/api/evaluation/models?force_refresh=true")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_models_endpoint_openapi_requires_http_bearer() -> None:
    upload_root = _make_upload_root()
    with _build_client("user-a", upload_root, with_auth=False) as client:
        openapi = client.get("/openapi.json").json()
        endpoint = openapi["paths"]["/api/evaluation/models"]["get"]
        assert endpoint["security"] == [{"HTTPBearer": []}]


def test_evaluation_data_is_isolated_by_user() -> None:
    upload_root = _make_upload_root()
    payload = {
        "id": "TC-A",
        "question": "Only user A should see this",
        "ground_truth": "A",
        "ground_truth_short": "A-short",
        "key_points": ["point-a"],
        "ragas_focus": ["faithfulness"],
        "category": "isolation",
        "difficulty": "easy",
        "source_docs": [],
        "requires_multi_doc_reasoning": False,
    }

    with _build_client("user-a", upload_root) as client_a:
        created = client_a.post("/api/evaluation/test-cases", json=payload)
        assert created.status_code == 200

    with _build_client("user-b", upload_root) as client_b:
        listed_b = client_b.get("/api/evaluation/test-cases")
        assert listed_b.status_code == 200
        assert listed_b.json() == []

    with _build_client("user-a", upload_root) as client_a_again:
        listed_a = client_a_again.get("/api/evaluation/test-cases")
        assert listed_a.status_code == 200
        assert len(listed_a.json()) == 1
        assert listed_a.json()[0]["id"] == "TC-A"
        assert listed_a.json()[0]["ground_truth_short"] == "A-short"


@pytest.mark.asyncio
async def test_rerun_route_returns_durable_job(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation import router as evaluation_router

    created = EvaluationJob(
        job_id="job-rerun",
        job_type=EvaluationJobType.RERUN,
        user_id="user-a",
        campaign_id="campaign-a",
        selection={"scope": "failed_only"},
    )
    engine = AsyncMock()
    engine.create_rerun.return_value = created
    monkeypatch.setattr(evaluation_router, "get_campaign_engine", lambda: engine)

    response = await evaluation_router.create_campaign_rerun(
        "campaign-a",
        evaluation_router.EvaluationRerunRequest(
            scope="failed_only", stages="execution_and_ragas"
        ),
        "user-a",
    )

    assert response.job_type is EvaluationJobType.RERUN
    engine.create_rerun.assert_awaited_once()


@pytest.mark.asyncio
async def test_attempt_history_route_rejects_unknown_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation import router as evaluation_router
    from core.errors import AppError, ErrorCode

    engine = AsyncMock()
    engine.list_attempts.side_effect = AppError(
        code=ErrorCode.NOT_FOUND, message="Attempt history not found", status_code=404
    )
    monkeypatch.setattr(evaluation_router, "get_campaign_engine", lambda: engine)

    with pytest.raises(AppError) as exc_info:
        await evaluation_router.get_work_item_attempts("missing", "user-b")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_job_items_route_returns_owned_item_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation import router as evaluation_router

    engine = AsyncMock()
    engine.list_job_items.return_value = [
        {
            "job_item_id": "item-1",
            "job_id": "job-1",
            "work_item_id": "work-1",
            "work_type": "dataset_execution",
            "status": "failed",
            "question_id": "Q1",
            "metric_name": None,
            "latest_attempt": None,
        }
    ]
    monkeypatch.setattr(evaluation_router, "get_campaign_engine", lambda: engine)

    response = await evaluation_router.list_evaluation_job_items("job-1", "user-a")

    assert response[0]["work_item_id"] == "work-1"
    engine.list_job_items.assert_awaited_once_with(user_id="user-a", job_id="job-1")


def test_mixed_succeeded_cancelled_job_is_completed_with_errors() -> None:
    status = EvaluationJobStore._derive_job_status(
        total=2,
        succeeded=1,
        failed=0,
        cancelled=1,
        unresolved=0,
    )
    assert status == "completed_with_errors"


@pytest.mark.asyncio
async def test_cancel_job_derives_running_campaign_lifecycle() -> None:
    campaign = type("Campaign", (), {"id": "campaign-a", "status": CampaignLifecycleStatus.RUNNING})()
    job = EvaluationJob(
        job_id="job-a",
        job_type=EvaluationJobType.RERUN,
        user_id="user-a",
        campaign_id="campaign-a",
    )
    store = AsyncMock()
    store.get_job.return_value = job
    store.get_job_work_types.return_value = [EvaluationWorkType.DATASET_EXECUTION]
    store.cancel_job.return_value = job.model_copy(update={"status": "cancelled"})
    repository = AsyncMock()
    repository.get.return_value = campaign
    engine = CampaignEngine(
        campaign_repository=repository,
        job_store=store,
        configure_worker=False,
        worker_notifier=lambda: None,
    )

    result = await engine.cancel_job(user_id="user-a", job_id="job-a")

    assert result.status == "cancelled"
    repository.derive_execution_state.assert_awaited_once_with(
        user_id="user-a", campaign_id="campaign-a"
    )
    repository.derive_ragas_state.assert_not_awaited()

