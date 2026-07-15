from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation.campaign_engine import CampaignEngine
from evaluation.observability_storage import EvaluationObservabilityRepository
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.trace_schemas import EvaluationLlmCall, EvaluationTraceEvent
from main import app


class FakeRagasEvaluator:
    async def evaluate_campaign(self, *, on_progress=None, **kwargs) -> str:
        if on_progress:
            await on_progress(1, 1, "Q-EXPORT", "agentic")
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


async def _seed_export_rows(*, run_id: str, campaign_id: str) -> None:
    repository = EvaluationObservabilityRepository()
    now = datetime.now(timezone.utc)
    await repository.record_trace_event(
        EvaluationTraceEvent(
            event_id=f"{run_id}-export-error",
            run_id=run_id,
            campaign_id=campaign_id,
            span_id=f"{run_id}-span",
            parent_event_id=None,
            parent_span_id=None,
            event_type="generation",
            event_schema_version="1.0",
            sequence=10,
            stage_type="generation",
            stage_name="answer_generation",
            started_at=now,
            ended_at=now,
            duration_ms=10,
            status="failed",
            retry_count=0,
            payload={"provider": "test"},
            error={"code": "PROVIDER_ERROR", "message": "apiKey=sk-secret exploded with stack trace"},
            created_at=now,
        )
    )
    await repository.record_llm_call(
        EvaluationLlmCall(
            llm_call_id=f"{run_id}-llm",
            run_id=run_id,
            campaign_id=campaign_id,
            purpose="campaign_generation",
            provider="google",
            model_name="gemini-2.5-flash",
            prompt_tokens=10,
            completion_tokens=6,
            total_tokens=16,
            prompt_hash="prompt-hash",
            prompt_preview="Question: preview only",
            response_hash="response-hash",
            latency_ms=12,
            status="failed",
            error={"message": "apiKey=sk-secret exploded with stack trace"},
            payload={"full_prompt": "SECRET FULL PROMPT", "other_field": "kept"},
            created_at=now,
        )
    )


def _campaign_payload() -> dict:
    return {
        "name": "Export",
        "test_case_ids": ["Q-EXPORT"],
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


def test_export_defaults_redact_full_prompts_and_errors_are_sanitized() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="Grounded answer",
            contexts=["SECRET RETRIEVED CONTEXT"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 16},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export")

    with _build_client("user-a", upload_root, db_path, engine) as client:
        created_case = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-EXPORT",
                "question": "What failed?",
                "ground_truth": "A safe answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        created = client.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client, campaign_id)
        run_id = client.get(f"/api/evaluation/campaigns/{campaign_id}/results").json()["results"][0]["id"]
        asyncio.run(_seed_export_rows(run_id=run_id, campaign_id=campaign_id))

        errors_response = client.get(f"/api/evaluation/campaigns/{campaign_id}/errors")
        assert errors_response.status_code == 200
        error_row = errors_response.json()["rows"][0]
        assert error_row["run_id"] == run_id
        assert error_row["stage_name"] == "answer_generation"
        assert "sk-secret" not in error_row["message"]
        assert "stack trace" not in error_row["message"].lower()

        default_export = client.post(f"/api/evaluation/campaigns/{campaign_id}/export", json={})
        assert default_export.status_code == 200
        export_body = default_export.json()
        assert export_body["redaction"]["include_full_prompts"] is False
        llm_call = next(item for item in export_body["llm_calls"] if item["llm_call_id"] == f"{run_id}-llm")
        assert llm_call["prompt_preview"] == "Question: preview only"
        assert "full_prompt" not in llm_call["payload"]

        full_export = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export",
            json={"include_full_prompts": True},
        )
        assert full_export.status_code == 200
        full_llm_call = next(
            item for item in full_export.json()["llm_calls"] if item["llm_call_id"] == f"{run_id}-llm"
        )
        assert full_llm_call["payload"]["full_prompt"] == "SECRET FULL PROMPT"

        redacted_export = client.post(
            f"/api/evaluation/campaigns/{campaign_id}/export",
            json={"include_answers": False, "include_retrieved_excerpts": False},
        )
        assert redacted_export.status_code == 200
        redacted_text = redacted_export.text
        assert "Grounded answer" not in redacted_text
        assert "A safe answer" not in redacted_text
        assert "SECRET RETRIEVED CONTEXT" not in redacted_text


def test_user_cannot_export_another_users_campaign() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer="Grounded answer",
            contexts=[],
            source_doc_ids=[],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 16},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("export")

    with _build_client("user-a", upload_root, db_path, engine) as client_a:
        created_case = client_a.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-EXPORT",
                "question": "What failed?",
                "ground_truth": "A safe answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert created_case.status_code == 200
        created = client_a.post("/api/evaluation/campaigns", json=_campaign_payload())
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]
        _wait_for_completed(client_a, campaign_id)

    with _build_client("user-b", upload_root, db_path, engine) as client_b:
        denied = client_b.post(f"/api/evaluation/campaigns/{campaign_id}/export", json={})
        assert denied.status_code == 404
