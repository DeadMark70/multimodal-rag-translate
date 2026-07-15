from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from core.auth import get_current_user_id
from evaluation.campaign_engine import CampaignEngine
from evaluation.campaign_schemas import CampaignConfig
from evaluation.rag_modes import BenchmarkExecutionResult
from evaluation.schemas import ModelConfig
from main import app


class FakeRagasEvaluator:
    async def evaluate_campaign(self, *, on_progress=None, **kwargs) -> str:
        if on_progress:
            await on_progress(1, 1, "Q-ABLATE", "naive")
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


def _model_config() -> ModelConfig:
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


def _make_workspace_paths(prefix: str) -> tuple[Path, Path]:
    root = Path.cwd() / "output" / "test_tmp" / f"{prefix}_{uuid4().hex}"
    return root / "uploads", root / "evaluation.db"


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


def test_campaign_config_accepts_ablation_conditions() -> None:
    config = CampaignConfig(
        test_case_ids=["Q-ABLATE"],
        modes=[],
        model_config=_model_config(),
        ablation_conditions=[
            {
                "condition_id": "visual_verifier",
                "label": "Visual + verifier",
                "mode": "agentic",
                "ablation_flags": {
                    "use_graph": False,
                    "use_visual_reexamine": True,
                    "use_claim_verifier": True,
                    "max_drilldown_depth": 1,
                },
                "budget": {"max_tokens": 12000},
            }
        ],
    )

    assert config.ablation_conditions[0].condition_id == "visual_verifier"
    assert config.ablation_conditions[0].budget == {"max_tokens": 12000}


def test_campaign_config_rejects_duplicate_ablation_condition_ids() -> None:
    with pytest.raises(ValueError, match="ablation condition_id values must be unique"):
        CampaignConfig(
            test_case_ids=["Q-ABLATE"],
            modes=[],
            model_config=_model_config(),
            ablation_conditions=[
                {"condition_id": "same", "label": "A", "mode": "naive"},
                {"condition_id": "same", "label": "B", "mode": "agentic"},
            ],
        )


def test_ablation_campaign_expands_conditions_and_persists_condition_metadata() -> None:
    async def runner(**kwargs) -> BenchmarkExecutionResult:
        test_case = kwargs["test_case"]
        return BenchmarkExecutionResult(
            question_id=test_case.id,
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            mode=kwargs["mode"],
            answer=f"answer-{kwargs['mode']}",
            contexts=["ctx-1"],
            source_doc_ids=["doc-1"],
            expected_sources=[],
            latency_ms=10,
            token_usage={"total_tokens": 20},
            category=test_case.category,
            difficulty=test_case.difficulty,
        )

    engine = CampaignEngine(runner=runner, ragas_evaluator=FakeRagasEvaluator())
    upload_root, db_path = _make_workspace_paths("ablation")

    with _build_client("user-a", upload_root, db_path, engine) as client:
        test_case_response = client.post(
            "/api/evaluation/test-cases",
            json={
                "id": "Q-ABLATE",
                "question": "Which setup works best?",
                "ground_truth": "Baseline answer",
                "source_docs": [],
                "requires_multi_doc_reasoning": False,
            },
        )
        assert test_case_response.status_code == 200

        created = client.post(
            "/api/evaluation/campaigns",
            json={
                "name": "Ablation",
                "test_case_ids": ["Q-ABLATE"],
                "modes": [],
                "ablation_conditions": [
                    {
                        "condition_id": "text_only",
                        "label": "Text only",
                        "mode": "naive",
                        "ablation_flags": {"use_graph": False},
                    },
                    {
                        "condition_id": "visual_verifier",
                        "label": "Visual + verifier",
                        "mode": "agentic",
                        "ablation_flags": {
                            "use_graph": False,
                            "use_visual_reexamine": True,
                            "use_claim_verifier": True,
                        },
                    },
                ],
                "model_config": _model_config().model_dump(mode="json"),
                "repeat_count": 2,
                "batch_size": 1,
                "rpm_limit": 60,
            },
        )
        assert created.status_code == 200
        campaign_id = created.json()["campaign_id"]

        _wait_for_completed(client, campaign_id)

        results_response = client.get(f"/api/evaluation/campaigns/{campaign_id}/results")
        assert results_response.status_code == 200
        results = results_response.json()["results"]
        assert len(results) == 4
        assert {
            (
                item["derived_metrics"]["condition_id"],
                item["derived_metrics"]["condition_label"],
                item["derived_metrics"]["repeat_number"],
                item["repeat_number"],
            )
            for item in results
        } == {
            ("text_only", "Text only", 1, 1),
            ("text_only", "Text only", 2, 2),
            ("visual_verifier", "Visual + verifier", 1, 1),
            ("visual_verifier", "Visual + verifier", 2, 2),
        }
        assert len({item["run_number"] for item in results}) == 4
        assert results[0]["system_version_snapshot"]["ablation_flags"] is not None

        ablation_response = client.get(f"/api/evaluation/campaigns/{campaign_id}/ablation")
        assert ablation_response.status_code == 200
        assert ablation_response.json()["summaries"]["condition_counts"] == {
            "text_only": 2,
            "visual_verifier": 2,
        }
