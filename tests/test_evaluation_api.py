"""API tests for evaluation phase 1 endpoints."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from evaluation.schemas import AvailableModel
from main import app


@contextmanager
def _build_client(user_id: str, upload_root: Path):
    """Build test client with auth override and startup stubs."""
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("evaluation.storage.BASE_UPLOAD_FOLDER", str(upload_root)),
    ):
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
