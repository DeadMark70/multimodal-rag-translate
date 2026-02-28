"""Graph router wording and response behavior tests."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from main import app

TEST_USER_ID = "test-user-graph"


class _MockStatus:
    node_count = 1


class _MockStore:
    def __init__(self, _user_id: str) -> None:
        self._status = _MockStatus()

    def get_status(self):  # noqa: ANN201
        return self._status


def _client() -> TestClient:
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    return TestClient(app)


def test_graph_rebuild_message_is_not_misleading() -> None:
    """Rebuild endpoint message should clearly state no source re-extraction."""
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("graph_rag.router.GraphStore", _MockStore),
        _client() as client,
    ):
        response = client.post("/graph/rebuild", json={"force": True})

    app.dependency_overrides = {}
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "started"
    assert "不重新抽取文件實體" in payload["message"]


def test_graph_rebuild_openapi_description_matches_behavior() -> None:
    """OpenAPI should mention rebuild does not re-extract source entities."""
    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        _client() as client,
    ):
        openapi = client.get("/openapi.json").json()

    app.dependency_overrides = {}
    description = openapi["paths"]["/graph/rebuild"]["post"]["description"]
    assert "不會重新從原始文件抽取新實體" in description
