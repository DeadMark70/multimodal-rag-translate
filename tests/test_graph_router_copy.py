"""Graph router wording and response behavior tests."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from graph_rag.router import _rebuild_graph_task
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
    """Rebuild endpoint message should clearly state safe rebuild behavior."""
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
    assert "不清空既有關係" in payload["message"]


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


@pytest.mark.asyncio
async def test_rebuild_task_does_not_clear_graph() -> None:
    """Background rebuild must preserve graph contents and only refresh optimization artifacts."""
    mock_status = _MockStatus()
    mock_status.node_count = 3
    from unittest.mock import MagicMock

    mock_store = MagicMock()
    mock_store.get_status.return_value = mock_status

    with (
        patch("graph_rag.router.GraphStore", return_value=mock_store),
        patch("graph_rag.router._optimize_existing_graph", new=AsyncMock(return_value=(2, 3))) as mock_optimize,
    ):
        await _rebuild_graph_task(TEST_USER_ID)

    mock_store.clear.assert_not_called()
    mock_optimize.assert_awaited_once_with(mock_store, regenerate_communities=True)
