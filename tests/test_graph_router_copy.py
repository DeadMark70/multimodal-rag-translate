"""Graph router wording and response behavior tests."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from graph_rag.maintenance import rebuild_graph_task
from graph_rag.maintenance import _copy_graph_sidecars, _replace_live_graph_files
from graph_rag.schemas import EvidenceAnchor, EntityType, GraphAssetLink
from graph_rag.store import GraphStore
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
        patch("graph_rag.maintenance.GraphStore", return_value=mock_store),
        patch(
            "graph_rag.maintenance.optimize_existing_graph",
            new=AsyncMock(return_value=(2, 3)),
        ) as mock_optimize,
    ):
        await rebuild_graph_task(TEST_USER_ID)

    mock_store.clear.assert_not_called()
    mock_optimize.assert_awaited_once_with(mock_store, regenerate_communities=True)


def test_copy_graph_sidecars_preserves_evidence_locator_state(tmp_path) -> None:
    source = GraphStore(TEST_USER_ID, storage_dir=tmp_path / "source")
    source_id = source.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")
    target_id = source.add_node_from_extraction("memory", EntityType.CONCEPT, "doc-1")
    source.add_edge_from_extraction(
        source_id,
        target_id,
        "uses",
        "doc-1",
    )
    edge_id = source.edge_id(source_id, target_id, "uses")
    source.record_edge_provenance(
        edge_id,
        [EvidenceAnchor(doc_id="doc-1", chunk_id="chunk-1", confidence=0.9)],
    )
    source.record_asset_link(
        GraphAssetLink(
            asset_id="asset-1",
            doc_id="doc-1",
            asset_type="table",
            asset_parse_status="parsed",
        )
    )
    source.save()
    destination = GraphStore(TEST_USER_ID, storage_dir=tmp_path / "destination")

    _copy_graph_sidecars(source, destination)
    copied = GraphStore(TEST_USER_ID, storage_dir=tmp_path / "destination")

    assert edge_id in copied.edge_provenance
    assert "asset-1" in copied.asset_links


def test_snapshot_promotion_retains_ready_node_vector_index(tmp_path) -> None:
    live = GraphStore(TEST_USER_ID, storage_dir=tmp_path / "live")
    temp = GraphStore(TEST_USER_ID, storage_dir=tmp_path / "temp")
    temp.add_node_from_extraction("MedSAM", EntityType.METHOD, "doc-1")
    temp.save()
    temp.node_vector_dirty = False
    temp._get_node_vector_map_path().write_text('{"node_ids": ["node-1"]}', encoding="utf-8")

    _replace_live_graph_files(temp, live)

    promoted = GraphStore(TEST_USER_ID, storage_dir=tmp_path / "live")
    assert promoted._get_node_vector_map_path().exists()
