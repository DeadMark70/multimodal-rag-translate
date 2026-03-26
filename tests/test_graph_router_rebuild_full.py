from pathlib import Path
from uuid import uuid4
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from graph_rag.router import (
    _rebuild_full_graph_task,
    _purge_graph_document_task,
    _retry_graph_document_task,
)
from graph_rag.schemas import EntityType, GraphDocumentStatus, GraphExtractionRunResult
from graph_rag.store import GraphStore
from main import app

TEST_USER_ID = "graph-router-user"


def _workspace_upload_root(test_name: str) -> Path:
    root = Path("output") / "test_tmp" / f"{test_name}_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _client() -> TestClient:
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    return TestClient(app)


def test_graph_documents_endpoint_returns_persisted_and_unattempted_rows() -> None:
    class _MockStore:
        def __init__(self, _user_id: str) -> None:
            self.document_statuses = {
                "doc-1": GraphDocumentStatus(
                    doc_id="doc-1",
                    status="failed",
                    chunk_count=2,
                    chunks_succeeded=1,
                    chunks_failed=1,
                    entities_added=3,
                    edges_added=2,
                    last_error="chunk 1: quota exceeded",
                )
            }

        def get_document_status(self, doc_id: str) -> GraphDocumentStatus | None:
            return self.document_statuses.get(doc_id)

        def list_eligible_document_ids(self) -> list[str]:
            return ["doc-1", "doc-2"]

        def get_documents(self) -> set[str]:
            return {"doc-1", "doc-orphan"}

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("graph_rag.router.GraphStore", _MockStore),
        patch(
            "graph_rag.router.list_pdf_documents",
            new=AsyncMock(
                return_value=[
                    {"id": "doc-1", "file_name": "failed.pdf", "original_path": "uploads/u/doc-1/failed.pdf"},
                    {"id": "doc-2", "file_name": "fresh.pdf", "original_path": "uploads/u/doc-2/fresh.pdf"},
                ]
            ),
        ),
        _client() as client,
    ):
        response = client.get("/graph/documents")

    app.dependency_overrides = {}
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert payload["documents"][0]["status"] == "failed"
    assert payload["documents"][1]["status"] == "skipped"
    assert payload["documents"][2]["doc_id"] == "doc-orphan"
    assert payload["documents"][2]["status"] == "indexed"
    assert payload["documents"][2]["is_eligible"] is False


def test_rebuild_full_endpoint_sets_active_job_and_starts_background_task() -> None:
    mock_store = Mock()
    mock_store.active_job_state = None
    mock_store.save_sidecars = Mock()
    mock_task = AsyncMock()

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("graph_rag.router.GraphStore", return_value=mock_store),
        patch(
            "graph_rag.router._list_graph_source_documents",
            new=AsyncMock(return_value=[{"doc_id": "doc-1", "file_name": "demo.pdf", "original_path": "uploads/u/doc-1/demo.pdf"}]),
        ),
        patch("graph_rag.router._rebuild_full_graph_task", new=mock_task),
        _client() as client,
    ):
        response = client.post("/graph/rebuild-full")

    app.dependency_overrides = {}
    assert response.status_code == 200
    assert response.json()["status"] == "started"
    mock_store.set_active_job_state.assert_called_once_with("rebuild_full")
    mock_store.save_sidecars.assert_called()
    mock_task.assert_awaited_once_with(TEST_USER_ID)


def test_purge_graph_document_endpoint_sets_active_job_and_starts_background_task() -> None:
    mock_store = Mock()
    mock_store.active_job_state = None
    mock_store.get_document_status.return_value = GraphDocumentStatus(doc_id="doc-1", status="failed")
    mock_store.get_documents.return_value = {"doc-1"}
    mock_store.save_sidecars = Mock()
    mock_task = AsyncMock()

    with (
        patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
        patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
        patch("graph_rag.router.GraphStore", return_value=mock_store),
        patch("graph_rag.router._purge_graph_document_task", new=mock_task),
        _client() as client,
    ):
        response = client.delete("/graph/documents/doc-1")

    app.dependency_overrides = {}
    assert response.status_code == 200
    assert response.json()["status"] == "started"
    mock_store.set_active_job_state.assert_called_once_with("purge:doc-1")
    mock_store.save_sidecars.assert_called()
    mock_task.assert_awaited_once_with(TEST_USER_ID, "doc-1")


@pytest.mark.asyncio
async def test_full_rebuild_keeps_old_graph_when_any_document_fails() -> None:
    upload_root = _workspace_upload_root("graph_full_rebuild_fail")
    artifact_dir = upload_root / TEST_USER_ID / "doc-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "extracted.md").write_text("demo", encoding="utf-8")

    with (
        patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("graph_rag.router.list_pdf_documents", new=AsyncMock(return_value=[
            {"id": "doc-1", "file_name": "demo.pdf", "original_path": str(artifact_dir / "demo.pdf")},
        ])),
        patch("graph_rag.router.load_ocr_artifacts", new=Mock(return_value=("demo", []))),
    ):
        live_store = GraphStore(TEST_USER_ID)
        live_store.add_node_from_extraction(
            label="Legacy Entity",
            entity_type=EntityType.CONCEPT,
            doc_id="doc-legacy",
            pending_resolution=False,
        )
        live_store.save()

        async def _fake_run_graph_extraction(*, store: GraphStore, doc_id: str, **_: object) -> GraphExtractionRunResult:
            store.upsert_document_status(
                GraphDocumentStatus(doc_id=doc_id, status="failed", last_error="boom")
            )
            store.save_sidecars()
            return GraphExtractionRunResult(doc_id=doc_id, status="failed", last_error="boom")

        with patch("graph_rag.router.run_graph_extraction", new=AsyncMock(side_effect=_fake_run_graph_extraction)):
            await _rebuild_full_graph_task(TEST_USER_ID)

        reloaded = GraphStore(TEST_USER_ID)

    assert reloaded.graph.number_of_nodes() == 1
    assert reloaded.get_all_nodes()[0].label == "Legacy Entity"
    assert reloaded.get_document_status("doc-1") is not None
    assert reloaded.get_document_status("doc-1").status == "failed"


@pytest.mark.asyncio
async def test_retry_graph_document_replaces_only_target_document_contribution() -> None:
    upload_root = _workspace_upload_root("graph_retry_success")
    artifact_dir = upload_root / TEST_USER_ID / "doc-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "extracted.md").write_text("demo", encoding="utf-8")

    with (
        patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch(
            "graph_rag.router.get_document",
            new=AsyncMock(return_value={"original_path": str(artifact_dir / "demo.pdf")}),
        ),
        patch("graph_rag.router.load_ocr_artifacts", new=Mock(return_value=("demo", []))),
    ):
        live_store = GraphStore(TEST_USER_ID)
        live_store.add_node_from_extraction(
            label="Old Entity",
            entity_type=EntityType.CONCEPT,
            doc_id="doc-1",
            pending_resolution=False,
        )
        live_store.add_node_from_extraction(
            label="Keep Entity",
            entity_type=EntityType.CONCEPT,
            doc_id="doc-2",
            pending_resolution=False,
        )
        live_store.save()

        async def _fake_run_graph_extraction(*, store: GraphStore, doc_id: str, **_: object) -> GraphExtractionRunResult:
            store.add_node_from_extraction(
                label="Retried Entity",
                entity_type=EntityType.METHOD,
                doc_id=doc_id,
                pending_resolution=False,
            )
            store.upsert_document_status(GraphDocumentStatus(doc_id=doc_id, status="indexed", entities_added=1))
            store.save()
            return GraphExtractionRunResult(doc_id=doc_id, status="indexed", entities_added=1)

        async def _fake_optimize(store: GraphStore, *, regenerate_communities: bool = True) -> tuple[int, int]:
            store.save()
            return (0, 0)

        with (
            patch("graph_rag.router.run_graph_extraction", new=AsyncMock(side_effect=_fake_run_graph_extraction)),
            patch("graph_rag.router._optimize_existing_graph", new=AsyncMock(side_effect=_fake_optimize)),
        ):
            await _retry_graph_document_task(TEST_USER_ID, "doc-1")

        reloaded = GraphStore(TEST_USER_ID)
        labels = {node.label for node in reloaded.get_all_nodes()}

    assert "Old Entity" not in labels
    assert "Retried Entity" in labels
    assert "Keep Entity" in labels
    assert reloaded.get_document_status("doc-1") is not None
    assert reloaded.get_document_status("doc-1").status == "indexed"


@pytest.mark.asyncio
async def test_purge_graph_document_removes_orphan_contribution() -> None:
    upload_root = _workspace_upload_root("graph_purge_success")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        live_store = GraphStore(TEST_USER_ID)
        live_store.add_node_from_extraction(
            label="Orphan Entity",
            entity_type=EntityType.CONCEPT,
            doc_id="doc-orphan",
            pending_resolution=False,
        )
        live_store.add_node_from_extraction(
            label="Keep Entity",
            entity_type=EntityType.CONCEPT,
            doc_id="doc-2",
            pending_resolution=False,
        )
        live_store.upsert_document_status(
            GraphDocumentStatus(doc_id="doc-orphan", status="failed", last_error="left behind")
        )
        live_store.save()

        await _purge_graph_document_task(TEST_USER_ID, "doc-orphan")

        reloaded = GraphStore(TEST_USER_ID)
        labels = {node.label for node in reloaded.get_all_nodes()}

    assert "Orphan Entity" not in labels
    assert "Keep Entity" in labels
    assert reloaded.get_document_status("doc-orphan") is None
