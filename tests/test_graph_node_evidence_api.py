from urllib.parse import quote
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from graph_rag.node_evidence import build_node_evidence_response
from graph_rag.schemas import EntityType, EvidenceAnchor
from graph_rag.store import GraphStore
from main import app


def _store(tmp_path):
    store = GraphStore("user-1", storage_dir=tmp_path)
    source = store.add_node_from_extraction("Transformer", EntityType.METHOD, "doc-1")
    store.graph.nodes[source]["doc_ids"] = ["doc-1", "doc-2"]
    target = store.add_node_from_extraction("Attention", EntityType.CONCEPT, "doc-1")
    store.add_edge_from_extraction(source, target, "uses", "doc-1")
    edge_id = store.edge_id(source, target, "uses")
    full = EvidenceAnchor(
        doc_id="doc-1", chunk_id="chunk-1", page=3,
        quote="Transformer uses self-attention.", quote_hash="quote-1",
        chunk_hash="chunk-1-hash", confidence=0.95,
    )
    partial = EvidenceAnchor(
        doc_id="doc-1", page=4, quote="A second source passage.", confidence=0.7,
    )
    store.record_edge_provenance(edge_id, [full, full, partial])
    return store, source


@pytest.mark.asyncio
async def test_build_node_evidence_deduplicates_and_keeps_source_only_documents(tmp_path):
    store, node_key = _store(tmp_path)

    async def fake_get_document(*, doc_id, user_id, columns="*"):
        assert user_id == "user-1"
        return {"id": doc_id, "file_name": f"{doc_id}.pdf"}

    with (
        patch("graph_rag.node_evidence.GraphStore", return_value=store),
        patch("graph_rag.node_evidence.get_document", side_effect=fake_get_document),
    ):
        response = await build_node_evidence_response(user_id="user-1", node_key=node_key)

    assert [item.provenance_status for item in response.evidence] == ["full", "partial"]
    assert [item.page for item in response.evidence] == [3, 4]
    assert [item.doc_id for item in response.source_documents] == ["doc-1", "doc-2"]
    assert response.source_documents[1].filename == "doc-2.pdf"


@pytest.mark.asyncio
async def test_build_node_evidence_uses_safe_not_found_for_missing_node(tmp_path):
    store = GraphStore("user-1", storage_dir=tmp_path)
    with patch("graph_rag.node_evidence.GraphStore", return_value=store):
        with pytest.raises(AppError) as exc_info:
            await build_node_evidence_response(user_id="user-1", node_key="node-other-user")

    assert exc_info.value.code is ErrorCode.NOT_FOUND
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_build_node_evidence_caps_rows_and_drops_mismatched_quotes(tmp_path):
    store, node_key = _store(tmp_path)
    edge = store.get_edges_for_node(node_key)[0]
    edge_id = store.edge_id(edge.source_id, edge.target_id, edge.relation)
    anchors = [
        EvidenceAnchor(
            doc_id="doc-1", chunk_id=f"chunk-{index}", page=index + 1,
            quote=f"Verified quote {index}", quote_hash=f"quote-{index}",
            chunk_hash=f"chunk-hash-{index}", confidence=0.9,
            verification_status="quote_match",
        )
        for index in range(25)
    ]
    anchors.append(EvidenceAnchor(
        doc_id="doc-1", chunk_id="bad", quote="Mismatched quote",
        confidence=0.9, verification_status="quote_mismatch",
    ))
    store.record_edge_provenance(edge_id, anchors)

    with (
        patch("graph_rag.node_evidence.GraphStore", return_value=store),
        patch("graph_rag.node_evidence.get_document", new=AsyncMock(
            return_value={"id": "doc-1", "file_name": "doc-1.pdf"}
        )),
    ):
        response = await build_node_evidence_response(user_id="user-1", node_key=node_key)

    assert len(response.evidence) == 20
    assert all(item.quote != "Mismatched quote" for item in response.evidence)


def test_get_graph_node_evidence_returns_authenticated_evidence(tmp_path):
    store, node_key = _store(tmp_path)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            patch("graph_rag.node_evidence.GraphStore", return_value=store),
            patch(
                "graph_rag.node_evidence.get_document",
                new=AsyncMock(
                    side_effect=lambda *, doc_id, **_: {
                        "id": doc_id,
                        "file_name": f"{doc_id}.pdf",
                    }
                ),
            ),
            TestClient(app) as client,
        ):
            response = client.get(f"/graph/nodes/{quote(node_key, safe='')}/evidence")
    finally:
        app.dependency_overrides = {}

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_key"] == node_key
    assert payload["evidence"][0]["page"] == 3
    assert payload["source_documents"] == [
        {"doc_id": "doc-1", "filename": "doc-1.pdf"},
        {"doc_id": "doc-2", "filename": "doc-2.pdf"},
    ]


def test_graph_data_keeps_label_id_and_exposes_internal_node_key(tmp_path):
    store, node_key = _store(tmp_path)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    try:
        with (
            patch("core.app_factory._initialize_rag_components", new=AsyncMock()),
            patch("core.app_factory._warm_up_pdf_ocr", new=AsyncMock()),
            patch("graph_rag.router.GraphStore", return_value=store),
            TestClient(app) as client,
        ):
            response = client.get("/graph/data")
    finally:
        app.dependency_overrides = {}

    assert response.status_code == 200
    node = next(item for item in response.json()["nodes"] if item["id"] == "Transformer")
    assert node["node_key"] == node_key
    assert node["source_docs"] == ["doc-1", "doc-2"]
