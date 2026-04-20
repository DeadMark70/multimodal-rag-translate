from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from graph_rag.node_vector_index import (
    NodeVectorSearchResult,
    search_nodes_by_vector,
    sync_node_vector_index,
)
from graph_rag.schemas import EntityType
from graph_rag.store import GraphStore


def _workspace_upload_root(test_name: str) -> Path:
    root = Path("output") / "test_tmp" / f"{test_name}_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.doc_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "nnu-net" in lowered or "nnunet" in lowered:
            return [1.0, 0.0, 0.0]
        if "swinunetr" in lowered:
            return [0.8, 0.2, 0.0]
        if "bert" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += len(texts)
        return [self._vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.mark.asyncio
async def test_sync_node_vector_index_reuses_unchanged_signatures() -> None:
    upload_root = _workspace_upload_root("graph_node_vector_sync")
    fake_embeddings = _FakeEmbeddings()

    with (
        patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("graph_rag.node_vector_index.get_embeddings", return_value=fake_embeddings),
        patch("graph_rag.node_vector_index.initialize_embeddings", new=AsyncMock()),
    ):
        store = GraphStore("vector-user")
        store.add_node_from_extraction(
            label="nnU-Net",
            entity_type=EntityType.METHOD,
            doc_id="doc-1",
            pending_resolution=False,
        )
        store.save()

        first = await sync_node_vector_index(user_id="vector-user", store=store)
        first_doc_calls = fake_embeddings.doc_calls
        second = await sync_node_vector_index(user_id="vector-user", store=store)

        reloaded = GraphStore("vector-user")

    assert first["index_state"] == "ready"
    assert first["changed_count"] == 1
    assert second["index_state"] == "ready"
    assert second["changed_count"] == 0
    assert second["reused_count"] == 1
    assert fake_embeddings.doc_calls == first_doc_calls
    assert reloaded.node_vector_dirty is False


@pytest.mark.asyncio
async def test_search_nodes_by_vector_returns_missing_index_fallback() -> None:
    upload_root = _workspace_upload_root("graph_node_vector_missing")

    with (
        patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("graph_rag.node_vector_index.schedule_node_vector_autosync", return_value=True),
    ):
        store = GraphStore("missing-user")
        result = await search_nodes_by_vector(
            store=store,
            query="nnunet",
            top_k=5,
            min_score=0.2,
        )

    assert isinstance(result, NodeVectorSearchResult)
    assert result.node_ids == []
    assert result.fallback_reason == "node_vector_index_missing"
    assert result.index_state == "missing"


@pytest.mark.asyncio
async def test_search_nodes_by_vector_respects_similarity_threshold() -> None:
    upload_root = _workspace_upload_root("graph_node_vector_threshold")
    fake_embeddings = _FakeEmbeddings()

    with (
        patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)),
        patch("graph_rag.node_vector_index.get_embeddings", return_value=fake_embeddings),
        patch("graph_rag.node_vector_index.initialize_embeddings", new=AsyncMock()),
    ):
        store = GraphStore("threshold-user")
        store.add_node_from_extraction(
            label="nnU-Net",
            entity_type=EntityType.METHOD,
            doc_id="doc-1",
            pending_resolution=False,
        )
        store.add_node_from_extraction(
            label="SwinUNETR",
            entity_type=EntityType.METHOD,
            doc_id="doc-2",
            pending_resolution=False,
        )
        store.save()
        expected_node_ids = set(store.find_nodes_by_label("nnu-net", fuzzy=True))
        await sync_node_vector_index(user_id="threshold-user", store=store)

        hit_result = await search_nodes_by_vector(
            store=store,
            query="nnunet segmentation",
            top_k=2,
            min_score=0.3,
        )
        miss_result = await search_nodes_by_vector(
            store=store,
            query="completely unrelated chemistry topic",
            top_k=2,
            min_score=0.3,
        )

    assert any(node_id in expected_node_ids for node_id in hit_result.node_ids)
    assert hit_result.vector_hit_count >= 1
    assert miss_result.node_ids == []
    assert miss_result.fallback_reason == "vector_score_below_threshold"
