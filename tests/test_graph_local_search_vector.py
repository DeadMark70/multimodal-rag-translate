from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from graph_rag.node_vector_index import NodeVectorSearchResult
from graph_rag.schemas import EntityType
from graph_rag.store import GraphStore


def _workspace_upload_root(test_name: str) -> Path:
    root = Path("output") / "test_tmp" / f"{test_name}_{uuid4().hex}" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.asyncio
async def test_local_search_prefers_vector_seeds_without_llm_entity_step() -> None:
    upload_root = _workspace_upload_root("graph_local_vector_prefer")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("vector-local-user")
        node_id = store.add_node_from_extraction(
            label="nnU-Net",
            entity_type=EntityType.METHOD,
            doc_id="doc-1",
            pending_resolution=False,
        )
        store.save()

        with (
            patch("graph_rag.local_search.node_vector_search_enabled", return_value=True),
            patch(
                "graph_rag.local_search.search_nodes_by_vector",
                new=AsyncMock(
                    return_value=NodeVectorSearchResult(
                        node_ids=[node_id],
                        vector_hit_count=1,
                        index_state="ready",
                        fallback_reason=None,
                        top_score=0.91,
                    )
                ),
            ),
            patch(
                "graph_rag.local_search.identify_query_entities",
                new=AsyncMock(return_value=["should-not-be-used"]),
            ) as mock_identify,
        ):
            from graph_rag.local_search import local_search

            context, nodes = await local_search(store, "nnunet", hops=1, max_nodes=5)

    assert node_id in nodes
    assert "nnU-Net" in context
    mock_identify.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_search_falls_back_to_legacy_llm_entity_matching() -> None:
    upload_root = _workspace_upload_root("graph_local_vector_fallback")

    with patch("core.uploads.BASE_UPLOAD_FOLDER", str(upload_root)):
        store = GraphStore("fallback-local-user")
        node_id = store.add_node_from_extraction(
            label="BERT",
            entity_type=EntityType.METHOD,
            doc_id="doc-1",
            pending_resolution=False,
        )
        store.save()

        with (
            patch("graph_rag.local_search.node_vector_search_enabled", return_value=True),
            patch(
                "graph_rag.local_search.search_nodes_by_vector",
                new=AsyncMock(
                    return_value=NodeVectorSearchResult(
                        node_ids=[],
                        vector_hit_count=0,
                        index_state="missing",
                        fallback_reason="node_vector_index_missing",
                    )
                ),
            ),
            patch(
                "graph_rag.local_search.identify_query_entities",
                new=AsyncMock(return_value=["BERT"]),
            ) as mock_identify,
        ):
            from graph_rag.local_search import local_search

            context, nodes = await local_search(store, "bert method", hops=1, max_nodes=5)

    assert node_id in nodes
    assert "BERT" in context
    mock_identify.assert_awaited_once()
