from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph_rag.community_builder import summarize_community
from graph_rag.global_search import query_community
from graph_rag.schemas import Community, EntityType, GraphNode


class _GraphStoreStub:
    def get_node(self, node_id: str) -> GraphNode | None:
        return GraphNode(
            id=node_id,
            label=f"Node {node_id}",
            entity_type=EntityType.CONCEPT,
            doc_ids=["doc-1"],
            description=f"description for {node_id}",
        )

    def get_edges_for_node(self, _node_id: str) -> list:
        return []


@pytest.mark.asyncio
async def test_query_community_handles_list_content_blocks() -> None:
    store = _GraphStoreStub()
    community = Community(id=7, node_ids=["n1"], summary="社群摘要", title="社群標題")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=[{"type": "text", "text": "這是社群回答"}],
        )
    )

    with patch("graph_rag.global_search.get_llm", return_value=mock_llm):
        answer = await query_community(store, community, "這個社群在說什麼？")

    assert answer == "這是社群回答"


@pytest.mark.asyncio
async def test_summarize_community_handles_list_content_blocks() -> None:
    store = _GraphStoreStub()
    community = Community(id=3, node_ids=["n1", "n2"])
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=[
                {
                    "type": "text",
                    "text": '{"title": "模型比較", "summary": "整理 SwinUNETR 與 nnU-Net 的對比"}',
                }
            ],
        )
    )

    with patch("graph_rag.community_builder.get_llm", return_value=mock_llm):
        updated = await summarize_community(store, community)

    assert updated.title == "模型比較"
    assert updated.summary == "整理 SwinUNETR 與 nnU-Net 的對比"


@pytest.mark.asyncio
async def test_summarize_community_uses_graph_rag_runtime_override() -> None:
    store = _GraphStoreStub()
    community = Community(id=5, node_ids=["n1", "n2"])
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"title": "總結", "summary": "使用統一 thinking override"}',
            usage_metadata={},
        )
    )
    override_calls: list[str] = []

    @contextmanager
    def _fake_override(purpose: str):
        override_calls.append(purpose)
        yield

    with (
        patch("graph_rag.community_builder.get_llm", return_value=mock_llm),
        patch("graph_rag.community_builder.graph_rag_llm_runtime_override", side_effect=_fake_override),
    ):
        updated = await summarize_community(store, community)

    assert override_calls == ["community_summary"]
    assert updated.title == "總結"
