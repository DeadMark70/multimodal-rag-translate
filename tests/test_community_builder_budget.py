from unittest.mock import AsyncMock, patch

import pytest

from graph_rag.community_builder import build_communities
from graph_rag.schemas import Community, EntityType, GraphNode


class _FakeStore:
    def __init__(self) -> None:
        self.user_id = "budget-test-user"
        self.communities = []
        self._optimized = False

    def get_node(self, node_id: str) -> GraphNode | None:
        return GraphNode(
            id=node_id,
            label=node_id.upper(),
            entity_type=EntityType.CONCEPT,
            doc_ids=["doc-1"],
            description=f"description for {node_id}",
        )

    def mark_optimized(self) -> None:
        self._optimized = True


@pytest.mark.asyncio
async def test_build_communities_caps_llm_summary_calls() -> None:
    store = _FakeStore()
    detected = [
        Community(id=1, node_ids=["a1", "a2", "a3"]),
        Community(id=2, node_ids=["b1", "b2", "b3"]),
        Community(id=3, node_ids=["c1", "c2"]),
        Community(id=4, node_ids=["d1", "d2"]),
    ]

    async def _fake_leaf_summary(_store: _FakeStore, community: Community) -> Community:
        community.title = f"leaf-{community.id}"
        community.summary = f"leaf-summary-{community.id}"
        return community

    with (
        patch("graph_rag.community_builder.detect_communities_leiden", new=AsyncMock(return_value=detected)),
        patch("graph_rag.community_builder.summarize_community", new=AsyncMock(side_effect=_fake_leaf_summary)) as mock_leaf,
        patch("graph_rag.community_builder._summarize_parent_community", new=AsyncMock()) as mock_parent,
    ):
        communities = await build_communities(
            store,
            generate_summaries=True,
            max_llm_summaries=2,
            summary_delay_seconds=0.0,
        )

    assert len(communities) == 6
    assert mock_leaf.await_count == 2
    mock_parent.assert_not_awaited()
    assert any(community.summary == "包含 2 個實體" for community in communities if community.level == 0)
    assert any((community.summary or "").startswith("leaf-summary") for community in communities if community.level == 0)
    assert store._optimized is True


@pytest.mark.asyncio
async def test_build_communities_throttles_between_llm_summaries() -> None:
    store = _FakeStore()
    detected = [
        Community(id=1, node_ids=["a1", "a2"]),
        Community(id=2, node_ids=["b1", "b2"]),
    ]
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    async def _fake_leaf_summary(_store: _FakeStore, community: Community) -> Community:
        community.title = f"leaf-{community.id}"
        community.summary = f"leaf-summary-{community.id}"
        return community

    with (
        patch("graph_rag.community_builder.detect_communities_leiden", new=AsyncMock(return_value=detected)),
        patch("graph_rag.community_builder.summarize_community", new=AsyncMock(side_effect=_fake_leaf_summary)),
        patch("graph_rag.community_builder.asyncio.sleep", new=AsyncMock(side_effect=_fake_sleep)),
    ):
        await build_communities(
            store,
            generate_summaries=True,
            max_llm_summaries=2,
            summary_delay_seconds=0.5,
        )

    assert sleep_calls == [0.5, 0.5]
