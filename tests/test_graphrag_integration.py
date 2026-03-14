import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from data_base.RAG_QA_service import _get_graph_context
from graph_rag.generic_mode import GraphEvidence, estimate_token_count

@pytest.mark.asyncio
async def test_graph_context_retrieval_auto():
    """Verify that auto mode aliases to generic routing and merges evidence."""
    user_id = "c1bae279-c099-4c45-ba19-2bb393ca4e4b"
    # Question with keywords that should trigger graph search
    question = "分析 SwinUNETR 與 nnU-Net 的關係"
    
    with patch("graph_rag.store.GraphStore") as MockStore:
        mock_instance = MockStore.return_value
        # Mock status showing graph exists
        mock_status = MagicMock()
        mock_status.has_graph = True
        mock_status.node_count = 10
        mock_status.community_count = 5
        mock_status.community_level_counts = {"0": 3, "1": 2}
        mock_status.needs_optimization = False
        mock_instance.get_status.return_value = mock_status
        
        with patch("graph_rag.local_search.local_search_evidence", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = ([
                GraphEvidence(
                    evidence_id="local-edge-1",
                    evidence_type="local_edge",
                    text="Local relation evidence",
                    score=0.9,
                    token_estimate=estimate_token_count("Local relation evidence"),
                )
            ], ["n1", "n2"])

            with patch("graph_rag.global_search.global_search_evidence", new_callable=AsyncMock) as mock_global:
                mock_global.return_value = (
                    "Global Answer",
                    [
                        GraphEvidence(
                            evidence_id="community-answer-1",
                            evidence_type="community_answer",
                            text="Global community evidence",
                            score=0.8,
                            token_estimate=estimate_token_count("Global community evidence"),
                        )
                    ],
                    [1],
                )

                context = await _get_graph_context(question, user_id, search_mode="auto")

                assert "Graph Evidence" in context
                assert "Local relation evidence" in context
                assert "Global community evidence" in context
                mock_local.assert_awaited_once()
                mock_global.assert_awaited_once()

@pytest.mark.asyncio
async def test_graph_context_retrieval_no_graph():
    """Verify that graph context is empty when no graph exists."""
    user_id = "test-user"
    question = "分析關係"
    
    with patch("graph_rag.store.GraphStore") as MockStore:
        mock_instance = MockStore.return_value
        mock_status = MagicMock()
        mock_status.has_graph = False
        mock_status.community_level_counts = {}
        mock_status.needs_optimization = False
        mock_instance.get_status.return_value = mock_status
        
        context = await _get_graph_context(question, user_id)
        assert context == ""

def test_should_use_graph_search_logic():
    """Test the keyword detection for graph search."""
    from data_base.RAG_QA_service import _should_use_graph_search
    
    assert _should_use_graph_search("這兩篇論文的趨勢") is True
    assert _should_use_graph_search("何謂 SwinUNETR") is False
    assert _should_use_graph_search("compare these papers") is True
