from unittest.mock import patch
from langchain_core.documents import Document
from data_base.reranker import DocumentReranker, rerank_documents

def test_reranker_singleton():
    """Verify that DocumentReranker is a singleton."""
    with patch("data_base.reranker.CrossEncoder") as MockEncoder:
        r1 = DocumentReranker()
        r2 = DocumentReranker()
        assert r1 is r2
        assert MockEncoder.call_count == 1

def test_reranking_logic():
    """Verify that reranker correctly reorders documents based on scores."""
    # Reset singleton for clean test
    DocumentReranker._instance = None
    
    with patch("data_base.reranker.CrossEncoder") as MockEncoder:
        mock_model = MockEncoder.return_value
        # Mock scores: [0.1, 0.9, 0.5]
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        
        reranker = DocumentReranker()
        
        query = "SwinUNETR優勢"
        docs = [
            Document(page_content="雜訊內容 A", metadata={"id": "noise1"}),
            Document(page_content="SwinUNETR 核心架構優勢...", metadata={"id": "relevant"}),
            Document(page_content="雜訊內容 B", metadata={"id": "noise2"}),
        ]
        
        reranked = reranker.rerank(query, docs, top_k=3)
        
        # Expected order: relevant (0.9), noise2 (0.5), noise1 (0.1)
        assert reranked[0].metadata["id"] == "relevant"
        assert reranked[1].metadata["id"] == "noise2"
        assert reranked[2].metadata["id"] == "noise1"

def test_reranker_not_initialized():
    """Verify fallback when reranker is not initialized."""
    DocumentReranker._instance = None
    
    docs = [Document(page_content="A"), Document(page_content="B")]
    # rerank_documents should return original order if not initialized
    result = rerank_documents("query", docs, enabled=True)
    assert result == docs
