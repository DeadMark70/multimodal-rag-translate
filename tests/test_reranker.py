"""
Unit Tests for Reranker Module

Tests the DocumentReranker class and related functions.
"""

# Standard library
from unittest.mock import MagicMock

# Third-party
from langchain_core.documents import Document


class TestDocumentRerankerUnit:
    """Unit tests for DocumentReranker without loading actual model."""

    def test_rerank_empty_list(self):
        """Tests reranking empty document list."""
        from data_base.reranker import rerank_documents
        
        result = rerank_documents("query", [], top_k=6, enabled=False)
        assert result == []

    def test_rerank_disabled(self):
        """Tests that disabled reranking returns original order."""
        from data_base.reranker import rerank_documents
        
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={}),
        ]
        
        result = rerank_documents("query", docs, top_k=2, enabled=False)
        
        assert len(result) == 2
        assert result[0].page_content == "Doc 1"
        assert result[1].page_content == "Doc 2"

    def test_rerank_top_k_truncation(self):
        """Tests that top_k limits results when disabled."""
        from data_base.reranker import rerank_documents
        
        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(10)]
        
        result = rerank_documents("query", docs, top_k=3, enabled=False)
        
        assert len(result) == 3


class TestDocumentRerankerWithMock:
    """Tests with mocked CrossEncoder model."""

    def test_rerank_with_mock_model(self):
        """Tests reranking with mocked CrossEncoder."""
        from data_base.reranker import DocumentReranker
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.9, 0.5, 0.7])
        
        docs = [
            Document(page_content="High relevance", metadata={"id": 1}),
            Document(page_content="Low relevance", metadata={"id": 2}),
            Document(page_content="Medium relevance", metadata={"id": 3}),
        ]
        
        # Create reranker with mocked model
        reranker = DocumentReranker.__new__(DocumentReranker)
        reranker._model = mock_model
        
        result = reranker.rerank("query", docs, top_k=2)
        
        # Should be sorted by score descending
        assert len(result) == 2
        assert result[0].metadata["id"] == 1  # 0.9 score
        assert result[1].metadata["id"] == 3  # 0.7 score

    def test_rerank_with_scores_mock(self):
        """Tests rerank_with_scores method."""
        from data_base.reranker import DocumentReranker
        
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.3, 0.8])
        
        docs = [
            Document(page_content="First", metadata={}),
            Document(page_content="Second", metadata={}),
        ]
        
        reranker = DocumentReranker.__new__(DocumentReranker)
        reranker._model = mock_model
        
        result = reranker.rerank_with_scores("query", docs, top_k=2)
        
        assert len(result) == 2
        # Highest score first
        assert result[0][0].page_content == "Second"
        assert result[0][1] == 0.8
        assert result[1][0].page_content == "First"
        assert result[1][1] == 0.3

    def test_rerank_error_handling(self):
        """Tests graceful error handling when model fails."""
        from data_base.reranker import DocumentReranker
        
        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=RuntimeError("Model error"))
        
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]
        
        reranker = DocumentReranker.__new__(DocumentReranker)
        reranker._model = mock_model
        
        result = reranker.rerank("query", docs, top_k=2)
        
        # Should return original order on error
        assert len(result) == 2
        assert result[0].page_content == "Doc 1"


class TestRerankerSingleton:
    """Tests for singleton pattern."""

    def test_is_initialized_before_creation(self):
        """Tests is_initialized returns False initially."""
        from data_base.reranker import DocumentReranker
        
        # Reset singleton for test
        DocumentReranker._instance = None
        DocumentReranker._model = None
        
        assert not DocumentReranker.is_initialized()
