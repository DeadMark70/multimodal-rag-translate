"""
Unit Tests for Query Transformer Module

Tests HyDE, multi-query generation, and RRF fusion.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
from langchain.schema import Document


class TestReciprocalRankFusion:
    """Tests for the RRF fusion function."""

    def test_rrf_single_list(self):
        """Tests RRF with single result list."""
        from data_base.query_transformer import reciprocal_rank_fusion
        
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]
        
        result = reciprocal_rank_fusion([docs])
        
        assert len(result) == 2
        assert result[0].page_content == "Doc 1"

    def test_rrf_multiple_lists_with_overlap(self):
        """Tests RRF with overlapping documents from multiple lists."""
        from data_base.query_transformer import reciprocal_rank_fusion
        
        # Same document appears in both lists at different ranks
        doc_a = Document(page_content="Common doc", metadata={})
        doc_b = Document(page_content="Only in list 1", metadata={})
        doc_c = Document(page_content="Only in list 2", metadata={})
        
        list1 = [doc_a, doc_b]  # doc_a at rank 0
        list2 = [doc_c, doc_a]  # doc_a at rank 1 (but same content)
        
        result = reciprocal_rank_fusion([list1, list2])
        
        # Common doc should have highest score (appears in both)
        assert result[0].page_content == "Common doc"

    def test_rrf_empty_lists(self):
        """Tests RRF with empty input."""
        from data_base.query_transformer import reciprocal_rank_fusion
        
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_deduplication(self):
        """Tests that RRF deduplicates by content."""
        from data_base.query_transformer import reciprocal_rank_fusion
        
        doc1 = Document(page_content="Same content", metadata={"id": 1})
        doc2 = Document(page_content="Same content", metadata={"id": 2})
        
        result = reciprocal_rank_fusion([[doc1], [doc2]])
        
        # Should be deduplicated to 1 document
        assert len(result) == 1


class TestQueryTransformerHyde:
    """Tests for HyDE query transformation."""

    @pytest.mark.asyncio
    async def test_hyde_generation(self):
        """Tests HyDE document generation with mock LLM."""
        from data_base.query_transformer import QueryTransformer
        
        mock_response = MagicMock()
        mock_response.content = "這是一段假設性文檔，包含答案相關的資訊。"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.query_transformer.get_llm", return_value=mock_llm):
            transformer = QueryTransformer()
            result = await transformer.generate_hyde_document("什麼是機器學習？")
        
        assert "假設性文檔" in result

    @pytest.mark.asyncio
    async def test_hyde_error_fallback(self):
        """Tests that HyDE falls back to original on error."""
        from data_base.query_transformer import QueryTransformer
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        with patch("data_base.query_transformer.get_llm", return_value=mock_llm):
            transformer = QueryTransformer()
            result = await transformer.generate_hyde_document("test question")
        
        # Should return original question on error
        assert result == "test question"

    @pytest.mark.asyncio
    async def test_transform_query_with_hyde_disabled(self):
        """Tests convenience function when disabled."""
        from data_base.query_transformer import transform_query_with_hyde
        
        result = await transform_query_with_hyde("test question", enabled=False)
        
        assert result == "test question"


class TestQueryTransformerMultiQuery:
    """Tests for multi-query generation."""

    @pytest.mark.asyncio
    async def test_multi_query_generation(self):
        """Tests multi-query generation with mock LLM."""
        from data_base.query_transformer import QueryTransformer
        
        mock_response = MagicMock()
        mock_response.content = """1. 什麼是深度學習的基本原理？
2. 深度學習如何處理圖像？
3. 深度學習的應用場景有哪些？"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.query_transformer.get_llm", return_value=mock_llm):
            transformer = QueryTransformer()
            result = await transformer.generate_multi_queries("深度學習是什麼？")
        
        # Should include original + parsed queries
        assert len(result) >= 2
        assert "深度學習是什麼？" in result[0]

    @pytest.mark.asyncio
    async def test_multi_query_error_fallback(self):
        """Tests that multi-query falls back to original on error."""
        from data_base.query_transformer import QueryTransformer
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        with patch("data_base.query_transformer.get_llm", return_value=mock_llm):
            transformer = QueryTransformer()
            result = await transformer.generate_multi_queries("test question")
        
        # Should return [original] on error
        assert result == ["test question"]

    @pytest.mark.asyncio
    async def test_transform_query_multi_disabled(self):
        """Tests convenience function when disabled."""
        from data_base.query_transformer import transform_query_multi
        
        result = await transform_query_multi("test question", enabled=False)
        
        assert result == ["test question"]

    @pytest.mark.asyncio
    async def test_multi_query_max_limit(self):
        """Tests that max_queries is respected."""
        from data_base.query_transformer import QueryTransformer
        
        mock_response = MagicMock()
        mock_response.content = """1. Query 1 is a longer query
2. Query 2 is also quite long
3. Query 3 another longer query
4. Query 4 more content here
5. Query 5 even more content"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.query_transformer.get_llm", return_value=mock_llm):
            transformer = QueryTransformer()
            result = await transformer.generate_multi_queries("test", max_queries=2)
        
        # Original + max 2 sub-queries = 3 max
        assert len(result) <= 3
