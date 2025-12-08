"""
Unit Tests for Context Enricher

Tests the ContextEnricher class and related functions.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
from langchain.schema import Document

# Local application
from data_base.context_enricher import (
    ContextEnricher,
    enrich_documents_with_context,
)


class TestContextEnricher:
    """Tests for the ContextEnricher class."""

    @pytest.mark.asyncio
    async def test_enrich_chunk_adds_context_prefix(self):
        """Tests that context prefix is added to chunk."""
        enricher = ContextEnricher(max_concurrent=3)
        
        doc = Document(
            page_content="它的營收增長了 20%。",
            metadata={"page": 1}
        )
        
        mock_response = MagicMock()
        mock_response.content = "本段討論 Tesla 的財務表現。"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.context_enricher.get_llm", return_value=mock_llm):
            result = await enricher.enrich_chunk(doc, "財務報告")
        
        assert "<context>" in result.page_content
        assert "</context>" in result.page_content
        assert "Tesla" in result.page_content
        assert result.metadata["has_context_prefix"] == True

    @pytest.mark.asyncio
    async def test_enrich_chunk_preserves_original_content(self):
        """Tests that original content is preserved."""
        enricher = ContextEnricher()
        
        original_content = "這是原始內容，應該被保留。"
        doc = Document(page_content=original_content, metadata={})
        
        mock_response = MagicMock()
        mock_response.content = "上下文說明"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.context_enricher.get_llm", return_value=mock_llm):
            result = await enricher.enrich_chunk(doc, "文檔")
        
        assert original_content in result.page_content

    @pytest.mark.asyncio
    async def test_enrich_chunk_error_returns_original(self):
        """Tests that errors return original document."""
        enricher = ContextEnricher()
        
        doc = Document(
            page_content="測試內容",
            metadata={"page": 1}
        )
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        with patch("data_base.context_enricher.get_llm", return_value=mock_llm):
            result = await enricher.enrich_chunk(doc, "文檔")
        
        # Should return original on error
        assert result.page_content == "測試內容"
        assert result.metadata.get("has_context_prefix") is None


class TestEnrichChunks:
    """Tests for the enrich_chunks method."""

    @pytest.mark.asyncio
    async def test_enrich_chunks_skips_short(self):
        """Tests that short chunks are skipped."""
        enricher = ContextEnricher()
        
        docs = [
            Document(page_content="短", metadata={"id": 1}),
            Document(page_content="這是足夠長的內容，應該被處理。", metadata={"id": 2}),
        ]
        
        mock_response = MagicMock()
        mock_response.content = "上下文"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.context_enricher.get_llm", return_value=mock_llm):
            result = await enricher.enrich_chunks(docs, "文檔", skip_if_short=10)
        
        assert len(result) == 2
        # First doc should be unchanged (too short)
        assert result[0].page_content == "短"
        # Second doc should be enriched
        assert "<context>" in result[1].page_content

    @pytest.mark.asyncio
    async def test_enrich_chunks_empty_list(self):
        """Tests enriching empty list."""
        enricher = ContextEnricher()
        
        result = await enricher.enrich_chunks([], "文檔")
        assert result == []


class TestEnrichDocumentsWithContext:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_original(self, sample_documents):
        """Tests that disabled enrichment returns unchanged."""
        result = await enrich_documents_with_context(
            sample_documents,
            document_title="測試",
            enabled=False,
        )
        
        assert result == sample_documents

    @pytest.mark.asyncio
    async def test_enabled_enriches_documents(self):
        """Tests that enabled enrichment processes documents."""
        # Content must be > 50 chars (skip_if_short default)
        long_content = "這是一段需要被增強的長文字內容。" * 5  # ~80+ chars
        docs = [
            Document(
                page_content=long_content,
                metadata={"page": 1}
            )
        ]
        
        mock_response = MagicMock()
        mock_response.content = "上下文說明"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.context_enricher.get_llm", return_value=mock_llm):
            result = await enrich_documents_with_context(
                docs,
                document_title="測試文檔",
                enabled=True,
            )
        
        assert len(result) == 1
        assert "<context>" in result[0].page_content


