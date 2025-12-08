"""
Unit Tests for Proposition Chunker

Tests the PropositionChunker class and related functions.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
from langchain.schema import Document

# Local application
from data_base.proposition_chunker import (
    PropositionChunker,
    extract_propositions_from_documents,
)


class TestPropositionChunkerParsing:
    """Tests for the proposition parsing logic."""

    def test_parse_numbered_propositions(self):
        """Tests parsing numbered proposition list."""
        chunker = PropositionChunker()
        
        response = """1. Tesla 在 2023 年營收增長了 20%。
2. Tesla 在歐洲的市場份額下降。
3. 這是第三個命題，內容較長一些。"""
        
        propositions = chunker._parse_propositions(response)
        
        assert len(propositions) == 3
        assert "Tesla" in propositions[0]
        assert "營收" in propositions[0]
        assert "歐洲" in propositions[1]

    def test_parse_with_parenthesis_numbers(self):
        """Tests parsing propositions with parenthesis numbers."""
        chunker = PropositionChunker()
        
        response = """1) 第一個命題說明了重要事實。
2) 第二個命題描述了另一個事實。"""
        
        propositions = chunker._parse_propositions(response)
        
        assert len(propositions) == 2

    def test_parse_filters_short_lines(self):
        """Tests that very short lines are filtered out."""
        chunker = PropositionChunker()
        
        response = """1. 這是一個有效的命題陳述。
2. 太短
3. 這是另一個有效的長命題。"""
        
        propositions = chunker._parse_propositions(response)
        
        # "太短" should be filtered (< 10 chars)
        assert len(propositions) == 2

    def test_parse_empty_response(self):
        """Tests parsing empty response."""
        chunker = PropositionChunker()
        
        propositions = chunker._parse_propositions("")
        assert propositions == []

    def test_parse_filters_headers(self):
        """Tests that header lines are filtered."""
        chunker = PropositionChunker()
        
        response = """# 這是標題
1. 這是第一個命題說明，內容較長一些。
## 這是子標題
2. 這是第二個命題說明，也很長。"""
        
        propositions = chunker._parse_propositions(response)
        
        # Headers starting with # should be filtered, numbered items extracted
        assert len(propositions) == 2
        assert not any(p.startswith("#") for p in propositions)


class TestPropositionChunkerExtraction:
    """Tests for the proposition extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_propositions_short_text(self):
        """Tests that short text is returned as-is."""
        chunker = PropositionChunker()
        
        doc = Document(
            page_content="短文",
            metadata={"page": 1}
        )
        
        result = await chunker.extract_propositions(doc, min_text_length=50)
        
        assert len(result) == 1
        assert result[0].page_content == "短文"

    @pytest.mark.asyncio
    async def test_extract_propositions_with_mock_llm(self):
        """Tests proposition extraction with mocked LLM."""
        chunker = PropositionChunker()
        
        doc = Document(
            page_content="Tesla 在 2023 年營收增長 20%，同時歐洲市場份額下降。這是一段足夠長的文字來觸發命題提取。",
            metadata={"page": 1, "unique_chunk_id": "chunk-1"}
        )
        
        mock_response = MagicMock()
        mock_response.content = """1. Tesla 在 2023 年營收增長了 20%。
2. Tesla 歐洲市場份額下降。"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.proposition_chunker.get_llm", return_value=mock_llm):
            result = await chunker.extract_propositions(doc, min_text_length=20)
        
        assert len(result) == 2
        for prop_doc in result:
            assert prop_doc.metadata["is_proposition"] == True
            assert prop_doc.metadata["parent_chunk_id"] == "chunk-1"

    @pytest.mark.asyncio
    async def test_extract_propositions_error_handling(self):
        """Tests that errors return original document."""
        chunker = PropositionChunker()
        
        doc = Document(
            page_content="這是一段足夠長的測試文字，用於測試錯誤處理邏輯。",
            metadata={"page": 1}
        )
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        with patch("data_base.proposition_chunker.get_llm", return_value=mock_llm):
            result = await chunker.extract_propositions(doc, min_text_length=10)
        
        # Should return original document on error
        assert len(result) == 1
        assert result[0].page_content == doc.page_content


class TestExtractPropositionsFromDocuments:
    """Tests for the batch extraction function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_original(self, sample_documents):
        """Tests that disabled extraction returns documents unchanged."""
        result = await extract_propositions_from_documents(
            sample_documents,
            enabled=False,
        )
        
        assert result == sample_documents

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Tests extraction from empty list."""
        result = await extract_propositions_from_documents([], enabled=True)
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_extraction_with_mock(self):
        """Tests batch extraction with mocked LLM."""
        docs = [
            Document(page_content="短", metadata={"page": 1}),
            Document(page_content="這是一段足夠長的文字來觸發命題提取。", metadata={"page": 2}),
        ]
        
        mock_response = MagicMock()
        mock_response.content = "1. 這是一個命題說明文字。"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.proposition_chunker.get_llm", return_value=mock_llm):
            result = await extract_propositions_from_documents(
                docs,
                enabled=True,
                min_text_length=10,
            )
        
        # First doc should be unchanged (too short)
        # Second doc should have propositions
        assert len(result) >= 2
