"""
Integration Tests for Phase 1

Tests that all Phase 1 components work together correctly.
"""

# Standard library
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
from langchain_core.documents import Document

# Local application
from data_base.word_chunk_strategy import split_markdown


class TestWordChunkStrategyIntegration:
    """Integration tests for word_chunk_strategy module."""

    @pytest.mark.asyncio
    async def test_split_markdown_recursive_default(self):
        """Tests default recursive chunking still works."""
        markdown = """[[PAGE_1]]
# 標題一

這是第一頁的內容。包含一些重要的資訊。

[[PAGE_2]]
# 標題二

這是第二頁的內容。
"""
        
        chunks = await split_markdown(
            markdown,
            pdf_title="測試文檔",
            original_doc_uid="doc-123",
            chunk_size=100,
            overlap=20,
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "book_title" in chunk.metadata
            assert "original_doc_uid" in chunk.metadata
            assert chunk.metadata["original_doc_uid"] == "doc-123"

    @pytest.mark.asyncio
    async def test_split_markdown_semantic_with_embeddings(self, mock_embeddings):
        """Tests semantic chunking with mock embeddings."""
        markdown = """[[PAGE_1]]
人工智慧是計算機科學的重要分支。它致力於創建智能系統。

機器學習是 AI 的核心技術。深度學習是機器學習的延伸。

[[PAGE_2]]
自然語言處理讓機器理解人類語言。這是一個複雜的任務。
"""
        
        chunks = await split_markdown(
            markdown,
            pdf_title="AI 導論",
            original_doc_uid="ai-doc",
            chunking_method="semantic",
            embeddings=mock_embeddings,
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["book_title"] == "AI 導論"

    @pytest.mark.asyncio
    async def test_split_markdown_no_page_markers(self):
        """Tests handling text without page markers."""
        markdown = """# 文檔標題

這是沒有頁標記的文檔內容。

## 第一節

這是第一節的內容。

## 第二節

這是第二節的內容。
"""
        
        chunks = await split_markdown(
            markdown,
            pdf_title="無頁碼文檔",
            original_doc_uid="no-page-doc",
        )
        
        assert len(chunks) > 0
        # All chunks should have page_number = 1
        for chunk in chunks:
            assert chunk.metadata["page_number"] == 1

    @pytest.mark.asyncio
    async def test_split_markdown_semantic_requires_embeddings(self):
        """Tests that semantic chunking requires embeddings."""
        with pytest.raises(ValueError, match="embeddings"):
            await split_markdown(
                "測試內容",
                pdf_title="測試",
                original_doc_uid="test",
                chunking_method="semantic",
                embeddings=None,  # Missing embeddings
            )


class TestFullPipelineIntegration:
    """Tests full Phase 1 pipeline integration."""

    @pytest.mark.asyncio
    async def test_chunking_then_enrichment(self, mock_embeddings):
        """Tests chunking followed by context enrichment."""
        from data_base.context_enricher import enrich_documents_with_context
        
        # Step 1: Chunk the document
        markdown = """[[PAGE_1]]
Tesla 是一家電動車公司。它由 Elon Musk 領導。

該公司在 2023 年表現強勁。營收大幅成長。
"""
        
        chunks = await split_markdown(
            markdown,
            pdf_title="Tesla 報告",
            original_doc_uid="tesla-report",
            chunking_method="semantic",
            embeddings=mock_embeddings,
        )
        
        # Step 2: Enrich with context
        mock_response = MagicMock()
        mock_response.content = "本段討論電動車公司 Tesla。"
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.context_enricher.get_llm", return_value=mock_llm):
            enriched = await enrich_documents_with_context(
                chunks,
                document_title="Tesla 報告",
                enabled=True,
            )
        
        assert len(enriched) > 0

    @pytest.mark.asyncio
    async def test_parent_child_with_proposition_indexing(self, tmp_path):
        """Tests parent-child indexing with proposition extraction."""
        from data_base.parent_child_store import (
            ParentDocumentStore,
            create_parent_child_chunks,
        )
        from data_base.proposition_chunker import extract_propositions_from_documents
        
        # Step 1: Create documents
        documents = [
            Document(
                page_content="Tesla 營收增長 20%，同時歐洲市場份額下降。公司計劃擴大產能。" * 5,
                metadata={"original_doc_uid": "tesla", "page": 1}
            )
        ]
        
        # Step 2: Create parent-child hierarchy
        parents, children = create_parent_child_chunks(
            documents,
            parent_chunk_size=200,
            child_chunk_size=50,
        )
        
        assert len(parents) > 0
        assert len(children) > 0
        
        # Step 3: Store parents
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            store.add_parents(parents)
            
            # Verify storage
            assert len(store._documents) == len(parents)
        
        # Step 4: Extract propositions from children (mocked)
        mock_response = MagicMock()
        mock_response.content = """1. Tesla 營收增長了 20%。
2. Tesla 歐洲市場份額下降。
3. Tesla 計劃擴大產能。"""
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("data_base.proposition_chunker.get_llm", return_value=mock_llm):
            propositions = await extract_propositions_from_documents(
                children[:1],  # Just first child
                enabled=True,
                min_text_length=20,
            )
        
        # Should have extracted some propositions
        assert len(propositions) >= 1


class TestBackwardCompatibility:
    """Tests that existing code still works."""

    @pytest.mark.asyncio
    async def test_old_function_signature_works(self):
        """Tests that old function calls still work."""
        # Old signature without new parameters
        chunks = await split_markdown(
            "[[PAGE_1]]\n這是測試內容。",
            "測試.pdf",
            "doc-uuid",
        )
        
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_explicit_recursive_still_works(self):
        """Tests explicit recursive method."""
        chunks = await split_markdown(
            "[[PAGE_1]]\n這是測試內容。" * 10,
            "測試.pdf",
            "doc-uuid",
            chunking_method="recursive",
            chunk_size=50,
            overlap=10,
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            # Default method doesn't add chunking_method metadata
            pass  # Just verify no errors
