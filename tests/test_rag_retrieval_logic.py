import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from data_base.RAG_QA_service import _expand_short_chunks, _should_use_graph_search

def test_should_use_graph_search():
    """Verify keyword-based graph search detection."""
    assert _should_use_graph_search("這幾篇論文有什麼關係？") is True
    assert _should_use_graph_search("Transformer 的定義是什麼？") is False
    assert _should_use_graph_search("分析跨文件的趨勢") is True

def test_expand_short_chunks_logic():
    """
    Test the Context Enricher logic.
    Verify that short chunks are expanded if parent exists.
    """
    # Setup mock documents
    short_doc = Document(
        page_content="Very short.",
        metadata={"parent_id": "p1", "doc_id": "d1"}
    )
    long_doc = Document(
        page_content="This is already a long enough document chunk that does not need expansion." * 5,
        metadata={"parent_id": "p2", "doc_id": "d1"}
    )
    
    docs = [short_doc, long_doc]
    user_id = "test_user"
    
    # Mock ParentDocumentStore
    with patch("data_base.RAG_QA_service.ParentDocumentStore") as MockStore:
        mock_instance = MockStore.return_value
        # Mock parent content
        mock_parent = Document(page_content="This is the full parent content which is much longer than the original short chunk.")
        mock_instance.get_parent.side_effect = lambda pid: mock_parent if pid == "p1" else None
        
        expanded_docs = _expand_short_chunks(docs, user_id)
        
        assert len(expanded_docs) == 2
        # First doc should be expanded
        assert expanded_docs[0].page_content == mock_parent.page_content
        assert expanded_docs[0].metadata["expanded_from_parent"] is True
        # Second doc should remain same
        assert expanded_docs[1].page_content == long_doc.page_content

def test_expand_short_chunks_token_limit():
    """
    Verify that expansion respects _MAX_TOTAL_CHARS limit.
    """
    from data_base.RAG_QA_service import _MAX_TOTAL_CHARS
    
    # Create a doc that is just below the limit, and another short one
    large_content = "A" * (_MAX_TOTAL_CHARS - 50)
    large_doc = Document(page_content=large_content, metadata={"doc_id": "d1"})
    short_doc = Document(page_content="short", metadata={"parent_id": "p1", "doc_id": "d1"})
    
    docs = [large_doc, short_doc]
    user_id = "test_user"
    
    with patch("data_base.RAG_QA_service.ParentDocumentStore") as MockStore:
        mock_instance = MockStore.return_value
        # Parent is very large, would exceed limit
        mock_parent = Document(page_content="B" * 1000)
        mock_instance.get_parent.return_value = mock_parent
        
        expanded_docs = _expand_short_chunks(docs, user_id)
        
        # Total chars would be (15000-50) + 1000 = 15950 > 15000
        # So it should NOT expand
        assert expanded_docs[1].page_content == "short"
        assert "expanded_from_parent" not in expanded_docs[1].metadata
