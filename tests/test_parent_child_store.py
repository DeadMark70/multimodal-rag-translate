"""
Unit Tests for Parent-Child Document Store

Tests the ParentDocumentStore class and related functions.
"""

# Standard library
import os
import pytest
import tempfile
from unittest.mock import patch

# Third-party
from langchain.schema import Document

# Local application
from data_base.parent_child_store import (
    ParentDocumentStore,
    create_parent_child_chunks,
    _get_parent_store_path,
)


class TestGetParentStorePath:
    """Tests for the path helper function."""

    def test_returns_correct_path(self):
        """Tests that correct path is returned."""
        path = _get_parent_store_path("test-user")
        
        assert "test-user" in path
        assert "rag_index" in path
        assert "parent_docs.pkl" in path


class TestParentDocumentStore:
    """Tests for the ParentDocumentStore class."""

    def test_init_creates_empty_store(self, tmp_path):
        """Tests initialization with no existing store."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("new-user")
            
            assert len(store._documents) == 0

    def test_add_parent(self, tmp_path):
        """Tests adding a single parent document."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            doc = Document(
                page_content="Parent content",
                metadata={"doc_id": "doc-1"}
            )
            
            store.add_parent("parent-1", doc)
            
            assert "parent-1" in store._documents
            assert store._documents["parent-1"].page_content == "Parent content"

    def test_add_parents_batch(self, tmp_path):
        """Tests adding multiple parents at once."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            parents = {
                "p1": Document(page_content="Content 1", metadata={}),
                "p2": Document(page_content="Content 2", metadata={}),
                "p3": Document(page_content="Content 3", metadata={}),
            }
            
            store.add_parents(parents)
            
            assert len(store._documents) == 3
            assert store.get_parent("p2").page_content == "Content 2"

    def test_get_parent(self, tmp_path):
        """Tests retrieving a parent document."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            doc = Document(page_content="Test", metadata={})
            store.add_parent("p1", doc)
            
            retrieved = store.get_parent("p1")
            
            assert retrieved.page_content == "Test"

    def test_get_parent_not_found(self, tmp_path):
        """Tests retrieving non-existent parent."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            result = store.get_parent("nonexistent")
            
            assert result is None

    def test_get_parents_for_children(self, tmp_path):
        """Tests looking up parents from child documents."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            # Add parent
            parent = Document(
                page_content="Full parent content",
                metadata={"type": "parent"}
            )
            store.add_parent("parent-1", parent)
            
            # Create children with parent_id
            children = [
                Document(page_content="Child 1", metadata={"parent_id": "parent-1"}),
                Document(page_content="Child 2", metadata={"parent_id": "parent-1"}),
                Document(page_content="Child 3", metadata={"parent_id": "nonexistent"}),
            ]
            
            result = store.get_parents_for_children(children)
            
            # Should get parent for first two, fallback for third
            assert len(result) == 2  # Deduplicated
            assert result[0].page_content == "Full parent content"

    def test_delete_by_doc_id(self, tmp_path):
        """Tests deleting parents by doc_id."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            # Add multiple parents
            store.add_parent("p1", Document(
                page_content="Doc 1 Parent 1",
                metadata={"original_doc_uid": "doc-1"}
            ))
            store.add_parent("p2", Document(
                page_content="Doc 1 Parent 2",
                metadata={"original_doc_uid": "doc-1"}
            ))
            store.add_parent("p3", Document(
                page_content="Doc 2 Parent 1",
                metadata={"original_doc_uid": "doc-2"}
            ))
            
            # Delete doc-1's parents
            deleted = store.delete_by_doc_id("doc-1")
            
            assert deleted == 2
            assert store.get_parent("p1") is None
            assert store.get_parent("p2") is None
            assert store.get_parent("p3") is not None

    def test_clear(self, tmp_path):
        """Tests clearing all parents."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            store = ParentDocumentStore("test-user")
            
            store.add_parent("p1", Document(page_content="Test", metadata={}))
            store.add_parent("p2", Document(page_content="Test", metadata={}))
            
            store.clear()
            
            assert len(store._documents) == 0

    def test_persistence(self, tmp_path):
        """Tests that store persists across instances."""
        with patch("data_base.parent_child_store.BASE_UPLOAD_FOLDER", str(tmp_path)):
            # Create and save
            store1 = ParentDocumentStore("test-user")
            store1.add_parent("p1", Document(page_content="Persisted", metadata={}))
            
            # Load in new instance
            store2 = ParentDocumentStore("test-user")
            
            assert store2.get_parent("p1").page_content == "Persisted"


class TestCreateParentChildChunks:
    """Tests for the create_parent_child_chunks function."""

    def test_creates_hierarchy(self):
        """Tests that parent-child hierarchy is created."""
        documents = [
            Document(
                page_content="這是一段很長的文字內容。" * 50,  # Long text
                metadata={"original_doc_uid": "doc-1", "page": 1}
            )
        ]
        
        parents, children = create_parent_child_chunks(
            documents,
            parent_chunk_size=500,
            child_chunk_size=100,
        )
        
        assert len(parents) > 0
        assert len(children) > 0
        assert len(children) >= len(parents)

    def test_children_have_parent_id(self):
        """Tests that children have parent_id metadata."""
        documents = [
            Document(
                page_content="測試文字內容。" * 100,
                metadata={"original_doc_uid": "doc-1"}
            )
        ]
        
        parents, children = create_parent_child_chunks(
            documents,
            parent_chunk_size=500,
            child_chunk_size=100,
        )
        
        for child in children:
            assert "parent_id" in child.metadata
            assert child.metadata["chunk_type"] == "child"

    def test_parents_have_correct_type(self):
        """Tests that parents have correct chunk_type."""
        documents = [
            Document(
                page_content="測試內容。" * 50,
                metadata={"original_doc_uid": "doc-1"}
            )
        ]
        
        parents, children = create_parent_child_chunks(
            documents,
            parent_chunk_size=200,
            child_chunk_size=50,
        )
        
        for parent in parents.values():
            assert parent.metadata["chunk_type"] == "parent"

    def test_empty_documents(self):
        """Tests with empty document list."""
        parents, children = create_parent_child_chunks([])
        
        assert parents == {}
        assert children == []
