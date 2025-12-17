"""
Parent-Child Document Store

Manages hierarchical document storage for parent-child retrieval patterns.
Small chunks are indexed for precision, parent chunks are returned for context.
"""

# Standard library
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

# Third-party
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

# Base folder for parent document storage
BASE_UPLOAD_FOLDER = "uploads"


def _get_parent_store_path(user_id: str) -> str:
    """
    Returns the path to user's parent document store.
    
    Args:
        user_id: User's ID.
        
    Returns:
        Path to parent_docs.pkl file.
    """
    return os.path.normpath(
        os.path.join(BASE_UPLOAD_FOLDER, user_id, "rag_index", "parent_docs.pkl")
    )


class ParentDocumentStore:
    """
    Manages parent documents for hierarchical retrieval.
    
    Stores full-context parent documents that can be retrieved
    when their child chunks are matched during vector search.
    """
    
    def __init__(self, user_id: str) -> None:
        """
        Initializes the parent document store for a user.
        
        Args:
            user_id: User's ID.
        """
        self._user_id = user_id
        self._store_path = _get_parent_store_path(user_id)
        self._documents: Dict[str, Document] = {}
        self._load()
    
    def _load(self) -> None:
        """Loads parent documents from disk."""
        if os.path.exists(self._store_path):
            try:
                with open(self._store_path, "rb") as f:
                    self._documents = pickle.load(f)
                logger.debug(f"Loaded {len(self._documents)} parent documents")
            except (pickle.PickleError, EOFError, IOError) as e:
                logger.warning(f"Failed to load parent store: {e}")
                self._documents = {}
        else:
            self._documents = {}
    
    def _save(self) -> None:
        """Saves parent documents to disk."""
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        try:
            with open(self._store_path, "wb") as f:
                pickle.dump(self._documents, f)
            logger.debug(f"Saved {len(self._documents)} parent documents")
        except (pickle.PickleError, IOError) as e:
            logger.error(f"Failed to save parent store: {e}")
    
    def add_parent(self, parent_id: str, document: Document) -> None:
        """
        Adds a parent document to the store.
        
        Args:
            parent_id: Unique identifier for the parent document.
            document: The parent Document object.
        """
        self._documents[parent_id] = document
        self._save()
    
    def add_parents(self, parents: Dict[str, Document]) -> None:
        """
        Adds multiple parent documents at once.
        
        Args:
            parents: Dictionary mapping parent_id to Document.
        """
        self._documents.update(parents)
        self._save()
        logger.info(f"Added {len(parents)} parent documents")
    
    def get_parent(self, parent_id: str) -> Optional[Document]:
        """
        Retrieves a parent document by ID.
        
        Args:
            parent_id: The parent document ID.
            
        Returns:
            The parent Document, or None if not found.
        """
        return self._documents.get(parent_id)
    
    def get_parents_for_children(
        self,
        child_documents: List[Document]
    ) -> List[Document]:
        """
        Retrieves parent documents for a list of child documents.
        
        Uses the 'parent_id' metadata field in child documents
        to look up corresponding parents.
        
        Args:
            child_documents: List of child Documents from retrieval.
            
        Returns:
            List of parent Documents (deduplicated).
        """
        parent_ids_seen = set()
        parents = []
        
        for child in child_documents:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids_seen:
                parent = self.get_parent(parent_id)
                if parent:
                    parents.append(parent)
                    parent_ids_seen.add(parent_id)
                else:
                    # Parent not found, use child as fallback
                    parents.append(child)
                    parent_ids_seen.add(parent_id)
        
        return parents
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Deletes all parent documents for a specific document ID.
        
        Args:
            doc_id: The document ID to delete.
            
        Returns:
            Number of parent documents deleted.
        """
        to_delete = [
            pid for pid, doc in self._documents.items()
            if doc.metadata.get("original_doc_uid") == doc_id or
               doc.metadata.get("doc_id") == doc_id
        ]
        
        for pid in to_delete:
            del self._documents[pid]
        
        if to_delete:
            self._save()
            logger.info(f"Deleted {len(to_delete)} parent documents for doc {doc_id}")
        
        return len(to_delete)
    
    def clear(self) -> None:
        """Clears all parent documents."""
        self._documents.clear()
        self._save()
        logger.info("Parent document store cleared")


def create_parent_child_chunks(
    documents: List[Document],
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 400,
) -> Tuple[Dict[str, Document], List[Document]]:
    """
    Creates parent and child chunks from documents.
    
    Parent chunks are larger for context, child chunks are smaller for precision.
    Each child chunk has a 'parent_id' metadata field linking to its parent.
    
    Args:
        documents: Original documents to split.
        parent_chunk_size: Target size for parent chunks.
        child_chunk_size: Target size for child chunks.
        
    Returns:
        Tuple of (parent_dict, child_list) where:
        - parent_dict maps parent_id to parent Document
        - child_list contains child Documents with parent_id metadata
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    parents: Dict[str, Document] = {}
    children: List[Document] = []
    
    # First, create parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    
    for doc_idx, doc in enumerate(documents):
        # Create parent chunks from document
        parent_chunks = parent_splitter.split_documents([doc])
        
        for p_idx, parent in enumerate(parent_chunks):
            parent_id = f"{doc.metadata.get('original_doc_uid', 'doc')}_{doc_idx}_parent_{p_idx}"
            
            # Store parent
            parent.metadata["chunk_type"] = "parent"
            parent.metadata["parent_id"] = parent_id
            parents[parent_id] = parent
            
            # Create child chunks from parent
            child_chunks = child_splitter.split_documents([parent])
            
            for c_idx, child in enumerate(child_chunks):
                child.metadata["chunk_type"] = "child"
                child.metadata["parent_id"] = parent_id
                child.metadata["child_index"] = c_idx
                children.append(child)
    
    logger.info(f"Created {len(parents)} parents and {len(children)} children")
    return parents, children
