import inspect
import pytest
import sys
from unittest.mock import MagicMock

# Mock dependencies that might be missing in CI/Analysis environment
sys.modules["networkx"] = MagicMock()
sys.modules["leidenalg"] = MagicMock()
sys.modules["igraph"] = MagicMock()

from graph_rag import community_builder, schemas, store
import graph_rag.local_search as local_search_module

def test_community_builder_structure():
    """Verify community_builder module structure."""
    assert hasattr(community_builder, "detect_communities_leiden")
    assert hasattr(community_builder, "build_communities")
    
    sig = inspect.signature(community_builder.detect_communities_leiden)
    assert "store" in sig.parameters

def test_local_search_structure():
    """Verify local_search module structure."""
    assert hasattr(local_search_module, "local_search")
    assert hasattr(local_search_module, "identify_query_entities")
    
    sig = inspect.signature(local_search_module.local_search)
    assert "store" in sig.parameters
    assert "question" in sig.parameters

def test_schemas_structure():
    """Verify schemas module."""
    assert hasattr(schemas, "Community")
    assert hasattr(schemas, "GraphNode")
    assert hasattr(schemas, "GraphEdge")

def test_store_structure():
    """Verify store module."""
    assert hasattr(store, "GraphStore")
    # Check for basic methods
    assert hasattr(store.GraphStore, "add_node")
    assert hasattr(store.GraphStore, "get_node")

def test_leiden_fallback_check():
    """
    Check if the fallback logic is structurally present in source code.
    We don't strictly require leidenalg to be installed for this test to pass,
    but we want to ensure the code *handles* it.
    """
    source = inspect.getsource(community_builder.detect_communities_leiden)
    assert "import leidenalg" in source or "leidenalg" in source
    assert "except ImportError" in source
    assert "nx.connected_components" in source
