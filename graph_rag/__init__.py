"""
GraphRAG Module

Provides knowledge graph-based retrieval augmented generation
for multi-document research and deep analysis.
"""

# Schemas - always safe to import
from graph_rag.schemas import (
    EntityType,
    RelationType,
    GraphNode,
    GraphEdge,
    Community,
    GraphStatusResponse,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

# Store - no complex dependencies
from graph_rag.store import GraphStore

__all__ = [
    # Schemas
    "EntityType",
    "RelationType",
    "GraphNode",
    "GraphEdge",
    "Community",
    "GraphStatusResponse",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    # Store
    "GraphStore",
    # Functions (lazy imports below)
    "get_extractor",
    "extract_from_chunk",
    "add_extraction_to_graph",
    "resolve_entities",
    "build_communities",
    "local_search",
    "global_search",
]


# ===== Lazy Import Functions =====
# These avoid circular dependencies and heavy imports at module load time

def get_extractor():
    """Get EntityRelationExtractor instance."""
    from graph_rag.extractor import EntityRelationExtractor
    return EntityRelationExtractor()


async def extract_from_chunk(text: str, doc_id: str, chunk_index: int = 0):
    """Extract entities and relations from a text chunk."""
    from graph_rag.extractor import extract_from_chunk as _extract
    return await _extract(text, doc_id, chunk_index)


async def add_extraction_to_graph(store: GraphStore, result: ExtractionResult):
    """Add extraction results to graph."""
    from graph_rag.extractor import add_extraction_to_graph as _add
    return await _add(store, result)


async def resolve_entities(store: GraphStore):
    """Resolve (deduplicate) entities in graph."""
    from graph_rag.entity_resolver import resolve_entities as _resolve
    return await _resolve(store)


async def build_communities(store: GraphStore, generate_summaries: bool = True):
    """Build communities from graph."""
    from graph_rag.community_builder import build_communities as _build
    return await _build(store, generate_summaries)


async def local_search(store: GraphStore, question: str, hops: int = 2, max_nodes: int = 30):
    """Perform entity-centric local search."""
    from graph_rag.local_search import local_search as _local
    return await _local(store, question, hops, max_nodes)


async def global_search(store: GraphStore, question: str, max_communities: int = 5):
    """Perform community-based global search."""
    from graph_rag.global_search import global_search as _global
    return await _global(store, question, max_communities)
