"""
GraphRAG Entity Resolver Module

Provides entity resolution (deduplication) using embedding similarity.
Merges similar entities that represent the same concept across documents.
"""

# Standard library
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Third-party
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity score (0-1).
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def _string_similarity(s1: str, s2: str) -> float:
    """
    Calculate simple string similarity (Jaccard on character trigrams).
    
    Args:
        s1: First string.
        s2: Second string.
        
    Returns:
        Similarity score (0-1).
    """
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    
    if s1 == s2:
        return 1.0
    
    if len(s1) < 3 or len(s2) < 3:
        # For short strings, use simple containment
        if s1 in s2 or s2 in s1:
            return 0.8
        return 0.0
    
    # Generate trigrams
    def trigrams(s):
        return set(s[i:i+3] for i in range(len(s) - 2))
    
    t1 = trigrams(s1)
    t2 = trigrams(s2)
    
    if not t1 or not t2:
        return 0.0
    
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    
    return intersection / union if union > 0 else 0.0


class EntityResolver:
    """
    Resolves (deduplicates) similar entities in the graph.
    
    Uses a combination of:
    1. String similarity (for exact/near matches)
    2. Embedding similarity (for semantic matches)
    
    Attributes:
        similarity_threshold: Minimum similarity to consider as match.
        use_embeddings: Whether to use embedding-based matching.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_embeddings: bool = True,
    ) -> None:
        """
        Initialize the resolver.
        
        Args:
            similarity_threshold: Minimum similarity for merging.
            use_embeddings: Use embedding similarity if available.
        """
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self._embeddings_model = None
    
    def _get_embeddings_model(self):
        """Lazy load embeddings model."""
        if self._embeddings_model is None:
            try:
                from data_base.vector_store_manager import get_embeddings
                self._embeddings_model = get_embeddings()
            except Exception as e:
                logger.warning(f"Could not load embeddings model: {e}")
                self._embeddings_model = False  # Mark as unavailable
        return self._embeddings_model if self._embeddings_model else None
    
    async def compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute embedding for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector or None if unavailable.
        """
        model = self._get_embeddings_model()
        if model is None:
            return None
        
        try:
            embedding = await model.aembed_query(text)
            return embedding
        except Exception as e:
            logger.warning(f"Embedding failed for '{text[:50]}...': {e}")
            return None
    
    def compute_similarity(
        self,
        label1: str,
        label2: str,
        embedding1: Optional[List[float]] = None,
        embedding2: Optional[List[float]] = None,
    ) -> float:
        """
        Compute similarity between two entities.
        
        Uses string similarity as base, enhanced by embedding similarity
        if available.
        
        Args:
            label1: First entity label.
            label2: Second entity label.
            embedding1: First entity embedding (optional).
            embedding2: Second entity embedding (optional).
            
        Returns:
            Combined similarity score (0-1).
        """
        # String similarity (always computed)
        str_sim = _string_similarity(label1, label2)
        
        # If very high string similarity, return early
        if str_sim >= 0.95:
            return str_sim
        
        # Embedding similarity (if available)
        if self.use_embeddings and embedding1 and embedding2:
            emb_sim = _cosine_similarity(embedding1, embedding2)
            # Weighted combination: 40% string, 60% embedding
            return 0.4 * str_sim + 0.6 * emb_sim
        
        return str_sim
    
    def find_similar_pairs(
        self,
        nodes: List[Tuple[str, str, Optional[List[float]]]],
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of similar nodes.
        
        Args:
            nodes: List of (node_id, label, embedding) tuples.
            
        Returns:
            List of (node_id1, node_id2, similarity) tuples.
        """
        pairs = []
        
        for i, (id1, label1, emb1) in enumerate(nodes):
            for j, (id2, label2, emb2) in enumerate(nodes):
                if i >= j:  # Skip self and duplicates
                    continue
                
                similarity = self.compute_similarity(label1, label2, emb1, emb2)
                
                if similarity >= self.similarity_threshold:
                    pairs.append((id1, id2, similarity))
        
        return pairs
    
    def build_merge_groups(
        self,
        pairs: List[Tuple[str, str, float]],
    ) -> List[Set[str]]:
        """
        Build groups of nodes to merge using Union-Find.
        
        Args:
            pairs: List of (node_id1, node_id2, similarity) tuples.
            
        Returns:
            List of sets, each containing node IDs to merge.
        """
        # Union-Find data structure
        parent: Dict[str, str] = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Build unions
        for id1, id2, _ in pairs:
            union(id1, id2)
        
        # Group by root
        groups: Dict[str, Set[str]] = defaultdict(set)
        for node_id in parent:
            root = find(node_id)
            groups[root].add(node_id)
        
        # Filter to groups with > 1 member
        return [group for group in groups.values() if len(group) > 1]


async def resolve_entities(store: "GraphStore") -> int:
    """
    Resolve entities in a graph store.
    
    Finds similar entities and merges them by:
    1. Combining doc_ids
    2. Redirecting edges
    3. Removing duplicate nodes
    
    Args:
        store: GraphStore to resolve.
        
    Returns:
        Number of merges performed.
    """
    from graph_rag.store import GraphStore
    
    resolver = EntityResolver()
    
    # Get all pending nodes
    nodes = []
    for node in store.get_all_nodes():
        if node.pending_resolution:
            nodes.append((node.id, node.label, node.embedding))
    
    if len(nodes) < 2:
        logger.info("No pending entities to resolve")
        store.clear_pending_resolution()
        return 0
    
    logger.info(f"Resolving {len(nodes)} pending entities")
    
    # Find similar pairs
    pairs = resolver.find_similar_pairs(nodes)
    logger.info(f"Found {len(pairs)} similar pairs")
    
    # Build merge groups
    groups = resolver.build_merge_groups(pairs)
    logger.info(f"Built {len(groups)} merge groups")
    
    merges = 0
    
    for group in groups:
        group_list = list(group)
        if len(group_list) < 2:
            continue
        
        # Keep the first node, merge others into it
        primary_id = group_list[0]
        primary_node = store.get_node(primary_id)
        
        if not primary_node:
            continue
        
        # Get primary node data
        primary_docs = set(primary_node.doc_ids)
        
        # Merge other nodes
        for other_id in group_list[1:]:
            other_node = store.get_node(other_id)
            if not other_node:
                continue
            
            # Merge doc_ids
            primary_docs.update(other_node.doc_ids)
            
            # Get edges from other node
            other_edges = store.get_edges_for_node(other_id)
            
            # Redirect edges to primary
            for edge in other_edges:
                if edge.source_id == other_id:
                    store.add_edge_from_extraction(
                        source_id=primary_id,
                        target_id=edge.target_id,
                        relation=edge.relation,
                        doc_id=edge.doc_ids[0] if edge.doc_ids else "",
                        description=edge.description,
                        weight=edge.weight,
                    )
                elif edge.target_id == other_id:
                    store.add_edge_from_extraction(
                        source_id=edge.source_id,
                        target_id=primary_id,
                        relation=edge.relation,
                        doc_id=edge.doc_ids[0] if edge.doc_ids else "",
                        description=edge.description,
                        weight=edge.weight,
                    )
            
            # Remove the merged node (this also removes its edges)
            store.graph.remove_node(other_id)
            merges += 1
            
            logger.debug(f"Merged '{other_node.label}' into '{primary_node.label}'")
        
        # Update primary node's doc_ids
        store.graph.nodes[primary_id]["doc_ids"] = list(primary_docs)
    
    # Clear pending resolution flags
    store.clear_pending_resolution()
    
    logger.info(f"Entity resolution complete: {merges} nodes merged")
    return merges
