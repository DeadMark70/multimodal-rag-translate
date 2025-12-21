"""
GraphRAG Local Search Module

Provides entity-centric search that expands from identified entities
in the query to their graph neighbors.
"""

# Standard library
import logging
from typing import List, Optional, Tuple

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm
from graph_rag.schemas import EntityType, GraphNode
from graph_rag.store import GraphStore

# Configure logging
logger = logging.getLogger(__name__)

# Prompt for entity identification in query
_ENTITY_IDENTIFICATION_PROMPT = """請從以下問題中識別可能存在於知識圖譜中的關鍵實體。

問題：{question}

只輸出實體名稱，每行一個，不要其他文字："""


async def identify_query_entities(question: str) -> List[str]:
    """
    Identify potential entities in a user question.
    
    Args:
        question: User's question.
        
    Returns:
        List of entity names mentioned in the question.
    """
    try:
        llm = get_llm("graph_extraction")
        prompt = _ENTITY_IDENTIFICATION_PROMPT.format(question=question)
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse response (one entity per line)
        entities = []
        for line in response.content.strip().split("\n"):
            entity = line.strip().strip("-").strip("•").strip()
            if entity and len(entity) > 1:
                entities.append(entity)
        
        logger.debug(f"Identified entities from query: {entities}")
        return entities
        
    except Exception as e:
        logger.warning(f"Entity identification failed: {e}")
        return []


def find_matching_nodes(
    store: GraphStore,
    entity_labels: List[str],
    fuzzy: bool = True,
) -> List[str]:
    """
    Find graph nodes matching the given entity labels.
    
    Args:
        store: GraphStore to search.
        entity_labels: Entity names to find.
        fuzzy: Use fuzzy matching.
        
    Returns:
        List of matching node IDs.
    """
    matched_nodes = []
    
    for label in entity_labels:
        nodes = store.find_nodes_by_label(label, fuzzy=fuzzy)
        matched_nodes.extend(nodes)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for node_id in matched_nodes:
        if node_id not in seen:
            seen.add(node_id)
            unique.append(node_id)
    
    return unique


def expand_to_neighbors(
    store: GraphStore,
    node_ids: List[str],
    hops: int = 1,
    max_nodes: int = 30,
) -> List[str]:
    """
    Expand from seed nodes to their neighbors.
    
    Args:
        store: GraphStore to search.
        node_ids: Starting node IDs.
        hops: Number of hops to expand.
        max_nodes: Maximum nodes to return.
        
    Returns:
        List of all relevant node IDs (including original).
    """
    all_nodes = set(node_ids)
    
    for node_id in node_ids:
        neighbors = store.get_neighbors(node_id, hops=hops, max_nodes=max_nodes)
        all_nodes.update(neighbors)
        
        if len(all_nodes) >= max_nodes:
            break
    
    return list(all_nodes)[:max_nodes]


def build_local_context(
    store: GraphStore,
    node_ids: List[str],
) -> str:
    """
    Build context string from local graph neighborhood.
    
    Args:
        store: GraphStore.
        node_ids: Node IDs to include.
        
    Returns:
        Formatted context string for LLM.
    """
    if not node_ids:
        return ""
    
    lines = ["=== 知識圖譜上下文 ===\n"]
    
    # Add node information
    lines.append("相關實體：")
    for node_id in node_ids[:20]:
        node = store.get_node(node_id)
        if node:
            line = f"• {node.label} ({node.entity_type.value})"
            if node.description:
                line += f": {node.description}"
            lines.append(line)
    
    # Add relationship information
    lines.append("\n關係：")
    node_set = set(node_ids)
    seen_edges = set()
    
    for node_id in node_ids[:15]:
        edges = store.get_edges_for_node(node_id)
        for edge in edges:
            # Only include edges within the node set
            if edge.source_id in node_set and edge.target_id in node_set:
                edge_key = (edge.source_id, edge.target_id, edge.relation)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    
                    source = store.get_node(edge.source_id)
                    target = store.get_node(edge.target_id)
                    
                    if source and target:
                        line = f"• {source.label} [{edge.relation}] {target.label}"
                        if edge.description:
                            line += f" ({edge.description})"
                        lines.append(line)
    
    return "\n".join(lines)


async def local_search(
    store: GraphStore,
    question: str,
    hops: int = 2,
    max_nodes: int = 30,
) -> Tuple[str, List[str]]:
    """
    Perform local graph search based on query entities.
    
    Args:
        store: GraphStore to search.
        question: User's question.
        hops: Number of hops for neighbor expansion.
        max_nodes: Maximum nodes to include.
        
    Returns:
        Tuple of (context_string, matched_node_ids).
    """
    # Step 1: Identify entities in question
    query_entities = await identify_query_entities(question)
    
    if not query_entities:
        logger.info("No entities identified in query, returning empty context")
        return "", []
    
    # Step 2: Find matching nodes
    matched_nodes = find_matching_nodes(store, query_entities, fuzzy=True)
    
    if not matched_nodes:
        logger.info("No matching nodes found in graph")
        return "", []
    
    logger.info(f"Found {len(matched_nodes)} matching nodes for query")
    
    # Step 3: Expand to neighbors
    expanded_nodes = expand_to_neighbors(store, matched_nodes, hops=hops, max_nodes=max_nodes)
    
    # Step 4: Build context
    context = build_local_context(store, expanded_nodes)
    
    return context, expanded_nodes


def local_search_by_node_ids(
    store: GraphStore,
    node_ids: List[str],
    hops: int = 1,
    max_nodes: int = 30,
) -> str:
    """
    Perform local search starting from specific node IDs.
    
    Synchronous version for direct node ID lookup.
    
    Args:
        store: GraphStore to search.
        node_ids: Starting node IDs.
        hops: Number of hops to expand.
        max_nodes: Maximum nodes.
        
    Returns:
        Context string.
    """
    expanded = expand_to_neighbors(store, node_ids, hops=hops, max_nodes=max_nodes)
    return build_local_context(store, expanded)
