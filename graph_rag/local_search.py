"""
GraphRAG Local Search Module

Provides entity-centric search that expands from identified entities
in the query to their graph neighbors.
"""

# Standard library
import logging
import re
from typing import List, Tuple

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import graph_rag_llm_runtime_override
from core.providers import get_llm
from graph_rag.llm_response import response_content_to_text
from graph_rag.generic_mode import GraphEvidence, estimate_token_count
from graph_rag.node_vector_index import (
    node_vector_min_score,
    node_vector_search_enabled,
    node_vector_top_k,
    search_nodes_by_vector,
)
from graph_rag.store import GraphStore

# Configure logging
logger = logging.getLogger(__name__)

# Prompt for entity identification in query
_ENTITY_IDENTIFICATION_PROMPT = """請從以下問題中識別可能存在於知識圖譜中的關鍵實體。

問題：{question}

只輸出實體名稱，每行一個，不要其他文字："""


def _question_terms(question: str) -> set[str]:
    return {token for token in re.findall(r"[\w\-]+", question.lower()) if len(token) > 1}


def _text_overlap_score(question_terms: set[str], *values: str) -> float:
    if not question_terms:
        return 0.0
    haystack = " ".join(value.lower() for value in values if value)
    hits = sum(1 for token in question_terms if token in haystack)
    return hits / max(len(question_terms), 1)


async def identify_query_entities(question: str) -> List[str]:
    """
    Identify potential entities in a user question.
    
    Args:
        question: User's question.
        
    Returns:
        List of entity names mentioned in the question.
    """
    try:
        with graph_rag_llm_runtime_override("graph_extraction"):
            llm = get_llm("graph_extraction")
            prompt = _ENTITY_IDENTIFICATION_PROMPT.format(question=question)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse response (one entity per line)
        entities = []
        for line in response_content_to_text(response.content).split("\n"):
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
    question: str = "",
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
    ranked_candidates: list[tuple[float, str]] = []
    question_terms = _question_terms(question)

    for node_id in node_ids:
        neighbors = store.get_neighbors(node_id, hops=hops, max_nodes=max_nodes)
        for neighbor_id in neighbors:
            score = 0.0
            neighbor = store.get_node(neighbor_id)
            if neighbor:
                score += _text_overlap_score(
                    question_terms,
                    neighbor.label,
                    neighbor.description or "",
                )
                score += min(len(neighbor.doc_ids), 5) / 10
            ranked_candidates.append((score, neighbor_id))

    for _, neighbor_id in sorted(ranked_candidates, reverse=True):
        all_nodes.add(neighbor_id)
        if len(all_nodes) >= max_nodes:
            break

    return list(all_nodes)[:max_nodes]


def build_local_evidence(
    store: GraphStore,
    question: str,
    node_ids: List[str],
    max_edges: int = 14,
) -> List[GraphEvidence]:
    """Build scored local evidence units for generic mode merging."""
    evidence: list[GraphEvidence] = []
    question_terms = _question_terms(question)
    node_set = set(node_ids)

    for node_id in node_ids[:20]:
        node = store.get_node(node_id)
        if not node:
            continue
        text = f"{node.label} ({node.entity_type.value})"
        if node.description:
            text += f": {node.description}"
        score = 0.55 + _text_overlap_score(question_terms, node.label, node.description or "")
        score += min(len(node.doc_ids), 5) / 20
        evidence.append(
            GraphEvidence(
                evidence_id=f"node:{node_id}",
                evidence_type="local_node",
                text=text,
                score=min(score, 1.0),
                token_estimate=estimate_token_count(text),
                metadata={"node_id": node_id, "doc_ids": node.doc_ids},
            )
        )

    seen_edges = set()
    edge_candidates: list[GraphEvidence] = []
    for node_id in node_ids:
        for edge in store.get_edges_for_node(node_id):
            if edge.source_id not in node_set or edge.target_id not in node_set:
                continue
            edge_key = (edge.source_id, edge.target_id, edge.relation)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            source = store.get_node(edge.source_id)
            target = store.get_node(edge.target_id)
            if not source or not target:
                continue
            text = f"{source.label} [{edge.relation}] {target.label}"
            if edge.description:
                text += f" ({edge.description})"
            score = 0.65 + _text_overlap_score(
                question_terms,
                source.label,
                target.label,
                edge.relation,
                edge.description or "",
            )
            score += min(edge.weight, 1.0) / 5
            edge_candidates.append(
                GraphEvidence(
                    evidence_id=f"edge:{edge.source_id}:{edge.target_id}:{edge.relation}",
                    evidence_type="local_edge",
                    text=text,
                    score=min(score, 1.0),
                    token_estimate=estimate_token_count(text),
                    metadata={
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "doc_ids": edge.doc_ids,
                    },
                )
            )

    edge_candidates.sort(key=lambda item: item.score, reverse=True)
    evidence.extend(edge_candidates[:max_edges])
    evidence.sort(key=lambda item: item.score, reverse=True)
    return evidence


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
    matched_nodes: list[str] = []
    vector_fallback_reason: str | None = None
    index_state = "disabled"
    vector_hit_count = 0

    # Step 1: Vector-first seed retrieval
    if node_vector_search_enabled():
        vector_result = await search_nodes_by_vector(
            store=store,
            query=question,
            top_k=node_vector_top_k(),
            min_score=node_vector_min_score(),
        )
        matched_nodes = list(vector_result.node_ids)
        vector_hit_count = vector_result.vector_hit_count
        vector_fallback_reason = vector_result.fallback_reason
        index_state = vector_result.index_state

        if matched_nodes:
            logger.info(
                "Local search vector seeds ready | vector_hit_count=%s | vector_fallback_reason=%s | index_state=%s | top_score=%s",
                vector_hit_count,
                vector_fallback_reason,
                index_state,
                vector_result.top_score,
            )

    # Step 2: Fallback to legacy LLM + fuzzy matching when vector seeds are missing
    if not matched_nodes:
        query_entities = await identify_query_entities(question)
        if not query_entities:
            logger.info(
                "No entities identified in query and no vector seeds | vector_hit_count=%s | vector_fallback_reason=%s | index_state=%s",
                vector_hit_count,
                vector_fallback_reason,
                index_state,
            )
            return "", []
        matched_nodes = find_matching_nodes(store, query_entities, fuzzy=True)
        if not matched_nodes:
            logger.info(
                "No matching nodes found in graph | vector_hit_count=%s | vector_fallback_reason=%s | index_state=%s",
                vector_hit_count,
                vector_fallback_reason,
                index_state,
            )
            return "", []
        logger.info(
            "Local search fallback matched nodes | vector_hit_count=%s | vector_fallback_reason=%s | index_state=%s | fallback_match_count=%s",
            vector_hit_count,
            vector_fallback_reason,
            index_state,
            len(matched_nodes),
        )
    else:
        logger.info(
            "Local search using vector seeds | vector_hit_count=%s | vector_fallback_reason=%s | index_state=%s",
            vector_hit_count,
            vector_fallback_reason,
            index_state,
        )
    
    # Step 3: Expand to neighbors
    expanded_nodes = expand_to_neighbors(
        store,
        matched_nodes,
        question=question,
        hops=hops,
        max_nodes=max_nodes,
    )
    
    # Step 4: Build context
    context = build_local_context(store, expanded_nodes)
    
    return context, expanded_nodes


async def local_search_evidence(
    store: GraphStore,
    question: str,
    hops: int = 2,
    max_nodes: int = 30,
    max_edges: int = 14,
) -> Tuple[List[GraphEvidence], List[str]]:
    """Return local evidence units for generic graph context merging."""
    context, expanded_nodes = await local_search(
        store=store,
        question=question,
        hops=hops,
        max_nodes=max_nodes,
    )
    if not context or not expanded_nodes:
        return [], expanded_nodes
    return build_local_evidence(store, question, expanded_nodes, max_edges=max_edges), expanded_nodes


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
