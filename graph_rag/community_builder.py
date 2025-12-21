"""
GraphRAG Community Builder Module

Provides community detection using Leiden algorithm and
LLM-based community summarization.
"""

# Standard library
import logging
from typing import List, Optional

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm
from graph_rag.schemas import Community
from graph_rag.store import GraphStore

# Configure logging
logger = logging.getLogger(__name__)

# Prompt for community summarization
_COMMUNITY_SUMMARY_PROMPT = """你是一個學術論文分析專家。請為以下知識圖譜社群生成一個簡潔的摘要。

社群包含以下實體及其關係：

{entities_and_relations}

請提供：
1. 一個簡短的標題 (10 字以內)
2. 一段摘要 (50-100 字)

以 JSON 格式輸出：
```json
{{"title": "社群標題", "summary": "社群摘要描述..."}}
```

JSON 輸出："""


async def detect_communities_leiden(store: GraphStore) -> List[Community]:
    """
    Detect communities using Leiden algorithm.
    
    Falls back to connected components if leidenalg is not available.
    
    Args:
        store: GraphStore to analyze.
        
    Returns:
        List of Community objects.
    """
    import networkx as nx
    
    if store.graph.number_of_nodes() < 2:
        return []
    
    try:
        # Try to use Leiden algorithm
        import igraph as ig
        import leidenalg as la
        
        # Convert NetworkX to igraph
        # Create undirected version for community detection
        undirected = store.graph.to_undirected()
        
        # Get node mapping
        node_list = list(undirected.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        # Create edge list for igraph
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in undirected.edges()]
        
        # Create igraph graph
        ig_graph = ig.Graph(edges=edges, directed=False)
        
        # Run Leiden algorithm
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        
        # Convert partition to Community objects
        communities = []
        for i, community_nodes in enumerate(partition):
            node_ids = [node_list[idx] for idx in community_nodes]
            communities.append(Community(
                id=i,
                node_ids=node_ids,
                level=0,
            ))
        
        logger.info(f"Leiden detected {len(communities)} communities")
        return communities
        
    except ImportError:
        logger.warning("leidenalg not available, falling back to connected components")
        
        # Fallback: use connected components
        undirected = store.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        communities = []
        for i, component in enumerate(components):
            communities.append(Community(
                id=i,
                node_ids=list(component),
                level=0,
            ))
        
        logger.info(f"Connected components found {len(communities)} communities")
        return communities


async def summarize_community(
    store: GraphStore,
    community: Community,
) -> Community:
    """
    Generate LLM summary for a community.
    
    Args:
        store: GraphStore containing nodes.
        community: Community to summarize.
        
    Returns:
        Community with summary and title populated.
    """
    if not community.node_ids:
        return community
    
    # Build context from nodes and edges
    lines = []
    
    # Add nodes
    for node_id in community.node_ids[:20]:  # Limit to 20 nodes
        node = store.get_node(node_id)
        if node:
            lines.append(f"- {node.label} ({node.entity_type.value})")
            if node.description:
                lines.append(f"  描述: {node.description}")
    
    # Add edges within community
    community_set = set(community.node_ids)
    edge_count = 0
    
    for node_id in community.node_ids:
        for edge in store.get_edges_for_node(node_id):
            if edge.target_id in community_set and edge_count < 15:
                source_node = store.get_node(edge.source_id)
                target_node = store.get_node(edge.target_id)
                if source_node and target_node:
                    lines.append(
                        f"- {source_node.label} [{edge.relation}] {target_node.label}"
                    )
                    edge_count += 1
    
    if not lines:
        return community
    
    context = "\n".join(lines)
    
    try:
        llm = get_llm("community_summary")
        prompt = _COMMUNITY_SUMMARY_PROMPT.format(entities_and_relations=context)
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse response
        import json
        import re
        
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            data = json.loads(json_match.group(0))
            community.title = data.get("title", f"社群 {community.id}")
            community.summary = data.get("summary", "")
        
    except Exception as e:
        logger.warning(f"Failed to summarize community {community.id}: {e}")
        community.title = f"社群 {community.id}"
        community.summary = f"包含 {len(community.node_ids)} 個實體"
    
    return community


async def build_communities(
    store: GraphStore,
    generate_summaries: bool = True,
) -> List[Community]:
    """
    Detect communities and optionally generate summaries.
    
    Args:
        store: GraphStore to analyze.
        generate_summaries: Whether to generate LLM summaries.
        
    Returns:
        List of Community objects with summaries.
    """
    # Detect communities
    communities = await detect_communities_leiden(store)
    
    # Generate summaries
    if generate_summaries:
        summarized = []
        for community in communities:
            if len(community.node_ids) > 1:  # Only summarize multi-node communities
                community = await summarize_community(store, community)
            else:
                # Single-node community: use node label as title
                node = store.get_node(community.node_ids[0]) if community.node_ids else None
                if node:
                    community.title = node.label
                    community.summary = node.description or ""
            summarized.append(community)
        communities = summarized
    
    # Update store with communities
    store.communities = communities
    
    logger.info(f"Built {len(communities)} communities for user {store.user_id}")
    return communities


async def rebuild_communities(store: GraphStore) -> List[Community]:
    """
    Force rebuild communities (clears old and regenerates).
    
    Args:
        store: GraphStore to rebuild.
        
    Returns:
        New list of communities.
    """
    store.communities.clear()
    communities = await build_communities(store, generate_summaries=True)
    store.save()
    return communities
