"""
GraphRAG Community Builder Module

Provides community detection using Leiden algorithm and
LLM-based community summarization.
"""

# Standard library
import logging
from typing import List

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.providers import get_llm
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

_PARENT_COMMUNITY_PROMPT = """你是一個學術知識圖譜整理專家。請把多個子社群整理成一個上層主題。

子社群資訊：
{child_summaries}

請輸出 JSON：
{{"title": "上層主題標題", "summary": "50-100 字摘要"}}
"""


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
            if community.title or community.summary:
                community.ranking_text = " ".join(
                    part for part in [community.title, community.summary] if part
                )
            summarized.append(community)
        communities = summarized

    communities = await _build_hierarchy_communities(
        store,
        communities,
        generate_summaries=generate_summaries,
    )

    # Update store with communities
    store.communities = communities
    store.mark_optimized()
    
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


async def _summarize_parent_community(
    community_id: int,
    children: List[Community],
) -> Community:
    """Build a lightweight level-1 summary community from child communities."""
    node_ids = []
    child_summary_lines = []
    child_ids = []
    for child in children:
        node_ids.extend(child.node_ids)
        child_ids.append(child.id)
        child_summary_lines.append(
            f"- {child.title or f'社群 {child.id}'}: {child.summary or '無摘要'}"
        )

    parent = Community(
        id=community_id,
        node_ids=list(dict.fromkeys(node_ids)),
        level=1,
        child_ids=child_ids,
    )

    try:
        llm = get_llm("community_summary")
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=_PARENT_COMMUNITY_PROMPT.format(
                        child_summaries="\n".join(child_summary_lines)
                    )
                )
            ]
        )
        import json
        import re

        json_match = re.search(r"\{[\s\S]*\}", response.content)
        if json_match:
            data = json.loads(json_match.group(0))
            parent.title = data.get("title", f"主題 {community_id}")
            parent.summary = data.get("summary", "")
    except Exception as exc:
        logger.warning("Failed to summarize parent community %s: %s", community_id, exc)
        parent.title = f"主題 {community_id}"
        parent.summary = "；".join(
            child.title or f"社群 {child.id}"
            for child in children[:4]
        )

    parent.ranking_text = " ".join(
        part for part in [parent.title, parent.summary] if part
    )
    return parent


async def _build_hierarchy_communities(
    store: GraphStore,
    leaf_communities: List[Community],
    *,
    generate_summaries: bool,
) -> List[Community]:
    """Build a lightweight two-level hierarchy on top of leaf communities."""
    if not leaf_communities:
        return []

    ordered_leaves = sorted(
        leaf_communities,
        key=lambda community: len(community.node_ids),
        reverse=True,
    )

    if len(ordered_leaves) <= 3:
        return ordered_leaves

    parent_groups: list[list[Community]] = []
    group_size = 3
    for index in range(0, len(ordered_leaves), group_size):
        parent_groups.append(ordered_leaves[index:index + group_size])

    all_communities = list(ordered_leaves)
    next_id = max(community.id for community in ordered_leaves) + 1

    for group in parent_groups:
        parent = await _summarize_parent_community(next_id, group) if generate_summaries else Community(
            id=next_id,
            node_ids=list(dict.fromkeys(node_id for child in group for node_id in child.node_ids)),
            level=1,
            child_ids=[child.id for child in group],
            title=f"主題 {next_id}",
            summary="",
            ranking_text="",
        )
        for child in group:
            child.parent_id = parent.id
            child.summary_version = 1
        parent.summary_version = 1
        all_communities.append(parent)
        next_id += 1

    return all_communities
