"""
GraphRAG Global Search Module

Provides community-based search using Map-Reduce pattern.
Queries each relevant community and synthesizes answers.
"""

# Standard library
import asyncio
import logging
from typing import List, Optional, Tuple

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm
from graph_rag.schemas import Community
from graph_rag.store import GraphStore

# Configure logging
logger = logging.getLogger(__name__)

# Prompt for community relevance check
_RELEVANCE_CHECK_PROMPT = """判斷以下社群是否與問題相關。

問題：{question}

社群標題：{title}
社群摘要：{summary}

回答 "相關" 或 "不相關"，不要其他文字："""

# Prompt for community-based answer (Map phase)
_COMMUNITY_ANSWER_PROMPT = """你是一個學術論文分析專家。請根據以下社群知識回答問題。

問題：{question}

社群知識：
標題：{title}
摘要：{summary}

相關實體：
{entities}

請根據社群知識提供簡短回答 (50-100 字)。如果社群知識無法回答問題，回答 "無法從此社群獲得相關資訊"。

回答："""

# Prompt for answer synthesis (Reduce phase)
_SYNTHESIS_PROMPT = """你是一個研究報告撰寫專家。請將以下多個社群的回答綜合成一個完整的答案。

原始問題：{question}

各社群回答：
{community_answers}

請提供：
1. 一個綜合性的完整回答
2. 整合各社群的觀點
3. 使用繁體中文
4. 保持學術嚴謹的語氣

綜合答案："""


async def check_community_relevance(
    community: Community,
    question: str,
) -> bool:
    """
    Check if a community is relevant to the question.
    
    Args:
        community: Community to check.
        question: User's question.
        
    Returns:
        True if relevant.
    """
    if not community.summary:
        return False
    
    try:
        llm = get_llm("graph_extraction")  # Use fast model
        prompt = _RELEVANCE_CHECK_PROMPT.format(
            question=question,
            title=community.title or f"社群 {community.id}",
            summary=community.summary,
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return "相關" in response.content
        
    except Exception as e:
        logger.warning(f"Relevance check failed: {e}")
        # Default to include if check fails
        return True


async def query_community(
    store: GraphStore,
    community: Community,
    question: str,
) -> Optional[str]:
    """
    Query a single community for an answer (Map phase).
    
    Args:
        store: GraphStore containing nodes.
        community: Community to query.
        question: User's question.
        
    Returns:
        Community's answer or None.
    """
    if not community.summary:
        return None
    
    # Get entity labels from community
    entities = []
    for node_id in community.node_ids[:10]:
        node = store.get_node(node_id)
        if node:
            entities.append(f"• {node.label}")
    
    entity_str = "\n".join(entities) if entities else "無詳細實體資訊"
    
    try:
        llm = get_llm("community_summary")
        prompt = _COMMUNITY_ANSWER_PROMPT.format(
            question=question,
            title=community.title or f"社群 {community.id}",
            summary=community.summary,
            entities=entity_str,
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        
        # Skip non-answers
        if "無法從此社群" in answer or "無相關資訊" in answer:
            return None
        
        return answer
        
    except Exception as e:
        logger.warning(f"Community query failed for community {community.id}: {e}")
        return None


async def synthesize_answers(
    question: str,
    community_answers: List[Tuple[str, str]],
) -> str:
    """
    Synthesize answers from multiple communities (Reduce phase).
    
    Args:
        question: Original question.
        community_answers: List of (community_title, answer) tuples.
        
    Returns:
        Synthesized answer.
    """
    if not community_answers:
        return "無法從知識圖譜中找到相關資訊來回答此問題。"
    
    if len(community_answers) == 1:
        return community_answers[0][1]
    
    # Format answers for synthesis
    answers_str = "\n\n".join([
        f"[{title}]\n{answer}"
        for title, answer in community_answers
    ])
    
    try:
        llm = get_llm("community_summary")
        prompt = _SYNTHESIS_PROMPT.format(
            question=question,
            community_answers=answers_str,
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        # Fallback: just concatenate
        return "\n\n".join([answer for _, answer in community_answers])


async def global_search(
    store: GraphStore,
    question: str,
    max_communities: int = 5,
) -> Tuple[str, List[int]]:
    """
    Perform global search using community Map-Reduce.
    
    Args:
        store: GraphStore to search.
        question: User's question.
        max_communities: Maximum communities to query.
        
    Returns:
        Tuple of (synthesized_answer, community_ids_used).
    """
    communities = store.communities
    
    if not communities:
        logger.info("No communities in graph, cannot perform global search")
        return "", []
    
    logger.info(f"Global search across {len(communities)} communities")
    
    # Step 1: Filter to relevant communities
    relevance_tasks = [
        check_community_relevance(c, question)
        for c in communities
    ]
    relevance_results = await asyncio.gather(*relevance_tasks)
    
    relevant_communities = [
        c for c, is_relevant in zip(communities, relevance_results)
        if is_relevant
    ][:max_communities]
    
    if not relevant_communities:
        logger.info("No relevant communities found")
        return "無法從知識圖譜中找到與問題相關的社群。", []
    
    logger.info(f"Found {len(relevant_communities)} relevant communities")
    
    # Step 2: Map - Query each community
    query_tasks = [
        query_community(store, c, question)
        for c in relevant_communities
    ]
    answers = await asyncio.gather(*query_tasks)
    
    # Collect valid answers with titles
    community_answers = []
    community_ids = []
    
    for community, answer in zip(relevant_communities, answers):
        if answer:
            title = community.title or f"社群 {community.id}"
            community_answers.append((title, answer))
            community_ids.append(community.id)
    
    if not community_answers:
        return "社群知識無法直接回答此問題。", []
    
    # Step 3: Reduce - Synthesize answers
    final_answer = await synthesize_answers(question, community_answers)
    
    logger.info(f"Global search completed using {len(community_ids)} communities")
    
    return final_answer, community_ids


def build_global_context(store: GraphStore) -> str:
    """
    Build global context from all community summaries.
    
    Useful for providing high-level overview to LLM.
    
    Args:
        store: GraphStore.
        
    Returns:
        Formatted context string.
    """
    if not store.communities:
        return ""
    
    lines = ["=== 知識圖譜社群概覽 ===\n"]
    
    for community in store.communities:
        if community.summary:
            lines.append(f"## {community.title or f'社群 {community.id}'}")
            lines.append(community.summary)
            lines.append("")
    
    return "\n".join(lines)
