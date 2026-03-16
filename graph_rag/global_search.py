"""
GraphRAG Global Search Module

Provides community-based search using Map-Reduce pattern.
Queries each relevant community and synthesizes answers.
"""

# Standard library
import asyncio
import logging
import re
from typing import List, Optional, Tuple

# Third-party
from langchain_core.messages import HumanMessage

# Local application
from core.providers import get_llm
from graph_rag.generic_mode import GraphEvidence, estimate_token_count
from graph_rag.llm_response import response_content_to_text
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


def _question_terms(question: str) -> set[str]:
    return {token for token in re.findall(r"[\w\-]+", question.lower()) if len(token) > 1}


def score_community_relevance(community: Community, question: str) -> float:
    """Cheap lexical relevance score used for community pruning."""
    question_terms = _question_terms(question)
    if not question_terms:
        return 0.1
    haystack = " ".join(
        value.lower()
        for value in (community.title or "", community.summary or "", community.ranking_text or "")
        if value
    )
    hits = sum(1 for token in question_terms if token in haystack)
    if hits == 0:
        return 0.0
    size_bonus = min(len(community.node_ids), 10) / 50
    level_bonus = 0.05 if community.level > 0 else 0.0
    return (hits / max(len(question_terms), 1)) + size_bonus + level_bonus


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
    return score_community_relevance(community, question) > 0.0


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
        answer = response_content_to_text(response.content)
        
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
        return response_content_to_text(response.content)
        
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        # Fallback: just concatenate
        return "\n\n".join([answer for _, answer in community_answers])


async def global_search(
    store: GraphStore,
    question: str,
    max_communities: int = 5,
    level: Optional[int] = None,
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
    communities = store.get_communities(level=level)
    
    if not communities:
        logger.info("No communities in graph, cannot perform global search")
        return "", []
    
    logger.info(f"Global search across {len(communities)} communities")
    
    scored_communities = sorted(
        (
            (community, score_community_relevance(community, question))
            for community in communities
            if community.summary
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    relevant_communities = [
        community
        for community, score in scored_communities
        if score > 0.0
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


async def global_search_evidence(
    store: GraphStore,
    question: str,
    max_communities: int = 3,
    level: Optional[int] = None,
) -> Tuple[str, List[GraphEvidence], List[int]]:
    """Return global answer plus structured community evidence."""
    communities = store.get_communities(level=level)
    if not communities:
        return "", [], []

    scored = sorted(
        (
            (community, score_community_relevance(community, question))
            for community in communities
            if community.summary
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:max_communities]

    relevant = [(community, score) for community, score in scored if score > 0.0]
    if not relevant:
        return "", [], []

    query_tasks = [query_community(store, community, question) for community, _ in relevant]
    answers = await asyncio.gather(*query_tasks)

    answer_pairs: list[tuple[str, str]] = []
    evidence: list[GraphEvidence] = []
    community_ids: list[int] = []

    for (community, score), answer in zip(relevant, answers):
        title = community.title or f"社群 {community.id}"
        summary_text = f"{title}: {community.summary}"
        evidence.append(
            GraphEvidence(
                evidence_id=f"community-summary:{community.id}",
                evidence_type="community_summary",
                text=summary_text,
                score=min(0.55 + score, 1.0),
                token_estimate=estimate_token_count(summary_text),
                metadata={"community_id": community.id, "level": community.level},
            )
        )
        if answer:
            answer_pairs.append((title, answer))
            community_ids.append(community.id)
            answer_text = f"{title}: {answer}"
            evidence.append(
                GraphEvidence(
                    evidence_id=f"community-answer:{community.id}",
                    evidence_type="community_answer",
                    text=answer_text,
                    score=min(0.7 + score, 1.0),
                    token_estimate=estimate_token_count(answer_text),
                    metadata={"community_id": community.id, "level": community.level},
                )
            )

    final_answer = await synthesize_answers(question, answer_pairs) if answer_pairs else ""
    evidence.sort(key=lambda item: item.score, reverse=True)
    return final_answer, evidence, community_ids


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
