"""
Query Transformer Module

Provides query transformation techniques for improved retrieval:
- HyDE (Hypothetical Document Embeddings)
- Multi-Query Generation with RRF Fusion
"""

# Standard library
import asyncio
import logging
from typing import List

# Third-party
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Local application
from core.providers import get_llm
from core.prompt_loader import format_rag_pipeline_prompt

# Configure logging
logger = logging.getLogger(__name__)


class QueryTransformer:
    """
    Transforms user queries for improved retrieval.
    
    Supports:
    - HyDE: Generate hypothetical answer, use for retrieval
    - Multi-Query: Split into sub-queries, fuse results with RRF
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
    """
    
    def __init__(self, max_concurrent: int = 3) -> None:
        """
        Initializes the query transformer.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_hyde_document(
        self,
        question: str,
    ) -> str:
        """
        Generates a hypothetical document using HyDE technique.
        
        Creates a synthetic document that might contain the answer,
        which is then used for retrieval instead of the raw question.
        
        Args:
            question: Original user question.
            
        Returns:
            Hypothetical document content.
        """
        async with self._semaphore:
            try:
                llm = get_llm("query_rewrite")
                
                prompt = format_rag_pipeline_prompt("hyde", question=question)
                message = HumanMessage(content=prompt)
                
                response = await llm.ainvoke([message])
                hyde_doc = response.content.strip()
                
                logger.debug(f"Generated HyDE document: {hyde_doc[:100]}...")
                return hyde_doc
                
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")
                return question  # Fallback to original question
    
    async def generate_multi_queries(
        self,
        question: str,
        max_queries: int = 4,
    ) -> List[str]:
        """
        Generates multiple query variations.
        
        Splits the question into sub-queries from different perspectives
        for multi-dimensional retrieval.
        
        Args:
            question: Original user question.
            max_queries: Maximum number of sub-queries to generate.
            
        Returns:
            List of query strings (includes original question).
        """
        async with self._semaphore:
            try:
                llm = get_llm("query_rewrite")
                
                prompt = format_rag_pipeline_prompt("multi_query", question=question)
                message = HumanMessage(content=prompt)
                
                response = await llm.ainvoke([message])
                
                # Parse numbered queries
                queries = [question]  # Always include original
                
                for line in response.content.strip().split('\n'):
                    line = line.strip()
                    if line and line[0].isdigit():
                        # Remove number prefix
                        query = line.lstrip('0123456789.)').strip()
                        if query and len(query) > 5:
                            queries.append(query)
                
                logger.debug(f"Generated {len(queries)} queries")
                return queries[:max_queries + 1]
                
            except Exception as e:
                logger.warning(f"Multi-query generation failed: {e}")
                return [question]


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Fuses multiple result lists using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank_i)) for each list where doc appears.
    
    Args:
        result_lists: List of document lists from different retrievals.
        k: Constant for RRF (default 60 is commonly used).
        
    Returns:
        Fused and deduplicated document list, sorted by RRF score.
    """
    if not result_lists:
        return []
    
    # Track scores by document content (use as key for dedup)
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    
    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            key = doc.page_content[:500]  # Use first 500 chars as key
            
            # RRF score
            score = 1.0 / (k + rank + 1)  # rank is 0-indexed
            
            if key in doc_scores:
                doc_scores[key] += score
            else:
                doc_scores[key] = score
                doc_map[key] = doc
    
    # Sort by score
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    return [doc_map[key] for key in sorted_keys]


async def transform_query_with_hyde(
    question: str,
    enabled: bool = True,
) -> str:
    """
    Convenience function for HyDE transformation.
    
    Args:
        question: Original question.
        enabled: If False, returns question unchanged.
        
    Returns:
        Transformed query (or original if disabled).
    """
    if not enabled:
        return question
    
    transformer = QueryTransformer()
    return await transformer.generate_hyde_document(question)


async def transform_query_multi(
    question: str,
    enabled: bool = True,
    max_queries: int = 4,
) -> List[str]:
    """
    Convenience function for multi-query generation.
    
    Args:
        question: Original question.
        enabled: If False, returns [question].
        max_queries: Maximum sub-queries.
        
    Returns:
        List of queries.
    """
    if not enabled:
        return [question]
    
    transformer = QueryTransformer()
    return await transformer.generate_multi_queries(question, max_queries)
