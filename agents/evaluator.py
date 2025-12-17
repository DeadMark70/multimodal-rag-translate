"""
Self-RAG Evaluator Module

Provides LLM-as-judge evaluation for RAG output quality:
- Retrieval relevance: Do the documents contain needed information?
- Faithfulness: Is the answer grounded in the documents?
"""

# Standard library
import asyncio
import logging
from enum import Enum
from typing import List, Optional

# Third-party
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalGrade(str, Enum):
    """Grades for retrieval relevance."""
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"


class FaithfulnessGrade(str, Enum):
    """Grades for answer faithfulness."""
    GROUNDED = "grounded"
    HALLUCINATED = "hallucinated"


class EvaluationResult(BaseModel):
    """Result of RAG pipeline evaluation."""
    retrieval_grade: RetrievalGrade
    faithfulness_grade: FaithfulnessGrade
    should_retry: bool
    retry_reason: Optional[str] = None
    confidence: float = 0.0


# Prompts for evaluation
_RETRIEVAL_EVAL_PROMPT = """你是一個檢索品質評估專家。請評估以下檢索結果是否與問題相關。

問題：{question}

檢索文檔：
{documents}

評估標準：
- 如果文檔包含回答問題所需的資訊，回答 "RELEVANT"
- 如果文檔與問題無關或資訊不足，回答 "NOT_RELEVANT"

請只回答 RELEVANT 或 NOT_RELEVANT，不要其他內容："""

_FAITHFULNESS_EVAL_PROMPT = """你是一個答案品質評估專家。請評估以下答案是否完全基於提供的文檔內容。

問題：{question}

參考文檔：
{documents}

生成的答案：
{answer}

評估標準：
- 如果答案的所有內容都可以從文檔中找到依據，回答 "GROUNDED"
- 如果答案包含文檔中沒有的資訊或推測，回答 "HALLUCINATED"

請只回答 GROUNDED 或 HALLUCINATED，不要其他內容："""


class RAGEvaluator:
    """
    Self-RAG evaluator using LLM-as-judge.
    
    Evaluates:
    1. Retrieval relevance: Are retrieved docs relevant to question?
    2. Faithfulness: Is generated answer grounded in retrieved docs?
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
    """
    
    def __init__(self, max_concurrent: int = 2) -> None:
        """
        Initializes the evaluator.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_retrieval(
        self,
        question: str,
        documents: List[Document],
    ) -> RetrievalGrade:
        """
        Evaluates if retrieved documents are relevant to the question.
        
        Args:
            question: User question.
            documents: Retrieved documents.
            
        Returns:
            RetrievalGrade indicating relevance.
        """
        if not documents:
            return RetrievalGrade.NOT_RELEVANT
        
        async with self._semaphore:
            try:
                llm = get_llm("evaluator")
                
                doc_text = "\n\n".join([
                    f"[{i+1}] {doc.page_content[:500]}"
                    for i, doc in enumerate(documents[:5])
                ])
                
                prompt = _RETRIEVAL_EVAL_PROMPT.format(
                    question=question,
                    documents=doc_text,
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                result = response.content.strip().upper()
                
                if "RELEVANT" in result and "NOT" not in result:
                    return RetrievalGrade.RELEVANT
                else:
                    return RetrievalGrade.NOT_RELEVANT
                    
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"Retrieval evaluation failed: {e}")
                return RetrievalGrade.RELEVANT  # Assume relevant on error
    
    async def evaluate_faithfulness(
        self,
        question: str,
        documents: List[Document],
        answer: str,
    ) -> FaithfulnessGrade:
        """
        Evaluates if the answer is grounded in the documents.
        
        Args:
            question: User question.
            documents: Retrieved documents.
            answer: Generated answer.
            
        Returns:
            FaithfulnessGrade indicating grounding.
        """
        if not answer or not documents:
            return FaithfulnessGrade.HALLUCINATED
        
        async with self._semaphore:
            try:
                llm = get_llm("evaluator")
                
                doc_text = "\n\n".join([
                    f"[{i+1}] {doc.page_content[:500]}"
                    for i, doc in enumerate(documents[:5])
                ])
                
                prompt = _FAITHFULNESS_EVAL_PROMPT.format(
                    question=question,
                    documents=doc_text,
                    answer=answer[:1000],
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                result = response.content.strip().upper()
                
                if "GROUNDED" in result and "HALLUCINATED" not in result:
                    return FaithfulnessGrade.GROUNDED
                else:
                    return FaithfulnessGrade.HALLUCINATED
                    
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"Faithfulness evaluation failed: {e}")
                return FaithfulnessGrade.GROUNDED  # Assume grounded on error
    
    async def evaluate(
        self,
        question: str,
        documents: List[Document],
        answer: str,
    ) -> EvaluationResult:
        """
        Performs full RAG evaluation.
        
        Args:
            question: User question.
            documents: Retrieved documents.
            answer: Generated answer.
            
        Returns:
            EvaluationResult with all grades and retry recommendation.
        """
        # Run both evaluations concurrently
        retrieval_task = self.evaluate_retrieval(question, documents)
        faithfulness_task = self.evaluate_faithfulness(question, documents, answer)
        
        retrieval_grade, faithfulness_grade = await asyncio.gather(
            retrieval_task, faithfulness_task
        )
        
        # Determine if retry is needed
        should_retry = False
        retry_reason = None
        confidence = 1.0
        
        if retrieval_grade == RetrievalGrade.NOT_RELEVANT:
            should_retry = True
            retry_reason = "檢索結果與問題不相關，建議重新檢索"
            confidence = 0.3
        elif faithfulness_grade == FaithfulnessGrade.HALLUCINATED:
            should_retry = True
            retry_reason = "答案可能包含未經證實的資訊"
            confidence = 0.5
        
        return EvaluationResult(
            retrieval_grade=retrieval_grade,
            faithfulness_grade=faithfulness_grade,
            should_retry=should_retry,
            retry_reason=retry_reason,
            confidence=confidence,
        )


async def evaluate_rag_result(
    question: str,
    documents: List[Document],
    answer: str,
    enabled: bool = True,
) -> EvaluationResult:
    """
    Convenience function to evaluate RAG result.
    
    Args:
        question: User question.
        documents: Retrieved documents.
        answer: Generated answer.
        enabled: If False, returns positive evaluation.
        
    Returns:
        EvaluationResult.
    """
    if not enabled:
        return EvaluationResult(
            retrieval_grade=RetrievalGrade.RELEVANT,
            faithfulness_grade=FaithfulnessGrade.GROUNDED,
            should_retry=False,
            confidence=1.0,
        )
    
    evaluator = RAGEvaluator()
    return await evaluator.evaluate(question, documents, answer)
