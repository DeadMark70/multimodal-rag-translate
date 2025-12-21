"""
Self-RAG Evaluator Module

Provides LLM-as-judge evaluation for RAG output quality:
- Retrieval relevance: Do the documents contain needed information?
- Faithfulness: Is the answer grounded in the documents?
- Detailed evaluation: 1-5 scoring with weighted confidence
"""

# Standard library
import asyncio
import json
import logging
from enum import Enum
from typing import List, Optional

# Third-party
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

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


class DetailedEvaluationResult(BaseModel):
    """
    Detailed evaluation result with 1-5 scoring.
    
    Attributes:
        relevance_score: How relevant are the documents to the question (1-5)
        groundedness_score: How well is the answer grounded in documents (1-5)
        completeness_score: How completely does the answer address the question (1-5)
        reason: Brief explanation of the evaluation
        confidence: Weighted confidence score (0.2-1.0)
        evaluation_failed: True if LLM evaluation itself failed
    """
    relevance_score: int = Field(default=3, ge=1, le=5)
    groundedness_score: int = Field(default=3, ge=1, le=5)
    completeness_score: int = Field(default=3, ge=1, le=5)
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evaluation_failed: bool = False
    
    @property
    def is_reliable(self) -> bool:
        """Returns True if the answer is considered reliable (confidence >= 0.7)."""
        return self.confidence >= 0.7 and not self.evaluation_failed
    
    @property
    def faithfulness_level(self) -> str:
        """Maps groundedness score to faithfulness level."""
        if self.evaluation_failed:
            return "evaluation_failed"
        if self.groundedness_score >= 4:
            return "grounded"
        elif self.groundedness_score >= 3:
            return "uncertain"
        else:
            return "hallucinated"


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

_DETAILED_EVAL_PROMPT = """你是 RAG 答案品質評估專家。請詳細評估以下答案的品質。

## 問題
{question}

## 參考文檔
{documents}

## 生成的答案
{answer}

## 評估標準 (1=最差, 5=最佳)

1. **relevance (相關性)**: 檢索到的文檔與問題的相關程度
   - 5: 文檔完全針對問題，包含所有需要的資訊
   - 3: 文檔部分相關，有一些有用資訊
   - 1: 文檔與問題無關

2. **groundedness (依據性)**: 答案有多少內容可以在文檔中驗證
   - 5: 答案的每個觀點都能在文檔中找到依據
   - 3: 部分內容有依據，部分是合理推論
   - 1: 答案包含大量文檔中沒有的資訊

3. **completeness (完整性)**: 答案是否完整回答了問題
   - 5: 完整且全面地回答了問題
   - 3: 回答了主要問題，但遺漏了一些細節
   - 1: 沒有真正回答問題

請用 JSON 格式回答 (只輸出 JSON，不要其他內容):
{{"relevance": <1-5>, "groundedness": <1-5>, "completeness": <1-5>, "reason": "<一句話解釋>"}}"""


class RAGEvaluator:
    """
    Self-RAG evaluator using LLM-as-judge.
    
    Evaluates:
    1. Retrieval relevance: Are retrieved docs relevant to question?
    2. Faithfulness: Is generated answer grounded in retrieved docs?
    3. Detailed evaluation: 1-5 scoring with confidence calculation
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
    """
    
    # Weights for confidence calculation
    WEIGHT_RELEVANCE = 0.3
    WEIGHT_GROUNDEDNESS = 0.5
    WEIGHT_COMPLETENESS = 0.2
    
    def __init__(self, max_concurrent: int = 2) -> None:
        """
        Initializes the evaluator.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    def _calculate_confidence(self, scores: dict) -> float:
        """
        Calculates weighted confidence score.
        
        Args:
            scores: Dict with relevance, groundedness, completeness (1-5 each)
        
        Returns:
            Confidence score between 0.2 and 1.0
        """
        relevance = scores.get("relevance", 3)
        groundedness = scores.get("groundedness", 3)
        completeness = scores.get("completeness", 3)
        
        weighted = (
            relevance * self.WEIGHT_RELEVANCE +
            groundedness * self.WEIGHT_GROUNDEDNESS +
            completeness * self.WEIGHT_COMPLETENESS
        ) / 5.0
        
        return max(0.2, min(1.0, weighted))
    
    def _parse_json_response(self, response: str) -> Optional[dict]:
        """
        Parses JSON from LLM response, handling potential formatting issues.
        
        Args:
            response: Raw LLM response string
        
        Returns:
            Parsed dict or None if parsing fails
        """
        # Try direct parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
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
    
    async def evaluate_detailed(
        self,
        question: str,
        documents: List[Document],
        answer: str,
    ) -> DetailedEvaluationResult:
        """
        Performs detailed evaluation with 1-5 scoring.
        
        Args:
            question: User question.
            documents: Retrieved documents.
            answer: Generated answer.
            
        Returns:
            DetailedEvaluationResult with scores and confidence.
        """
        # Return failed result if no documents or answer
        if not documents or not answer:
            return DetailedEvaluationResult(
                relevance_score=1,
                groundedness_score=1,
                completeness_score=1,
                reason="無法評估：缺少文檔或答案",
                confidence=0.2,
                evaluation_failed=True,
            )
        
        async with self._semaphore:
            try:
                llm = get_llm("evaluator")
                
                doc_text = "\n\n".join([
                    f"[{i+1}] {doc.page_content[:500]}"
                    for i, doc in enumerate(documents[:5])
                ])
                
                prompt = _DETAILED_EVAL_PROMPT.format(
                    question=question,
                    documents=doc_text,
                    answer=answer[:1500],
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                # Parse JSON response
                scores = self._parse_json_response(response.content)
                
                if not scores:
                    logger.warning(f"Failed to parse evaluation JSON: {response.content[:200]}")
                    return DetailedEvaluationResult(
                        reason="評估失敗：無法解析 LLM 回應",
                        evaluation_failed=True,
                    )
                
                # Validate and clamp scores
                relevance = max(1, min(5, int(scores.get("relevance", 3))))
                groundedness = max(1, min(5, int(scores.get("groundedness", 3))))
                completeness = max(1, min(5, int(scores.get("completeness", 3))))
                reason = str(scores.get("reason", ""))[:200]
                
                confidence = self._calculate_confidence({
                    "relevance": relevance,
                    "groundedness": groundedness,
                    "completeness": completeness,
                })
                
                logger.debug(
                    f"Detailed evaluation: rel={relevance}, gnd={groundedness}, "
                    f"cmp={completeness}, conf={confidence:.2f}"
                )
                
                return DetailedEvaluationResult(
                    relevance_score=relevance,
                    groundedness_score=groundedness,
                    completeness_score=completeness,
                    reason=reason,
                    confidence=confidence,
                    evaluation_failed=False,
                )
                    
            except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.error(f"Detailed evaluation failed: {e}")
                return DetailedEvaluationResult(
                    reason=f"評估失敗：{str(e)[:100]}",
                    evaluation_failed=True,
                )
    
    async def evaluate(
        self,
        question: str,
        documents: List[Document],
        answer: str,
    ) -> EvaluationResult:
        """
        Performs full RAG evaluation (legacy method).
        
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


async def evaluate_rag_detailed(
    question: str,
    documents: List[Document],
    answer: str,
    enabled: bool = True,
) -> DetailedEvaluationResult:
    """
    Convenience function for detailed RAG evaluation.
    
    Args:
        question: User question.
        documents: Retrieved documents.
        answer: Generated answer.
        enabled: If False, returns default positive evaluation.
        
    Returns:
        DetailedEvaluationResult with scores and confidence.
    """
    if not enabled:
        return DetailedEvaluationResult(
            relevance_score=5,
            groundedness_score=5,
            completeness_score=5,
            reason="評估已停用",
            confidence=1.0,
            evaluation_failed=False,
        )
    
    evaluator = RAGEvaluator()
    return await evaluator.evaluate_detailed(question, documents, answer)
