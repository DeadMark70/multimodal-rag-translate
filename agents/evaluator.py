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
from core.providers import get_llm
from core.llm_usage_context import llm_accounting_phase
from core.prompt_loader import format_agentic_rag_prompt

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
    Detailed evaluation result with 1-10 scoring (Fine-Grained Evaluation).
    
    Implements the Phase 4 academic evaluation engine with 3 dimensions:
    - Accuracy (50% weight): Data precision, citation correctness, no hallucination
    - Completeness (30% weight): Coverage of all sub-aspects of the question
    - Clarity (20% weight): Logical structure and expression quality
    
    Attributes:
        accuracy: D1 - Data precision score (1-10, 50% weight)
        completeness: D2 - Coverage completeness score (1-10, 30% weight)
        clarity: D3 - Logical expression score (1-10, 20% weight)
        reason: Detailed analysis and evaluation reasoning
        suggestion: Improvement suggestions for smart retry
        confidence: Weighted confidence score (0.1-1.0)
        evaluation_failed: True if LLM evaluation itself failed
    """
    # New 1-10 scale dimensions
    accuracy: float = Field(default=5.0, ge=1, le=10, description="D1: 數據精確度 (50%權重)")
    completeness: float = Field(default=5.0, ge=1, le=10, description="D2: 完整覆蓋率 (30%權重)")
    clarity: float = Field(default=5.0, ge=1, le=10, description="D3: 邏輯表達 (20%權重)")
    
    reason: str = Field(default="", description="詳細評分理由")
    suggestion: str = Field(default="", description="改進建議 (用於 Smart Retry)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evaluation_failed: bool = False
    
    @property
    def weighted_score(self) -> float:
        """Calculates weighted total score (1-10 scale)."""
        return self.accuracy * 0.5 + self.completeness * 0.3 + self.clarity * 0.2
    
    @property
    def is_passing(self) -> bool:
        """Returns True if accuracy >= 7 (high standard for academic use)."""
        return self.accuracy >= 7.0 and not self.evaluation_failed
    
    @property
    def is_reliable(self) -> bool:
        """Returns True if the answer is considered reliable (confidence >= 0.7)."""
        return self.confidence >= 0.7 and not self.evaluation_failed
    
    @property
    def faithfulness_level(self) -> str:
        """Maps accuracy score to faithfulness level."""
        if self.evaluation_failed:
            return "evaluation_failed"
        if self.accuracy >= 8:
            return "grounded"
        elif self.accuracy >= 6:
            return "uncertain"
        else:
            return "hallucinated"



class RAGEvaluator:
    """
    Self-RAG evaluator using LLM-as-judge.
    
    Evaluates:
    1. Retrieval relevance: Are retrieved docs relevant to question?
    2. Faithfulness: Is generated answer grounded in retrieved docs?
    3. Fine-Grained evaluation: 1-10 scoring with academic rubric
    
    Dimensions (Phase 4 Academic Evaluation Engine):
    - Accuracy (50%): Data precision, no hallucination
    - Completeness (30%): Coverage of all sub-aspects
    - Clarity (20%): Logical structure
    
    Thresholds for Deep Research:
    - < 6: Must retry (poor quality)
    - 6-8: Acceptable (could be better)
    - >= 8: Perfect (high quality)
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
    """
    
    # Weights for confidence calculation (Phase 4: Academic Evaluation)
    WEIGHT_ACCURACY = 0.5
    WEIGHT_COMPLETENESS = 0.3
    WEIGHT_CLARITY = 0.2
    
    # Thresholds for retry logic
    THRESHOLD_MUST_RETRY = 6.0    # Accuracy < 6 must retry
    THRESHOLD_PERFECT = 8.0       # Accuracy >= 8 is considered perfect
    
    
    def __init__(self, max_concurrent: int = 2) -> None:
        """
        Initializes the evaluator.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    def _calculate_confidence(self, scores: dict) -> float:
        """
        Calculates weighted confidence score from 1-10 evaluation scores.
        
        Args:
            scores: Dict with accuracy, completeness, clarity (1-10 each)
        
        Returns:
            Confidence score between 0.1 and 1.0
        """
        accuracy = scores.get("accuracy", 5)
        completeness = scores.get("completeness", 5)
        clarity = scores.get("clarity", 5)
        
        weighted = (
            accuracy * self.WEIGHT_ACCURACY +
            completeness * self.WEIGHT_COMPLETENESS +
            clarity * self.WEIGHT_CLARITY
        ) / 10.0
        
        return max(0.1, min(1.0, weighted))
    
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
                
                prompt = format_agentic_rag_prompt(
                    "retrieval_eval",
                    question=question,
                    documents=doc_text,
                )
                
                message = HumanMessage(content=prompt)
                with llm_accounting_phase("agent_planning"):
                    response = await llm.ainvoke([message])
                
                result = response.content.strip().upper()
                
                if "RELEVANT" in result and "NOT" not in result:
                    return RetrievalGrade.RELEVANT
                else:
                    return RetrievalGrade.NOT_RELEVANT
                    
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"Retrieval evaluation failed: {e}")
                return RetrievalGrade.RELEVANT  # Assume relevant on error

    async def grade_documents(
        self,
        question: str,
        documents: List[Document],
    ) -> bool:
        """
        Lightweight boolean retrieval grader for CRAG guard.

        Returns:
            True when retrieved documents are relevant enough to proceed.
        """
        grade = await self.evaluate_retrieval(question=question, documents=documents)
        return grade == RetrievalGrade.RELEVANT
    
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
                
                prompt = format_agentic_rag_prompt(
                    "faithfulness_eval",
                    question=question,
                    documents=doc_text,
                    answer=answer[:1000],
                )
                
                message = HumanMessage(content=prompt)
                with llm_accounting_phase("agent_planning"):
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
        Performs fine-grained evaluation with 1-10 scoring.
        
        Uses Chain-of-Thought prompting with error analysis for accurate
        academic-grade evaluation.
        
        Args:
            question: User question.
            documents: Retrieved documents (can be empty for Pure LLM mode).
            answer: Generated answer.
            
        Returns:
            DetailedEvaluationResult with scores, confidence, and suggestion.
        """
        # Return failed result if no answer
        if not answer:
            return DetailedEvaluationResult(
                accuracy=1.0,
                completeness=1.0,
                clarity=1.0,
                reason="無法評估：缺少答案",
                suggestion="請提供有效的答案",
                confidence=0.1,
                evaluation_failed=True,
            )
        
        # Handle no documents case (still evaluate based on answer quality)
        if not documents:
            return DetailedEvaluationResult(
                accuracy=5.0,  # Neutral score when no docs to verify
                completeness=5.0,
                clarity=5.0,
                reason="無參考文獻，無法驗證準確度",
                suggestion="請提供參考文獻以進行完整評估",
                confidence=0.5,
                evaluation_failed=False,
            )
        
        async with self._semaphore:
            try:
                llm = get_llm("evaluator")
                
                # Use top 10 chunks for better context (long context model)
                doc_text = "\n\n".join([
                    f"[文獻 {i+1}] {doc.page_content[:800]}"
                    for i, doc in enumerate(documents[:10])
                ])
                
                prompt = format_agentic_rag_prompt(
                    "detailed_eval",
                    question=question,
                    documents=doc_text,
                    answer=answer[:2000],  # Increased for academic answers
                )
                
                message = HumanMessage(content=prompt)
                with llm_accounting_phase("agent_planning"):
                    response = await llm.ainvoke([message])
                
                # Parse JSON response
                scores = self._parse_json_response(response.content)
                
                if not scores:
                    logger.warning(f"Failed to parse evaluation JSON: {response.content[:200]}")
                    return DetailedEvaluationResult(
                        reason="評估失敗：無法解析 LLM 回應",
                        suggestion="請重試評估",
                        evaluation_failed=True,
                    )
                
                # Validate and clamp scores (1-10 scale)
                accuracy = max(1.0, min(10.0, float(scores.get("accuracy", 5))))
                completeness = max(1.0, min(10.0, float(scores.get("completeness", 5))))
                clarity = max(1.0, min(10.0, float(scores.get("clarity", 5))))
                reason = str(scores.get("reason", ""))[:300]
                suggestion = str(scores.get("suggestion", ""))[:300]
                analysis = str(scores.get("analysis", ""))[:500]
                
                # Combine analysis with reason if available
                full_reason = f"{analysis}\n評語: {reason}" if analysis else reason
                
                confidence = self._calculate_confidence({
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "clarity": clarity,
                })
                
                logger.debug(
                    f"Fine-grained evaluation: acc={accuracy:.1f}, cmp={completeness:.1f}, "
                    f"clr={clarity:.1f}, conf={confidence:.2f}"
                )
                
                return DetailedEvaluationResult(
                    accuracy=accuracy,
                    completeness=completeness,
                    clarity=clarity,
                    reason=full_reason,
                    suggestion=suggestion,
                    confidence=confidence,
                    evaluation_failed=False,
                )
                    
            except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.error(f"Fine-grained evaluation failed: {e}")
                return DetailedEvaluationResult(
                    reason=f"評估失敗：{str(e)[:100]}",
                    suggestion="請檢查輸入格式並重試",
                    evaluation_failed=True,
                )
    
    async def evaluate_pure_llm(
        self,
        question: str,
        pure_llm_answer: str,
        ground_truth: str,
    ) -> DetailedEvaluationResult:
        """
        Evaluates a Pure LLM answer against Ground Truth (RAG/Deep Research result).
        
        This is used for comparing RAG vs Pure LLM performance in experiments.
        The Ground Truth is typically the answer from Deep Research with verified sources.
        
        Args:
            question: User question.
            pure_llm_answer: Answer generated by Pure LLM (without RAG).
            ground_truth: Reference answer from RAG/Deep Research.
            
        Returns:
            DetailedEvaluationResult with comparison scores.
        """
        if not pure_llm_answer:
            return DetailedEvaluationResult(
                accuracy=1.0,
                completeness=1.0,
                clarity=1.0,
                reason="無法評估：缺少 Pure LLM 答案",
                suggestion="請提供有效的 Pure LLM 答案",
                confidence=0.1,
                evaluation_failed=True,
            )
        
        if not ground_truth:
            return DetailedEvaluationResult(
                accuracy=5.0,
                completeness=5.0,
                clarity=5.0,
                reason="無 Ground Truth，無法驗證準確度",
                suggestion="請先執行 Deep Research 以獲取參考答案",
                confidence=0.5,
                evaluation_failed=False,
            )
        
        async with self._semaphore:
            try:
                llm = get_llm("evaluator")
                
                prompt = format_agentic_rag_prompt(
                    "pure_llm_eval",
                    question=question,
                    answer=pure_llm_answer[:2000],
                    ground_truth=ground_truth[:2000],
                )
                
                message = HumanMessage(content=prompt)
                with llm_accounting_phase("agent_planning"):
                    response = await llm.ainvoke([message])
                
                # Parse JSON response
                scores = self._parse_json_response(response.content)
                
                if not scores:
                    logger.warning(f"Failed to parse Pure LLM evaluation JSON: {response.content[:200]}")
                    return DetailedEvaluationResult(
                        reason="評估失敗：無法解析 LLM 回應",
                        suggestion="請重試評估",
                        evaluation_failed=True,
                    )
                
                # Validate and clamp scores (1-10 scale)
                accuracy = max(1.0, min(10.0, float(scores.get("accuracy", 5))))
                completeness = max(1.0, min(10.0, float(scores.get("completeness", 5))))
                clarity = max(1.0, min(10.0, float(scores.get("clarity", 5))))
                reason = str(scores.get("reason", ""))[:300]
                suggestion = str(scores.get("suggestion", ""))[:300]
                analysis = str(scores.get("analysis", ""))[:500]
                
                full_reason = f"{analysis}\n評語: {reason}" if analysis else reason
                
                confidence = self._calculate_confidence({
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "clarity": clarity,
                })
                
                logger.debug(
                    f"Pure LLM evaluation: acc={accuracy:.1f}, cmp={completeness:.1f}, "
                    f"clr={clarity:.1f}, conf={confidence:.2f}"
                )
                
                return DetailedEvaluationResult(
                    accuracy=accuracy,
                    completeness=completeness,
                    clarity=clarity,
                    reason=full_reason,
                    suggestion=suggestion,
                    confidence=confidence,
                    evaluation_failed=False,
                )
                    
            except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.error(f"Pure LLM evaluation failed: {e}")
                return DetailedEvaluationResult(
                    reason=f"評估失敗：{str(e)[:100]}",
                    suggestion="請檢查輸入格式並重試",
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

async def compare_rag_vs_pure_llm(
    question: str,
    rag_answer: str,
    pure_llm_answer: str,
    documents: List[Document],
) -> dict:
    """
    Compares RAG and Pure LLM answers for experimental evaluation.
    
    This is the main function for Arena experiments. It evaluates both
    answers using their respective evaluation methods and returns a
    comparison result.
    
    Args:
        question: The research question.
        rag_answer: Answer from RAG/Deep Research with sources.
        pure_llm_answer: Answer from Pure LLM without sources.
        documents: Retrieved documents (for RAG evaluation only).
        
    Returns:
        Dict with 'rag_eval', 'pure_llm_eval', and 'winner' keys.
    """
    evaluator = RAGEvaluator()
    
    # Evaluate RAG answer with documents
    rag_eval = await evaluator.evaluate_detailed(
        question=question,
        documents=documents,
        answer=rag_answer,
    )
    
    # Evaluate Pure LLM answer using RAG answer as ground truth
    pure_llm_eval = await evaluator.evaluate_pure_llm(
        question=question,
        pure_llm_answer=pure_llm_answer,
        ground_truth=rag_answer,  # Use RAG answer as ground truth
    )
    
    # Determine winner based on weighted score
    rag_score = rag_eval.weighted_score
    pure_llm_score = pure_llm_eval.weighted_score
    
    if rag_score > pure_llm_score + 0.5:  # RAG wins with margin
        winner = "rag"
    elif pure_llm_score > rag_score + 0.5:  # Pure LLM wins with margin
        winner = "pure_llm"
    else:
        winner = "tie"
    
    return {
        "rag_eval": rag_eval.model_dump(),
        "pure_llm_eval": pure_llm_eval.model_dump(),
        "rag_score": rag_score,
        "pure_llm_score": pure_llm_score,
        "winner": winner,
        "score_difference": rag_score - pure_llm_score,
    }
