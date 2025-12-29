"""
Result Synthesizer Module

Combines results from multiple sub-task RAG queries into a coherent
research report.
"""

# Standard library
import asyncio
import logging
from typing import List, Optional

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)


class SubTaskResult(BaseModel):
    """Result from a single sub-task."""
    task_id: int
    question: str
    answer: str
    sources: List[str] = []
    confidence: float = 1.0
    status: str = "completed"  # For future SSE: pending, loading, completed, failed


class ResearchReport(BaseModel):
    """Final synthesized research report."""
    original_question: str
    summary: str
    detailed_answer: str
    sub_results: List[SubTaskResult]
    all_sources: List[str]
    confidence: float = 1.0


# Prompt for result synthesis
_SYNTHESIZER_PROMPT = """你是一個研究報告撰寫專家。請根據以下子問題的回答，綜合生成一份完整的研究報告。

原始研究問題：{original_question}

子問題與回答：
{sub_results}

要求：
1. 首先提供一段摘要（2-3句話）
2. 然後提供完整的綜合回答
3. 如果各子回答有矛盾，請說明並嘗試調和
4. 使用繁體中文
5. 保持學術嚴謹的語氣

請按以下格式輸出：

## 摘要
[摘要內容]

## 詳細分析
[完整綜合回答]"""


class ResultSynthesizer:
    """
    Synthesizes results from multiple sub-task queries.
    
    Combines answers from multiple RAG queries into a coherent
    research report, handling potential contradictions and
    providing a unified narrative.
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
    """
    
    def __init__(self, max_concurrent: int = 2) -> None:
        """
        Initializes the synthesizer.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    def _format_sub_results(self, results: List[SubTaskResult]) -> str:
        """
        Formats sub-results for the synthesis prompt.
        
        Args:
            results: List of sub-task results.
            
        Returns:
            Formatted string of results.
        """
        formatted = []
        for r in results:
            formatted.append(
                f"### 問題 {r.task_id}: {r.question}\n"
                f"**回答**: {r.answer}\n"
            )
        return "\n".join(formatted)
    
    def _parse_report(
        self,
        response: str,
        original_question: str,
        sub_results: List[SubTaskResult],
    ) -> ResearchReport:
        """
        Parses LLM response into ResearchReport.
        
        Args:
            response: LLM response text.
            original_question: Original research question.
            sub_results: Sub-task results.
            
        Returns:
            ResearchReport.
        """
        # Extract summary and detailed answer
        summary = ""
        detailed = response
        
        if "## 摘要" in response:
            parts = response.split("## 詳細分析")
            if len(parts) >= 2:
                summary_part = parts[0].replace("## 摘要", "").strip()
                summary = summary_part
                detailed = parts[1].strip()
        
        if not summary:
            # Use first paragraph as summary
            paragraphs = response.split("\n\n")
            summary = paragraphs[0] if paragraphs else ""
        
        # Collect all sources
        all_sources = []
        for r in sub_results:
            all_sources.extend(r.sources)
        all_sources = list(set(all_sources))  # Deduplicate
        
        # Calculate confidence as average
        avg_confidence = (
            sum(r.confidence for r in sub_results) / len(sub_results)
            if sub_results else 1.0
        )
        
        return ResearchReport(
            original_question=original_question,
            summary=summary,
            detailed_answer=detailed,
            sub_results=sub_results,
            all_sources=all_sources,
            confidence=avg_confidence,
        )
    
    async def synthesize(
        self,
        original_question: str,
        sub_results: List[SubTaskResult],
    ) -> ResearchReport:
        """
        Synthesizes sub-task results into a research report.
        
        Args:
            original_question: Original research question.
            sub_results: Results from sub-task RAG queries.
            
        Returns:
            ResearchReport with synthesized answer.
        """
        if not sub_results:
            return ResearchReport(
                original_question=original_question,
                summary="無法生成報告：沒有子任務結果。",
                detailed_answer="",
                sub_results=[],
                all_sources=[],
                confidence=0.0,
            )
        
        # If only one result, use it directly
        if len(sub_results) == 1:
            r = sub_results[0]
            return ResearchReport(
                original_question=original_question,
                summary=r.answer[:200] + "..." if len(r.answer) > 200 else r.answer,
                detailed_answer=r.answer,
                sub_results=sub_results,
                all_sources=r.sources,
                confidence=r.confidence,
            )
        
        async with self._semaphore:
            try:
                llm = get_llm("synthesizer")
                
                formatted_results = self._format_sub_results(sub_results)
                
                prompt = _SYNTHESIZER_PROMPT.format(
                    original_question=original_question,
                    sub_results=formatted_results,
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                report = self._parse_report(
                    response.content,
                    original_question,
                    sub_results,
                )
                
                logger.info(f"Synthesized report from {len(sub_results)} sub-results")
                
                return report
                
            except Exception as e:
                logger.warning(f"Result synthesis failed: {e}")
                # Fallback: concatenate results
                combined = "\n\n".join([
                    f"**{r.question}**\n{r.answer}"
                    for r in sub_results
                ])
                return ResearchReport(
                    original_question=original_question,
                    summary="綜合回答（無法完成自動綜合）",
                    detailed_answer=combined,
                    sub_results=sub_results,
                    all_sources=list(set(
                        s for r in sub_results for s in r.sources
                    )),
                    confidence=0.5,
                )


async def synthesize_results(
    original_question: str,
    sub_results: List[SubTaskResult],
    enabled: bool = True,
) -> ResearchReport:
    """
    Convenience function to synthesize results.
    
    Args:
        original_question: Original research question.
        sub_results: Sub-task results.
        enabled: If False, returns simple concatenation.
        
    Returns:
        ResearchReport.
    """
    # Skip LLM synthesis for ≤2 results - use direct concatenation instead
    if not enabled or len(sub_results) <= 2:
        if sub_results:
            # Combine all results without LLM call
            combined_answer = "\n\n".join([
                f"### {r.question}\n{r.answer}" for r in sub_results
            ])
            all_sources = list(set(s for r in sub_results for s in r.sources))
            avg_confidence = sum(r.confidence for r in sub_results) / len(sub_results)
            
            return ResearchReport(
                original_question=original_question,
                summary=sub_results[0].answer[:200] if sub_results[0].answer else "",
                detailed_answer=combined_answer,
                sub_results=sub_results,
                all_sources=all_sources,
                confidence=avg_confidence,
            )
        return ResearchReport(
            original_question=original_question,
            summary="",
            detailed_answer="",
            sub_results=[],
            all_sources=[],
            confidence=0.0,
        )
    
    synthesizer = ResultSynthesizer()
    return await synthesizer.synthesize(original_question, sub_results)
