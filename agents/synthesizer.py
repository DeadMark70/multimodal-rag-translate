"""
Result Synthesizer Module

Combines results from multiple sub-task RAG queries into a coherent
research report.
"""

# Standard library
import asyncio
import logging
from typing import Any, List, Optional

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Local application
from agents.planner import QuestionIntent, classify_question_intent
from core.providers import get_llm
from core.llm_usage_context import llm_accounting_phase
from core.prompt_loader import format_agentic_rag_prompt

# Configure logging
logger = logging.getLogger(__name__)

_NO_CONFLICT_SENTINEL = "NO_CONFLICT"


def _is_no_conflict_statement(statement: str) -> bool:
    text = (statement or "").strip()
    if not text:
        return True
    upper = text.upper()
    return (
        _NO_CONFLICT_SENTINEL in upper
        or "無衝突" in text
        or "沒有衝突" in text
    )


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


def _synthesis_guidance_for_intent(question_intent: QuestionIntent) -> str:
    if question_intent == "comparison_disambiguation":
        return """
## 題型聚焦：比較 / 概念消歧
- 第一段必須直接回答「A 與 B 的本質差異」。
- 必須明確指出容易混淆之處，使用「不要把 X 和 Y 混為一談」或同等明確措辭。
- 除非原題明問，禁止把資料集規模、年份、背景脈絡寫成主體。
"""
    if question_intent == "figure_flow":
        return """
## 題型聚焦：架構流程 / Figure
- 第一段必須直接輸出有序流程，可用 `A -> B -> C` 或等價順序表述。
- 先輸出流程主鏈，再補充分支，不可混成敘事段落。
- 僅保留 block/branch/order 相關資訊。
- 禁止新增子結果未出現的元件或步驟名稱。
- 除非原題明問，禁止補充尺寸、背景、訓練設定。
- 第一段禁止題目回顯或標題式開頭（例如：`### What is ...`）。
"""
    if question_intent == "benchmark_data":
        return """
## 題型聚焦：Benchmark / 數據
- 第一段先列「指標-模型-數值-來源」（至少 2 列，若可得）。
- 每列都要附來源標記，禁止只寫排名不寫數值。
- 若沒有明確數字，必須誠實標示「資料不足」。
- 不確定時收斂答案，只保留可被子結果直接支持的句子。
"""
    if question_intent == "enumeration_definition":
        return """
## 題型聚焦：列舉 / 定義
- 第一段必須先列出完整項目數量或完整清單。
- 後續條列每項只保留一句短特徵，避免延伸成背景介紹。
"""
    return ""


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

    def _format_arbitration_input(self, results: List[SubTaskResult]) -> str:
        lines: list[str] = []
        for result in results:
            sources = ", ".join(result.sources) if result.sources else "(unknown)"
            answer = (result.answer or "").strip()
            if len(answer) > 800:
                answer = f"{answer[:797]}..."
            lines.append(
                f"[Task {result.task_id}] Question: {result.question}\n"
                f"Sources: {sources}\n"
                f"Answer: {answer}"
            )
        return "\n\n".join(lines)

    async def _detect_and_arbitrate_conflicts(
        self,
        *,
        llm: Any,
        sub_results: List[SubTaskResult],
    ) -> str:
        if len(sub_results) < 2:
            return _NO_CONFLICT_SENTINEL

        prompt = format_agentic_rag_prompt(
            "conflict_arbitration",
            sub_results=self._format_arbitration_input(sub_results),
        )
        try:
            with llm_accounting_phase("agent_synthesis"):
                response = await llm.ainvoke([HumanMessage(content=prompt)])
            statement = (getattr(response, "content", "") or "").strip()
            if _is_no_conflict_statement(statement):
                return _NO_CONFLICT_SENTINEL
            return statement
        except Exception as exc:  # noqa: BLE001
            logger.warning("Conflict arbitration failed; fallback to NO_CONFLICT: %s", exc)
            return _NO_CONFLICT_SENTINEL
    
    def _strip_think_tags(self, text: str) -> str:
        """
        Removes <think>...</think> blocks from LLM output.
        
        Phase 5: The think tags contain the conflict reasoning process
        which should not be shown to end users.
        
        Args:
            text: Raw LLM response with potential think tags.
            
        Returns:
            Cleaned text without think blocks.
        """
        import re
        # Remove <think>...</think> blocks (including multiline and case insensitive)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Clean up extra whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
    
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
        # Phase 5: Strip <think> tags before processing
        response = self._strip_think_tags(response)
        
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
        
        # Calculate base confidence as average
        avg_confidence = (
            sum(r.confidence for r in sub_results) / len(sub_results)
            if sub_results else 1.0
        )
        
        # Phase 6.2: 衝突懲罰機制
        # 當報告中存在衝突觀點時，降低信心分數以反映不確定性
        conflict_keywords = [
            "衝突", "不一致", "相矛盾", "一方面", "另一方面",
            "conflict", "inconsistent", "contradictory", "on the other hand",
            "然而", "但是", "相反", "However", "contrary"
        ]
        
        has_conflict = any(kw in detailed for kw in conflict_keywords)
        conflict_penalty = 0.8 if has_conflict else 1.0
        final_confidence = avg_confidence * conflict_penalty
        
        if has_conflict:
            logger.info(
                f"Phase 6.2 Conflict detected: applying {conflict_penalty:.0%} penalty "
                f"(confidence: {avg_confidence:.2f} -> {final_confidence:.2f})"
            )
        
        return ResearchReport(
            original_question=original_question,
            summary=summary,
            detailed_answer=detailed,
            sub_results=sub_results,
            all_sources=all_sources,
            confidence=final_confidence,  # Phase 6.2: 使用校準後的信心度
        )
    
    async def synthesize(
        self,
        original_question: str,
        sub_results: List[SubTaskResult],
        use_academic_template: bool = False,
        question_intent: Optional[QuestionIntent] = None,
        force_llm_for_single: bool = False,
        enable_conflict_arbitration: bool = True,
    ) -> ResearchReport:
        """
        Synthesizes sub-task results into a research report.
        
        Args:
            original_question: Original research question.
            sub_results: Results from sub-task RAG queries.
            use_academic_template: If True, use structured academic report format.
                                   Recommended for Deep Research flows.
            
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
        
        # If only one result, use it directly unless caller requests
        # a lightweight normalization pass.
        if len(sub_results) == 1 and not force_llm_for_single:
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
                effective_intent = question_intent or classify_question_intent(
                    original_question
                )
                arbitration_statement = _NO_CONFLICT_SENTINEL
                if enable_conflict_arbitration:
                    arbitration_statement = await self._detect_and_arbitrate_conflicts(
                        llm=llm,
                        sub_results=sub_results,
                    )
                
                # Select prompt template based on use_academic_template
                prompt = format_agentic_rag_prompt(
                    "academic_report" if use_academic_template else "synthesizer",
                    original_question=original_question,
                    sub_results=formatted_results,
                )
                prompt = (
                    f"{prompt}\n\n"
                    f"{_synthesis_guidance_for_intent(effective_intent)}"
                )
                if not _is_no_conflict_statement(arbitration_statement):
                    prompt = (
                        f"{prompt}\n\n"
                        "## 衝突仲裁指示\n"
                        "子結果存在衝突，請嚴格遵守以下仲裁結論撰寫最終報告：\n"
                        f"{arbitration_statement}\n"
                        "- 禁止和稀泥式結論；若證據權重已明確，必須明確選邊。"
                    )
                    logger.info("Conflict arbitration injected into synthesizer prompt")
                
                message = HumanMessage(content=prompt)
                with llm_accounting_phase("agent_synthesis"):
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


    def _is_failed_answer(self, text: str) -> bool:
        """Checks if the answer indicates a RAG retrieval failure."""
        failure_markers = ["抱歉", "找不到", "無法回答", "未提及", "沒有資料", "not found", "unable to answer"]
        return any(marker in text for marker in failure_markers) and len(text) < 150

async def synthesize_results(
    original_question: str,
    sub_results: List[SubTaskResult],
    enabled: bool = True,
    use_academic_template: bool = False,
    question_intent: Optional[QuestionIntent] = None,
    force_llm_for_single: bool = False,
    enable_conflict_arbitration: bool = True,
) -> ResearchReport:
    """
    Convenience function to synthesize results.
    
    Args:
        original_question: Original research question.
        sub_results: Sub-task results.
        enabled: If False, returns simple concatenation.
        use_academic_template: If True, use structured academic report format.
        
    Returns:
        ResearchReport.
    """
    # Check for general failure across all sub-results
    all_failed = all(ResultSynthesizer()._is_failed_answer(r.answer) for r in sub_results) if sub_results else False
    
    # Skip LLM synthesis ONLY if disabled or all failed. We want synthesis even for small results to enforce BLUF.
    if not enabled or all_failed:
        if sub_results:
            if all_failed:
                summary = "檢索失敗：在目前的知識庫中找不到相關資訊。"
                detailed_answer = "抱歉，系統嘗試分析了多個子面向，但在現有文件中均未發現確切證據。建議：\n1. 確認是否已上傳正確的 PDF 文件\n2. 嘗試調整問題的關鍵字\n3. 檢查文件 OCR 處理是否完整。"
            else:
                summary = sub_results[0].answer[:200] if sub_results[0].answer else ""
                detailed_answer = "\n\n".join([
                    f"### {r.question}\n{r.answer}" for r in sub_results
                ])
            
            all_sources = list(set(s for r in sub_results for s in r.sources))
            avg_confidence = sum(r.confidence for r in sub_results) / len(sub_results)
            
            return ResearchReport(
                original_question=original_question,
                summary=summary,
                detailed_answer=detailed_answer,
                sub_results=sub_results,
                all_sources=all_sources,
                confidence=0.0 if all_failed else avg_confidence,
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
    return await synthesizer.synthesize(
        original_question, 
        sub_results, 
        use_academic_template=use_academic_template,
        question_intent=question_intent,
        force_llm_for_single=force_llm_for_single,
        enable_conflict_arbitration=enable_conflict_arbitration,
    )
