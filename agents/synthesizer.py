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


# Academic report template for Deep Research (Phase 5: Conflict Arbitration)
_ACADEMIC_REPORT_PROMPT = """你是一位專業的學術報告撰寫專家。請根據以下子問題的回答，綜合生成一份結構完整的學術研究報告。

原始研究問題：{original_question}

子問題與回答：
{sub_results}

## 衝突處理守則 (Critical: 必須遵循)

在撰寫報告前，請先在 <think> 標籤內執行以下推理步驟（此部分不顯示於報告正文）：

<think>
### 觀點盤點
列出各來源的核心論點：
- [來源 1]: {{觀點}} (支持/反駁)
- [來源 2]: {{觀點}} (支持/反駁)

### 證據權重判斷
根據以下優先順序：
1. 大規模基準測試 (Benchmark) > 單一實驗
2. 較新發表年份 > 較舊發表年份
3. 多來源共識 > 單一來源聲稱

若 Metadata 無發表年份：
- 嘗試從文檔標題、頁首、頁尾推斷
- 若無法推斷，依賴「Benchmark vs 單一實驗」邏輯

### 結論選擇
採信: {{來源X}}，原因: {{理由}}
</think>

## 衝突處理格式（報告正文中若有衝突必須使用）

"一方面，{{來源A}} 主張...；
另一方面，{{來源B}} 的 {{Benchmark/實驗類型}} 顯示...。
**根據證據權重**，較可信的結論是...，理由是...。"

⚠️ 禁止模糊結論：不可使用「兩者互有優劣」「效果因情況而異」等和稀泥的表述。

---

## 報告結構（請嚴格遵循此格式）

### 1. Executive Summary (執行摘要)
- 用 2-3 句話總結關鍵發現
- 直接回答原始問題的核心
- 若發現衝突，明確表態採信哪方結論

### 2. Key Findings (主要發現)
- 以條列點整理最重要的發現
- 每個發現應有明確的資料支撐
- 標註衝突觀點（若有）

### 3. Detailed Analysis (詳細分析)
- 深入解釋每個發現
- 如有圖表數據，請使用 Markdown 格式引用圖片：`![圖表說明](圖片路徑)`
- 衝突觀點必須使用「衝突處理格式」呈現

### 4. Research Gaps (知識缺口)
- 指出目前資料庫中缺少的拼圖
- 建議後續研究方向

### 5. References (參考來源)
- 列出引用的所有來源文件
- 標註年份（若可得）

## 格式要求
1. 使用繁體中文
2. 保持學術嚴謹的語氣
3. 數學公式使用 LaTeX 格式
4. 若引用圖片摘要內容，務必以 `![描述](路徑)` 格式插入圖片

請開始撰寫報告："""


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
        # Remove <think>...</think> blocks (including multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
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
                
                # Select prompt template based on use_academic_template
                template = (
                    _ACADEMIC_REPORT_PROMPT if use_academic_template 
                    else _SYNTHESIZER_PROMPT
                )
                
                prompt = template.format(
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
    use_academic_template: bool = False,
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
    return await synthesizer.synthesize(
        original_question, 
        sub_results, 
        use_academic_template=use_academic_template,
    )
