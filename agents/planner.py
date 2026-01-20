"""
Task Planner Module

Provides task decomposition for complex research questions.
Breaks down complex questions into manageable sub-tasks for RAG.
Supports GraphRAG-aware planning for multi-document research.
"""

# Standard library
import asyncio
import logging
import re
from typing import List, Literal, Optional

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)


# Task types
TaskType = Literal["rag", "graph_analysis"]


class SubTask(BaseModel):
    """A single sub-task in the research plan."""
    id: int
    question: str
    depends_on: List[int] = []
    priority: int = 1
    task_type: TaskType = Field(
        default="rag",
        description="Task type: 'rag' for vector search, 'graph_analysis' for graph traversal"
    )


class ResearchPlan(BaseModel):
    """A complete research plan with sub-tasks."""
    original_question: str
    sub_tasks: List[SubTask]
    estimated_complexity: str = "medium"


# Prompt for task decomposition (standard RAG)
_PLANNER_PROMPT = """你是一個研究規劃專家。請將以下複雜問題分解為 2-5 個可獨立回答的子問題。
這個問題通常涉及多個實體或概念的比較、分析或綜合。

原始問題：{question}

要求：
1. **專注性 (Focus)**：子問題必須直接有助於回答原始問題。避免生成過於寬泛或教科書式的定義問題（如「什麼是 X？」），除非該定義對區分 X 和 Y 至關重要。
2. **比較性 (Comparison)**：如果原問題是 A vs B，請確保生成針對「A 與 B 的差異」、「A 與 B 的基準測試比較」或「A 與 B 的優缺點對比」的子問題。
3. **查證性 (Verification)**：若需要具體數據，請生成「查詢 X 在 Y 任務上的具體性能數據」此類的子問題。
4. **依賴關係**：如果某個子問題依賴另一個的答案，請標註。
5. 以數字編號列出。
6. **語言要求 (Language)**：為了確保對英文學術論文的檢索效果，請**務必將所有子問題翻譯為英文** (Translate sub-questions to English)。

## 視覺查證規範 (Strict Visual Requirement)
- 如果原始問題涉及「圖表」、「Figure」、「Figure 1」、「圖片中的位置/細節」，且你認為文字檢索可能不足以提供精確數據，你必須在子問題中明確要求「查證圖片內容」。
- 嚴禁在未嘗試調用工具的情況下回答「不知道」。

## 負面約束 (Negative Constraints)
- **除非原始問題明確詢問**，否則嚴禁生成關於 "Interactive Segmentation"、"SAM (Segment Anything Model)"、"SegVol" 或 "Annotation" 相關的子問題。
- 我們只關注原始問題中提及的模型或技術的直接比較。

輸出格式（每行一個英文子問題）：
1. [Sub-question 1 in English]
2. [Sub-question 2 in English]
3. [Sub-question 3 in English]

子問題列表："""


# Prompt for GraphRAG-aware planning
_GRAPH_PLANNER_PROMPT = """你是一個研究規劃專家。請將以下複雜問題分解為子問題，並標註每個問題適合的查詢方式。

原始問題：{question}

可用的查詢方式：
- [RAG] 向量檢索：適合查找特定事實、數據、定義、引用
- [GRAPH] 圖譜分析：適合分析實體關係、跨文件比較、趨勢分析

要求：
1. 每個子問題應該是具體、可回答的
2. 為每個子問題選擇最適合的查詢方式
3. 關係類問題用 [GRAPH]，事實類問題用 [RAG]
4. **語言要求**：為了匹配學術文獻，請將**所有子問題寫成英文** (Write all sub-questions in English)。
5. 以數字編號列出，格式：[查詢方式] 子問題

## 視覺查證規範 (Strict Visual Requirement)
- 如果問題涉及「圖片」、「圖表」、「Figure X」，且需要精確細節，請生成一個 [RAG] 子任務來專門執行「視覺查證」。
- 即使檢索到文字摘要，若摘要不含具體位置或數值，仍須標註需要看圖。

輸出格式：
1. [RAG] What is the definition of X?
2. [GRAPH] Analyze the relationship between A and B
3. [RAG] What is the specific data for X?

子問題列表："""


# Prompt for follow-up task generation (drill-down)
_FOLLOWUP_PROMPT = """你正在協助研究以下問題：
{original_question}

目前已找到的資訊：
{current_findings}

已經問過的問題：
{existing_questions}

請判斷是否有任何概念、數據、專有名詞或主張需要進一步在文件中「查證」或「挖掘細節」？

## Phase 6.3: 對抗性思維 (Critical - 必須遵循)

針對每個核心論點，請同時考慮：
1. 支持該論點的證據
2. **限制或反對該論點的證據** (必要)

## 視覺查證規範 (Strict Visual Requirement)
- 如果目前的發現中提到「如圖所示」、「Figure 1 顯示了...」但沒有寫出圖中具體數據，**你必須**生成一個追問子任務來要求「檢視該圖片以獲取具體細節」。
- 嚴禁跳過圖片細節的查證。

例如：
- 若論點是「方法 A 效果好」→ 同時生成查詢「What are the limitations of Method A?」
- 若論點是「X 優於 Y」→ 同時生成查詢「Is there evidence that Y outperforms X?」

## 注意事項
1. 我們只能查閱現有的文件，不能上網搜尋
2. 不要重複已經問過的問題
3. 只針對文件中「提到但未詳細解釋」的內容追問
4. **必須包含至少一個「反面/限制」相關的查詢**
5. **語言要求**：請用**英文**撰寫追問子任務 (Write follow-up tasks in English)。

如果資訊已經足夠完整（包含正反面觀點），請回覆：無需追加查詢

如果需要深入查詢，請列出 1-3 個子任務，格式：
1. [RAG] Specific question in English
2. [GRAPH] Specific analysis question in English

子任務列表："""


# Prompt for smart query refinement based on evaluation failure
_REFINE_QUERY_PROMPT = """你是一個搜尋策略專家。一個 AI 系統剛剛嘗試回答問題但品質不佳。

原始問題：{original_question}
之前的回答：{failed_answer}
評估失敗原因：{evaluation_reason}

請根據失敗原因，生成一個**修正過的搜尋查詢**來補救問題。

## 策略指南
- 如果原因提到「資料太舊」或「outdated」→ 加入「latest」、「recent」、「2024」等時間限定詞
- 如果原因提到「缺乏數據」或「no data」→ 改為搜尋「statistics」、「data」、「percentage」、「quantitative」
- 如果原因提到「圖片細節不足」或「視覺資訊不足」→ 加入「visual verification」、「details of Figure X」、「check image」
- 如果原因提到「定義不清」或「unclear」→ 增加「definition」、「what is」
- 如果原因提到「缺乏比較」或「no comparison」→ 增加「versus」、「comparison」、「difference」
- 如果原因提到「證據不足」或「insufficient」→ 聚焦「evidence」、「study」、「paper」
- 如果原因提到「範圍太廣」或「too broad」→ 縮小搜尋範圍，更具體化
- 如果原因提到「不完整」或「incomplete」→ 擴大搜尋範圍，增加相關細節

## 要求
1. 新的查詢必須與原始問題有**明顯差異**
2. 針對失敗原因做出有針對性的修正
3. 保持查詢簡潔（30 字以內），適合向量檢索
4. **務必使用英文 (English)** 撰寫新的查詢，以獲得最佳學術檢索結果

請直接輸出修正後的搜尋查詢，不要其他內容："""


class TaskPlanner:
    """
    Decomposes complex research questions into sub-tasks.
    
    Uses LLM to identify the components of a complex question
    and break them into manageable, answerable sub-questions.
    Supports GraphRAG-aware planning for research queries.
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
        max_subtasks: Maximum number of sub-tasks to generate.
        enable_graph_planning: Whether to use graph-aware prompts.
    """
    
    def __init__(
        self,
        max_concurrent: int = 2,
        max_subtasks: int = 5,
        enable_graph_planning: bool = False,
    ) -> None:
        """
        Initializes the task planner.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
            max_subtasks: Maximum number of sub-tasks.
            enable_graph_planning: Use graph-aware planning prompts.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.max_subtasks = max_subtasks
        self.enable_graph_planning = enable_graph_planning
    
    def _parse_subtasks(self, response: str) -> List[SubTask]:
        """
        Parses LLM response into sub-tasks.
        
        Handles both standard and graph-aware formats.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            List of SubTask objects.
        """
        subtasks = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Match numbered items with optional [TAG]
            # Format: "1. [GRAPH] question" or "1. question"
            match = re.match(r'^(\d+)[\.\)]\s*(?:\[(\w+)\])?\s*(.+)$', line)
            if match:
                task_id = int(match.group(1))
                tag = match.group(2)  # GRAPH or RAG or None
                question = match.group(3).strip()
                
                # Determine task type from tag
                task_type: TaskType = "rag"
                if tag and tag.upper() == "GRAPH":
                    task_type = "graph_analysis"
                
                if question and len(question) >= 3:
                    subtasks.append(SubTask(
                        id=task_id,
                        question=question,
                        priority=task_id,
                        task_type=task_type,
                    ))
        
        return subtasks[:self.max_subtasks]
    
    async def plan(self, question: str) -> ResearchPlan:
        """
        Decomposes a complex question into sub-tasks.
        
        Args:
            question: Complex research question.
            
        Returns:
            ResearchPlan with sub-tasks.
        """
        async with self._semaphore:
            try:
                llm = get_llm("planner")
                
                # Choose prompt based on graph planning mode
                if self.enable_graph_planning:
                    prompt = _GRAPH_PLANNER_PROMPT.format(question=question)
                else:
                    prompt = _PLANNER_PROMPT.format(question=question)
                
                message = HumanMessage(content=prompt)
                
                response = await llm.ainvoke([message])
                subtasks = self._parse_subtasks(response.content)
                
                if not subtasks:
                    # Fallback: use original question as single task
                    subtasks = [SubTask(id=1, question=question)]
                
                # Estimate complexity
                complexity = "simple"
                if len(subtasks) >= 4:
                    complexity = "complex"
                elif len(subtasks) >= 2:
                    complexity = "medium"
                
                # Log task type breakdown
                graph_count = sum(1 for t in subtasks if t.task_type == "graph_analysis")
                logger.info(
                    f"Planned {len(subtasks)} sub-tasks for research "
                    f"({graph_count} graph, {len(subtasks) - graph_count} rag)"
                )
                
                return ResearchPlan(
                    original_question=question,
                    sub_tasks=subtasks,
                    estimated_complexity=complexity,
                )
                
            except Exception as e:
                logger.warning(f"Task planning failed: {e}")
                # Fallback: return original question as single task
                return ResearchPlan(
                    original_question=question,
                    sub_tasks=[SubTask(id=1, question=question)],
                    estimated_complexity="simple",
                )
    
    def needs_planning(self, question: str) -> bool:
        """
        Heuristic check if question needs decomposition.
        
        Args:
            question: User question.
            
        Returns:
            True if question appears complex.
        """
        # Complex question indicators
        complex_indicators = [
            "比較", "對比", "分析", "評估", "研究",
            "以及", "並且", "同時", "還有", "甚至",
            "怎麼", "為什麼", "如何", "哪些", "什麼是",
            "compare", "analyze", "evaluate", "research",
            "and", "as well as", "also", "including",
        ]
        
        question_lower = question.lower()
        
        # Check for complexity indicators
        indicator_count = sum(
            1 for ind in complex_indicators
            if ind in question_lower
        )
        
        # Long questions or multiple indicators suggest complexity
        return len(question) > 40 or indicator_count >= 2
    
    def needs_graph_analysis(self, question: str) -> bool:
        """
        Heuristic check if question benefits from graph analysis.
        
        Args:
            question: User question.
            
        Returns:
            True if question likely benefits from graph analysis.
        """
        graph_indicators = [
            "關係", "連結", "趨勢", "比較", "對比",
            "這些論文", "這幾篇", "跨文件", "綜合", "關聯",
            "relationship", "connection", "trend", "compare",
            "across", "these papers", "multi-document", "global",
        ]
        
        question_lower = question.lower()
        
        return any(ind in question_lower for ind in graph_indicators)
    
    async def create_followup_tasks(
        self,
        original_question: str,
        current_findings: str,
        existing_tasks: List[SubTask],
    ) -> List[SubTask]:
        """
        Generates follow-up tasks based on knowledge gaps in current findings.
        
        This method analyzes the current research results and identifies
        concepts, data, or claims that need further investigation.
        Used for recursive drill-down in deep research.
        
        Args:
            original_question: The original research question.
            current_findings: Summary of current findings.
            existing_tasks: List of already executed tasks.
            
        Returns:
            List of follow-up SubTask objects, or empty list if no gaps found.
        """
        async with self._semaphore:
            try:
                llm = get_llm("planner")
                
                # Build list of existing questions to avoid duplicates
                existing_questions = [t.question for t in existing_tasks]
                existing_list = "\n".join(
                    f"- {q}" for q in existing_questions
                )
                
                prompt = _FOLLOWUP_PROMPT.format(
                    original_question=original_question,
                    current_findings=current_findings,
                    existing_questions=existing_list,
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                # Check if no follow-up needed
                content = response.content.strip()
                if "無需追加" in content or "不需要" in content or "已經足夠" in content:
                    logger.info("No knowledge gaps identified")
                    return []
                
                # Parse follow-up tasks
                followup_tasks = self._parse_subtasks(content)
                
                # Filter out duplicate questions
                filtered_tasks = [
                    task for task in followup_tasks
                    if not any(
                        self._is_similar_question(task.question, existing)
                        for existing in existing_questions
                    )
                ]
                
                logger.info(f"Generated {len(filtered_tasks)} follow-up tasks")
                return filtered_tasks[:3]  # Limit to 3 follow-ups per iteration
                
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Follow-up task generation failed: {e}")
                return []
    
    def _is_similar_question(self, q1: str, q2: str) -> bool:
        """
        Checks if two questions are similar enough to be considered duplicates.
        
        Uses character bigrams for better CJK language support.
        
        Args:
            q1: First question.
            q2: Second question.
            
        Returns:
            True if questions are similar.
        """
        # Normalize: lowercase and remove punctuation
        import re
        q1_clean = re.sub(r'[^\w\s]', '', q1.lower())
        q2_clean = re.sub(r'[^\w\s]', '', q2.lower())
        
        # Generate character bigrams for CJK support
        def get_bigrams(text: str) -> set:
            text = text.replace(' ', '')  # Remove spaces for Chinese
            if len(text) < 2:
                return {text} if text else set()
            return {text[i:i+2] for i in range(len(text) - 1)}
        
        bigrams1 = get_bigrams(q1_clean)
        bigrams2 = get_bigrams(q2_clean)
        
        if not bigrams1 or not bigrams2:
            return False
        
        # Jaccard similarity
        overlap = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return overlap / union > 0.5 if union > 0 else False
    
    async def refine_query_from_evaluation(
        self,
        original_question: str,
        evaluation_reason: str,
        failed_answer: str,
    ) -> str:
        """
        Generates a refined search query based on evaluation failure reason.
        
        This method enables "smart retry" - instead of repeating the same query,
        it analyzes the evaluation reason and modifies the search strategy
        accordingly.
        
        Args:
            original_question: The original question that produced low-quality answer.
            evaluation_reason: The reason from Evaluator explaining why quality is low.
            failed_answer: The answer that was evaluated as insufficient.
            
        Returns:
            A refined search query that addresses the evaluation failure.
            Falls back to original question if refinement fails.
        """
        async with self._semaphore:
            try:
                llm = get_llm("planner")
                
                # Truncate long inputs
                truncated_answer = failed_answer[:500] if len(failed_answer) > 500 else failed_answer
                truncated_reason = evaluation_reason[:200] if len(evaluation_reason) > 200 else evaluation_reason
                
                prompt = _REFINE_QUERY_PROMPT.format(
                    original_question=original_question,
                    evaluation_reason=truncated_reason,
                    failed_answer=truncated_answer,
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                refined_query = response.content.strip()
                
                # Validate refined query
                if not refined_query or len(refined_query) < 5:
                    logger.warning("Refined query too short, using original")
                    return original_question
                
                # Check if query is actually different
                if self._is_similar_question(refined_query, original_question):
                    logger.warning("Refined query too similar to original, appending modifier")
                    # Append a modifier based on common patterns
                    if "數據" not in refined_query and "data" not in refined_query.lower():
                        refined_query = f"{refined_query} 具體數據"
                
                logger.info(f"Smart retry: '{original_question[:30]}...' -> '{refined_query[:30]}...'")
                return refined_query
                
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Query refinement failed: {e}")
                return original_question


async def plan_research(
    question: str,
    enabled: bool = True,
    max_subtasks: int = 5,
    enable_graph_planning: bool = False,
) -> ResearchPlan:
    """
    Convenience function to plan research.
    
    Args:
        question: Research question.
        enabled: If False, returns single-task plan.
        max_subtasks: Maximum sub-tasks.
        enable_graph_planning: Use graph-aware planning.
        
    Returns:
        ResearchPlan.
    """
    if not enabled:
        return ResearchPlan(
            original_question=question,
            sub_tasks=[SubTask(id=1, question=question)],
            estimated_complexity="simple",
        )
    
    planner = TaskPlanner(
        max_subtasks=max_subtasks,
        enable_graph_planning=enable_graph_planning,
    )
    return await planner.plan(question)
