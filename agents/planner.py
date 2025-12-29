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

原始問題：{question}

要求：
1. 每個子問題應該是具體、可回答的
2. 子問題應涵蓋原問題的不同面向
3. 如果某個子問題依賴另一個的答案，請標註
4. 以數字編號列出

輸出格式（每行一個子問題）：
1. [子問題1]
2. [子問題2]
3. [子問題3]

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
4. 以數字編號列出，格式：[查詢方式] 子問題

輸出格式：
1. [RAG] 查找 X 的定義
2. [GRAPH] 分析 A 與 B 的關係
3. [RAG] X 的具體數據是什麼

子問題列表："""


# Prompt for follow-up task generation (drill-down)
_FOLLOWUP_PROMPT = """你正在協助研究以下問題：
{original_question}

目前已找到的資訊：
{current_findings}

已經問過的問題：
{existing_questions}

請判斷是否有任何概念、數據、專有名詞或主張需要進一步在文件中「查證」或「挖掘細節」？

注意：
1. 我們只能查閱現有的文件，不能上網搜尋
2. 不要重複已經問過的問題
3. 只針對文件中「提到但未詳細解釋」的內容追問

如果資訊已經足夠完整，請回覆：無需追加查詢

如果需要深入查詢，請列出 1-3 個子任務，格式：
1. [RAG] 具體查詢問題
2. [GRAPH] 具體分析問題

子任務列表："""


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
                
                if question and len(question) > 5:
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
            "以及", "並且", "同時", "還有",
            "怎麼", "為什麼", "如何",
            "compare", "analyze", "evaluate", "research",
            "and", "as well as", "also",
        ]
        
        question_lower = question.lower()
        
        # Check for complexity indicators
        indicator_count = sum(
            1 for ind in complex_indicators
            if ind in question_lower
        )
        
        # Long questions or multiple indicators suggest complexity
        return len(question) > 100 or indicator_count >= 2
    
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
            "這些論文", "這幾篇", "跨文件", "綜合",
            "relationship", "connection", "trend", "compare",
            "across", "these papers", "multi-document",
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
        
        Simple heuristic: checks for significant word overlap.
        
        Args:
            q1: First question.
            q2: Second question.
            
        Returns:
            True if questions are similar.
        """
        # Simple word overlap check
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        
        # Remove common stopwords
        stopwords = {"的", "是", "什麼", "如何", "為什麼", "嗎", "呢", "了", "在", "有",
                     "the", "is", "what", "how", "why", "a", "an", "in", "of"}
        words1 -= stopwords
        words2 -= stopwords
        
        if not words1 or not words2:
            return False
        
        # Check overlap ratio
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        
        return overlap / min_len > 0.7 if min_len > 0 else False


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
