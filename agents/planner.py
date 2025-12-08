"""
Task Planner Module

Provides task decomposition for complex research questions.
Breaks down complex questions into manageable sub-tasks for RAG.
"""

# Standard library
import asyncio
import logging
import re
from typing import List, Optional

# Third-party
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)


class SubTask(BaseModel):
    """A single sub-task in the research plan."""
    id: int
    question: str
    depends_on: List[int] = []
    priority: int = 1


class ResearchPlan(BaseModel):
    """A complete research plan with sub-tasks."""
    original_question: str
    sub_tasks: List[SubTask]
    estimated_complexity: str = "medium"


# Prompt for task decomposition
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


class TaskPlanner:
    """
    Decomposes complex research questions into sub-tasks.
    
    Uses LLM to identify the components of a complex question
    and break them into manageable, answerable sub-questions.
    
    Attributes:
        _semaphore: Rate limiter for LLM calls.
        max_subtasks: Maximum number of sub-tasks to generate.
    """
    
    def __init__(
        self,
        max_concurrent: int = 2,
        max_subtasks: int = 5,
    ) -> None:
        """
        Initializes the task planner.
        
        Args:
            max_concurrent: Maximum concurrent LLM calls.
            max_subtasks: Maximum number of sub-tasks.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.max_subtasks = max_subtasks
    
    def _parse_subtasks(self, response: str) -> List[SubTask]:
        """
        Parses LLM response into sub-tasks.
        
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
            
            # Match numbered items
            match = re.match(r'^(\d+)[\.\)]\s*(.+)$', line)
            if match:
                task_id = int(match.group(1))
                question = match.group(2).strip()
                
                if question and len(question) > 5:
                    subtasks.append(SubTask(
                        id=task_id,
                        question=question,
                        priority=task_id,
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
                
                logger.info(f"Planned {len(subtasks)} sub-tasks for research")
                
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


async def plan_research(
    question: str,
    enabled: bool = True,
    max_subtasks: int = 5,
) -> ResearchPlan:
    """
    Convenience function to plan research.
    
    Args:
        question: Research question.
        enabled: If False, returns single-task plan.
        max_subtasks: Maximum sub-tasks.
        
    Returns:
        ResearchPlan.
    """
    if not enabled:
        return ResearchPlan(
            original_question=question,
            sub_tasks=[SubTask(id=1, question=question)],
            estimated_complexity="simple",
        )
    
    planner = TaskPlanner(max_subtasks=max_subtasks)
    return await planner.plan(question)
