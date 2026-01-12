"""
Deep Research Schemas

Pydantic models for Interactive Deep Research API endpoints.
Supports Human-in-the-loop planning and recursive drill-down.
"""

# Standard library
from typing import List, Literal, Optional

# Third-party
from pydantic import BaseModel, Field


class EditableSubTask(BaseModel):
    """
    A user-editable sub-task in the research plan.
    
    Attributes:
        id: Unique task identifier.
        question: The sub-question to research.
        task_type: Query method - 'rag' for vector search, 'graph_analysis' for graph.
        enabled: Whether this task is enabled (user can toggle).
    """
    id: int
    question: str
    task_type: Literal["rag", "graph_analysis"] = "rag"
    enabled: bool = True


class ResearchPlanRequest(BaseModel):
    """
    Request model for generating a research plan.
    
    Attributes:
        question: The complex research question.
        doc_ids: Optional document IDs to restrict search scope.
        enable_graph_planning: Use graph-aware task planning prompts.
    """
    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: Optional[List[str]] = None
    enable_graph_planning: bool = False


class ResearchPlanResponse(BaseModel):
    """
    Response model for research plan generation.
    
    Returned by POST /rag/plan for user confirmation.
    
    Attributes:
        status: Always 'waiting_confirmation' to indicate pending user action.
        original_question: The original question submitted.
        sub_tasks: List of planned sub-tasks, editable by user.
        estimated_complexity: Estimated complexity level.
        doc_ids: Document IDs the research is scoped to.
    """
    status: Literal["waiting_confirmation"] = "waiting_confirmation"
    original_question: str
    sub_tasks: List[EditableSubTask]
    estimated_complexity: Literal["simple", "medium", "complex"] = "medium"
    doc_ids: Optional[List[str]] = None


class ExecutePlanRequest(BaseModel):
    """
    Request model for executing a confirmed research plan.
    
    Attributes:
        original_question: The original research question.
        sub_tasks: User-confirmed (possibly modified) sub-tasks.
        doc_ids: Document IDs to restrict search scope.
        max_iterations: Maximum drill-down iterations (1-5).
        enable_reranking: Enable cross-encoder reranking.
        enable_drilldown: Enable recursive drill-down for knowledge gaps.
        enable_deep_image_analysis: Enable deep image analysis for specific questions.
    """
    original_question: str
    sub_tasks: List[EditableSubTask]
    doc_ids: Optional[List[str]] = None
    max_iterations: int = Field(default=2, ge=1, le=5)
    enable_reranking: bool = True
    enable_drilldown: bool = True
    enable_deep_image_analysis: bool = Field(
        default=False,
        description="啟用進階圖片查證（針對特定問題重新分析圖片，會增加 API 調用）"
    )


class DrillDownTask(BaseModel):
    """
    A follow-up task generated during drill-down.
    
    Attributes:
        id: Task identifier (continues from original tasks).
        question: The drill-down question.
        task_type: Query method.
        source_task_id: ID of the task that triggered this drill-down.
        iteration: Which drill-down iteration this was created in.
    """
    id: int
    question: str
    task_type: Literal["rag", "graph_analysis"] = "rag"
    source_task_id: int
    iteration: int


class SubTaskExecutionResult(BaseModel):
    """
    Result from executing a single sub-task.
    
    Attributes:
        id: Task identifier.
        question: The question asked.
        answer: The answer generated.
        sources: Document IDs referenced.
        contexts: Text chunks used for the answer.
        is_drilldown: Whether this was a drill-down task.
        iteration: Drill-down iteration (0 for original tasks).
        usage: Token usage information.
    """
    id: int
    question: str
    answer: str
    sources: List[str] = []
    contexts: List[str] = []
    is_drilldown: bool = False
    iteration: int = 0
    usage: dict = Field(default_factory=lambda: {"total_tokens": 0})



class ExecutePlanResponse(BaseModel):
    """
    Response model for research plan execution.
    
    Attributes:
        question: Original research question.
        summary: Executive summary of findings.
        detailed_answer: Full research report.
        sub_tasks: All executed sub-tasks (original + drill-down).
        all_sources: Deduplicated list of all document sources.
        confidence: Overall confidence score (0.0-1.0).
        total_iterations: Number of drill-down iterations performed.
    """
    question: str
    summary: str
    detailed_answer: str
    sub_tasks: List[SubTaskExecutionResult]
    all_sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    total_iterations: int = 0
