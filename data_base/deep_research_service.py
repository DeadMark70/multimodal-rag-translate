"""
Deep Research Service

Provides interactive deep research capabilities with:
1. Human-in-the-loop plan confirmation
2. Breadth-first task execution
3. Knowledge gap identification
4. Recursive drill-down
5. Result synthesis
"""

# Standard library
import asyncio
import logging
from typing import List, Optional, Tuple

# Local application
from data_base.schemas_deep_research import (
    EditableSubTask,
    ExecutePlanRequest,
    ExecutePlanResponse,
    ResearchPlanResponse,
    SubTaskExecutionResult,
)
from data_base.RAG_QA_service import rag_answer_question
from agents.planner import TaskPlanner, SubTask, ResearchPlan
from agents.synthesizer import synthesize_results, SubTaskResult

# Configure logging
logger = logging.getLogger(__name__)


class DeepResearchService:
    """
    Deep Research Service for interactive plan-and-execute workflow.
    
    Implements a two-phase research process:
    1. Planning phase (generate_plan): Creates a research plan for user confirmation
    2. Execution phase (execute_plan): Runs the confirmed plan with drill-down
    
    Attributes:
        max_concurrent_tasks: Maximum concurrent RAG queries.
        default_max_iterations: Default drill-down iterations.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        default_max_iterations: int = 2,
    ) -> None:
        """
        Initializes the Deep Research Service.
        
        Args:
            max_concurrent_tasks: Maximum concurrent RAG queries.
            default_max_iterations: Default maximum drill-down iterations.
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_max_iterations = default_max_iterations
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def generate_plan(
        self,
        question: str,
        user_id: str,
        doc_ids: Optional[List[str]] = None,
        enable_graph_planning: bool = False,
    ) -> ResearchPlanResponse:
        """
        Generates a research plan for user confirmation.
        
        This is the first phase of interactive deep research.
        The plan is returned to the user for review and modification
        before execution.
        
        Args:
            question: The complex research question.
            user_id: Authenticated user's ID.
            doc_ids: Optional document IDs to restrict scope.
            enable_graph_planning: Use graph-aware planning prompts.
            
        Returns:
            ResearchPlanResponse with editable sub-tasks.
        """
        logger.info(f"Generating research plan for user {user_id}: {question[:50]}...")
        
        # Create planner and generate plan
        planner = TaskPlanner(
            max_subtasks=5,
            enable_graph_planning=enable_graph_planning,
        )
        
        plan = await planner.plan(question)
        
        # Convert SubTask to EditableSubTask
        editable_tasks = [
            EditableSubTask(
                id=task.id,
                question=task.question,
                task_type=task.task_type,
                enabled=True,
            )
            for task in plan.sub_tasks
        ]
        
        logger.info(f"Generated plan with {len(editable_tasks)} sub-tasks")
        
        return ResearchPlanResponse(
            status="waiting_confirmation",
            original_question=question,
            sub_tasks=editable_tasks,
            estimated_complexity=plan.estimated_complexity,
            doc_ids=doc_ids,
        )
    
    async def execute_plan(
        self,
        request: ExecutePlanRequest,
        user_id: str,
    ) -> ExecutePlanResponse:
        """
        Executes a user-confirmed research plan with drill-down.
        
        This is the second phase of interactive deep research.
        Implements:
        1. Breadth-first execution of all enabled sub-tasks
        2. Knowledge gap identification
        3. Recursive drill-down (up to max_iterations)
        4. Final synthesis
        
        Args:
            request: The confirmed execution request.
            user_id: Authenticated user's ID.
            
        Returns:
            ExecutePlanResponse with complete research results.
        """
        logger.info(
            f"Executing research plan for user {user_id}: "
            f"{len(request.sub_tasks)} tasks, max_iter={request.max_iterations}"
        )
        
        # Filter enabled tasks only
        enabled_tasks = [t for t in request.sub_tasks if t.enabled]
        
        if not enabled_tasks:
            logger.warning("No enabled tasks in plan")
            return ExecutePlanResponse(
                question=request.original_question,
                summary="沒有啟用的子任務。",
                detailed_answer="請至少啟用一個子任務後重試。",
                sub_tasks=[],
                all_sources=[],
                confidence=0.0,
                total_iterations=0,
            )
        
        # Phase 1: Breadth-first execution
        all_results: List[SubTaskExecutionResult] = []
        current_results = await self._execute_tasks(
            tasks=enabled_tasks,
            user_id=user_id,
            doc_ids=request.doc_ids,
            enable_reranking=request.enable_reranking,
            iteration=0,
        )
        all_results.extend(current_results)
        
        # Phase 2: Drill-down loop (if enabled AND answers not already complete)
        total_iterations = 0
        should_skip = self._should_skip_drilldown(all_results)
        
        if request.enable_drilldown and request.max_iterations > 0 and not should_skip:
            total_iterations = await self._drill_down_loop(
                original_question=request.original_question,
                current_results=all_results,
                user_id=user_id,
                doc_ids=request.doc_ids,
                enable_reranking=request.enable_reranking,
                max_iterations=request.max_iterations,
            )
        
        # Phase 3: Synthesize results
        synthesizer_results = [
            SubTaskResult(
                task_id=r.id,
                question=r.question,
                answer=r.answer,
                sources=r.sources,
                confidence=1.0 if r.answer else 0.0,
            )
            for r in all_results
        ]
        
        report = await synthesize_results(
            original_question=request.original_question,
            sub_results=synthesizer_results,
            enabled=len(synthesizer_results) > 1,
        )
        
        # Collect all unique sources
        all_sources = list(set(
            src for r in all_results for src in r.sources
        ))
        
        logger.info(
            f"Research complete: {len(all_results)} tasks, "
            f"{total_iterations} drill-down iterations, "
            f"{len(all_sources)} sources"
        )
        
        return ExecutePlanResponse(
            question=request.original_question,
            summary=report.summary,
            detailed_answer=report.detailed_answer,
            sub_tasks=all_results,
            all_sources=all_sources,
            confidence=report.confidence,
            total_iterations=total_iterations,
        )
    
    async def _execute_tasks(
        self,
        tasks: List[EditableSubTask],
        user_id: str,
        doc_ids: Optional[List[str]],
        enable_reranking: bool,
        iteration: int,
    ) -> List[SubTaskExecutionResult]:
        """
        Executes a batch of tasks concurrently.
        
        Args:
            tasks: List of tasks to execute.
            user_id: User ID for RAG queries.
            doc_ids: Document filter.
            enable_reranking: Enable reranking.
            iteration: Current drill-down iteration.
            
        Returns:
            List of execution results.
        """
        async def execute_single(task: EditableSubTask) -> SubTaskExecutionResult:
            async with self._semaphore:
                try:
                    use_graph = task.task_type == "graph_analysis"
                    
                    answer, sources = await rag_answer_question(
                        question=task.question,
                        user_id=user_id,
                        doc_ids=doc_ids,
                        enable_reranking=enable_reranking,
                        enable_graph_rag=use_graph,
                        graph_search_mode="hybrid" if use_graph else "auto",
                    )
                    
                    return SubTaskExecutionResult(
                        id=task.id,
                        question=task.question,
                        answer=answer,
                        sources=sources,
                        is_drilldown=iteration > 0,
                        iteration=iteration,
                    )
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Task {task.id} failed: {e}")
                    return SubTaskExecutionResult(
                        id=task.id,
                        question=task.question,
                        answer=f"無法回答此問題: {str(e)[:100]}",
                        sources=[],
                        is_drilldown=iteration > 0,
                        iteration=iteration,
                    )
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_single(task) for task in tasks]
        )
        
        return list(results)
    
    async def _drill_down_loop(
        self,
        original_question: str,
        current_results: List[SubTaskExecutionResult],
        user_id: str,
        doc_ids: Optional[List[str]],
        enable_reranking: bool,
        max_iterations: int,
    ) -> int:
        """
        Performs recursive drill-down to fill knowledge gaps.
        
        Args:
            original_question: The original research question.
            current_results: Results accumulated so far.
            user_id: User ID for RAG queries.
            doc_ids: Document filter.
            enable_reranking: Enable reranking.
            max_iterations: Maximum iterations allowed.
            
        Returns:
            Number of iterations performed.
        """
        planner = TaskPlanner(max_subtasks=3, enable_graph_planning=False)
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Drill-down iteration {iteration}/{max_iterations}")
            
            # Build current findings summary
            findings_summary = self._build_findings_summary(current_results)
            
            # Ask planner for follow-up tasks
            followup_tasks = await planner.create_followup_tasks(
                original_question=original_question,
                current_findings=findings_summary,
                existing_tasks=[
                    SubTask(id=r.id, question=r.question)
                    for r in current_results
                ],
            )
            
            if not followup_tasks:
                logger.info(f"No knowledge gaps found at iteration {iteration}")
                return iteration - 1
            
            logger.info(f"Found {len(followup_tasks)} follow-up tasks")
            
            # Convert to EditableSubTask with new IDs
            max_id = max(r.id for r in current_results)
            editable_followups = [
                EditableSubTask(
                    id=max_id + i + 1,
                    question=task.question,
                    task_type=task.task_type,
                    enabled=True,
                )
                for i, task in enumerate(followup_tasks)
            ]
            
            # Execute follow-up tasks
            new_results = await self._execute_tasks(
                tasks=editable_followups,
                user_id=user_id,
                doc_ids=doc_ids,
                enable_reranking=enable_reranking,
                iteration=iteration,
            )
            
            current_results.extend(new_results)
        
        return max_iterations
    
    def _build_findings_summary(
        self,
        results: List[SubTaskExecutionResult],
    ) -> str:
        """
        Builds a summary of current findings for drill-down analysis.
        
        Args:
            results: Current execution results.
            
        Returns:
            Formatted summary string.
        """
        lines = []
        for r in results:
            # Truncate long answers
            answer_preview = r.answer[:300] + "..." if len(r.answer) > 300 else r.answer
            lines.append(f"【問題 {r.id}】{r.question}")
            lines.append(f"【回答】{answer_preview}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _should_skip_drilldown(
        self,
        results: List[SubTaskExecutionResult],
        min_answer_length: int = 200,
        min_complete_ratio: float = 0.8,
    ) -> bool:
        """
        Determines whether to skip the drill-down phase based on answer completeness.
        
        Smart termination conditions:
        1. All answers are sufficiently long (not shallow responses)
        2. No answers contain failure markers (e.g., "無法回答", "找不到")
        
        Args:
            results: Current execution results.
            min_answer_length: Minimum length for a "complete" answer.
            min_complete_ratio: Ratio of complete answers required to skip drill-down.
            
        Returns:
            True if drill-down should be skipped.
        """
        if not results:
            return False
        
        # Failure markers indicating incomplete or failed answers
        failure_markers = [
            "無法回答", "找不到", "沒有相關", "抱歉", "無法找到",
            "unable to answer", "not found", "no relevant", "sorry",
            "無法確定", "資料不足", "沒有足夠"
        ]
        
        complete_count = 0
        for r in results:
            answer_lower = r.answer.lower()
            
            # Check for failure markers
            has_failure = any(marker in answer_lower for marker in failure_markers)
            
            # Check answer length
            is_long_enough = len(r.answer) >= min_answer_length
            
            if not has_failure and is_long_enough:
                complete_count += 1
        
        complete_ratio = complete_count / len(results)
        
        should_skip = complete_ratio >= min_complete_ratio
        if should_skip:
            logger.info(
                f"Smart termination: {complete_count}/{len(results)} answers complete "
                f"({complete_ratio:.0%}), skipping drill-down"
            )
        
        return should_skip
    
    async def execute_plan_streaming(
        self,
        request: ExecutePlanRequest,
        user_id: str,
    ):
        """
        Executes research plan with SSE streaming progress updates.
        
        This is an async generator that yields SSE events during execution.
        Use with EventSourceResponse for SSE streaming.
        
        Args:
            request: The confirmed execution request.
            user_id: Authenticated user's ID.
            
        Yields:
            Dict with 'event' and 'data' keys for SSE.
        """
        from data_base.sse_events import (
            SSEEventType,
            PlanConfirmedData,
            TaskStartData,
            TaskDoneData,
            DrilldownStartData,
            SynthesisStartData,
            ErrorData,
            format_sse_event,
        )
        
        # Filter enabled tasks
        enabled_tasks = [t for t in request.sub_tasks if t.enabled]
        
        if not enabled_tasks:
            yield format_sse_event(
                SSEEventType.ERROR,
                ErrorData(message="沒有啟用的子任務")
            )
            return
        
        # Emit plan confirmed
        yield format_sse_event(
            SSEEventType.PLAN_CONFIRMED,
            PlanConfirmedData(
                task_count=len(request.sub_tasks),
                enabled_count=len(enabled_tasks),
            )
        )
        
        all_results: List[SubTaskExecutionResult] = []
        
        # Phase 1: Execute initial tasks (sequential for streaming)
        for task in enabled_tasks:
            # Emit task start
            yield format_sse_event(
                SSEEventType.TASK_START,
                TaskStartData(
                    id=task.id,
                    question=task.question,
                    task_type=task.task_type,
                    iteration=0,
                )
            )
            
            # Execute task
            result = await self._execute_single_task(
                task=task,
                user_id=user_id,
                doc_ids=request.doc_ids,
                enable_reranking=request.enable_reranking,
                iteration=0,
            )
            all_results.append(result)
            
            # Emit task done
            yield format_sse_event(
                SSEEventType.TASK_DONE,
                TaskDoneData(
                    id=result.id,
                    question=result.question,
                    answer=result.answer,
                    sources=result.sources,
                    iteration=0,
                )
            )
        
        # Phase 2: Drill-down loop (with smart termination check)
        total_iterations = 0
        should_skip = self._should_skip_drilldown(all_results)
        
        if request.enable_drilldown and request.max_iterations > 0 and not should_skip:
            planner = TaskPlanner(max_subtasks=3, enable_graph_planning=False)
            
            for iteration in range(1, request.max_iterations + 1):
                findings_summary = self._build_findings_summary(all_results)
                
                followup_tasks = await planner.create_followup_tasks(
                    original_question=request.original_question,
                    current_findings=findings_summary,
                    existing_tasks=[
                        SubTask(id=r.id, question=r.question)
                        for r in all_results
                    ],
                )
                
                if not followup_tasks:
                    break
                
                total_iterations = iteration
                
                # Emit drilldown start
                yield format_sse_event(
                    SSEEventType.DRILLDOWN_START,
                    DrilldownStartData(
                        iteration=iteration,
                        new_task_count=len(followup_tasks),
                    )
                )
                
                # Execute follow-up tasks
                max_id = max(r.id for r in all_results)
                for i, task in enumerate(followup_tasks):
                    editable_task = EditableSubTask(
                        id=max_id + i + 1,
                        question=task.question,
                        task_type=task.task_type,
                        enabled=True,
                    )
                    
                    yield format_sse_event(
                        SSEEventType.DRILLDOWN_TASK_START,
                        TaskStartData(
                            id=editable_task.id,
                            question=editable_task.question,
                            task_type=editable_task.task_type,
                            iteration=iteration,
                        )
                    )
                    
                    result = await self._execute_single_task(
                        task=editable_task,
                        user_id=user_id,
                        doc_ids=request.doc_ids,
                        enable_reranking=request.enable_reranking,
                        iteration=iteration,
                    )
                    all_results.append(result)
                    
                    yield format_sse_event(
                        SSEEventType.DRILLDOWN_TASK_DONE,
                        TaskDoneData(
                            id=result.id,
                            question=result.question,
                            answer=result.answer,
                            sources=result.sources,
                            iteration=iteration,
                        )
                    )
        
        # Phase 3: Synthesis
        yield format_sse_event(
            SSEEventType.SYNTHESIS_START,
            SynthesisStartData(total_tasks=len(all_results))
        )
        
        synthesizer_results = [
            SubTaskResult(
                task_id=r.id,
                question=r.question,
                answer=r.answer,
                sources=r.sources,
                confidence=1.0 if r.answer else 0.0,
            )
            for r in all_results
        ]
        
        report = await synthesize_results(
            original_question=request.original_question,
            sub_results=synthesizer_results,
            enabled=len(synthesizer_results) > 1,
        )
        
        all_sources = list(set(
            src for r in all_results for src in r.sources
        ))
        
        # Emit complete with full response
        final_response = ExecutePlanResponse(
            question=request.original_question,
            summary=report.summary,
            detailed_answer=report.detailed_answer,
            sub_tasks=all_results,
            all_sources=all_sources,
            confidence=report.confidence,
            total_iterations=total_iterations,
        )
        
        yield format_sse_event(
            SSEEventType.COMPLETE,
            final_response.model_dump()
        )
    
    async def _execute_single_task(
        self,
        task: EditableSubTask,
        user_id: str,
        doc_ids: Optional[List[str]],
        enable_reranking: bool,
        iteration: int,
    ) -> SubTaskExecutionResult:
        """
        Executes a single task.
        
        Args:
            task: The task to execute.
            user_id: User ID for RAG queries.
            doc_ids: Document filter.
            enable_reranking: Enable reranking.
            iteration: Current drill-down iteration.
            
        Returns:
            SubTaskExecutionResult.
        """
        try:
            use_graph = task.task_type == "graph_analysis"
            
            answer, sources = await rag_answer_question(
                question=task.question,
                user_id=user_id,
                doc_ids=doc_ids,
                enable_reranking=enable_reranking,
                enable_graph_rag=use_graph,
                graph_search_mode="hybrid" if use_graph else "auto",
            )
            
            return SubTaskExecutionResult(
                id=task.id,
                question=task.question,
                answer=answer,
                sources=sources,
                is_drilldown=iteration > 0,
                iteration=iteration,
            )
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Task {task.id} failed: {e}")
            return SubTaskExecutionResult(
                id=task.id,
                question=task.question,
                answer=f"無法回答此問題: {str(e)[:100]}",
                sources=[],
                is_drilldown=iteration > 0,
                iteration=iteration,
            )


# Module-level singleton for convenience
_deep_research_service: Optional[DeepResearchService] = None


def get_deep_research_service() -> DeepResearchService:
    """
    Gets or creates the Deep Research Service singleton.
    
    Returns:
        DeepResearchService instance.
    """
    global _deep_research_service
    if _deep_research_service is None:
        _deep_research_service = DeepResearchService()
    return _deep_research_service
