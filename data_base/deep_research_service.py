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
from typing import List, Optional

# Local application
from core.errors import AppError
from data_base.repository import persist_research_conversation
from data_base.schemas_deep_research import (
    EditableSubTask,
    ExecutePlanRequest,
    ExecutePlanResponse,
    ResearchPlanResponse,
    SubTaskExecutionResult,
)
from data_base.RAG_QA_service import rag_answer_question
from agents.planner import TaskPlanner, SubTask
from agents.synthesizer import synthesize_results, SubTaskResult
from agents.evaluator import RAGEvaluator

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
            enable_deep_image_analysis=request.enable_deep_image_analysis,
        )
        all_results.extend(current_results)
        
        # Phase 2: Drill-down loop (if enabled AND answers not already complete)
        total_iterations = 0
        # Phase 6.1B: 傳入 current_iteration=0 以強制至少一次 Drill-down
        should_skip = self._should_skip_drilldown(all_results, current_iteration=0)
        
        if request.enable_drilldown and request.max_iterations > 0 and not should_skip:
            total_iterations = await self._drill_down_loop(
                original_question=request.original_question,
                current_results=all_results,
                user_id=user_id,
                doc_ids=request.doc_ids,
                enable_reranking=request.enable_reranking,
                max_iterations=request.max_iterations,
                enable_deep_image_analysis=request.enable_deep_image_analysis,
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
            use_academic_template=False,  # Simplified format for better RAGAS compatibility and latency
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
        
        # Phase 1: Persistence - Update conversation in Supabase if conversation_id provided
        if request.conversation_id:
            try:
                metadata_payload = {
                    "summary": report.summary,
                    "detailed_answer": report.detailed_answer,
                    "sub_tasks": [t.model_dump() for t in all_results],
                    "all_sources": all_sources,
                    "confidence": report.confidence,
                    "total_iterations": total_iterations,
                    "question": request.original_question,
                }

                await persist_research_conversation(
                    conversation_id=request.conversation_id,
                    user_id=user_id,
                    title=request.original_question[:100]
                    if request.original_question
                    else None,
                    metadata=metadata_payload,
                )
                logger.info(
                    "Persisted research results to conversation %s",
                    request.conversation_id,
                )
            except AppError as e:
                logger.error(f"Failed to persist research results: {e}", exc_info=True)

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
        enable_deep_image_analysis: bool = False,
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
                    # Phase 6.1A: 預設開啟 GraphRAG 以提升抗噪能力
                    # 大規模文檔環境下，GraphRAG 可捕捉隱藏關聯
                    
                    result = await rag_answer_question(
                        question=task.question,
                        user_id=user_id,
                        doc_ids=doc_ids,
                        enable_reranking=enable_reranking,
                        enable_graph_rag=True,  # Phase 6: 預設開啟
                        graph_search_mode="hybrid",  # Phase 6: 混合模式
                        enable_visual_verification=enable_deep_image_analysis,  # Phase 9
                        return_docs=True, # Capture documents for context
                    )
                    
                    # Handle RAGResult
                    if hasattr(result, 'answer'):
                        answer = result.answer
                        sources = result.source_doc_ids
                        contexts = [d.page_content for d in result.documents]
                        usage = result.usage or {"total_tokens": 0}
                        thought_process = result.thought_process
                        tool_calls = result.tool_calls
                    else:
                        # Fallback if return_docs ignored (shouldn't happen with updated service)
                        answer, sources = result
                        contexts = []
                        usage = {"total_tokens": 0}
                        thought_process = None
                        tool_calls = []
                    
                    return SubTaskExecutionResult(
                        id=task.id,
                        question=task.question,
                        answer=answer,
                        sources=sources,
                        contexts=contexts,
                        is_drilldown=iteration > 0,
                        iteration=iteration,
                        usage=usage,
                        thought_process=thought_process,
                        tool_calls=tool_calls
                    )


                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Task {task.id} failed: {e}")
                    return SubTaskExecutionResult(
                        id=task.id,
                        question=task.question,
                        answer=f"無法回答此問題: {str(e)[:100]}",
                        sources=[],
                        contexts=[],
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
        enable_deep_image_analysis: bool = False,
    ) -> int:
        """
        Performs recursive drill-down to fill knowledge gaps with evaluation-driven retry.
        
        Implements smart retry: when an answer has low completeness score,
        the planner refines the query based on the evaluation reason rather
        than simply repeating the same question.
        
        Args:
            original_question: The original research question.
            current_results: Results accumulated so far.
            user_id: User ID for RAG queries.
            doc_ids: Document filter.
            enable_reranking: Enable reranking.
            max_iterations: Maximum iterations allowed.
            enable_deep_image_analysis: Enable deep image analysis for specific questions.
            
        Returns:
            Number of iterations performed.
        """
        planner = TaskPlanner(max_subtasks=3, enable_graph_planning=False)
        evaluator = RAGEvaluator()
        
        # Constants for evaluation-driven retry (Phase 4: 1-10 scale)
        MIN_ACCURACY_SCORE = 6.0   # Accuracy < 6 must retry (poor quality)
        MAX_RETRIES_PER_TASK = 2   # Prevent infinite loops
        
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
            
            # Process each follow-up task with evaluation-driven retry
            for i, task in enumerate(followup_tasks):
                task_id = max_id + i + 1
                current_question = task.question
                retry_count = 0
                
                while retry_count <= MAX_RETRIES_PER_TASK:
                    # Execute the task
                    EditableSubTask(
                        id=task_id,
                        question=current_question,
                        task_type=task.task_type,
                        enabled=True,
                    )
                    
                    # Get answer with return_docs for evaluation
                    try:
                        use_graph = task.task_type == "graph_analysis"
                        result = await rag_answer_question(
                            question=current_question,
                            user_id=user_id,
                            doc_ids=doc_ids,
                            enable_reranking=enable_reranking,
                            enable_graph_rag=use_graph,
                            graph_search_mode="hybrid" if use_graph else "auto",
                            return_docs=True,  # Get documents for evaluation
                            enable_visual_verification=enable_deep_image_analysis,  # Phase 9
                        )
                        
                        # Handle return format (answer, sources) or RAGResult
                        if hasattr(result, 'answer'):
                            answer = result.answer
                            sources = result.source_doc_ids
                            documents = result.documents
                            contexts = [d.page_content for d in documents]
                            usage = result.usage or {"total_tokens": 0}
                            thought_process = result.thought_process
                            tool_calls = result.tool_calls
                        else:
                            answer, sources = result
                            documents = []
                            contexts = []
                            usage = {"total_tokens": 0}
                            thought_process = None
                            tool_calls = []
                        
                    except (RuntimeError, ValueError) as e:
                        logger.warning(f"Task {task_id} failed: {e}")
                        answer = f"無法回答此問題: {str(e)[:100]}"
                        sources = []
                        documents = []
                        contexts = []
                    
                    # Evaluate answer quality (only if we have documents)
                    if documents and retry_count < MAX_RETRIES_PER_TASK:
                        evaluation = await evaluator.evaluate_detailed(
                            question=current_question,
                            documents=documents,
                            answer=answer,
                        )
                        
                        # Check if retry is needed (Phase 4: use accuracy threshold)
                        if evaluation.accuracy < MIN_ACCURACY_SCORE:
                            logger.info(
                                f"Task {task_id} low accuracy ({evaluation.accuracy:.1f}/10), "
                                f"reason: {evaluation.reason[:50]}..."
                            )
                            
                            # Smart retry: use suggestion from evaluation if available
                            retry_hint = evaluation.suggestion or evaluation.reason
                            refined_query = await planner.refine_query_from_evaluation(
                                original_question=current_question,
                                evaluation_reason=retry_hint,
                                failed_answer=answer,
                            )
                            
                            if refined_query != current_question:
                                logger.info(f"Smart retry #{retry_count + 1} with refined query")
                                current_question = refined_query
                                retry_count += 1
                                continue  # Retry with new query
                            else:
                                logger.info("Could not refine query, accepting current answer")
                    
                    # Accept the answer (either good score or max retries reached)
                    current_results.append(SubTaskExecutionResult(
                        id=task_id,
                        question=task.question,  # Use original question for display
                        answer=answer,
                        sources=sources,
                        contexts=contexts,
                        is_drilldown=True,
                        iteration=iteration,
                        usage=usage,
                        thought_process=thought_process,
                        tool_calls=tool_calls
                    ))


                    
                    if retry_count > 0:
                        logger.info(f"Task {task_id} accepted after {retry_count} retry(s)")
                    
                    break  # Exit retry loop
        
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
        current_iteration: int = -1,  # Phase 6.1B: 新增參數
    ) -> bool:
        """
        Determines whether to skip the drill-down phase based on answer completeness.
        
        Smart termination conditions:
        1. All answers are sufficiently long (not shallow responses)
        2. No answers contain failure markers (e.g., "無法回答", "找不到")
        3. Phase 6.1B: Iteration 0 is never skipped (forced drill-down)
        
        Args:
            results: Current execution results.
            min_answer_length: Minimum length for a "complete" answer.
            min_complete_ratio: Ratio of complete answers required to skip drill-down.
            current_iteration: Current iteration number. If 0, drill-down is forced.
            
        Returns:
            True if drill-down should be skipped.
        """
        # Phase 6.1B: 強制至少執行一次 Drill-down
        if current_iteration == 0:
            logger.info("Phase 6 Forced Drill-down: iteration 0 requires at least one drill-down")
            return False
        
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
        
        # Phase 14: Self-Stop Condition (User Feedback)
        # If we found specific quantitative data (DSC), stop early to avoid drift
        has_quantitative_data = any(
            "dsc" in r.answer.lower() and any(c.isdigit() for c in r.answer)
            for r in results
        )
        
        if has_quantitative_data and complete_ratio > 0.5:
             logger.info("Self-Stop Triggered: Found quantitative data (DSC) and incomplete ratio > 0.5")
             return True
        
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
                    contexts=result.contexts,
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
                            contexts=result.contexts,
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
            use_academic_template=False,  # Simplified format for better RAGAS compatibility and latency
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
            
            result = await rag_answer_question(
                question=task.question,
                user_id=user_id,
                doc_ids=doc_ids,
                enable_reranking=enable_reranking,
                enable_graph_rag=use_graph,
                graph_search_mode="hybrid" if use_graph else "auto",
                return_docs=True, # MUST be True to capture tool_calls and diagnostics
            )
            
            # Handle RAGResult
            if hasattr(result, 'answer'):
                answer = result.answer
                sources = result.source_doc_ids
                contexts = [d.page_content for d in result.documents]
                usage = result.usage or {"total_tokens": 0}
                thought_process = result.thought_process
                tool_calls = result.tool_calls
            else:
                answer, sources = result
                contexts = []
                usage = {"total_tokens": 0}
                thought_process = None
                tool_calls = []
            
            return SubTaskExecutionResult(
                id=task.id,
                question=task.question,
                answer=answer,
                sources=sources,
                contexts=contexts,
                is_drilldown=iteration > 0,
                iteration=iteration,
                usage=usage,
                thought_process=thought_process,
                tool_calls=tool_calls
            )


        except (RuntimeError, ValueError) as e:
            logger.warning(f"Task {task.id} failed: {e}")
            return SubTaskExecutionResult(
                id=task.id,
                question=task.question,
                answer=f"無法回答此問題: {str(e)[:100]}",
                sources=[],
                contexts=[],
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
