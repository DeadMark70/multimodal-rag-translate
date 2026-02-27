"""
RAG Question Answering Router

Provides API endpoints for RAG-based question answering,
including enhanced research mode with Plan-and-Solve agents.
"""

# Standard library
import logging
import time
from typing import Optional, List

# Third-party
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel, Field

# Local application
from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from data_base.RAG_QA_service import initialize_llm_service, rag_answer_question, RAGResult
from data_base.repository import insert_chat_log, insert_query_log
from data_base.vector_store_manager import initialize_embeddings
from data_base.schemas import (
    AskRequest,
    EnhancedAskResponse,
    SourceDetail,
    EvaluationMetrics,
    FaithfulnessLevel,
)
from agents.planner import plan_research
from agents.synthesizer import synthesize_results, SubTaskResult
from agents.evaluator import RAGEvaluator
from data_base.schemas_deep_research import (
    ResearchPlanRequest,
    ResearchPlanResponse,
    ExecutePlanRequest,
    ExecutePlanResponse,
)
from data_base.deep_research_service import get_deep_research_service

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Constants ---

MAX_HISTORY_LENGTH = 10  # Maximum conversation history messages to accept


class ResearchRequest(BaseModel):
    """Request model for research queries."""
    question: str
    max_subtasks: int = 5
    enable_reranking: bool = True
    enable_graph_planning: bool = False  # Enable graph-aware task planning


class SubTaskResponse(BaseModel):
    """Response for a single sub-task."""
    id: int
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Response model for research queries."""
    question: str
    summary: str
    detailed_answer: str
    sub_tasks: list[SubTaskResponse]
    all_sources: list[str]
    confidence: float


# --- Startup ---

async def on_startup_rag_init() -> None:
    """
    Initializes RAG components during application startup.

    Initializes:
    - Embedding model (Google Gemini Embedding API)
    - LLM service (Gemini)
    """
    logger.info("=== Initializing RAG components ===")
    try:
        # 1. Initialize Embedding Model (Google API)
        await initialize_embeddings()

        # 2. Initialize LLM (API Client)
        await initialize_llm_service()

        logger.info("=== RAG components ready ===")
    except (RuntimeError, ImportError, OSError) as e:
        logger.error(f"RAG initialization failed: {e}", exc_info=True)
        raise


# --- Endpoints ---

@router.post("/ask", response_model=EnhancedAskResponse)
async def ask_question_with_context(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id)
) -> EnhancedAskResponse:
    """
    Context-aware question answering with conversation history.

    This enhanced endpoint supports:
    - Conversation history for multi-turn dialogue
    - Advanced retrieval options (HyDE, Multi-Query)
    - Document filtering
    - Optional Self-RAG evaluation (enable_evaluation=True)

    When enable_evaluation=True, returns EnhancedAskResponse with:
    - Detailed source citations (page, snippet, score)
    - Responsibility metrics (faithfulness, confidence)

    Args:
        request: AskRequest with question, history, and options.
        background_tasks: FastAPI BackgroundTasks for async logging.
        user_id: Authenticated user ID (injected).

    Returns:
        EnhancedAskResponse.

    Raises:
        HTTPException: 400 if history too long, 500 if answering fails.
    """
    t1 = time.perf_counter()

    # Validate history length
    if request.history and len(request.history) > MAX_HISTORY_LENGTH:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message=f"History too long; max {MAX_HISTORY_LENGTH} messages",
            status_code=400,
        )

    history_count = len(request.history) if request.history else 0
    logger.info(
        f"Context-aware RAG for user {user_id}: "
        f"history={history_count}, hyde={request.enable_hyde}, "
        f"multi_query={request.enable_multi_query}, "
        f"evaluation={request.enable_evaluation}, "
        f"graph_rag={request.enable_graph_rag}"
    )

    try:
        # Get RAG answer
        answer, sources = await rag_answer_question(
            question=request.question,
            user_id=user_id,
            doc_ids=request.doc_ids,
            history=request.history,
            enable_reranking=request.enable_reranking,
            enable_hyde=request.enable_hyde,
            enable_multi_query=request.enable_multi_query,
            enable_graph_rag=request.enable_graph_rag,
            graph_search_mode=request.graph_search_mode,
        )

        t2 = time.perf_counter()
        logger.info(f"Context-aware answer in {t2 - t1:.2f}s, sources: {sources}")

        # Background task: Log analytics and chat history.
        background_tasks.add_task(
            _log_query_to_supabase,
            user_id=user_id,
            question=request.question,
            answer=answer,
            has_history=history_count > 0,
        )

        # Build default source details for unified response shape.
        source_details = [
            SourceDetail(
                doc_id=doc_id,
                filename=None,
                page=None,
                snippet=answer[:200],
                score=0.0,
            )
            for doc_id in sources
        ]
        if not request.enable_evaluation:
            return EnhancedAskResponse(
                question=request.question,
                answer=answer,
                sources=source_details,
                metrics=None,
            )

        # --- Enhanced Response with Evaluation ---
        logger.info("Running Self-RAG evaluation with documents...")
        
        # Re-fetch with documents for evaluation
        result_with_docs = await rag_answer_question(
            question=request.question,
            user_id=user_id,
            doc_ids=request.doc_ids,
            history=request.history,
            enable_reranking=request.enable_reranking,
            enable_hyde=request.enable_hyde,
            enable_multi_query=request.enable_multi_query,
            return_docs=True,
            enable_graph_rag=request.enable_graph_rag,
            graph_search_mode=request.graph_search_mode,
        )
        
        # Unpack result
        if isinstance(result_with_docs, RAGResult):
            answer = result_with_docs.answer
            sources = result_with_docs.source_doc_ids
            docs = result_with_docs.documents
        else:
            # Fallback if somehow not RAGResult
            answer, sources = result_with_docs
            docs = []

        # Build SourceDetail list with document snippets
        source_details = []
        for i, doc_id in enumerate(sources):
            # Find matching document for snippet
            snippet = ""
            for doc in docs:
                if doc.metadata.get("doc_id") == doc_id or doc.metadata.get("original_doc_uid") == doc_id:
                    snippet = doc.page_content[:200]
                    break
            source_details.append(SourceDetail(
                doc_id=doc_id,
                filename=None,
                page=None,
                snippet=snippet if snippet else answer[:200],
                score=0.7  # Placeholder
            ))

        # Run detailed evaluation with real documents
        try:
            evaluator = RAGEvaluator()
            eval_result = await evaluator.evaluate_detailed(
                question=request.question,
                documents=docs,
                answer=answer
            )
            
            # Map evaluation result to API schema (Phase 4: 1-10 scale)
            if eval_result.evaluation_failed:
                faithfulness = FaithfulnessLevel.evaluation_failed
            elif eval_result.accuracy >= 8:
                faithfulness = FaithfulnessLevel.grounded
            elif eval_result.accuracy >= 6:
                faithfulness = FaithfulnessLevel.uncertain
            else:
                faithfulness = FaithfulnessLevel.hallucinated
            
            metrics = EvaluationMetrics(
                faithfulness=faithfulness,
                confidence_score=eval_result.confidence,
                evaluation_reason=eval_result.reason if eval_result.reason else None,
                # Phase 4: New 1-10 scale fields for radar chart
                accuracy=eval_result.accuracy,
                completeness=eval_result.completeness,
                clarity=eval_result.clarity,
                weighted_score=eval_result.weighted_score,
                suggestion=eval_result.suggestion if eval_result.suggestion else None,
                is_passing=eval_result.is_passing,
            )

            t3 = time.perf_counter()
            logger.info(
                f"Evaluation complete in {t3 - t2:.2f}s: "
                f"{faithfulness.value}, confidence={eval_result.confidence:.2f}"
            )

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Evaluation failed: {e}")
            metrics = EvaluationMetrics(
                faithfulness=FaithfulnessLevel.evaluation_failed,
                confidence_score=0.5,
                evaluation_reason="Evaluation failed"
            )

        return EnhancedAskResponse(
            question=request.question,
            answer=answer,
            sources=source_details,
            metrics=metrics
        )

    except (RuntimeError, ValueError) as e:
        logger.error(f"Context-aware RAG failed: {e}", exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to answer question with context",
            status_code=500,
        )


async def _log_query_to_supabase(
    user_id: str,
    question: str,
    answer: str,
    has_history: bool,
    faithfulness: Optional[str] = None,
    confidence: Optional[float] = None,
    response_time_ms: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
) -> None:
    """
    Background task to log query to Supabase.

    Logs to both chat_logs (legacy) and query_logs (new analytics).

    Args:
        user_id: User's ID.
        question: The question asked.
        answer: The answer generated.
        has_history: Whether conversation history was used.
        faithfulness: Evaluation result ('grounded', 'hallucinated', 'uncertain').
        confidence: Confidence score (0.0-1.0).
        response_time_ms: Response time in milliseconds.
        doc_ids: List of document IDs used.
    """
    # Log to chat_logs (legacy table)
    try:
        await insert_chat_log(
            user_id=user_id,
            question=question,
            answer=answer,
        )
        logger.debug("Chat log saved to Supabase (background)")
    except AppError as e:
        logger.warning(f"Failed to save chat log: {e}")

    # Log to query_logs (new analytics table)
    try:
        await insert_query_log(
            user_id=user_id,
            question=question,
            answer=answer[:500] if answer else None,  # Truncate for storage
            has_history=has_history,
            faithfulness=faithfulness,
            confidence=confidence,
            response_time_ms=response_time_ms,
            doc_ids=doc_ids,
        )
        logger.debug("Query log saved to Supabase (background)")
    except AppError as e:
        # Non-fatal: table might not exist yet
        logger.debug(f"Failed to save to query_logs (may not exist): {e}")




@router.post("/research", response_model=ResearchResponse)
async def research_question(
    request: ResearchRequest,
    user_id: str = Depends(get_current_user_id)
) -> ResearchResponse:
    """
    Performs deep research on a complex question using Plan-and-Solve.

    This endpoint:
    1. Decomposes the question into sub-tasks
    2. Runs RAG on each sub-task
    3. Synthesizes results into a research report

    Args:
        request: Research request with question and options.
        user_id: Authenticated user ID (injected).

    Returns:
        ResearchResponse with summary, detailed answer, and sub-task results.

    Raises:
        HTTPException: 500 if research fails.
    """
    t1 = time.perf_counter()
    logger.info(f"Research query for user {user_id}: {request.question[:50]}...")

    try:
        # Step 1: Plan - decompose into sub-tasks
        plan = await plan_research(
            question=request.question,
            enabled=True,
            max_subtasks=request.max_subtasks,
            enable_graph_planning=request.enable_graph_planning,
        )
        
        logger.info(f"Planned {len(plan.sub_tasks)} sub-tasks")

        # Step 2: Execute - run RAG on each sub-task
        sub_results: List[SubTaskResult] = []
        
        for task in plan.sub_tasks:
            try:
                # Enable GraphRAG for graph_analysis type subtasks
                use_graph = task.task_type == "graph_analysis"
                
                answer, sources = await rag_answer_question(
                    question=task.question,
                    user_id=user_id,
                    enable_reranking=request.enable_reranking,
                    enable_graph_rag=use_graph,
                    graph_search_mode="hybrid" if use_graph else "auto",
                )
                
                sub_results.append(SubTaskResult(
                    task_id=task.id,
                    question=task.question,
                    answer=answer,
                    sources=sources,
                    confidence=1.0,
                ))
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Sub-task {task.id} failed: {e}")
                sub_results.append(SubTaskResult(
                    task_id=task.id,
                    question=task.question,
                    answer=f"無法回答此子問題: {e}",
                    sources=[],
                    confidence=0.0,
                ))

        # Step 3: Synthesize - combine results
        report = await synthesize_results(
            original_question=request.question,
            sub_results=sub_results,
            enabled=len(sub_results) > 1,
        )

        t2 = time.perf_counter()
        logger.info(f"Research completed in {t2 - t1:.2f}s")

        # Build response
        sub_task_responses = [
            SubTaskResponse(
                id=r.task_id,
                question=r.question,
                answer=r.answer,
                sources=r.sources,
            )
            for r in sub_results
        ]

        return ResearchResponse(
            question=request.question,
            summary=report.summary,
            detailed_answer=report.detailed_answer,
            sub_tasks=sub_task_responses,
            all_sources=report.all_sources,
            confidence=report.confidence,
        )

    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Research request failed",
            status_code=500,
        )


# --- Interactive Deep Research Endpoints ---

@router.post("/plan", response_model=ResearchPlanResponse)
async def generate_research_plan(
    request: ResearchPlanRequest,
    user_id: str = Depends(get_current_user_id)
) -> ResearchPlanResponse:
    """
    Generates a research plan for user confirmation.
    
    This is the first phase of interactive deep research.
    Returns a list of sub-tasks that the user can review, modify,
    or remove before execution.
    
    Args:
        request: Research plan request with question and options.
        user_id: Authenticated user ID (injected).
        
    Returns:
        ResearchPlanResponse with editable sub-tasks.
        
    Raises:
        HTTPException: 500 if planning fails.
    """
    t1 = time.perf_counter()
    logger.info(f"Generating research plan for user {user_id}: {request.question[:50]}...")
    
    try:
        service = get_deep_research_service()
        plan = await service.generate_plan(
            question=request.question,
            user_id=user_id,
            doc_ids=request.doc_ids,
            enable_graph_planning=request.enable_graph_planning,
        )
        
        t2 = time.perf_counter()
        logger.info(f"Plan generated in {t2 - t1:.2f}s with {len(plan.sub_tasks)} tasks")
        
        return plan
        
    except (RuntimeError, ValueError) as e:
        logger.error(f"Plan generation failed: {e}", exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to generate research plan",
            status_code=500,
        )


@router.post("/execute", response_model=ExecutePlanResponse)
async def execute_research_plan(
    request: ExecutePlanRequest,
    user_id: str = Depends(get_current_user_id)
) -> ExecutePlanResponse:
    """
    Executes a user-confirmed research plan with drill-down.
    
    This is the second phase of interactive deep research.
    Runs the confirmed sub-tasks, identifies knowledge gaps,
    and performs recursive drill-down up to max_iterations.
    
    Args:
        request: Confirmed execution request with sub-tasks.
        user_id: Authenticated user ID (injected).
        
    Returns:
        ExecutePlanResponse with complete research results.
        
    Raises:
        HTTPException: 400 if no tasks provided, 500 if execution fails.
    """
    t1 = time.perf_counter()
    enabled_count = sum(1 for t in request.sub_tasks if t.enabled)
    logger.info(
        f"Executing research plan for user {user_id}: "
        f"{enabled_count}/{len(request.sub_tasks)} tasks enabled, "
        f"max_iter={request.max_iterations}"
    )
    
    if not request.sub_tasks:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="At least one sub-task is required",
            status_code=400,
        )
    
    try:
        service = get_deep_research_service()
        result = await service.execute_plan(
            request=request,
            user_id=user_id,
        )
        
        t2 = time.perf_counter()
        logger.info(
            f"Research executed in {t2 - t1:.2f}s: "
            f"{len(result.sub_tasks)} tasks, "
            f"{result.total_iterations} drill-down iterations"
        )
        
        return result
        
    except (RuntimeError, ValueError) as e:
        logger.error(f"Research execution failed: {e}", exc_info=True)
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message="Failed to execute research plan",
            status_code=500,
        )


@router.post("/execute/stream")
async def execute_research_plan_stream(
    request: ExecutePlanRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Executes research plan with SSE streaming progress updates.
    
    This endpoint uses Server-Sent Events (SSE) to stream real-time
    progress updates during deep research execution.
    
    Event types:
    - plan_confirmed: Research started
    - task_start: Sub-task execution started
    - task_done: Sub-task completed
    - drilldown_start: Drill-down iteration started
    - drilldown_task_start: Drill-down task started
    - drilldown_task_done: Drill-down task completed
    - synthesis_start: Final synthesis started
    - complete: Research complete (includes full response)
    - error: Error occurred
    
    Args:
        request: Confirmed execution request with sub-tasks.
        user_id: Authenticated user ID (injected).
        
    Returns:
        EventSourceResponse with SSE stream.
        
    Raises:
        HTTPException: 400 if no tasks provided.
    """
    from sse_starlette.sse import EventSourceResponse
    
    logger.info(
        f"Starting streaming research for user {user_id}: "
        f"{len(request.sub_tasks)} tasks"
    )
    
    if not request.sub_tasks:
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="At least one sub-task is required",
            status_code=400,
        )
    
    service = get_deep_research_service()
    
    async def event_generator():
        try:
            async for event in service.execute_plan_streaming(
                request=request,
                user_id=user_id,
            ):
                yield event
        except (RuntimeError, ValueError) as e:
            logger.error(f"Streaming research failed: {e}", exc_info=True)
            from data_base.sse_events import SSEEventType, ErrorData, format_sse_event
            yield format_sse_event(
                SSEEventType.ERROR,
                ErrorData(message="Streaming execution failed")
            )
    
    return EventSourceResponse(event_generator())
