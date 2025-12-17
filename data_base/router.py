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
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from postgrest.exceptions import APIError as PostgrestAPIError

# Local application
from core.auth import get_current_user_id
from supabase_client import supabase
from data_base.RAG_QA_service import initialize_llm_service, rag_answer_question
from data_base.vector_store_manager import initialize_embeddings
from data_base.schemas import AskRequest, AskResponse
from agents.planner import plan_research, SubTask
from agents.synthesizer import synthesize_results, SubTaskResult

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Constants ---

MAX_HISTORY_LENGTH = 10  # Maximum conversation history messages to accept


# --- Pydantic Models ---

class QuestionRequest(BaseModel):
    """Request model for questions."""
    question: str


class AnswerResponse(BaseModel):
    """Response model for answers."""
    question: str
    answer: str
    sources: list[str] = []  # Document IDs used in the response


class ResearchRequest(BaseModel):
    """Request model for research queries."""
    question: str
    max_subtasks: int = 5
    enable_reranking: bool = True


class SubTaskResponse(BaseModel):
    """Response for a single sub-task."""
    id: int
    question: str
    answer: str
    sources: list[str] = []


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

@router.get("/ask", response_model=AnswerResponse)
async def ask_question(
    question: str,
    doc_ids: Optional[str] = Query(
        default=None,
        description="Comma-separated document IDs to filter (e.g., 'uuid1,uuid2'). "
                    "Leave empty to query all documents."
    ),
    user_id: str = Depends(get_current_user_id)
) -> AnswerResponse:
    """
    Answers a question using the user's knowledge base.

    Args:
        question: The question to answer.
        doc_ids: Optional comma-separated document IDs to filter results.
        user_id: Authenticated user ID (injected).

    Returns:
        AnswerResponse with question, answer, and source document IDs.

    Raises:
        HTTPException: 500 if answering fails.
    """
    t1 = time.perf_counter()
    
    # Parse doc_ids
    doc_id_list: Optional[list[str]] = None
    if doc_ids:
        doc_id_list = [d.strip() for d in doc_ids.split(",") if d.strip()]
        logger.info(f"RAG query for user {user_id} with {len(doc_id_list)} doc filter(s)")
    else:
        logger.info(f"RAG query for user {user_id} (all documents)")

    try:
        answer, sources = await rag_answer_question(
            question=question,
            user_id=user_id,
            doc_ids=doc_id_list
        )

        t2 = time.perf_counter()
        logger.info(f"Answered in {t2 - t1:.2f}s, sources: {sources}")

        # Log to Supabase (non-fatal if fails)
        if supabase:
            try:
                log_data = {
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                }
                supabase.table("chat_logs").insert(log_data).execute()
                logger.debug("Chat log saved to Supabase")
            except PostgrestAPIError as e:
                logger.warning(f"Failed to save chat log: {e}")

        return AnswerResponse(question=question, answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"RAG answering failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


@router.post("/ask", response_model=AskResponse)
async def ask_question_with_context(
    request: AskRequest,
    user_id: str = Depends(get_current_user_id)
) -> AskResponse:
    """
    Context-aware question answering with conversation history.

    This enhanced endpoint supports:
    - Conversation history for multi-turn dialogue
    - Advanced retrieval options (HyDE, Multi-Query)
    - Document filtering

    Args:
        request: AskRequest with question, history, and options.
        user_id: Authenticated user ID (injected).

    Returns:
        AskResponse with question, answer, and source document IDs.

    Raises:
        HTTPException: 400 if history too long, 500 if answering fails.
    """
    t1 = time.perf_counter()

    # Validate history length
    if request.history and len(request.history) > MAX_HISTORY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"對話歷史過長，最多允許 {MAX_HISTORY_LENGTH} 條訊息"
        )

    history_count = len(request.history) if request.history else 0
    logger.info(
        f"Context-aware RAG for user {user_id}: "
        f"history={history_count}, hyde={request.enable_hyde}, "
        f"multi_query={request.enable_multi_query}"
    )

    try:
        answer, sources = await rag_answer_question(
            question=request.question,
            user_id=user_id,
            doc_ids=request.doc_ids,
            history=request.history,
            enable_reranking=request.enable_reranking,
            enable_hyde=request.enable_hyde,
            enable_multi_query=request.enable_multi_query,
        )

        t2 = time.perf_counter()
        logger.info(f"Context-aware answer in {t2 - t1:.2f}s, sources: {sources}")

        # Log to Supabase (non-fatal if fails)
        if supabase:
            try:
                log_data = {
                    "user_id": user_id,
                    "question": request.question,
                    "answer": answer,
                    "has_history": history_count > 0,
                }
                supabase.table("chat_logs").insert(log_data).execute()
                logger.debug("Chat log saved to Supabase")
            except PostgrestAPIError as e:
                logger.warning(f"Failed to save chat log: {e}")

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )

    except (RuntimeError, ValueError) as e:
        logger.error(f"Context-aware RAG failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"回答問題時發生錯誤: {str(e)}")


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
        )
        
        logger.info(f"Planned {len(plan.sub_tasks)} sub-tasks")

        # Step 2: Execute - run RAG on each sub-task
        sub_results: List[SubTaskResult] = []
        
        for task in plan.sub_tasks:
            try:
                answer, sources = await rag_answer_question(
                    question=task.question,
                    user_id=user_id,
                    enable_reranking=request.enable_reranking,
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
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")