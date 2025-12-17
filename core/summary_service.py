"""
Executive Summary Generation Service

Provides LLM-based document summarization for generating executive briefings.
Supports both synchronous generation and async background processing.
"""

# Standard library
import asyncio
import logging
from typing import Optional

# Third-party
from langchain_core.messages import HumanMessage
from postgrest.exceptions import APIError as PostgrestAPIError

# Local application
from core.llm_factory import get_llm
from supabase_client import supabase

# Configure logging
logger = logging.getLogger(__name__)

# Executive briefing prompt template
_SUMMARY_PROMPT = """You are an expert research assistant. Provide a structured Executive Briefing for the following document.

Structure your response with these sections:
## 核心問題與假設 (Core Problem & Hypothesis)
[Identify the main problem being addressed and any hypotheses]

## 研究方法 (Methodology)
[Describe the approach, methods, or framework used]

## 關鍵發現 (Key Findings)
[List the most important results and insights]

## 影響與應用 (Industry/Academic Impact)
[Explain the significance and potential applications]

IMPORTANT:
- Keep the entire briefing under 500 words
- Use Traditional Chinese (繁體中文)
- Be concise but comprehensive
- If the document is not academic, adapt the sections appropriately

Document content:
{content}

Executive Briefing:"""


async def generate_executive_summary(
    text_content: str,
    max_length: int = 8000,
) -> Optional[str]:
    """
    Generates an executive summary for document text content.

    Uses LLM to create a structured briefing with problem, methodology,
    findings, and impact sections.

    Args:
        text_content: The document text to summarize.
        max_length: Maximum characters to send to LLM (to control token usage).

    Returns:
        Generated summary string, or None if generation fails.
    """
    if not text_content or len(text_content.strip()) < 100:
        logger.warning("Text content too short for summary generation")
        return None

    try:
        llm = get_llm("summary")
    except (RuntimeError, KeyError, ValueError) as e:
        logger.error(f"Failed to get LLM for summary: {e}")
        return None

    # Truncate content if too long
    truncated_content = text_content[:max_length]
    if len(text_content) > max_length:
        truncated_content += "\n\n[... 內容已截斷 ...]"
        logger.debug(f"Content truncated from {len(text_content)} to {max_length} chars")

    prompt = _SUMMARY_PROMPT.format(content=truncated_content)
    message = HumanMessage(content=prompt)

    try:
        response = await llm.ainvoke([message])
        summary = response.content.strip()
        logger.info(f"Generated summary: {len(summary)} chars")
        return summary
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        return None


async def update_document_summary(
    doc_id: str,
    summary: str,
) -> bool:
    """
    Updates the executive_summary field in Supabase documents table.

    Args:
        doc_id: Document UUID.
        summary: Generated executive summary.

    Returns:
        True if update successful, False otherwise.
    """
    if not supabase:
        logger.warning("Supabase client not available, cannot update summary")
        return False

    try:
        supabase.table("documents").update({
            "executive_summary": summary
        }).eq("id", doc_id).execute()

        logger.info(f"Updated summary for doc {doc_id}")
        return True

    except PostgrestAPIError as e:
        logger.error(f"Failed to update summary in Supabase: {e}")
        return False


async def get_document_summary(doc_id: str) -> Optional[str]:
    """
    Retrieves the executive summary for a document from Supabase.

    Args:
        doc_id: Document UUID.

    Returns:
        Summary string if exists, None otherwise.
    """
    if not supabase:
        logger.warning("Supabase client not available")
        return None

    try:
        result = supabase.table("documents").select(
            "executive_summary"
        ).eq("id", doc_id).single().execute()

        return result.data.get("executive_summary") if result.data else None

    except PostgrestAPIError as e:
        logger.error(f"Failed to get summary from Supabase: {e}")
        return None


async def generate_summary_background(
    doc_id: str,
    text_content: str,
    user_id: str,
) -> None:
    """
    Background task to generate and save document summary.

    This function is designed to be called via asyncio.create_task()
    to run non-blocking after PDF upload/extraction.

    Args:
        doc_id: Document UUID.
        text_content: Full text content of the document.
        user_id: User ID (for logging purposes).
    """
    logger.info(f"[Background] Starting summary generation for doc {doc_id}")

    try:
        # Generate summary
        summary = await generate_executive_summary(text_content)

        if summary:
            # Save to database
            success = await update_document_summary(doc_id, summary)
            if success:
                logger.info(f"[Background] Summary saved for doc {doc_id}")
            else:
                logger.warning(f"[Background] Failed to save summary for doc {doc_id}")
        else:
            logger.warning(f"[Background] No summary generated for doc {doc_id}")

    except Exception as e:
        # Catch all exceptions to prevent background task from crashing
        logger.error(f"[Background] Summary task failed for doc {doc_id}: {e}", exc_info=True)


def schedule_summary_generation(
    doc_id: str,
    text_content: str,
    user_id: str,
) -> None:
    """
    Schedules background summary generation as a non-blocking task.

    Safe to call from sync or async context. Creates a new task
    that will complete independently.

    Args:
        doc_id: Document UUID.
        text_content: Full text content of the document.
        user_id: User ID (for logging).
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(generate_summary_background(doc_id, text_content, user_id))
        logger.debug(f"Scheduled background summary for doc {doc_id}")
    except RuntimeError:
        # No running event loop - this shouldn't happen in FastAPI context
        logger.warning("No running event loop, summary generation skipped")
