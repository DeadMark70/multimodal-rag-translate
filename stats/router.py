"""
Dashboard Statistics Router

Provides API endpoints for dashboard analytics and statistics.
"""

# Standard library
import logging
from datetime import datetime, timedelta
from typing import Optional, List

# Third-party
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from postgrest.exceptions import APIError as PostgrestAPIError

# Local application
from core.auth import get_current_user_id
from supabase_client import supabase

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic Models ---

class DocumentStat(BaseModel):
    """Document usage statistics."""
    doc_id: str
    filename: Optional[str] = None
    query_count: int


class DashboardStats(BaseModel):
    """Dashboard statistics response model."""
    total_queries: int
    accuracy_rate: float
    grounded_count: int
    hallucinated_count: int
    uncertain_count: int
    avg_confidence: float
    queries_last_7_days: List[int]
    top_documents: List[DocumentStat]


# --- Endpoints ---

@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    user_id: str = Depends(get_current_user_id)
) -> DashboardStats:
    """
    Returns dashboard statistics for the user.

    Aggregates query_logs data to provide:
    - Total query count
    - Accuracy metrics (grounded/hallucinated rates)
    - 7-day query trend
    - Most queried documents

    Args:
        user_id: Authenticated user ID (injected).

    Returns:
        DashboardStats with analytics data.

    Raises:
        HTTPException: 500 if database query fails.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        # Get all user's query logs
        result = supabase.table("query_logs")\
            .select("*")\
            .eq("user_id", user_id)\
            .execute()

        logs = result.data if result.data else []
        total = len(logs)

        if total == 0:
            # Return empty stats if no data
            return DashboardStats(
                total_queries=0,
                accuracy_rate=0.0,
                grounded_count=0,
                hallucinated_count=0,
                uncertain_count=0,
                avg_confidence=0.0,
                queries_last_7_days=[0] * 7,
                top_documents=[]
            )

        # Count faithfulness levels
        grounded = sum(1 for log in logs if log.get("faithfulness") == "grounded")
        hallucinated = sum(1 for log in logs if log.get("faithfulness") == "hallucinated")
        uncertain = total - grounded - hallucinated

        # Calculate accuracy rate (grounded / total with evaluation)
        evaluated = grounded + hallucinated
        accuracy_rate = grounded / evaluated if evaluated > 0 else 0.0

        # Calculate average confidence
        confidences = [log.get("confidence") for log in logs if log.get("confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Calculate 7-day trend
        now = datetime.utcnow()
        queries_last_7_days = []
        for i in range(6, -1, -1):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0)
            day_end = day_start + timedelta(days=1)
            count = sum(
                1 for log in logs
                if log.get("created_at") and
                day_start.isoformat() <= log["created_at"] < day_end.isoformat()
            )
            queries_last_7_days.append(count)

        # Get top documents by query count
        doc_counts: dict = {}
        for log in logs:
            doc_ids = log.get("doc_ids") or []
            for doc_id in doc_ids:
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        top_documents = [
            DocumentStat(doc_id=doc_id, query_count=count)
            for doc_id, count in sorted(
                doc_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        ]

        logger.info(f"Dashboard stats for user {user_id}: {total} queries, {accuracy_rate:.1%} accuracy")

        return DashboardStats(
            total_queries=total,
            accuracy_rate=accuracy_rate,
            grounded_count=grounded,
            hallucinated_count=hallucinated,
            uncertain_count=uncertain,
            avg_confidence=avg_confidence,
            queries_last_7_days=queries_last_7_days,
            top_documents=top_documents
        )

    except PostgrestAPIError as e:
        logger.error(f"Failed to get dashboard stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
