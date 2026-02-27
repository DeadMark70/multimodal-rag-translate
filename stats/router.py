"""
Dashboard Statistics Router

Provides API endpoints for dashboard analytics and statistics.
"""

# Standard library
import logging

# Third-party
from fastapi import APIRouter, Depends

# Local application
from core.auth import get_current_user_id
from stats.schemas import DashboardStats
from stats.service import build_dashboard_stats

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


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
    stats = await build_dashboard_stats(user_id=user_id)
    logger.info(
        "Dashboard stats for user %s: %s queries, %.1f%% accuracy",
        user_id,
        stats.total_queries,
        stats.accuracy_rate * 100,
    )
    return stats
