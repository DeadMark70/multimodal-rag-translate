"""
Centralized Authentication Module

Provides shared authentication dependency for all routers.
"""

# Standard library
import logging
import os

# Third-party
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Local application
from core.auth_repository import fetch_user_id_from_token
from core.errors import AppError, ErrorCode

# Configure logging
logger = logging.getLogger(__name__)

# Feature flag for testing (set DEV_MODE=true in env to bypass auth)
_DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """
    Validates Supabase JWT token and extracts user ID.

    This is the centralized authentication dependency used by all routers.
    Set DEV_MODE=true in environment to bypass authentication for testing.

    Args:
        authorization: Bearer token from Authorization header.

    Returns:
        The authenticated user's ID.

    Raises:
        AppError: 401 if token is missing/invalid, 500 if auth service unavailable.
    """
    # Dev mode bypass (for testing only)
    if _DEV_MODE:
        logger.warning("DEV_MODE enabled - using test user ID")
        return "00000000-0000-0000-0000-000000000001"

    if not credentials or not credentials.credentials:
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            message="Missing Authorization header",
            status_code=401,
        )

    token = credentials.credentials

    try:
        return await fetch_user_id_from_token(token)

    except AppError:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            message="Authentication failed",
            status_code=401,
        )
