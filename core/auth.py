"""
Centralized Authentication Module

Provides shared authentication dependency for all routers.
"""

# Standard library
import logging
import os

# Third-party
from fastapi import Header, HTTPException

# Local application
from supabase_client import supabase

# Configure logging
logger = logging.getLogger(__name__)

# Feature flag for testing (set DEV_MODE=true in env to bypass auth)
_DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"


async def get_current_user_id(authorization: str = Header(None)) -> str:
    """
    Validates Supabase JWT token and extracts user ID.

    This is the centralized authentication dependency used by all routers.
    Set DEV_MODE=true in environment to bypass authentication for testing.

    Args:
        authorization: Bearer token from Authorization header.

    Returns:
        The authenticated user's ID.

    Raises:
        HTTPException: 401 if token is missing/invalid, 500 if Supabase unavailable.
    """
    # Dev mode bypass (for testing only)
    if _DEV_MODE:
        logger.warning("DEV_MODE enabled - using test user ID")
        return "test-user-id-001"

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")

    # Validate header format
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization Header format")

    token = parts[1]

    if not supabase:
        logger.error("Supabase client not initialized")
        raise HTTPException(status_code=500, detail="Authentication service unavailable")

    try:
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid Token")

        return user_response.user.id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Authentication Failed")
