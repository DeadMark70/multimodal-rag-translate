"""
Supabase Client Configuration

Initializes the Supabase client for database and authentication operations.
"""

# Standard library
import logging
import os
from threading import Lock

# Third-party
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logger = logging.getLogger(__name__)

_client_lock = Lock()
_env_loaded = False


def _load_env_once() -> None:
    """Loads config.env exactly once (lazy, not at import-time)."""
    global _env_loaded
    if _env_loaded:
        return

    dotenv_path = os.path.join(os.path.dirname(__file__), "config.env")
    load_dotenv(dotenv_path=dotenv_path)
    _env_loaded = True


class LazySupabaseClient:
    """Lazy proxy that initializes Supabase client only when first accessed."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self._init_attempted = False

    def initialize(self, force: bool = False) -> Client | None:
        """Initializes and caches the Supabase client."""
        with _client_lock:
            if self._client is not None and not force:
                return self._client
            if self._init_attempted and not force:
                return self._client

            _load_env_once()
            self._init_attempted = True

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                logger.warning(
                    "SUPABASE_URL or SUPABASE_KEY not found - database features disabled"
                )
                self._client = None
                return None

            try:
                self._client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            except Exception as exc:
                logger.error(f"Failed to initialize Supabase client: {exc}")
                self._client = None

            return self._client

    def get_client(self) -> Client | None:
        """Returns a lazily initialized Supabase client."""
        if self._client is None and not self._init_attempted:
            return self.initialize()
        return self._client

    def reset(self) -> None:
        """Resets client state (used by tests)."""
        with _client_lock:
            self._client = None
            self._init_attempted = False

    def __bool__(self) -> bool:
        return self.get_client() is not None

    def __getattr__(self, attr: str):
        client = self.get_client()
        if client is None:
            raise AttributeError(
                "Supabase client is not initialized. "
                "Check SUPABASE_URL and SUPABASE_KEY."
            )
        return getattr(client, attr)


supabase = LazySupabaseClient()


def init_supabase(force: bool = False) -> Client | None:
    """Explicit initialization entrypoint for app startup."""
    return supabase.initialize(force=force)


def get_supabase() -> Client | None:
    """Returns current Supabase client (lazy-initialized)."""
    return supabase.get_client()


def reset_supabase_for_tests() -> None:
    """Test helper for resetting cached Supabase state."""
    supabase.reset()
