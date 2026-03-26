"""Shared Google GenAI client helpers for control-plane integrations.

Architecture note:
- direct `google-genai` usage belongs in control-plane modules only
- runtime chat/model access stays behind `core.providers.get_llm(...)`
- runtime embedding creation stays centralized in `data_base.vector_store_manager`
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

_CLIENT: genai.Client | None = None
_CLIENT_API_KEY: str | None = None


def resolve_google_api_key() -> str | None:
    """Return the active Google API key, supporting the compatible fallback env."""
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        api_key = (os.getenv(env_name) or "").strip().strip('"')
        if api_key:
            return api_key
    return None


def get_google_genai_client() -> genai.Client | None:
    """Return a cached Google GenAI client for control-plane SDK operations."""
    global _CLIENT
    global _CLIENT_API_KEY

    api_key = resolve_google_api_key()
    if not api_key:
        _CLIENT = None
        _CLIENT_API_KEY = None
        return None

    if _CLIENT is not None and _CLIENT_API_KEY == api_key:
        return _CLIENT

    _CLIENT = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=10_000),
    )
    _CLIENT_API_KEY = api_key
    return _CLIENT


def reset_google_genai_client_cache() -> None:
    """Clear cached direct-SDK client state for tests or env changes."""
    global _CLIENT
    global _CLIENT_API_KEY

    _CLIENT = None
    _CLIENT_API_KEY = None
