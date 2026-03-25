"""Gemini model discovery with TTL cache."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types

from evaluation.schemas import AvailableModel

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600
_cache: tuple[float, list[AvailableModel]] | None = None
_cache_lock = asyncio.Lock()
_client: genai.Client | None = None
_client_api_key: str | None = None

_FALLBACK_MODELS: list[AvailableModel] = [
    AvailableModel(
        name="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        description="Fallback model list (dynamic discovery unavailable)",
        supported_actions=["generateContent"],
    ),
    AvailableModel(
        name="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
        description="Fallback model list (dynamic discovery unavailable)",
        supported_actions=["generateContent"],
    ),
    AvailableModel(
        name="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        description="Fallback model list (dynamic discovery unavailable)",
        supported_actions=["generateContent"],
    ),
]


def _read_api_key() -> str | None:
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        api_key = (os.getenv(env_name) or "").strip().strip('"')
        if api_key:
            return api_key
    return None


def _get_genai_client() -> genai.Client | None:
    """Return a cached Google GenAI client for the active API key."""
    global _client
    global _client_api_key

    api_key = _read_api_key()
    if not api_key:
        _client = None
        _client_api_key = None
        return None

    if _client is not None and _client_api_key == api_key:
        return _client

    _client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=10_000),
    )
    _client_api_key = api_key
    return _client


def _fetch_models_sync() -> list[Any]:
    """Fetch and materialize model pager synchronously.

    Iterates manually so that a single model entry that fails to deserialize
    does not abort the entire listing.
    """
    client = _get_genai_client()
    if client is None:
        return []

    pager = client.models.list(config=types.ListModelsConfig(page_size=100))
    results: list[Any] = []
    iterator = iter(pager)
    while True:
        try:
            raw = next(iterator)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unparseable model entry: %s", exc)
            continue
        results.append(raw)
    return results


def _normalize_model(raw_model: Any) -> AvailableModel | None:
    supported_actions = list(getattr(raw_model, "supported_generation_methods", []) or [])
    if "generateContent" not in supported_actions:
        return None

    raw_name = str(getattr(raw_model, "name", ""))
    model_name = raw_name.replace("models/", "", 1) if raw_name else ""
    if not model_name:
        return None

    return AvailableModel(
        name=model_name,
        display_name=getattr(raw_model, "display_name", None),
        description=getattr(raw_model, "description", None),
        input_token_limit=getattr(raw_model, "input_token_limit", None),
        output_token_limit=getattr(raw_model, "output_token_limit", None),
        supported_actions=supported_actions,
    )


async def list_available_models(force_refresh: bool = False) -> list[AvailableModel]:
    """Return available Gemini models for the current API key."""

    global _cache

    async with _cache_lock:
        now = time.time()
        if (
            not force_refresh
            and _cache is not None
            and (now - _cache[0]) < _CACHE_TTL_SECONDS
        ):
            return [item.model_copy(deep=True) for item in _cache[1]]

        normalized: list[AvailableModel] = []
        try:
            raw_models = await asyncio.to_thread(_fetch_models_sync)
            for raw_model in raw_models:
                model = _normalize_model(raw_model)
                if model is not None:
                    normalized.append(model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dynamic Gemini model discovery failed (%s: %s), using fallback list.",
                type(exc).__name__,
                exc,
            )

        if not normalized:
            normalized = [item.model_copy(deep=True) for item in _FALLBACK_MODELS]

        normalized.sort(key=lambda item: (item.display_name or item.name).lower())
        _cache = (now, normalized)
        return [item.model_copy(deep=True) for item in normalized]
