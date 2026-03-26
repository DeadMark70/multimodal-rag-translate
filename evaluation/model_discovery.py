"""Gemini model discovery with TTL cache.

Architecture note:
- this module is control-plane code and may consume direct `google-genai` helpers
- it should not instantiate runtime LangChain chat or embedding clients
- model normalization stays here; direct SDK client lifecycle lives in
  `core.google_genai_client`
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.google_genai_client import get_google_genai_client
from evaluation.schemas import AvailableModel

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600
_cache: tuple[float, list[AvailableModel]] | None = None
_cache_lock = asyncio.Lock()

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


def _fetch_models_sync() -> list[Any]:
    """Fetch and materialize model pager synchronously.

    Iterates manually so that a single model entry that fails to deserialize
    does not abort the entire listing.
    """
    client = get_google_genai_client()
    if client is None:
        return []

    pager = client.models.list(config={"page_size": 100})
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
