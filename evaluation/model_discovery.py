"""Gemini model discovery with TTL cache."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import google.generativeai as genai

from evaluation.schemas import AvailableModel

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600
_cache: tuple[float, list[AvailableModel]] | None = None
_cache_lock = asyncio.Lock()
_configured_api_key: str | None = None

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
    """Fetch and materialize model pager synchronously."""
    _ensure_genai_configured()
    pager = genai.list_models(request_options={"timeout": 10})
    return list(pager)


def _ensure_genai_configured() -> None:
    """Configure google.generativeai client from env key when available."""
    global _configured_api_key
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"')
    if not api_key:
        return
    if _configured_api_key == api_key:
        return
    genai.configure(api_key=api_key)
    _configured_api_key = api_key


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
                "Dynamic Gemini model discovery unavailable, using fallback list: %s",
                exc,
            )

        if not normalized:
            normalized = [item.model_copy(deep=True) for item in _FALLBACK_MODELS]

        normalized.sort(key=lambda item: (item.display_name or item.name).lower())
        _cache = (now, normalized)
        return [item.model_copy(deep=True) for item in normalized]
