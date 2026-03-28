"""Gemini model discovery with resilient caching and curated fallback models.

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
from typing import Any, Iterable

from core.google_genai_client import get_google_genai_client
from evaluation.schemas import AvailableModel

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600
_FALLBACK_CACHE_TTL_SECONDS = 60
_cache: tuple[float, list[AvailableModel], bool] | None = None
_cache_lock = asyncio.Lock()

_FALLBACK_MODELS: list[AvailableModel] = [
    AvailableModel(
        name="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        description="Curated fallback list when dynamic discovery is temporarily unavailable.",
        input_token_limit=1_048_576,
        output_token_limit=65_536,
        supported_actions=["generateContent"],
    ),
    AvailableModel(
        name="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        description="Curated fallback list when dynamic discovery is temporarily unavailable.",
        input_token_limit=1_048_576,
        output_token_limit=8_192,
        supported_actions=["generateContent"],
    ),
    AvailableModel(
        name="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
        description="Curated fallback list when dynamic discovery is temporarily unavailable.",
        input_token_limit=1_048_576,
        output_token_limit=8_192,
        supported_actions=["generateContent"],
    ),
    AvailableModel(
        name="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        description="Curated fallback list when dynamic discovery is temporarily unavailable.",
        input_token_limit=1_048_576,
        output_token_limit=8_192,
        supported_actions=["generateContent"],
    ),
]


def _clone_models(models: list[AvailableModel]) -> list[AvailableModel]:
    return [item.model_copy(deep=True) for item in models]


def _cache_ttl(entry: tuple[float, list[AvailableModel], bool]) -> int:
    return _CACHE_TTL_SECONDS if entry[2] else _FALLBACK_CACHE_TTL_SECONDS


def _read_field(raw_model: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(raw_model, dict):
        return raw_model.get(field_name, default)
    return getattr(raw_model, field_name, default)


def _coerce_actions(raw_model: Any) -> list[str]:
    actions = _read_field(raw_model, "supported_generation_methods")
    if actions is None:
        actions = _read_field(raw_model, "supported_actions")
    if isinstance(actions, str):
        return [actions]
    if isinstance(actions, Iterable):
        return [str(item) for item in actions if item]
    return []


def _iter_models(pager: Any) -> list[Any]:
    results: list[Any] = []
    iterator = iter(pager)
    while True:
        try:
            results.append(next(iterator))
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unparseable model entry: %s", exc)
            continue
    return results


def _fetch_models_sync() -> list[Any]:
    """Fetch and materialize the base model pager synchronously."""
    client = get_google_genai_client()
    if client is None:
        return []

    pager = client.models.list(config={"page_size": 100})
    return _iter_models(pager)


def _normalize_model(raw_model: Any) -> AvailableModel | None:
    supported_actions = _coerce_actions(raw_model)
    if "generateContent" not in supported_actions:
        return None

    raw_name = str(_read_field(raw_model, "name", "") or "")
    model_name = raw_name.replace("models/", "", 1) if raw_name.startswith("models/") else raw_name
    if not model_name or not model_name.startswith("gemini-"):
        return None

    display_name = _read_field(raw_model, "display_name") or model_name
    description = _read_field(raw_model, "description")
    input_token_limit = _read_field(raw_model, "input_token_limit")
    output_token_limit = _read_field(raw_model, "output_token_limit")

    return AvailableModel(
        name=model_name,
        display_name=str(display_name) if display_name else None,
        description=str(description) if description else None,
        input_token_limit=int(input_token_limit) if input_token_limit is not None else None,
        output_token_limit=int(output_token_limit) if output_token_limit is not None else None,
        supported_actions=supported_actions,
    )


async def list_available_models(force_refresh: bool = False) -> list[AvailableModel]:
    """Return available Gemini models for the current API key."""

    global _cache

    async with _cache_lock:
        now = time.time()
        if not force_refresh and _cache is not None and (now - _cache[0]) < _cache_ttl(_cache):
            return _clone_models(_cache[1])

        previous_dynamic = _clone_models(_cache[1]) if _cache is not None and _cache[2] else None
        normalized: list[AvailableModel] = []

        try:
            raw_models = await asyncio.to_thread(_fetch_models_sync)
            seen: set[str] = set()
            for raw_model in raw_models:
                model = _normalize_model(raw_model)
                if model is None or model.name in seen:
                    continue
                seen.add(model.name)
                normalized.append(model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dynamic Gemini model discovery failed (%s: %s), using cached/fallback list.",
                type(exc).__name__,
                exc,
            )

        if normalized:
            normalized.sort(key=lambda item: (item.display_name or item.name).lower())
            _cache = (now, normalized, True)
            return _clone_models(normalized)

        if previous_dynamic:
            logger.info("Dynamic discovery unavailable; reusing last successful Gemini model cache.")
            _cache = (now, previous_dynamic, True)
            return _clone_models(previous_dynamic)

        fallback = _clone_models(_FALLBACK_MODELS)
        fallback.sort(key=lambda item: (item.display_name or item.name).lower())
        _cache = (now, fallback, False)
        return fallback
