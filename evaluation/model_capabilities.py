"""Model-family thinking capability rules for evaluation presets."""

from __future__ import annotations

from typing import Any, cast

from evaluation.schemas import ThinkingCapability, ThinkingLevel

_DEFAULT_LEVELS: list[ThinkingLevel] = ["minimal", "low", "medium", "high"]
_BASE_RUNTIME_KEYS = {
    "model_name",
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
}


def _normalize_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip()
    return normalized.replace("models/", "", 1) if normalized.startswith("models/") else normalized


def get_thinking_capability(model_name: str) -> ThinkingCapability:
    """Return model-specific thinking control metadata."""
    normalized = _normalize_model_name(model_name)

    if normalized.startswith("gemini-3"):
        return ThinkingCapability(
            supported=True,
            control_type="level",
            levels=list(_DEFAULT_LEVELS),
            default_level="medium",
            guidance="Gemini 3 models should use thinking_level rather than thinking_budget.",
        )

    if normalized.startswith("gemini-2.5-flash-lite"):
        return ThinkingCapability(
            supported=True,
            control_type="budget",
            budget_min=0,
            budget_max=24576,
            supports_disable=True,
            supports_dynamic=True,
            default_budget=8192,
            guidance="Gemini 2.5 Flash Lite uses thinking_budget; -1 enables dynamic thinking.",
        )

    if normalized.startswith("gemini-2.5-flash"):
        return ThinkingCapability(
            supported=True,
            control_type="budget",
            budget_min=0,
            budget_max=24576,
            supports_disable=True,
            supports_dynamic=True,
            default_budget=8192,
            guidance="Gemini 2.5 Flash uses thinking_budget; 0 disables thinking and -1 enables dynamic thinking.",
        )

    if normalized.startswith("gemini-2.5-pro"):
        return ThinkingCapability(
            supported=True,
            control_type="budget",
            budget_min=128,
            budget_max=32768,
            supports_disable=False,
            supports_dynamic=True,
            default_budget=8192,
            guidance="Gemini 2.5 Pro uses thinking_budget and does not support disabling thinking.",
        )

    if normalized.startswith("gemma-4"):
        return ThinkingCapability(
            supported=True,
            control_type="level",
            levels=list(_DEFAULT_LEVELS),
            default_level="medium",
            guidance="Gemma 4 is configured through thinking_level in this evaluation center.",
        )

    return ThinkingCapability(
        supported=False,
        control_type="none",
        guidance="This model has no configurable thinking controls in the evaluation center.",
    )


def _clamp_budget(raw_value: Any, capability: ThinkingCapability) -> int | None:
    if raw_value is None:
        return capability.default_budget

    try:
        budget = int(raw_value)
    except (TypeError, ValueError):
        return capability.default_budget

    if budget == -1 and capability.supports_dynamic:
        return budget
    if budget == 0 and capability.supports_disable:
        return budget

    min_budget = capability.budget_min
    max_budget = capability.budget_max
    if min_budget is not None and budget < min_budget:
        return min_budget
    if max_budget is not None and budget > max_budget:
        return max_budget
    return budget


def _coerce_level(raw_value: Any, capability: ThinkingCapability) -> ThinkingLevel | None:
    if isinstance(raw_value, str) and raw_value in capability.levels:
        return cast(ThinkingLevel, raw_value)
    return capability.default_level


def normalize_model_config_for_runtime(model_config: dict[str, Any]) -> dict[str, Any]:
    """Return request-scoped LLM overrides with valid thinking parameters only."""
    runtime = {
        key: model_config.get(key)
        for key in _BASE_RUNTIME_KEYS
        if model_config.get(key) is not None
    }
    capability = get_thinking_capability(str(model_config.get("model_name") or ""))

    if not model_config.get("thinking_mode") or not capability.supported:
        return runtime

    if capability.control_type == "budget":
        budget = _clamp_budget(model_config.get("thinking_budget"), capability)
        if budget is not None:
            runtime["thinking_budget"] = budget
            runtime["include_thoughts"] = bool(model_config.get("thinking_include_thoughts", False))
        return runtime

    if capability.control_type == "level":
        level = _coerce_level(model_config.get("thinking_level"), capability)
        if level is not None:
            runtime["thinking_level"] = level
            runtime["include_thoughts"] = bool(model_config.get("thinking_include_thoughts", False))
        return runtime

    return runtime


def normalize_model_config_for_storage(model_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize persisted model config while keeping old presets readable."""
    normalized = dict(model_config)
    capability = get_thinking_capability(str(normalized.get("model_name") or ""))

    normalized.setdefault("thinking_mode", False)
    normalized.setdefault("thinking_include_thoughts", False)

    if not capability.supported:
        normalized["thinking_mode"] = False
        normalized["thinking_budget"] = None
        normalized["thinking_level"] = None
        return normalized

    if capability.control_type == "budget":
        normalized["thinking_budget"] = _clamp_budget(normalized.get("thinking_budget"), capability)
        normalized["thinking_level"] = None
        return normalized

    if capability.control_type == "level":
        normalized["thinking_level"] = _coerce_level(normalized.get("thinking_level"), capability)
        normalized["thinking_budget"] = None
        return normalized

    return normalized
