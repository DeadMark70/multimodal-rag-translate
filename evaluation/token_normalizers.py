"""Provider-specific token usage normalization for evaluation accounting."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NormalizedTokenUsage(BaseModel):
    """Non-overlapping token categories reconciled with provider totals."""

    input_tokens: int = Field(default=0, ge=0)
    output_text_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    other_tokens: int = Field(default=0, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    usage_status: Literal["measured", "missing"]
    reconciliation_status: Literal["balanced", "partial", "unavailable"]


def _coerce_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return None


def first_int(payload: dict[str, Any], *keys: str) -> int | None:
    """Return the first usable non-negative integer among provider aliases."""
    for key in keys:
        value = _coerce_int(payload.get(key))
        if value is not None:
            return value
    return None


def extract_usage_dict(raw_usage: object) -> dict[str, Any]:
    """Extract callback usage metadata without retaining provider objects."""
    usage = getattr(raw_usage, "usage_metadata", raw_usage)
    if isinstance(raw_usage, dict) and "usage_metadata" in raw_usage:
        usage = raw_usage["usage_metadata"]
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    return usage if isinstance(usage, dict) else {}


def normalize_provider_usage(
    provider: str | None, raw_usage: object
) -> NormalizedTokenUsage:
    """Normalize provider usage while preserving missing and partial measurements."""
    payload = extract_usage_dict(raw_usage)
    if not payload:
        return NormalizedTokenUsage(
            usage_status="missing", reconciliation_status="unavailable"
        )

    input_tokens = first_int(
        payload, "input_tokens", "prompt_tokens", "prompt_token_count"
    )
    completion = first_int(
        payload, "output_tokens", "completion_tokens", "candidates_token_count"
    )
    details = payload.get("output_token_details")
    output_details = details if isinstance(details, dict) else {}
    reasoning = first_int(payload, "reasoning_tokens", "thoughts_token_count")
    if reasoning is None:
        reasoning = first_int(output_details, "reasoning")
    total = first_int(payload, "total_tokens", "total_token_count")

    if all(value is None for value in (input_tokens, completion, reasoning, total)):
        return NormalizedTokenUsage(
            usage_status="missing", reconciliation_status="unavailable"
        )

    input_value = input_tokens or 0
    completion_value = completion or 0
    reasoning_value = reasoning or 0
    provider_key = (provider or "").lower()
    output_text = (
        max(completion_value - reasoning_value, 0)
        if provider_key == "openai"
        else completion_value
    )

    if total is None:
        return NormalizedTokenUsage(
            input_tokens=input_value,
            output_text_tokens=output_text,
            reasoning_tokens=reasoning_value,
            usage_status="missing",
            reconciliation_status="partial",
        )

    if provider_key == "openai" and reasoning_value > completion_value:
        return NormalizedTokenUsage(
            input_tokens=input_value,
            output_text_tokens=output_text,
            reasoning_tokens=reasoning_value,
            reported_total_tokens=total,
            usage_status="measured",
            reconciliation_status="partial",
        )

    known = input_value + output_text + reasoning_value
    if known > total:
        return NormalizedTokenUsage(
            input_tokens=input_value,
            output_text_tokens=output_text,
            reasoning_tokens=reasoning_value,
            reported_total_tokens=total,
            usage_status="measured",
            reconciliation_status="partial",
        )
    return NormalizedTokenUsage(
        input_tokens=input_value,
        output_text_tokens=output_text,
        reasoning_tokens=reasoning_value,
        other_tokens=total - known,
        reported_total_tokens=total,
        usage_status="measured",
        reconciliation_status="balanced",
    )
