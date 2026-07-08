"""Token usage normalization and local price estimation for evaluation runs."""

from __future__ import annotations

from typing import Any


DEFAULT_PRICE_SNAPSHOT: dict[str, Any] = {
    "snapshot_id": "local-unknown",
    "currency": "USD",
    "usd_to_twd": None,
    "models": {},
}


def _coerce_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _extract_usage_payload(raw_usage: Any) -> dict[str, Any]:
    usage = getattr(raw_usage, "usage_metadata", raw_usage)
    if isinstance(raw_usage, dict) and "usage_metadata" in raw_usage:
        usage = raw_usage["usage_metadata"]
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    return usage if isinstance(usage, dict) else {}


def normalize_llm_usage(raw_usage: Any) -> dict[str, int]:
    """Normalize provider token fields to prompt/completion/total/reasoning tokens."""
    usage = _extract_usage_payload(raw_usage)
    output_details = usage.get("output_token_details") or {}
    prompt_tokens = _coerce_int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
    )
    completion_tokens = _coerce_int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("candidates_token_count")
    )
    total_tokens = _coerce_int(usage.get("total_tokens") or usage.get("total_token_count"))
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    reasoning_tokens = _coerce_int(
        usage.get("reasoning_tokens")
        or output_details.get("reasoning")
        or usage.get("thoughts_token_count")
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def price_llm_usage(
    *,
    model_name: str | None,
    usage: dict[str, int],
    price_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate cost from a caller-supplied price snapshot.

    The default snapshot intentionally has no rates; callers can pass audited
    research price snapshots without hard-coding fast-changing provider prices.
    """
    snapshot = price_snapshot or DEFAULT_PRICE_SNAPSHOT
    rates = (snapshot.get("models") or {}).get(model_name or "")
    if not isinstance(rates, dict):
        return {
            "estimated_cost_usd": None,
            "estimated_cost_twd": None,
            "price_snapshot_id": snapshot.get("snapshot_id"),
        }

    input_rate = float(rates.get("input_per_1m_usd") or 0)
    output_rate = float(rates.get("output_per_1m_usd") or 0)
    estimated_cost_usd = (
        (usage.get("prompt_tokens", 0) / 1_000_000) * input_rate
        + (usage.get("completion_tokens", 0) / 1_000_000) * output_rate
    )
    usd_to_twd = snapshot.get("usd_to_twd")
    estimated_cost_twd = (
        estimated_cost_usd * float(usd_to_twd)
        if usd_to_twd not in (None, "")
        else None
    )
    return {
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_cost_twd": estimated_cost_twd,
        "price_snapshot_id": snapshot.get("snapshot_id"),
    }
