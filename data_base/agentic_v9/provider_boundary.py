"""Shared production provider construction for Agentic v9 contract planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.providers import bind_json_schema, get_llm


def build_contract_planning_provider(
    *, response_schema: Mapping[str, Any]
) -> Any:
    """Build the production planner model and bind its selected JSON schema."""
    return bind_json_schema(
        get_llm("synthesizer"),
        schema=dict(response_schema),
    )
