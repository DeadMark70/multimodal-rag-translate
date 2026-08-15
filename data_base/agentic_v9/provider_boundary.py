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


def evidence_qualification_response_schema() -> dict[str, Any]:
    """Return the strict JSON shape accepted from evidence qualification."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "packets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_evidence_id": {"type": "string", "minLength": 1},
                        "slot_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "statement": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "source_evidence_id",
                        "slot_ids",
                        "statement",
                    ],
                },
            }
        },
        "required": ["packets"],
    }


def build_evidence_qualification_provider(
    *, response_schema: Mapping[str, Any]
) -> Any:
    """Build the production evidence model with strict JSON output."""
    return bind_json_schema(
        get_llm("synthesizer"),
        schema=dict(response_schema),
    )
