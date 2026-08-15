"""Shared production provider construction and response normalization for Agentic v9."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.providers import bind_json_schema, get_llm


def provider_response_text(response: Any) -> str | None:
    """Return provider text without serializing non-text content blocks."""
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        content = response.get("content")
        return content if isinstance(content, str) else None
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    text = getattr(response, "text", None)
    return str(text) if isinstance(text, str) else None


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
