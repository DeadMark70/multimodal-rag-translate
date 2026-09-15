"""Small provider-response helpers shared by answer-producing paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_TEXT_BLOCK_TYPES = frozenset({"text", "output_text"})
_NON_ANSWER_BLOCK_TYPES = frozenset(
    {
        "thinking",
        "reasoning",
        "analysis",
        "signature",
        "tool_call",
        "tool_use",
        "function_call",
    }
)


def final_answer_text(response: Any) -> str:
    """Return visible provider text without exposing reasoning or tool payloads.

    LangChain's ``AIMessage.text`` is the preferred projection when available.
    Some provider adapters instead retain a list of typed content blocks, so this
    helper also accepts only explicit text blocks. Missing visible text remains
    empty; callers decide whether that is a valid partial response or a failure.
    """
    if isinstance(response, str):
        return response.strip()

    text = _mapping_or_attribute(response, "text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = _mapping_or_attribute(response, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(_text_block_value(block) for block in content).strip()
    return ""


def _mapping_or_attribute(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _text_block_value(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, Mapping):
        return ""
    block_type = block.get("type")
    if isinstance(block_type, str):
        normalized_type = block_type.strip().lower()
        if normalized_type in _NON_ANSWER_BLOCK_TYPES:
            return ""
        if normalized_type not in _TEXT_BLOCK_TYPES:
            return ""
    text = block.get("text")
    return text if isinstance(text, str) else ""


def safe_validation_fields(error: BaseException) -> list[str]:
    """Project at most eight validation locations without retaining values/text."""
    errors = getattr(error, "errors", None)
    if not callable(errors):
        return []
    try:
        details = errors()
    except (TypeError, ValueError):
        return []
    if not isinstance(details, list):
        return []
    fields: list[str] = []
    for item in details:
        location = item.get("loc") if isinstance(item, Mapping) else None
        if not isinstance(location, (tuple, list)):
            continue
        field = ".".join(
            str(part) for part in location if isinstance(part, (str, int))
        )
        if field and field not in fields:
            fields.append(field)
        if len(fields) >= 8:
            break
    return fields


__all__ = ["final_answer_text", "safe_validation_fields"]
