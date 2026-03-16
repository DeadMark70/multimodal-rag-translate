"""Utilities for normalizing LLM response payloads to plain text."""

from __future__ import annotations

from typing import Any


def response_content_to_text(content: Any) -> str:
    """Best-effort conversion of LangChain/Gemini content blocks into text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = response_content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        for key in ("text", "content", "output"):
            value = content.get(key)
            if value is not None:
                return response_content_to_text(value)
        return ""

    for attr in ("text", "content"):
        value = getattr(content, attr, None)
        if value is not None:
            return response_content_to_text(value)

    return str(content).strip()
