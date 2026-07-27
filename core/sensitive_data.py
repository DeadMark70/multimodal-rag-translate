"""Shared, conservative predicates for credential-bearing structured data."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


_EXACT_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "token",
    }
)
_CREDENTIAL_SUFFIXES = (
    "token",
    "secret",
    "password",
    "passwd",
    "privatekey",
    "credential",
    "credentials",
)
_MAX_JSON_STRING_CHARS = 64 * 1024
_MAX_JSON_STRING_DEPTH = 64
_ASSIGNMENT_KEY_PATTERN = re.compile(
    r"""(?P<key>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_][A-Za-z0-9_-]*)\s*[:=]\s*"""
)


def is_sensitive_credential_key(value: Any) -> bool:
    """Return whether a structured key denotes a credential, not a public ID."""
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized in _EXACT_CREDENTIAL_KEYS or normalized.endswith(
        _CREDENTIAL_SUFFIXES
    )


def sanitize_credential_assignments_text(value: str) -> tuple[str, bool]:
    """Redact bounded credential assignments in otherwise unstructured text."""
    if len(value) > _MAX_JSON_STRING_CHARS:
        return "[REDACTED]", True

    parts: list[str] = []
    previous_end = 0
    redacted = False
    for match in _ASSIGNMENT_KEY_PATTERN.finditer(value):
        key = match.group("key")
        if not is_sensitive_credential_key(key[1:-1] if key[0] in "\"'" else key):
            continue
        value_end = _assignment_value_end(value, match.end())
        if value_end == match.end():
            continue
        parts.extend((value[previous_end : match.end()], "[REDACTED]"))
        previous_end = value_end
        redacted = True
    if not redacted:
        return value, False
    parts.append(value[previous_end:])
    return "".join(parts), True


def _assignment_value_end(value: str, start: int) -> int:
    if start >= len(value):
        return start
    quote = value[start]
    if quote in "\"'":
        escaped = False
        for index in range(start + 1, len(value)):
            character = value[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                return index + 1
        return len(value)
    end = start
    while end < len(value) and value[end] not in "\t\r\n ,;} ]":
        end += 1
    return end


def sanitize_json_object_text(value: str) -> tuple[str | None, bool]:
    """Safely sanitize a JSON object/list encoded within a text value.

    Oversized or deeply nested JSON-shaped strings are replaced rather than
    parsed, preventing an unbounded structured credential from reaching capture
    or export evidence. Invalid JSON remains ``None`` so callers can apply their
    ordinary text-redaction rules.
    """
    stripped = value.lstrip()
    if not stripped.startswith(("{", "[")):
        return None, False
    if len(value) > _MAX_JSON_STRING_CHARS or not _json_nesting_is_bounded(value):
        return "[REDACTED]", True
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, RecursionError, ValueError):
        return None, False
    if not isinstance(parsed, (dict, list)):
        return None, False
    sanitized, redacted = sanitize_credential_value(parsed)
    return (
        json.dumps(
            sanitized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        redacted,
    )


def sanitize_credential_value(value: Any) -> tuple[Any, bool]:
    """Recursively redact credential values in structured JSON-compatible data."""
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        redacted = False
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_credential_key(key_text):
                sanitized[key_text] = "[REDACTED]"
                redacted = True
                continue
            sanitized_item, item_redacted = sanitize_credential_value(item)
            sanitized[key_text] = sanitized_item
            redacted = redacted or item_redacted
        return sanitized, redacted
    if isinstance(value, (list, tuple)):
        items = []
        redacted = False
        for item in value:
            sanitized_item, item_redacted = sanitize_credential_value(item)
            items.append(sanitized_item)
            redacted = redacted or item_redacted
        return items, redacted
    return value, False


def _json_nesting_is_bounded(value: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    for character in value:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "{[":
            depth += 1
            if depth > _MAX_JSON_STRING_DEPTH:
                return False
        elif character in "}]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted
