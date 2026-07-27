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


def is_sensitive_credential_key(value: Any) -> bool:
    """Return whether a structured key denotes a credential, not a public ID."""
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized in _EXACT_CREDENTIAL_KEYS or normalized.endswith(
        _CREDENTIAL_SUFFIXES
    )


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
