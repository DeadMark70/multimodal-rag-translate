"""Shared, conservative predicates for credential-bearing structured data."""

from __future__ import annotations

import re
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


def is_sensitive_credential_key(value: Any) -> bool:
    """Return whether a structured key denotes a credential, not a public ID."""
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized in _EXACT_CREDENTIAL_KEYS or normalized.endswith(
        _CREDENTIAL_SUFFIXES
    )
