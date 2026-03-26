"""Helpers for canonical document metadata access."""

from __future__ import annotations

from typing import Any

LEGACY_DOC_ID_KEY = "original_doc_uid"
CANONICAL_DOC_ID_KEY = "doc_id"


def get_document_id(metadata: dict[str, Any] | None) -> str | None:
    """Return the canonical or legacy document identifier from metadata."""
    if not metadata:
        return None
    doc_id = metadata.get(CANONICAL_DOC_ID_KEY)
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    legacy_doc_id = metadata.get(LEGACY_DOC_ID_KEY)
    if isinstance(legacy_doc_id, str) and legacy_doc_id:
        return legacy_doc_id
    return None


def matches_document_id(metadata: dict[str, Any] | None, doc_id: str) -> bool:
    """Return True when metadata belongs to the requested document id."""
    return get_document_id(metadata) == doc_id


def with_document_id(metadata: dict[str, Any] | None, doc_id: str) -> dict[str, Any]:
    """Return metadata normalized to the canonical document-id key."""
    normalized = dict(metadata or {})
    normalized[CANONICAL_DOC_ID_KEY] = doc_id
    normalized.pop(LEGACY_DOC_ID_KEY, None)
    return normalized
