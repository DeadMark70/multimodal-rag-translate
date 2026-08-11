"""Build truthful citations from retrieved document evidence."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document

from core.errors import AppError
from data_base.document_metadata import get_document_id
from data_base.repository import fetch_document_filenames
from data_base.schemas import SourceDetail

logger = logging.getLogger(__name__)


def _positive_page(metadata: Mapping[str, Any]) -> int | None:
    for key in ("page", "page_number"):
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    return None


def _measured_score(metadata: Mapping[str, Any]) -> float | None:
    for key in ("relevance_score", "reranker_score", "score"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            measured = float(value)
            return (
                measured
                if math.isfinite(measured) and 0.0 <= measured <= 1.0
                else None
            )
    return None


def _normalized_bbox(
    metadata: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    value = metadata.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        return None
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return (x1, y1, x2, y2)


def _metadata_filename(metadata: Mapping[str, Any]) -> str | None:
    for key in ("file_name", "source_file", "filename"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def build_source_details(
    documents: Sequence[Document], source_doc_ids: Sequence[str], *, user_id: str
) -> list[SourceDetail]:
    """Project retrieved evidence into the public citation wire contract."""
    requested_ids = set(source_doc_ids)
    try:
        filenames = await fetch_document_filenames(
            user_id=user_id,
            doc_ids=list(source_doc_ids),
        )
    except AppError as exc:
        logger.warning(
            "Citation filename lookup failed: %s: %s",
            exc.code.value,
            exc.message,
        )
        filenames = {}

    details: list[SourceDetail] = []
    seen_evidence: set[tuple[str, int | None, str | None]] = set()
    represented_doc_ids: set[str] = set()
    for document in documents:
        doc_id = get_document_id(document.metadata)
        if doc_id is None or doc_id not in requested_ids:
            continue
        text = document.page_content.strip()[:200]
        page = _positive_page(document.metadata)
        snippet = text or None
        evidence_key = (doc_id, page, snippet)
        if evidence_key in seen_evidence:
            continue
        seen_evidence.add(evidence_key)
        represented_doc_ids.add(doc_id)
        details.append(
            SourceDetail(
                doc_id=doc_id,
                filename=(
                    _metadata_filename(document.metadata) or filenames.get(doc_id)
                ),
                page=page,
                snippet=snippet,
                score=_measured_score(document.metadata),
                bbox=_normalized_bbox(document.metadata),
            )
        )

    for doc_id in source_doc_ids:
        if doc_id in represented_doc_ids or doc_id not in filenames:
            continue
        represented_doc_ids.add(doc_id)
        details.append(SourceDetail(doc_id=doc_id, filename=filenames.get(doc_id)))

    return details
