"""Deterministic-first selective CRAG classification for Agentic v9."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel

from data_base.agentic_v9.schemas import ResolvedSourceScope


RetrievalAction = Literal["pass", "correct", "llm_judge"]
_LOW_RELEVANCE_SCORE = 0.25
_HIGH_RELEVANCE_SCORE = 0.75


class CragDecision(BaseModel):
    """One deterministic retrieval disposition before any optional LLM grade."""

    action: RetrievalAction
    reason: str
    requires_llm_judge: bool = False


def classify_retrieval_action(
    *,
    documents: Sequence[Mapping[str, Any]],
    source_scope: ResolvedSourceScope,
    locator_hints: Sequence[str],
) -> CragDecision:
    """Classify retrieval without fail-open relevance or answer synthesis.

    Exact authorized locator hits pass locally. Missing, unauthorized, and
    clearly low-scoring retrieval correct locally. Only authorized evidence
    that is plausibly relevant yet partial, close-scored, or conflicting reaches
    the optional 96-token retrieval judge in the caller's budgeted boundary.
    """
    if not documents:
        return CragDecision(action="correct", reason="no_documents")

    authorized_ids = set(source_scope.authorized_doc_ids)
    authorized = [
        document
        for document in documents
        if _document_doc_id(document) in authorized_ids
    ]
    if not authorized:
        return CragDecision(action="correct", reason="no_authorized_documents")
    if any(bool(document.get("conflicting", False)) for document in authorized):
        return _judge("conflicting_evidence")

    scores = [_score(document) for document in authorized]
    known_scores = [score for score in scores if score is not None]
    if known_scores and max(known_scores) <= _LOW_RELEVANCE_SCORE:
        return CragDecision(action="correct", reason="low_relevance")

    normalized_hints = tuple(hint.strip().casefold() for hint in locator_hints if hint.strip())
    has_exact_locator = bool(normalized_hints) and any(
        _has_all_locators(document, normalized_hints) for document in authorized
    )
    if has_exact_locator and (not known_scores or max(known_scores) >= _HIGH_RELEVANCE_SCORE):
        return CragDecision(action="pass", reason="exact_authorized_locator")
    return _judge("partial_or_scope_ambiguous_evidence")


async def resolve_selective_crag(
    *,
    decision: CragDecision,
    judge: Callable[[], Awaitable[bool]] | None,
    judge_budget_available: bool,
) -> CragDecision:
    """Resolve the ambiguity-only judge without ever spending final reserve.

    The caller passes ``judge_budget_available`` only after its budget boundary
    has retained the final-call envelope. Any unavailable, malformed, or failed
    judge result corrects conservatively instead of treating evidence as relevant.
    """
    if decision.action != "llm_judge":
        return decision
    if not judge_budget_available:
        return CragDecision(action="correct", reason="final_budget_protected")
    if judge is None:
        return CragDecision(action="correct", reason="llm_judge_unavailable")
    try:
        accepted = await judge()
    except Exception:
        return CragDecision(action="correct", reason="llm_judge_failed")
    if accepted is True:
        return CragDecision(action="pass", reason="llm_judge_passed")
    return CragDecision(action="correct", reason="llm_judge_rejected")


def _judge(reason: str) -> CragDecision:
    return CragDecision(action="llm_judge", reason=reason, requires_llm_judge=True)


def _document_doc_id(document: Mapping[str, Any]) -> str | None:
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        doc_id = metadata.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            return doc_id.strip()
    doc_id = document.get("doc_id")
    return doc_id.strip() if isinstance(doc_id, str) and doc_id.strip() else None


def _score(document: Mapping[str, Any]) -> float | None:
    value = document.get("score")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _has_all_locators(document: Mapping[str, Any], hints: Sequence[str]) -> bool:
    searchable = " ".join(_string_values(document)).casefold()
    return all(hint in searchable for hint in hints)


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _string_values(item)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [text for item in value for text in _string_values(item)]
    return []


__all__ = [
    "CragDecision",
    "RetrievalAction",
    "classify_retrieval_action",
    "resolve_selective_crag",
]
