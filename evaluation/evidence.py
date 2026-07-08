"""Evidence matching helpers for evaluation observability."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    return max(len(text.split()), 0)


def normalize_doc_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def expected_evidence_matches_doc(
    *,
    doc_id: str | None,
    expected_evidence: list[dict[str, Any]],
    expected_sources: list[str],
) -> bool:
    normalized_doc_id = normalize_doc_id(doc_id)
    if normalized_doc_id is None:
        return False
    if normalized_doc_id in {str(item) for item in expected_sources}:
        return True
    return any(str(item.get("doc_id") or item.get("source_doc") or "") == normalized_doc_id for item in expected_evidence)


def text_mentions_fact(text: str, fact_text: str | None) -> bool:
    if not fact_text:
        return False
    normalized_text = text.lower()
    normalized_fact = fact_text.lower().strip()
    if normalized_fact and normalized_fact in normalized_text:
        return True
    fact_tokens = {token.lower() for token in _WORD_RE.findall(fact_text)}
    if not fact_tokens:
        return False
    text_tokens = {token.lower() for token in _WORD_RE.findall(text)}
    return len(fact_tokens & text_tokens) >= max(1, min(len(fact_tokens), 3))


def build_gold_fact_attrition(
    *,
    atomic_facts: list[dict[str, Any]],
    expected_evidence: list[dict[str, Any]],
    source_doc_ids: list[str],
    contexts: list[str],
    answer: str,
) -> list[dict[str, Any]]:
    source_doc_set = {str(item) for item in source_doc_ids}
    packed_text = "\n".join(contexts)
    rows: list[dict[str, Any]] = []
    for index, fact in enumerate(atomic_facts, start=1):
        fact_id = str(fact.get("atomic_fact_id") or fact.get("id") or f"fact-{index}")
        fact_text = str(fact.get("fact_text") or fact.get("text") or fact.get("claim") or "")
        linked_evidence = [
            item
            for item in expected_evidence
            if str(item.get("atomic_fact_id") or item.get("fact_id") or "") in {"", fact_id}
        ]
        expected_docs = {
            str(item.get("doc_id") or item.get("source_doc") or "")
            for item in linked_evidence
            if item.get("doc_id") or item.get("source_doc")
        }
        retrieved = bool(expected_docs & source_doc_set) if expected_docs else text_mentions_fact(packed_text, fact_text)
        packed = retrieved and text_mentions_fact(packed_text, fact_text)
        mentioned = text_mentions_fact(answer, fact_text)
        rows.append(
            {
                "atomic_fact_id": fact_id,
                "fact_text": fact_text,
                "retrieved": retrieved,
                "packed": packed,
                "mentioned": mentioned,
                "cited": retrieved and mentioned,
                "expected_doc_ids": sorted(expected_docs),
            }
        )
    return rows
