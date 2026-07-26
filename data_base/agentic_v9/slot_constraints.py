"""Shared canonical source and locator constraints for atomic v9 slots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from data_base.agentic_v9.schemas import RequiredSlot, ResolvedSourceScope


_LOCATOR_PATTERN = re.compile(
    r"^(figure|fig\.?|table|equation|eq\.?|formula|theorem|appendix|section)"
    r"\s*[-:#.]?\s*(.*)$",
    re.IGNORECASE,
)
_LOCATOR_TYPE_ALIASES = {
    "fig": "figure",
    "fig.": "figure",
    "eq": "formula",
    "eq.": "formula",
    "equation": "formula",
}
_CONTENT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
_CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "describe",
    "explain",
    "fact",
    "for",
    "from",
    "give",
    "identify",
    "in",
    "metric",
    "of",
    "on",
    "report",
    "result",
    "retrieve",
    "score",
    "state",
    "the",
    "to",
    "value",
}
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)(?!\.\d)"
)
_STRUCTURED_LOCATOR_PATTERN = re.compile(
    r"\b(?:figure|fig\.?|table|equation|eq\.?|formula|theorem|appendix|section)"
    r"\s*[-:#.]?\s*[a-z0-9.]+\b",
    re.IGNORECASE,
)
_DEFINITION_PATTERN = re.compile(
    r"\b(?:is|are)\s+defined\s+as\b|\bmeans\b|\brefers\s+to\b",
    re.IGNORECASE,
)


def authorized_doc_ids_for_slot(
    slot: RequiredSlot, scope: ResolvedSourceScope
) -> list[str]:
    """Resolve one slot against direct IDs and authoritative source-name pairs."""
    direct_ids = set(slot.authorized_source_doc_ids)
    named_ids = {
        doc_id
        for source_name in slot.source_name_hints
        for doc_id in scope.source_name_to_doc_ids.get(source_name, ())
    }
    if not direct_ids and not named_ids:
        candidates = set(scope.authorized_doc_ids)
    elif direct_ids and named_ids:
        candidates = direct_ids.intersection(named_ids)
    else:
        candidates = direct_ids or named_ids
    return [
        doc_id for doc_id in scope.authorized_doc_ids if doc_id in candidates
    ]


def canonical_locator_set(hints: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return order-, case-, and whitespace-insensitive locator constraints."""
    return tuple(
        sorted(
            {
                constraint
                for hint in hints
                if (constraint := canonical_locator(hint)) is not None
            }
        )
    )


def canonical_locator(value: object) -> tuple[str, str] | None:
    """Normalize a bounded locator into a type and identifier."""
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return None
    match = _LOCATOR_PATTERN.match(normalized)
    if match is None:
        return ("text", normalized.casefold())
    locator_type = match.group(1).casefold()
    locator_type = _LOCATOR_TYPE_ALIASES.get(locator_type, locator_type)
    identifier = " ".join(match.group(2).split()).casefold()
    return (locator_type, identifier)


def display_locator_hints(hints: Iterable[str]) -> list[str]:
    """Deduplicate equivalent locators while retaining the first readable form."""
    displayed: list[str] = []
    seen: set[tuple[str, str]] = set()
    for hint in hints:
        normalized = " ".join(hint.split()).strip()
        constraint = canonical_locator(normalized)
        if not normalized or constraint is None or constraint in seen:
            continue
        seen.add(constraint)
        displayed.append(normalized)
    return displayed


def canonical_term_set(terms: Iterable[str]) -> tuple[str, ...]:
    """Normalize compatible retrieval terms as an unordered semantic set."""
    return tuple(
        sorted(
            {
                " ".join(term.split()).casefold()
                for term in terms
                if " ".join(term.split())
            }
        )
    )


def locator_hints_match_chunk(
    hints: Iterable[str], chunk: Mapping[str, Any]
) -> bool:
    """Require at least one declared locator to match structured chunk metadata."""
    expected = set(canonical_locator_set(hints))
    if not expected:
        return True
    actual = _chunk_locator_set(chunk)
    return bool(expected.intersection(actual))


def slot_content_matches_chunk(
    *,
    slot: RequiredSlot,
    peer_slots: Iterable[RequiredSlot],
    text: str,
) -> bool:
    """Disambiguate co-located atomic slots without consulting answer values."""
    peers = [peer for peer in peer_slots if peer.slot_id != slot.slot_id]
    if not peers:
        return True

    all_slots = [slot, *peers]
    discriminators = _unique_discriminators(slot, all_slots)
    if not discriminators:
        return False

    signal_spans = _answer_signal_spans(slot.expected_answer_type, text)
    if not signal_spans:
        return False
    if slot.expected_answer_type not in {"number", "equation", "definition"}:
        return discriminators.issubset(set(_content_tokens(text)))

    competitor_discriminators = [
        _unique_discriminators(peer, all_slots) for peer in peers
    ]
    for signal_span in signal_spans:
        slot_distance = _association_distance(discriminators, text, signal_span)
        if slot_distance is None:
            continue
        competitor_distances = [
            distance
            for terms in competitor_discriminators
            if terms
            if (distance := _association_distance(terms, text, signal_span))
            is not None
        ]
        if all(slot_distance < distance for distance in competitor_distances):
            return True
    return False


def _unique_discriminators(
    slot: RequiredSlot, all_slots: Iterable[RequiredSlot]
) -> set[str]:
    other_terms = {
        term
        for other in all_slots
        if other.slot_id != slot.slot_id
        for term in _slot_descriptor_terms(other)
    }
    return _slot_descriptor_terms(slot).difference(other_terms)


def _answer_signal_spans(
    expected_answer_type: str, text: str
) -> tuple[tuple[int, int], ...]:
    without_locators = _STRUCTURED_LOCATOR_PATTERN.sub(
        lambda match: " " * len(match.group(0)), text
    )
    if expected_answer_type == "number":
        return tuple(match.span() for match in _NUMBER_PATTERN.finditer(without_locators))
    if expected_answer_type == "equation":
        return tuple((index, index + 1) for index, char in enumerate(text) if char == "=")
    if expected_answer_type == "definition":
        return tuple(match.span() for match in _DEFINITION_PATTERN.finditer(text))
    if expected_answer_type == "comparison":
        match = re.search(r"\b(?:than|versus|vs\.?|compared)\b", text, re.IGNORECASE)
        return (match.span(),) if match is not None else ()
    return ((0, len(text)),) if text.strip() else ()


def _association_distance(
    terms: set[str],
    text: str,
    signal_span: tuple[int, int],
) -> float | None:
    token_spans: dict[str, list[tuple[int, int]]] = {}
    for match in _CONTENT_TOKEN_PATTERN.finditer(text):
        token_spans.setdefault(match.group(0).casefold(), []).append(match.span())
    if any(term not in token_spans for term in terms):
        return None
    signal_center = sum(signal_span) / 2
    distances = [
        min(abs((start + end) / 2 - signal_center) for start, end in token_spans[term])
        for term in terms
    ]
    return sum(distances) / len(distances)


def _slot_descriptor_terms(slot: RequiredSlot) -> set[str]:
    terms = set(_content_tokens(slot.description))
    for entity_id in slot.entity_ids:
        terms.update(_content_tokens(entity_id))
    return terms


def _content_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (
            match.group(0).casefold() for match in _CONTENT_TOKEN_PATTERN.finditer(value)
        )
        if token not in _CONTENT_STOPWORDS
    )


def _chunk_locator_set(chunk: Mapping[str, Any]) -> set[tuple[str, str]]:
    actual: set[tuple[str, str]] = set()
    typed_values = (
        ("figure", chunk.get("figure_id")),
        ("table", chunk.get("table_id")),
        ("formula", chunk.get("formula_id")),
    )
    for locator_type, value in typed_values:
        if not isinstance(value, str) or not value.strip():
            continue
        parsed = canonical_locator(value)
        if parsed is not None and parsed[0] == locator_type:
            actual.add(parsed)
        else:
            actual.add((locator_type, " ".join(value.split()).casefold()))
    section = canonical_locator(chunk.get("section"))
    if section is not None:
        actual.add(section)
    return actual


__all__ = [
    "authorized_doc_ids_for_slot",
    "canonical_locator",
    "canonical_locator_set",
    "canonical_term_set",
    "display_locator_hints",
    "locator_hints_match_chunk",
    "slot_content_matches_chunk",
]
