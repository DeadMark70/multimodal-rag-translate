"""Shared canonical source and locator constraints for atomic v9 slots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from data_base.agentic_v9.schemas import RequiredSlot, ResolvedSourceScope


_LOCATOR_PATTERN = re.compile(
    r"^((?i:figure|fig\.?|table|equation|eq\.?|formula|theorem|appendix|section))"
    r"\s*[-:#.]?\s*"
    r"(\(?\d+[A-Za-z]{0,3}\)?(?:\.\d+[A-Za-z]{0,3}){0,4}(?:\([A-Za-z0-9]{1,3}\))?|[A-Z]{1,3}\d*)$"
)
_NAMED_SECTION_PATTERN = re.compile(r"^(?i:section)\s+([A-Za-z][A-Za-z0-9]*)$")
_LOCATOR_TYPE_ALIASES = {
    "fig": "figure",
    "fig.": "figure",
    "eq": "formula",
    "eq.": "formula",
    "equation": "formula",
}

StructuredLocatorState = Literal[
    "not_requested", "matched", "mismatched", "unavailable"
]


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
                if (constraint := canonical_structured_locator(hint)) is not None
            }
        )
    )


def canonical_locator(value: object) -> tuple[str, str] | None:
    """Normalize an explicit structured locator into a type and identifier."""
    return canonical_structured_locator(value)


def canonical_structured_locator(value: object) -> tuple[str, str] | None:
    """Normalize an explicit structured locator, excluding ordinary prose."""
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return None
    match = _LOCATOR_PATTERN.match(normalized)
    if match is not None:
        locator_type = match.group(1).casefold()
        locator_type = _LOCATOR_TYPE_ALIASES.get(locator_type, locator_type)
        identifier = " ".join(match.group(2).split()).casefold()
        return (locator_type, identifier)
    named_section_match = _NAMED_SECTION_PATTERN.match(normalized)
    if named_section_match is None:
        return None
    return ("section", named_section_match.group(1).casefold())


def display_locator_hints(hints: Iterable[str]) -> list[str]:
    """Deduplicate equivalent locators while retaining the first readable form."""
    displayed: list[str] = []
    seen: set[tuple[str, str]] = set()
    for hint in hints:
        normalized = " ".join(hint.split()).strip()
        constraint = canonical_structured_locator(normalized)
        display_key = constraint or ("text", normalized.casefold())
        if not normalized or display_key in seen:
            continue
        seen.add(display_key)
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
    """Return whether structured locator metadata does not contradict hints."""
    return structured_locator_state(hints, chunk) != "mismatched"


def structured_locator_state(
    hints: Iterable[str], chunk: Mapping[str, Any]
) -> StructuredLocatorState:
    """Classify a chunk's structured locator evidence for the requested hints."""
    expected = set(canonical_locator_set(hints))
    if not expected:
        return "not_requested"
    actual = _chunk_locator_set(chunk)
    if expected.intersection(actual):
        return "matched"
    expected_types = {locator_type for locator_type, _identifier in expected}
    if any(locator_type in expected_types for locator_type, _identifier in actual):
        return "mismatched"
    return "unavailable"


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
        parsed = canonical_structured_locator(value)
        if parsed is not None and parsed[0] == locator_type:
            actual.add(parsed)
        else:
            actual.add((locator_type, " ".join(value.split()).casefold()))
    section_value = chunk.get("section")
    section = canonical_structured_locator(section_value)
    if section is not None:
        actual.add(section)
    elif isinstance(section_value, str) and section_value.strip():
        actual.add(("section", " ".join(section_value.split()).casefold()))
    return actual


__all__ = [
    "authorized_doc_ids_for_slot",
    "canonical_locator",
    "canonical_locator_set",
    "canonical_structured_locator",
    "canonical_term_set",
    "display_locator_hints",
    "locator_hints_match_chunk",
    "StructuredLocatorState",
    "structured_locator_state",
]
