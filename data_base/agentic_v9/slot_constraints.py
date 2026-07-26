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
]
