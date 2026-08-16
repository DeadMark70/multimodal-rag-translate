"""Shared canonical source and locator constraints for atomic v9 slots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from data_base.agentic_v9.claim_verifier import numeric_tokens
from data_base.agentic_v9.schemas import EvidencePacket, RequiredSlot, ResolvedSourceScope


_LOCATOR_PATTERN = re.compile(
    r"^((?i:figure|fig\.?|table|equation|eq\.?|formula|theorem|appendix|section))"
    r"\s*[-:#.]?\s*"
    r"(\(?\d+[A-Za-z]{0,3}\)?(?:\.\d+[A-Za-z]{0,3}){0,4}(?:\([A-Za-z0-9]{1,3}\))?|[A-Z]{1,3}\d*)$"
)
_NAMED_SECTION_PATTERN = re.compile(r"^(?i:section)\s+([A-Z][A-Za-z0-9]*)$")
_LOCATOR_TYPE_ALIASES = {
    "fig": "figure",
    "fig.": "figure",
    "eq": "formula",
    "eq.": "formula",
    "equation": "formula",
}
_MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", re.MULTILINE)
_TABLE_CAPTION_PATTERN = re.compile(
    r"\bTable\s+([0-9]+[A-Za-z]{0,3}(?:\.[0-9]+[A-Za-z]{0,3})*)\b",
    re.IGNORECASE,
)
_HARD_LOCATOR_PATTERN = re.compile(
    r"(?P<kind>(?i:algorithm|table|figure|fig\.|section))\s*[-:#.]?\s*"
    r"(?P<identifier>\(?\d+[A-Za-z]{0,3}\)?"
    r"(?:\.\d+[A-Za-z]{0,3}){0,4}(?:\([A-Za-z0-9]{1,3}\))?"
    r"|[A-Z][A-Za-z0-9]*)",
)
_HARD_LOCATOR_FULL_PATTERN = re.compile(
    rf"^{_HARD_LOCATOR_PATTERN.pattern}$"
)
_HARD_REGION_PATTERN = re.compile(r"\b(Abstract|Contribution|Method)\b", re.IGNORECASE)
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<identifier>[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)(?![A-Za-z0-9])"
)
_HARD_LOCATOR_ALIASES = {"fig.": "figure"}
_NON_TECHNICAL_IDENTIFIERS = frozenset(
    {
        "abstract",
        "algorithm",
        "contribution",
        "extract",
        "figure",
        "fig",
        "method",
        "section",
        "table",
    }
)


class SlotHardAnchors(BaseModel):
    """Explicit slot constraints that a curated source span must contain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    locators: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    numeric_tokens: tuple[tuple[str, str], ...] = ()
    identifiers: tuple[str, ...] = ()


def infer_markdown_table_id(text: str | None) -> str | None:
    """Conservatively infer a table identifier only from an explicit markdown pipe table with caption."""
    if not isinstance(text, str) or "|" not in text:
        return None
    if not _MARKDOWN_TABLE_SEPARATOR.search(text):
        return None
    caption_match = _TABLE_CAPTION_PATTERN.search(text)
    if caption_match is None:
        return None
    return f"Table {caption_match.group(1)}"


def derive_slot_hard_anchors(
    *, question: str, slot: RequiredSlot
) -> SlotHardAnchors:
    """Derive only explicit, deterministic anchors for one required slot.

    Slot-local locator hints and description text are authoritative.  A
    structured locator from the question is inherited only when the slot has
    no structured locator of its own.
    """

    slot_text = slot.description or ""
    local_locators = _hard_locators_from_hints(slot.locator_hints)
    local_locators = _stable_unique(
        [*local_locators, *_hard_locators_from_text(slot_text)]
    )
    locators = local_locators or _stable_unique(_hard_locators_from_text(question))
    regions = _stable_unique(
        match.group(1).casefold() for match in _HARD_REGION_PATTERN.finditer(slot_text)
    )
    numeric = tuple(
        sorted(
            token
            for token in numeric_tokens(slot_text)
            if not _is_generic_ordinal(slot_text, token)
        )
    )
    identifiers = _stable_unique(
        _technical_identifiers(slot_text)
    )
    return SlotHardAnchors(
        locators=tuple(locators),
        regions=tuple(regions),
        numeric_tokens=numeric,
        identifiers=tuple(identifiers),
    )


def candidate_satisfies_hard_anchors(
    *, question: str, slot: RequiredSlot, packet: EvidencePacket
) -> bool:
    """Return whether a canonical candidate contains every explicit anchor."""

    anchors = derive_slot_hard_anchors(question=question, slot=slot)
    if not any(
        (
            anchors.locators,
            anchors.regions,
            anchors.numeric_tokens,
            anchors.identifiers,
        )
    ):
        return True

    projection = _candidate_projection(packet)
    actual_locators = _candidate_locators(packet, projection)
    expected_locators = {
        locator
        for value in anchors.locators
        if (locator := _hard_locator_key(value)) is not None
    }
    if expected_locators:
        for locator in expected_locators:
            if locator in actual_locators:
                continue
            if not _contains_text_token(projection, _format_hard_locator(locator)):
                return False

    if any(
        not _contains_text_token(projection, region) for region in anchors.regions
    ):
        return False

    actual_numeric_tokens = numeric_tokens(projection)
    if not set(anchors.numeric_tokens).issubset(actual_numeric_tokens):
        return False

    return all(
        _contains_text_token(projection, identifier)
        for identifier in anchors.identifiers
    )


def _hard_locators_from_hints(hints: Iterable[str]) -> list[str]:
    locators: list[str] = []
    for hint in hints:
        if not isinstance(hint, str):
            continue
        value = _hard_locator_key(hint)
        if value is not None:
            locators.append(_format_hard_locator(value))
    return locators


def _hard_locators_from_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    locators: list[str] = []
    for match in _HARD_LOCATOR_PATTERN.finditer(text):
        value = _hard_locator_key(
            f"{match.group('kind')} {match.group('identifier')}"
        )
        if value is not None:
            locators.append(_format_hard_locator(value))
    return locators


def _hard_locator_key(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return None
    match = _HARD_LOCATOR_FULL_PATTERN.fullmatch(normalized)
    if match is None:
        return None
    kind = _HARD_LOCATOR_ALIASES.get(match.group("kind").casefold(), match.group("kind").casefold())
    identifier = match.group("identifier").casefold()
    return kind, identifier


def _format_hard_locator(value: tuple[str, str]) -> str:
    return f"{value[0]} {value[1]}"


def _technical_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    for match in _TECHNICAL_IDENTIFIER_PATTERN.finditer(text):
        candidate = match.group("identifier")
        letters = [char for char in candidate if char.isalpha()]
        if len(candidate) < 2 or not letters:
            continue
        normalized = candidate.casefold()
        if normalized in _NON_TECHNICAL_IDENTIFIERS:
            continue
        title_case = letters[0].isupper() and all(
            char.islower() for char in letters[1:]
        )
        mixed_case = (
            any(char.islower() for char in letters)
            and any(char.isupper() for char in letters)
            and not title_case
        )
        acronym = len(letters) >= 2 and all(char.isupper() for char in letters)
        alpha_numeric = any(char.isdigit() for char in candidate) and len(letters) >= 1
        if alpha_numeric and any(
            kind == "ratio" for _value, kind in numeric_tokens(candidate)
        ):
            alpha_numeric = False
        if mixed_case or acronym or alpha_numeric:
            identifiers.append(normalized)
    return identifiers


def _is_generic_ordinal(text: str, token: tuple[str, str]) -> bool:
    value, kind = token
    if kind != "scalar":
        return False
    escaped_value = re.escape(value)
    return re.search(
        rf"\b(?:slot|step|item|task|question)\s+{escaped_value}\b",
        text,
        re.IGNORECASE,
    ) is not None


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _candidate_projection(packet: Any) -> str:
    values: list[str] = []
    statement = getattr(packet, "statement", "")
    if isinstance(statement, str):
        values.append(" ".join(statement.split()).casefold())
    locator = getattr(packet, "locator", None)
    if locator is not None:
        try:
            locator_values = locator.model_dump(mode="python")
        except AttributeError:
            locator_values = vars(locator)
        values.extend(
            " ".join(str(locator_values.get(field)).split()).casefold()
            for field in ("section", "table_id", "figure_id")
            if locator_values.get(field) is not None
        )
    return " ".join(values)


def _candidate_locators(packet: Any, projection: str) -> set[tuple[str, str]]:
    values = set(
        _hard_locator_key(match.group(0))
        for match in _HARD_LOCATOR_PATTERN.finditer(projection)
    )
    values.discard(None)
    locator = getattr(packet, "locator", None)
    if locator is None:
        return values  # type: ignore[return-value]
    for field, kind in (
        ("table_id", "table"),
        ("figure_id", "figure"),
        ("section", "section"),
    ):
        field_value = getattr(locator, field, None)
        if not isinstance(field_value, str) or not field_value.strip():
            continue
        parsed = _hard_locator_key(field_value)
        values.add(parsed if parsed is not None else (kind, " ".join(field_value.split()).casefold()))
    return values  # type: ignore[return-value]


def _contains_text_token(projection: str, value: str) -> bool:
    normalized = " ".join(value.split()).casefold()
    if not normalized:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
    return re.search(pattern, projection.casefold()) is not None

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
    "SlotHardAnchors",
    "authorized_doc_ids_for_slot",
    "candidate_satisfies_hard_anchors",
    "canonical_locator",
    "canonical_locator_set",
    "canonical_structured_locator",
    "canonical_term_set",
    "derive_slot_hard_anchors",
    "display_locator_hints",
    "locator_hints_match_chunk",
    "StructuredLocatorState",
    "structured_locator_state",
]
