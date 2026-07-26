"""Shared canonical source and locator constraints for atomic v9 slots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal
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
_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)(?!\.\d)")
_CONSTRAINT_NUMBER_PATTERN = re.compile(
    r"(?<!\d)[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\d)(?!\.\d)"
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
_UNAVAILABLE_PATTERN = re.compile(
    r"\bunavailable\b"
    r"|\bnot\s+(?:available|reported|provided|given|found|stated|shown|listed)\b"
    r"|\bno\s+(?:(?:numeric|requested)\s+)?"
    r"(?:result|value|score|metric|dice)\b",
    re.IGNORECASE,
)
_NUMERIC_ANSWER_LINK_PATTERN = re.compile(
    r"(?:\b(?:is|are|was|were|equals?|scored|reached|achieved)\b"
    r"|\b(?:reported\s+as|measured\s+at|as|of)\b|[=:])\s*$",
    re.IGNORECASE,
)
_NUMERIC_CONDITION_CUES = {
    "condition",
    "epoch",
    "epochs",
    "fold",
    "folds",
    "level",
    "model",
    "noise",
    "release",
    "seed",
    "setting",
    "version",
    "year",
}
_FOLLOWING_NUMERIC_UNIT_PATTERN = re.compile(
    r"\s*(?:[-–—]\s*|\(\s*)?(%|[a-z]+)",
    re.IGNORECASE,
)
_NUMERIC_UNIT_ALIASES = {
    "case": "case",
    "cases": "case",
    "epoch": "epoch",
    "epochs": "epoch",
    "figure": "figure",
    "figures": "figure",
    "fold": "fold",
    "folds": "fold",
    "image": "image",
    "images": "image",
    "iteration": "iteration",
    "iterations": "iteration",
    "page": "page",
    "pages": "page",
    "patient": "patient",
    "patients": "patient",
    "percent": "percent",
    "percentage": "percent",
    "percentages": "percent",
    "pct": "percent",
    "run": "run",
    "runs": "run",
    "sample": "sample",
    "samples": "sample",
    "seed": "seed",
    "seeds": "seed",
    "subject": "subject",
    "subjects": "subject",
    "table": "table",
    "tables": "table",
    "year": "year",
    "years": "year",
}
_PERCENT_RESULT_TERMS = {
    "accuracy",
    "percentage",
    "percent",
    "rate",
    "result",
    "score",
    "value",
}
_COUNT_SHAPE_TERMS = {"count", "number", "size", "total"}
_IN_MEASUREMENT_UNITS = {"epoch", "iteration", "year"}
_YEAR_RESULT_TERMS = {"calendar", "publication", "release"}


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
    return [doc_id for doc_id in scope.authorized_doc_ids if doc_id in candidates]


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


def locator_hints_match_chunk(hints: Iterable[str], chunk: Mapping[str, Any]) -> bool:
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

    competitor_discriminators = [
        _unique_discriminators(peer, all_slots) for peer in peers
    ]
    signal_spans = _answer_signal_spans(
        slot,
        text,
        discriminators=discriminators,
        competitor_discriminators=competitor_discriminators,
    )
    if not signal_spans:
        return False
    if slot.expected_answer_type not in {"number", "equation", "definition"}:
        return discriminators.issubset(set(_content_tokens(text)))

    for signal_span in signal_spans:
        slot_distance = _association_distance(discriminators, text, signal_span)
        if slot_distance is None:
            continue
        competitor_distances = [
            distance
            for terms in competitor_discriminators
            if terms
            if (distance := _association_distance(terms, text, signal_span)) is not None
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
    slot: RequiredSlot,
    text: str,
    *,
    discriminators: set[str],
    competitor_discriminators: Iterable[set[str]],
) -> tuple[tuple[int, int], ...]:
    without_locators = _STRUCTURED_LOCATOR_PATTERN.sub(
        lambda match: " " * len(match.group(0)), text
    )
    if slot.expected_answer_type == "number":
        condition_numbers = _slot_condition_numbers(slot)
        spans: list[tuple[int, int]] = []
        for match in _NUMBER_PATTERN.finditer(without_locators):
            span = match.span()
            independently_reported = _is_independently_reported_number(
                slot,
                text,
                span,
                discriminators=discriminators,
            )
            value = _canonical_number(match.group(0))
            if value in condition_numbers and not independently_reported:
                continue
            if not independently_reported:
                continue
            if _numeric_candidate_has_unrequested_unit(slot, text, match):
                continue
            if _numeric_signal_is_unavailable(
                text,
                span,
                discriminators=discriminators,
                competitor_discriminators=competitor_discriminators,
            ):
                continue
            spans.append(span)
        return tuple(spans)
    if slot.expected_answer_type == "equation":
        return tuple(
            (index, index + 1) for index, char in enumerate(text) if char == "="
        )
    if slot.expected_answer_type == "definition":
        return tuple(match.span() for match in _DEFINITION_PATTERN.finditer(text))
    if slot.expected_answer_type == "comparison":
        match = re.search(r"\b(?:than|versus|vs\.?|compared)\b", text, re.IGNORECASE)
        return (match.span(),) if match is not None else ()
    return ((0, len(text)),) if text.strip() else ()


def _slot_condition_numbers(slot: RequiredSlot) -> set[str]:
    values: set[str] = set()
    constraint_texts = (
        slot.description,
        *slot.entity_ids,
        *slot.locator_hints,
        *slot.source_name_hints,
        *slot.authorized_source_doc_ids,
    )
    for value in constraint_texts:
        values.update(
            _canonical_number(match.group(0))
            for match in _CONSTRAINT_NUMBER_PATTERN.finditer(value)
        )
    return values


def _canonical_number(value: str) -> str:
    suffix = "%" if value.endswith("%") else ""
    number = Decimal(value.removesuffix("%"))
    return f"{number.normalize():f}{suffix}"


def _numeric_candidate_has_unrequested_unit(
    slot: RequiredSlot,
    text: str,
    match: re.Match[str],
) -> bool:
    unit = _numeric_candidate_unit(text, match)
    if unit is None:
        return False
    return unit not in _requested_numeric_units(slot)


def _numeric_candidate_unit(
    text: str,
    match: re.Match[str],
) -> str | None:
    if match.group(0).endswith("%"):
        return "percent"
    unit_match = _FOLLOWING_NUMERIC_UNIT_PATTERN.match(text, match.end())
    if unit_match is None:
        return None
    raw_unit = unit_match.group(1).casefold()
    if raw_unit == "%":
        return "percent"
    return _NUMERIC_UNIT_ALIASES.get(raw_unit)


def _requested_numeric_units(slot: RequiredSlot) -> set[str]:
    if slot.expected_answer_type != "number":
        return set()

    tokens = [
        match.group(0).casefold()
        for match in _CONTENT_TOKEN_PATTERN.finditer(slot.description)
    ]
    requested: set[str] = set()
    for index, token in enumerate(tokens):
        unit = _NUMERIC_UNIT_ALIASES.get(token)
        if unit is None:
            continue
        previous = tokens[index - 1] if index >= 1 else ""
        previous_two = tokens[index - 2 : index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        following_two = tokens[index + 1 : index + 3]

        if unit == "percent" and (
            previous in _PERCENT_RESULT_TERMS
            or following in _PERCENT_RESULT_TERMS
            or previous == "in"
            or previous_two == ["as", "a"]
        ):
            requested.add(unit)
            continue
        if (
            previous in _COUNT_SHAPE_TERMS
            or following in _COUNT_SHAPE_TERMS
            or (len(previous_two) == 2 and previous_two[0] in _COUNT_SHAPE_TERMS)
            or previous_two == ["how", "many"]
            or (previous == "in" and unit in _IN_MEASUREMENT_UNITS)
        ):
            requested.add(unit)
            continue
        if unit == "year" and (
            previous in _YEAR_RESULT_TERMS or following_two == ["of", "publication"]
        ):
            requested.add(unit)
    return requested


def _numeric_signal_is_unavailable(
    text: str,
    signal_span: tuple[int, int],
    *,
    discriminators: set[str],
    competitor_discriminators: Iterable[set[str]],
) -> bool:
    start = _associated_discriminator_start(discriminators, text, signal_span)
    _, end = _sentence_bounds(text, signal_span)
    competitor_start = _next_discriminator_start(
        competitor_discriminators,
        text,
        signal_span[1],
    )
    if competitor_start is not None:
        end = min(end, competitor_start)
    return _UNAVAILABLE_PATTERN.search(text[start:end]) is not None


def _is_independently_reported_number(
    slot: RequiredSlot,
    text: str,
    signal_span: tuple[int, int],
    *,
    discriminators: set[str],
) -> bool:
    clause_start, _ = _clause_bounds(text, signal_span)
    prefix = text[clause_start : signal_span[0]]
    link = _NUMERIC_ANSWER_LINK_PATTERN.search(prefix)
    if link is None:
        return False
    subject_tokens = _content_tokens(prefix[: link.start()])
    if not subject_tokens:
        return False
    if not discriminators.issubset(set(subject_tokens)):
        return False
    closest_subject = subject_tokens[-1]
    return (
        closest_subject not in _NUMERIC_CONDITION_CUES
        and closest_subject in _slot_descriptor_terms(slot)
    )


def _associated_discriminator_start(
    discriminators: set[str],
    text: str,
    signal_span: tuple[int, int],
) -> int:
    starts_by_term: dict[str, int] = {}
    for match in _CONTENT_TOKEN_PATTERN.finditer(text, 0, signal_span[0]):
        term = match.group(0).casefold()
        if term in discriminators:
            starts_by_term[term] = match.start()
    if discriminators.issubset(starts_by_term):
        return min(starts_by_term.values())
    clause_start, _ = _clause_bounds(text, signal_span)
    return clause_start


def _next_discriminator_start(
    discriminator_sets: Iterable[set[str]],
    text: str,
    start: int,
) -> int | None:
    terms = {term for discriminators in discriminator_sets for term in discriminators}
    return next(
        (
            match.start()
            for match in _CONTENT_TOKEN_PATTERN.finditer(text, start)
            if match.group(0).casefold() in terms
        ),
        None,
    )


def _clause_bounds(text: str, signal_span: tuple[int, int]) -> tuple[int, int]:
    start = 0
    for index in range(signal_span[0] - 1, -1, -1):
        if _is_clause_boundary(text, index):
            start = index + 1
            break
    end = len(text)
    for index in range(signal_span[1], len(text)):
        if _is_clause_boundary(text, index):
            end = index
            break
    return start, end


def _sentence_bounds(text: str, signal_span: tuple[int, int]) -> tuple[int, int]:
    start = 0
    for index in range(signal_span[0] - 1, -1, -1):
        if _is_sentence_boundary(text, index):
            start = index + 1
            break
    end = len(text)
    for index in range(signal_span[1], len(text)):
        if _is_sentence_boundary(text, index):
            end = index
            break
    return start, end


def _is_clause_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in ";\r\n!?":
        return True
    if char != ".":
        return False
    previous_is_digit = index > 0 and text[index - 1].isdigit()
    next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
    return not (previous_is_digit and next_is_digit)


def _is_sentence_boundary(text: str, index: int) -> bool:
    return text[index] != ";" and _is_clause_boundary(text, index)


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
            match.group(0).casefold()
            for match in _CONTENT_TOKEN_PATTERN.finditer(value)
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
