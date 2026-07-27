"""Unit coverage for Agentic v9 structured locator constraints."""

from __future__ import annotations

import pytest

from data_base.agentic_v9.slot_constraints import (
    canonical_structured_locator,
    structured_locator_state,
)


def test_ordinary_text_is_not_a_structured_locator() -> None:
    assert canonical_structured_locator("The reported score is 0.91.") is None
    assert structured_locator_state(
        ["Gemini 2.5 Flash"], {"table_id": "Table 3"}
    ) == "not_requested"


@pytest.mark.parametrize(
    "locator",
    [
        "Table",
        "Table discusses ablation results",
        "Section",
        "Section explains architecture",
    ],
)
def test_type_prefixed_prose_is_not_a_structured_locator(locator: str) -> None:
    assert canonical_structured_locator(locator) is None


@pytest.mark.parametrize(
    ("locator", "expected"),
    [
        ("Table 3", ("table", "3")),
        ("Figure 1(a)", ("figure", "1(a)")),
        ("Equation 2", ("formula", "2")),
        ("Theorem 1", ("theorem", "1")),
        ("Appendix A", ("appendix", "a")),
        ("Section 3.2", ("section", "3.2")),
        ("Section Results", ("section", "results")),
        ("Section Methods", ("section", "methods")),
        ("Section Overview", ("section", "overview")),
        ("Section Dataset", ("section", "dataset")),
        ("Section Ablation", ("section", "ablation")),
    ],
)
def test_representative_structured_locators_are_canonical(
    locator: str, expected: tuple[str, str]
) -> None:
    assert canonical_structured_locator(locator) == expected


def test_matching_table_locator_is_matched() -> None:
    assert canonical_structured_locator("  TABLE   3 ") == ("table", "3")
    assert structured_locator_state(
        ["Table 3"], {"table_id": "Table 3"}
    ) == "matched"


def test_different_relevant_table_is_mismatched() -> None:
    assert structured_locator_state(
        ["Table 3"], {"table_id": "Table 4"}
    ) == "mismatched"


def test_different_named_section_is_mismatched() -> None:
    assert structured_locator_state(
        ["Section Overview"], {"section": "Section Methods"}
    ) == "mismatched"


def test_absent_requested_table_metadata_is_unavailable() -> None:
    assert structured_locator_state(
        ["Table 3"], {"section": "Results"}
    ) == "unavailable"
