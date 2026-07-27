"""Unit coverage for Agentic v9 structured locator constraints."""

from __future__ import annotations

from data_base.agentic_v9.slot_constraints import (
    canonical_structured_locator,
    structured_locator_state,
)


def test_ordinary_text_is_not_a_structured_locator() -> None:
    assert canonical_structured_locator("The reported score is 0.91.") is None
    assert structured_locator_state(
        ["Gemini 2.5 Flash"], {"table_id": "Table 3"}
    ) == "not_requested"


def test_matching_table_locator_is_matched() -> None:
    assert canonical_structured_locator("  TABLE   3 ") == ("table", "3")
    assert structured_locator_state(
        ["Table 3"], {"table_id": "Table 3"}
    ) == "matched"


def test_different_relevant_table_is_mismatched() -> None:
    assert structured_locator_state(
        ["Table 3"], {"table_id": "Table 4"}
    ) == "mismatched"


def test_absent_requested_table_metadata_is_unavailable() -> None:
    assert structured_locator_state(
        ["Table 3"], {"section": "Results"}
    ) == "unavailable"
