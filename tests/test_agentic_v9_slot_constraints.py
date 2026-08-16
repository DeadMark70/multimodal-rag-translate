"""Unit coverage for Agentic v9 structured locator constraints."""

from __future__ import annotations

import pytest

from data_base.agentic_v9.slot_constraints import (
    candidate_satisfies_hard_anchors,
    canonical_structured_locator,
    derive_slot_hard_anchors,
    structured_locator_state,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    RequiredSlot,
    SourceLocator,
)


def _packet_for_anchor_test(
    statement: str,
    *,
    section: str | None = None,
    table_id: str | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id="evidence-anchor",
        task_id="task-anchor",
        round_id="round-anchor",
        query_id="query-anchor",
        slot_ids=["S1"],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(doc_id="document-anchor", chunk_id="chunk-anchor"),
        scope=EvidenceScope(),
        locator=SourceLocator(section=section, table_id=table_id),
    )


def test_ordinary_text_is_not_a_structured_locator() -> None:
    assert canonical_structured_locator("The reported score is 0.91.") is None
    assert structured_locator_state(
        ["Gemini 2.5 Flash"], {"table_id": "Table 3"}
    ) == "not_requested"


def test_structured_question_locator_is_inherited_when_slot_has_none() -> None:
    anchors = derive_slot_hard_anchors(
        question="According to Algorithm 2, explain the update flow.",
        slot=RequiredSlot(
            slot_id="S1", description="Explain the final update step"
        ),
    )
    assert anchors.locators == ("algorithm 2",)


def test_slot_local_locator_prevents_question_locator_inheritance() -> None:
    anchors = derive_slot_hard_anchors(
        question="According to Algorithm 2, explain the update flow.",
        slot=RequiredSlot(
            slot_id="S1",
            description="Extract the Table 4 update result.",
        ),
    )

    assert anchors.locators == ("table 4",)


def test_slot_local_region_and_ratio_anchors_are_preserved() -> None:
    anchors = derive_slot_hard_anchors(
        question="Compare the efficiency statements.",
        slot=RequiredSlot(
            slot_id="S1",
            description=(
                "Extract the Abstract statement reporting 33x and 13× reductions"
            ),
        ),
    )
    assert anchors.regions == ("abstract",)
    assert set(anchors.numeric_tokens) == {("33", "ratio"), ("13", "ratio")}


def test_hard_anchors_require_each_region_and_numeric_token() -> None:
    slot = RequiredSlot(
        slot_id="S1",
        description="Extract the Abstract statement reporting 33x and 13× reductions",
    )
    mismatched = _packet_for_anchor_test(
        "The Contribution reports 34× and 13× reductions.",
        section="Contribution",
    )
    exact = _packet_for_anchor_test(
        "The Abstract reports 33× and 13× reductions.",
        section="Abstract",
    )

    assert not candidate_satisfies_hard_anchors(
        question="Compare the efficiency statements.", slot=slot, packet=mismatched
    )
    assert candidate_satisfies_hard_anchors(
        question="Compare the efficiency statements.", slot=slot, packet=exact
    )


def test_numeric_anchor_ignores_unrelated_page_and_schema_metadata() -> None:
    slot = RequiredSlot(slot_id="S1", description="Report the scalar value 7.")
    packet = _packet_for_anchor_test(
        "The source reports a scalar value.",
        section="Results",
    ).model_copy(
        update={
            "locator": SourceLocator(
                pdf_page_index=7,
                section="Results",
                citation_format_version="7",
            )
        }
    )

    assert not candidate_satisfies_hard_anchors(
        question="Report the scalar value.", slot=slot, packet=packet
    )


def test_locator_anchor_matching_is_case_insensitive_for_lettered_ids() -> None:
    slot = RequiredSlot(slot_id="S1", description="Extract the Table S1 result.")
    packet = _packet_for_anchor_test(
        "table s1 reports the requested result.",
        table_id="table s1",
    )

    assert candidate_satisfies_hard_anchors(
        question="Extract the result.", slot=slot, packet=packet
    )


@pytest.mark.parametrize(
    "locator",
    [
        "Table",
        "Table discusses ablation results",
        "Section",
        "Section explains",
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


def test_prose_section_predicate_is_not_requested() -> None:
    assert structured_locator_state(
        ["Section explains"], {"section": "Section Results"}
    ) == "not_requested"


def test_absent_requested_table_metadata_is_unavailable() -> None:
    assert structured_locator_state(
        ["Table 3"], {"section": "Results"}
    ) == "unavailable"
