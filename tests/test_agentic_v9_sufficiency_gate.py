"""Focused contracts for Agentic v9 required-slot sufficiency."""

from __future__ import annotations

from decimal import Decimal

import pytest

from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RequiredSlot,
    SlotResolution,
    SourceLocator,
)
from data_base.agentic_v9.sufficiency_gate import evaluate_sufficiency


def _contract(*, include_optional_slot: bool = False) -> QueryContract:
    slots = [
        RequiredSlot(slot_id="score", description="reported score"),
        RequiredSlot(slot_id="baseline", description="baseline score"),
    ]
    if include_optional_slot:
        slots.append(
            RequiredSlot(
                slot_id="optional-context",
                description="optional context",
                required=False,
            )
        )
    return QueryContract(
        route="bounded_compare",
        intent="Compare reported scores.",
        required_slots=slots,
    )


def _packet(evidence_id: str, *slot_ids: str, valid: bool = True) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=list(slot_ids),
        statement="The reported score is 0.91.",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id=evidence_id),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1, table_id="table-1"),
        raw_value=Decimal("0.91"),
        validation_status="deterministic_valid" if valid else "invalid",
    )


def test_gate_derives_complete_response_from_validated_evidence_for_every_required_slot() -> None:
    result = evaluate_sufficiency(
        _contract(),
        [_packet("score-evidence", "score"), _packet("baseline-evidence", "baseline")],
    )

    assert [resolution.slot_id for resolution in result.slot_resolutions] == [
        "score",
        "baseline",
    ]
    assert [resolution.status for resolution in result.slot_resolutions] == [
        "supported",
        "supported",
    ]
    assert result.report.evidence_complete is True
    assert result.report.answerable is True
    assert result.report.response_status == "complete"
    assert result.repairable_slot_ids == ()
    assert result.repair_stopped_slot_ids == ()


def test_explicit_unavailability_stops_repair_but_cannot_complete_evidence() -> None:
    result = evaluate_sufficiency(
        _contract(),
        [_packet("score-evidence", "score")],
        [
            SlotResolution(
                slot_id="baseline",
                status="explicitly_unavailable",
                reason="The authorized source does not report a baseline.",
            )
        ],
    )

    assert result.report.evidence_complete is False
    assert result.report.answerable is True
    assert result.report.response_status == "qualified_partial"
    assert result.report.explicitly_unavailable_slot_ids == ["baseline"]
    assert result.report.stop_reason == "explicitly_unavailable"
    assert result.repairable_slot_ids == ()
    assert result.repair_stopped_slot_ids == ("baseline",)


def test_gate_persists_not_found_resolution_as_repairable_missing_slot() -> None:
    result = evaluate_sufficiency(_contract(), [_packet("score-evidence", "score")])

    assert result.slot_resolutions[-1].model_dump() == {
        "slot_id": "baseline",
        "status": "not_found",
        "evidence_ids": [],
        "reason": "No valid evidence or persisted resolution is available.",
        "resolution_stage": "sufficiency_gate",
    }
    assert result.report.not_found_slot_ids == ["baseline"]
    assert result.report.response_status == "qualified_partial"
    assert result.repairable_slot_ids == ("baseline",)


def test_conflicted_slot_remains_persisted_and_zero_supported_slots_are_insufficient() -> None:
    result = evaluate_sufficiency(
        _contract(),
        [_packet("first", "score"), _packet("second", "score")],
        [
            SlotResolution(
                slot_id="score",
                status="conflicted",
                evidence_ids=["first", "second"],
                reason="Same-scope values are incompatible.",
            )
        ],
    )

    assert result.slot_resolutions[0].status == "conflicted"
    assert result.report.conflicted_slot_ids == ["score"]
    assert result.report.evidence_complete is False
    assert result.report.answerable is False
    assert result.report.response_status == "insufficient"
    assert result.repairable_slot_ids == ("baseline",)


def test_optional_missing_slot_is_persisted_without_downgrading_required_completion() -> None:
    result = evaluate_sufficiency(
        _contract(include_optional_slot=True),
        [_packet("score-evidence", "score"), _packet("baseline-evidence", "baseline")],
    )

    assert result.slot_resolutions[-1].status == "not_found"
    assert result.report.evidence_complete is True
    assert result.report.response_status == "complete"
    assert result.repairable_slot_ids == ()


def test_gate_rejects_persisted_supported_resolution_without_known_valid_evidence() -> None:
    with pytest.raises(ValueError, match="unknown or invalid evidence"):
        evaluate_sufficiency(
            _contract(),
            [_packet("score-evidence", "score")],
            [
                SlotResolution(
                    slot_id="baseline",
                    status="supported",
                    evidence_ids=["not-a-packet"],
                )
            ],
        )
