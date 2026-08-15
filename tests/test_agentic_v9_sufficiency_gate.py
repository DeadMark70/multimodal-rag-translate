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
    ResponseConstraint,
    SlotResolution,
    SourceLocator,
    SynthesisObligation,
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
        source=EvidenceSource(
            doc_id="doc-1",
            chunk_id=evidence_id,
            source_span_hash="hash-1" if valid else None,
        ),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1, table_id="table-1"),
        raw_value=Decimal("0.91"),
        extractor_version="v9-deterministic-1" if valid else None,
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


def test_synthesis_obligations_and_constraints_never_appear_in_sufficiency_or_repairables() -> None:
    contract = QueryContract(
        contract_version="2",
        route="bounded_compare",
        intent="Compare Model A and Model B.",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Model A score"),
            RequiredSlot(slot_id="S2", description="Model B score"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare scores",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
        response_constraints=[
            ResponseConstraint(
                constraint_id="C1",
                kind="output_format",
                description="Table output",
            )
        ],
    )

    result = evaluate_sufficiency(
        contract,
        [_packet("p1", "S1")],
    )

    assert [r.slot_id for r in result.slot_resolutions] == ["S1", "S2"]
    assert result.report.supported_slot_ids == ["S1"]
    assert result.report.not_found_slot_ids == ["S2"]
    assert result.repairable_slot_ids == ("S2",)
    assert all("O1" not in res.slot_id for res in result.slot_resolutions)
    assert all("C1" not in res.slot_id for res in result.slot_resolutions)
    assert "O1" not in result.repairable_slot_ids
    assert "C1" not in result.repairable_slot_ids


def test_gate_rejects_packet_lacking_source_span_hash() -> None:
    # A packet marked deterministic_valid without source_span_hash cannot satisfy slots
    packet = EvidencePacket(
        schema_version="1",
        evidence_id="e1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="The reported score is 0.91.",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c1", source_span_hash=None),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        raw_value=Decimal("0.91"),
        extractor_version="v9-deterministic-1",
        validation_status="deterministic_valid",
    )
    result = evaluate_sufficiency(_contract(), [packet])
    assert result.slot_resolutions[0].status == "not_found"
    assert result.report.evidence_complete is False


def test_gate_rejects_packet_lacking_extractor_provenance() -> None:
    # A packet marked deterministic_valid without extractor_version cannot satisfy slots
    packet = EvidencePacket(
        schema_version="1",
        evidence_id="e1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="The reported score is 0.91.",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c1", source_span_hash="hash123"),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        raw_value=Decimal("0.91"),
        extractor_version=None,
        validation_status="deterministic_valid",
    )
    result = evaluate_sufficiency(_contract(), [packet])
    assert result.slot_resolutions[0].status == "not_found"
    assert result.report.evidence_complete is False


def test_gate_rejects_raw_task_inherited_packet_without_qualification() -> None:
    # Raw chunk with validation_status="invalid" cannot satisfy slot
    packet = EvidencePacket(
        schema_version="1",
        evidence_id="raw-1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="Raw chunk text",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c1"),
        scope=EvidenceScope(),
        locator=SourceLocator(pdf_page_index=1),
        validation_status="invalid",
    )
    result = evaluate_sufficiency(_contract(), [packet])
    assert result.slot_resolutions[0].status == "not_found"
    assert result.report.evidence_complete is False


def test_gate_accepts_quote_bound_and_deterministic_structured_packets() -> None:
    quote_packet = EvidencePacket(
        schema_version="1",
        evidence_id="quote-1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="The reported score is 0.91.",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c1", source_span_hash="hash-quote"),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        extractor_version="v9-prose-curator-1",
        validation_status="quote_bound",
    )
    struct_packet = EvidencePacket(
        schema_version="1",
        evidence_id="struct-1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["baseline"],
        statement="Table 1 | Baseline: 0.85",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c2", source_span_hash="hash-struct"),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        raw_value=Decimal("0.85"),
        extractor_version="v9-deterministic-1",
        validation_status="deterministic_valid",
    )
    result = evaluate_sufficiency(_contract(), [quote_packet, struct_packet])
    assert [r.status for r in result.slot_resolutions] == ["supported", "supported"]
    assert result.report.evidence_complete is True


def test_gate_calculated_packet_usable_only_when_all_premises_are_qualified_direct_evidence() -> None:
    premise1 = EvidencePacket(
        schema_version="1",
        evidence_id="p1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["score"],
        statement="Score 0.90",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c1", source_span_hash="hash-p1"),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        raw_value=Decimal("0.90"),
        extractor_version="v9-deterministic-1",
        validation_status="deterministic_valid",
    )
    premise2 = EvidencePacket(
        schema_version="1",
        evidence_id="p2",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["baseline"],
        statement="Baseline 0.80",
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c2", source_span_hash="hash-p2"),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        raw_value=Decimal("0.80"),
        extractor_version="v9-deterministic-1",
        validation_status="deterministic_valid",
    )
    calc_packet = EvidencePacket(
        schema_version="1",
        evidence_id="calc-1",
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["baseline"],
        statement="Difference between p1 and p2: 0.10",
        support_type="calculated",
        source=EvidenceSource(doc_id="doc-1", chunk_id="c1", source_span_hash="hash-p1"),
        scope=EvidenceScope(dataset="Dataset A", metric="Dice"),
        locator=SourceLocator(pdf_page_index=1),
        raw_value=Decimal("0.10"),
        normalized_value=Decimal("0.10"),
        calculation_operation="difference",
        premise_evidence_ids=["p1", "p2"],
        extractor_version="v9-deterministic-1",
        validation_status="derived_non_evidence",
    )

    # All premises present and qualified
    result = evaluate_sufficiency(_contract(), [premise1, premise2, calc_packet])
    assert result.slot_resolutions[1].status == "supported"

    # One premise missing -> calculated packet is not usable
    result_missing = evaluate_sufficiency(_contract(), [premise1, calc_packet])
    assert result_missing.slot_resolutions[1].status == "not_found"

    # One premise invalid (lacks source_span_hash) -> calculated packet is not usable
    invalid_p2 = premise2.model_copy(
        update={"source": EvidenceSource(doc_id="doc-1", chunk_id="c2", source_span_hash=None)}
    )
    result_invalid = evaluate_sufficiency(_contract(), [premise1, invalid_p2, calc_packet])
    assert result_invalid.slot_resolutions[1].status == "not_found"


