"""Tests for compact final synthesis context projection."""

from __future__ import annotations

from decimal import Decimal
import json

from data_base.agentic_v9.final_synthesis_context import (
    build_final_synthesis_context,
)
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


def _make_packet(
    evidence_id: str,
    slot_ids: list[str],
    statement: str,
    *,
    support_type: str = "direct",
    premise_evidence_ids: list[str] | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=slot_ids,
        statement=statement,
        support_type=support_type,  # type: ignore[arg-type]
        source=EvidenceSource(
            doc_id="doc-123",
            document_name="benchmark_paper.pdf",
            source_span_hash=f"hash-{evidence_id}",
        ),
        scope=EvidenceScope(metric="Dice"),
        locator=SourceLocator(pdf_page_index=3, table_id="Table 1"),
        raw_value=Decimal("85.5"),
        normalized_value=Decimal("85.5"),
        unit="%",
        premise_evidence_ids=premise_evidence_ids or [],
        extractor_version="v9-deterministic-1",
        validation_status="deterministic_valid",
    )


def test_build_final_synthesis_context_contains_exact_compact_keys() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare method A and B",
        required_slots=[
            RequiredSlot(slot_id="S1", description="Method A DSC", expected_answer_type="number"),
            RequiredSlot(slot_id="S2", description="Method B DSC", expected_answer_type="number"),
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare Method A vs B DSC",
                depends_on_slot_ids=["S1", "S2"],
            )
        ],
        response_constraints=[
            ResponseConstraint(
                constraint_id="C1",
                kind="output_format",
                description="Report exact delta percentage",
            )
        ],
    )
    p1 = _make_packet("E1", ["S1"], "Method A achieved 85.5% DSC.")
    p2 = _make_packet("E2", ["S2"], "Method B achieved 82.0% DSC.")
    resolutions = [
        SlotResolution(slot_id="S1", status="supported", evidence_ids=["E1"]),
        SlotResolution(slot_id="S2", status="supported", evidence_ids=["E2"]),
    ]

    context = build_final_synthesis_context(
        question="Which method has higher DSC?",
        contract=contract,
        packets=[p1, p2],
        slot_resolutions=resolutions,
    )

    data = json.loads(context.model_dump_json())
    expected_top_keys = {
        "question",
        "required_slots",
        "slot_resolutions",
        "synthesis_obligations",
        "response_constraints",
        "unresolved_requirements",
        "packed_evidence",
        "arbitration",
    }
    assert set(data.keys()) == expected_top_keys
    assert len(data["required_slots"]) == 2
    assert data["required_slots"][0]["slot_id"] == "S1"
    assert data["required_slots"][1]["slot_id"] == "S2"
    assert len(data["synthesis_obligations"]) == 1
    assert data["synthesis_obligations"][0]["obligation_id"] == "O1"
    assert len(data["packed_evidence"]) == 2
    assert data["packed_evidence"][0]["evidence_id"] == "E1"
    assert data["packed_evidence"][0]["doc_id"] == "doc-123"
    assert data["packed_evidence"][0]["statement"] == "Method A achieved 85.5% DSC."


def test_final_synthesis_context_is_materially_smaller_than_raw_dump() -> None:
    contract = QueryContract(
        route="bounded_compare",
        intent="Compare method A and B across several metrics",
        required_slots=[
            RequiredSlot(slot_id=f"S{i}", description=f"Metric {i}", expected_answer_type="number")
            for i in range(1, 6)
        ],
        synthesis_obligations=[
            SynthesisObligation(
                obligation_id="O1",
                kind="comparison",
                description="Compare metrics across methods",
                depends_on_slot_ids=["S1", "S2", "S3", "S4", "S5"],
            )
        ],
    )
    packets = [
        _make_packet(f"E{i}", [f"S{i}"], f"Method data statement for metric {i} with extensive details.")
        for i in range(1, 6)
    ]
    resolutions = [
        SlotResolution(slot_id=f"S{i}", status="supported", evidence_ids=[f"E{i}"])
        for i in range(1, 6)
    ]

    raw_payload = json.dumps(
        {
            "contract": contract.model_dump(mode="json"),
            "packets": [p.model_dump(mode="json") for p in packets],
            "resolutions": [r.model_dump(mode="json") for r in resolutions],
        }
    )

    context = build_final_synthesis_context(
        question="Detailed benchmark query?",
        contract=contract,
        packets=packets,
        slot_resolutions=resolutions,
    )
    compact_payload = context.model_dump_json()

    # The compact context must omit task_id, round_id, query_id, raw_value, normalized_value,
    # unit, extractor_version, validation_status, scope, source_span_hash, etc.
    assert len(compact_payload) < len(raw_payload) * 0.75
