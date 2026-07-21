"""Source-binding contracts for Agentic v9 prose evidence."""

from __future__ import annotations

from data_base.agentic_v9.evidence_extractor import extract_numeric_packets
from data_base.agentic_v9.evidence_pool import EvidencePoolItem
from data_base.agentic_v9.evidence_validator import (
    validate_deterministic_packet,
    validate_prose_packet,
)
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    RequiredSlot,
    SourceLocator,
)


def _packet(
    evidence_id: str,
    statement: str,
    *,
    scope: EvidenceScope | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["method"],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="chunk-1"),
        scope=scope or EvidenceScope(dataset="Dataset A", model_variant="Model A"),
        locator=SourceLocator(pdf_page_index=3, section="Results"),
    )


def test_quote_bound_prose_persists_normalized_source_span_hash() -> None:
    source = _packet(
        "E1", "The method uses a two-stage decoder for small lesions."
    )
    candidate = _packet(
        "curated:E1:method", "uses a two-stage\n decoder for small lesions"
    )

    result = validate_prose_packet(
        candidate,
        source=source,
        source_text=source.statement,
    )

    assert result.status == "quote_bound"
    assert result.packet is not None
    assert result.packet.validation_status == "quote_bound"
    assert result.packet.source.source_span_hash == (
        "4400aae0bafd3bc53d85c9916f92c19e1f47460eb5f51c1b4f2cb2d1db42b6b9"
    )


def test_validator_rejects_rewritten_scope_and_unsourced_number() -> None:
    source = _packet("E1", "Model A reached 0.87 on Dataset A.")
    candidate = _packet(
        "curated:E1:method",
        "Model A reached 0.88 on Dataset A.",
        scope=EvidenceScope(dataset="Dataset B", model_variant="Model A"),
    )

    result = validate_prose_packet(
        candidate,
        source=source,
        source_text=source.statement,
    )

    assert result.status == "invalid"
    assert result.packet is None
    assert result.final_claim is None


def test_quote_bound_high_risk_abstraction_becomes_final_claim_with_premise() -> None:
    source = _packet("E1", "Model A outperforms Model B on Dataset A.")
    candidate = _packet(
        "curated:E1:method", "Model A outperforms Model B on Dataset A."
    )

    result = validate_prose_packet(
        candidate,
        source=source,
        source_text=source.statement,
    )

    assert result.status == "quote_bound"
    assert result.packet is None
    assert result.final_claim is not None
    assert result.final_claim.premise_evidence_ids == ["E1"]


def test_deterministic_extraction_marks_exact_source_span_valid() -> None:
    source = _packet("E1", "Table 1 | Dice 0.877")
    item = EvidencePoolItem(source, metadata={"text": source.statement})

    packets = extract_numeric_packets(
        slot=RequiredSlot(slot_id="method", description="Extract Dice."),
        items=[item],
    )
    result = validate_deterministic_packet(packets[0], source_text=source.statement)

    assert result.status == "deterministic_valid"
    assert result.packet is not None
    assert result.packet.validation_status == "deterministic_valid"
    assert result.packet.source.source_span_hash
