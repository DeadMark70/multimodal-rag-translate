"""Focused contracts for deterministic-first Agentic v9 evidence extraction."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from data_base.agentic_v9.evidence_extractor import (
    EvidenceExtractor,
    calculate_difference,
    extract_numeric_packets,
)
from data_base.agentic_v9.evidence_pool import EvidencePoolItem
from data_base.agentic_v9.evidence_validator import source_span_hash
from data_base.agentic_v9.schemas import (
    EvidencePacket,
    EvidenceScope,
    EvidenceSource,
    QueryContract,
    RequiredSlot,
    SourceLocator,
)


def _contract(*slots: RequiredSlot) -> QueryContract:
    return QueryContract(
        route="exact_structured",
        intent="Extract source-bound evidence",
        required_slots=list(slots),
    )


def _slot(slot_id: str, description: str) -> RequiredSlot:
    return RequiredSlot(slot_id=slot_id, description=description)


def _item(
    evidence_id: str,
    statement: str,
    *,
    slot_ids: list[str],
    table_id: str | None = "Table 1",
    source_span_hash: str | None = None,
) -> EvidencePoolItem:
    return EvidencePoolItem(
        EvidencePacket(
            schema_version="1",
            evidence_id=evidence_id,
            task_id="task-1",
            round_id="round-1",
            query_id="query-1",
            slot_ids=slot_ids,
            statement=statement,
            support_type="direct",
            source=EvidenceSource(
                doc_id="doc-polyp",
                chunk_id="chunk-table-1",
                source_span_hash=source_span_hash,
            ),
            scope=EvidenceScope(metric="Dice"),
            locator=SourceLocator(pdf_page_index=4, table_id=table_id, section="Results"),
        ),
        metadata={"text": statement},
        retrieval_scores={"reranker": 0.91},
    )


def test_numeric_extraction_preserves_exact_values_units_locator_and_source() -> None:
    packets = extract_numeric_packets(
        slot=_slot("dice_values", "Extract the reported Dice values."),
        items=[
            _item(
                "E1",
                "Table 1 | Seen Dice 0.877 | Unseen Dice 0.837 | latency 5 ms",
                slot_ids=["dice_values"],
            )
        ],
    )

    assert [str(packet.raw_value) for packet in packets] == ["0.877", "0.837", "5"]
    assert [packet.unit for packet in packets] == [None, None, "ms"]
    assert all(packet.source.doc_id == "doc-polyp" for packet in packets)
    assert all(packet.locator.table_id == "Table 1" for packet in packets)
    assert all(packet.round_id == "round-1" for packet in packets)


def test_structured_extraction_preserves_formula_theorem_range_table_row_and_enumeration() -> None:
    extractor = EvidenceExtractor()
    packets = extractor.extract_deterministic(
        _contract(
            _slot("structure", "Extract the formula, Theorem 1 m range, table row, and enumeration.")
        ),
        [
            _item(
                "E1",
                "Theorem 1: m in [1, n]. Equation: L = L_ce + lambda L_dice. "
                "Table 2 | Model A | 0.91. Steps: (a) encode; (b) decode.",
                slot_ids=["structure"],
                table_id="Table 2",
            )
        ],
    )

    statements = [packet.statement for packet in packets]
    assert any("m in [1, n]" in statement for statement in statements)
    assert any("L = L_ce + lambda L_dice" in statement for statement in statements)
    assert any("Table 2 | Model A | 0.91" in statement for statement in statements)
    assert any("(a) encode; (b) decode" in statement for statement in statements)
    assert all(packet.source.chunk_id == "chunk-table-1" for packet in packets)


def test_calculation_references_direct_premises_without_inventing_precision() -> None:
    packet = calculate_difference(
        slot=_slot("dice_gap", "Calculate the Dice gap."),
        left=_item(
            "E1",
            "Seen 0.877",
            slot_ids=["dice_gap"],
            source_span_hash=source_span_hash("Seen 0.877"),
        ).packet,
        right=_item(
            "E2",
            "Unseen 0.837",
            slot_ids=["dice_gap"],
            source_span_hash=source_span_hash("Unseen 0.837"),
        ).packet,
    )

    assert packet.raw_value == Decimal("0.040")
    assert packet.support_type == "calculated"
    assert packet.calculation_operation == "difference"
    assert packet.premise_evidence_ids == ["E1", "E2"]
    assert packet.validation_status == "derived_non_evidence"


def test_calculation_rejects_unvalidated_raw_pool_premises() -> None:
    with pytest.raises(ValueError, match="validated span-hashed direct premises"):
        calculate_difference(
            slot=_slot("dice_gap", "Calculate the Dice gap."),
            left=_item("E1", "Seen 0.877", slot_ids=["dice_gap"]).packet,
            right=_item("E2", "Unseen 0.837", slot_ids=["dice_gap"]).packet,
        )


class _RecordingInvoker:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke(
        self, *, phase: str, purpose: str, messages: list[dict[str, Any]]
    ) -> Any:
        self.calls.append({"phase": phase, "purpose": purpose, "messages": messages})
        return self.response


@pytest.mark.asyncio
async def test_prose_curator_runs_once_only_after_repair_and_derives_source_bound_packet() -> None:
    item = _item(
        "E1",
        "The method uses a two-stage decoder for small lesions.",
        slot_ids=["method"],
    )
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["method"],
                    "statement": "The method uses a two-stage decoder for small lesions.",
                }
            ]
        }
    )
    extractor = EvidenceExtractor(invoker)
    contract = _contract(_slot("method", "Describe the decoder architecture."))

    deferred = await extractor.extract(
        contract, [item], repairs_complete=False, question="What decoder is used?"
    )
    result = await extractor.extract(
        contract, [item], repairs_complete=True, question="What decoder is used?"
    )

    assert deferred == []
    assert len(invoker.calls) == 1
    assert invoker.calls[0]["phase"] == "evidence_extract"
    assert invoker.calls[0]["purpose"] == "evidence_extraction"
    assert result[0].source.doc_id == "doc-polyp"
    assert result[0].statement == item.packet.statement
    assert result[0].slot_ids == ["method"]


@pytest.mark.asyncio
async def test_invalid_curator_packet_is_dropped_without_a_second_repair_call() -> None:
    item = _item("E1", "The decoder has two stages.", slot_ids=["method"])
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "unknown-id",
                    "slot_ids": ["method"],
                    "statement": "The decoder has three stages.",
                }
            ]
        }
    )

    result = await EvidenceExtractor(invoker).extract(
        _contract(_slot("method", "Describe the decoder architecture.")),
        [item],
        repairs_complete=True,
        question="What decoder is used?",
    )

    assert result == []
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_high_risk_curator_prose_is_handed_to_final_claims_not_evidence() -> None:
    item = _item(
        "E1",
        "Model A outperforms Model B on the held-out dataset.",
        slot_ids=["method"],
    )
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["method"],
                    "statement": "Model A outperforms Model B on the held-out dataset.",
                }
            ]
        }
    )
    extractor = EvidenceExtractor(invoker)

    result = await extractor.extract(
        _contract(_slot("method", "Describe the source conclusion.")),
        [item],
        repairs_complete=True,
    )

    assert result == []
    assert extractor.final_claims[0].premise_evidence_ids == ["E1"]
