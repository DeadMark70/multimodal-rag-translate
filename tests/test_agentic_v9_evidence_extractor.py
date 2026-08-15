"""Focused contracts for deterministic-first Agentic v9 evidence extraction."""

from __future__ import annotations

from decimal import Decimal
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

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
            RequiredSlot(
                slot_id="structure",
                description="Extract the formula, Theorem 1 m range, table row, and enumeration.",
                locator_hints=["Table 2"],
            )
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
async def test_prose_curator_batches_two_generic_slots_in_one_evidence_extract_call() -> None:
    first = _item(
        "E1",
        "The alpha source fact is stated verbatim.",
        slot_ids=["S1"],
    )
    second = _item(
        "E2",
        "The beta source fact is stated verbatim.",
        slot_ids=["S2"],
    )
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["S1"],
                    "statement": first.packet.statement,
                },
                {
                    "source_evidence_id": "E2",
                    "slot_ids": ["S2"],
                    "statement": second.packet.statement,
                },
            ]
        }
    )

    result = await EvidenceExtractor(invoker).extract(
        _contract(
            _slot("S1", "Describe the alpha source fact."),
            _slot("S2", "Describe the beta source fact."),
        ),
        [first, second],
        repairs_complete=True,
    )

    assert [(packet.source.doc_id, packet.slot_ids) for packet in result] == [
        ("doc-polyp", ["S1"]),
        ("doc-polyp", ["S2"]),
    ]
    assert len(invoker.calls) == 1


@pytest.mark.asyncio
async def test_structured_locator_unavailable_falls_back_to_batch_without_mismatched_candidate() -> None:
    unavailable = _item(
        "E-unavailable",
        "The requested result is stated in ordinary retrieved prose.",
        slot_ids=["S1"],
        table_id=None,
    )
    mismatched = _item(
        "E-mismatched",
        "The wrong table contains a different result.",
        slot_ids=["S1"],
        table_id="Table 4",
    )
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E-unavailable",
                    "slot_ids": ["S1"],
                    "statement": unavailable.packet.statement,
                }
            ]
        }
    )

    result = await EvidenceExtractor(invoker).extract(
        _contract(
            RequiredSlot(
                slot_id="S1",
                description="State the requested Table 3 result.",
                locator_hints=["Table 3"],
            )
        ),
        [unavailable, mismatched],
        repairs_complete=True,
    )

    assert [packet.evidence_id for packet in result] == ["curated:E-unavailable:S1"]
    assert "E-unavailable" in invoker.calls[0]["messages"][0]["content"]
    assert "E-mismatched" not in invoker.calls[0]["messages"][0]["content"]


class _FailingInvoker:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, **_kwargs: Any) -> Any:
        self.calls += 1
        raise TimeoutError("evidence extraction timed out")


@pytest.mark.asyncio
async def test_curator_failure_fails_closed_without_supporting_generic_slot() -> None:
    invoker = _FailingInvoker()

    result = await EvidenceExtractor(invoker).extract(
        _contract(_slot("S1", "Describe the generic source fact.")),
        [_item("E1", "The generic source fact is present.", slot_ids=["S1"])],
        repairs_complete=True,
    )

    assert result == []
    assert invoker.calls == 1


@pytest.mark.asyncio
async def test_extractor_retains_prevalidated_packets_without_retaining_invalid_raw_candidates() -> (
    None
):
    accepted = _item(
        "E-valid", "Verified prose.", slot_ids=["S1"], table_id=None
    ).packet.model_copy(update={"validation_status": "quote_bound"})
    raw = _item(
        "E-raw", "Unverified prose.", slot_ids=["S2"], table_id=None
    ).packet.model_copy(update={"validation_status": "invalid"})

    result = await EvidenceExtractor().extract(
        _contract(_slot("S1", "First"), _slot("S2", "Second")),
        [accepted, raw],
        repairs_complete=True,
    )

    assert [packet.evidence_id for packet in result] == ["E-valid"]


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
async def test_invalid_curator_packet_does_not_discard_valid_sibling() -> None:
    statement = "The decoder has two stages."
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "unknown-id",
                    "slot_ids": ["method"],
                    "statement": statement,
                },
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["other"],
                    "statement": statement,
                },
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["method"],
                    "statement": "The decoder has three stages.",
                },
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["method"],
                    "statement": statement,
                },
            ]
        }
    )

    outcome = await EvidenceExtractor(invoker).extract_with_outcome(
        _contract(
            _slot("method", "Describe the decoder architecture."),
            _slot("other", "Describe another source fact."),
        ),
        [_item("E1", statement, slot_ids=["method"])],
        repairs_complete=True,
        question="What decoder is used?",
    )

    assert outcome.status == "provider_qualified"
    assert [packet.evidence_id for packet in outcome.packets] == [
        "curated:E1:method"
    ]
    assert outcome.qualification_unknown_source_id_count == 1
    assert outcome.qualification_unauthorized_source_slot_count == 1
    assert outcome.qualification_statement_not_verbatim_count == 1


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


@pytest.mark.asyncio
async def test_unrelated_verbatim_quote_rejected_for_unauthorized_slot() -> None:
    item = _item(
        "E1",
        "The model achieves 0.91 on polyp segmentation. The authors used PyTorch 2.0.",
        slot_ids=["dice_score"],
    )
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["unrelated_framework"],
                    "statement": "The authors used PyTorch 2.0.",
                }
            ]
        }
    )
    contract = _contract(
        _slot("dice_score", "Extract the Dice score."),
        _slot("unrelated_framework", "Extract framework info."),
    )

    result = await EvidenceExtractor(invoker).extract(
        contract,
        [item],
        repairs_complete=True,
        question="What framework was used?",
    )

    assert result == []
    assert len(invoker.calls) == 1


def test_direct_packets_have_source_span_hash_and_extractor_and_prompt_version() -> None:
    source = _packet_for_test(
        "E1",
        "Table 1 | Seen Dice 0.877",
        table_id="Table 1",
    )
    item = EvidencePoolItem(source, metadata={"text": source.statement})

    # Numeric extraction
    num_packets = extract_numeric_packets(
        slot=RequiredSlot(
            slot_id="dice", description="Extract Dice.", locator_hints=["Table 1"]
        ),
        items=[item],
    )
    assert len(num_packets) == 1
    assert num_packets[0].support_type == "direct"
    assert num_packets[0].validation_status == "deterministic_valid"
    assert num_packets[0].source.source_span_hash is not None
    assert num_packets[0].extractor_version == "v9-deterministic-1"
    assert num_packets[0].prompt_version is None

    # Structured extraction
    struct_source = _packet_for_test(
        "E2",
        "Theorem 1: m in [1, n].",
        table_id="Table 1",
    )
    struct_item = EvidencePoolItem(
        struct_source, metadata={"text": struct_source.statement}
    )
    extractor = EvidenceExtractor()
    struct_packets = extractor.extract_deterministic(
        _contract(
            RequiredSlot(
                slot_id="theorem",
                description="Extract theorem range.",
                locator_hints=["Table 1"],
            )
        ),
        [struct_item],
    )
    assert len(struct_packets) >= 1
    assert struct_packets[0].support_type == "direct"
    assert struct_packets[0].validation_status == "deterministic_valid"
    assert struct_packets[0].source.source_span_hash is not None
    assert struct_packets[0].extractor_version == "v9-deterministic-1"
    assert struct_packets[0].prompt_version is None


@pytest.mark.asyncio
async def test_one_batch_call_handles_all_unresolved_slots_without_per_slot_calls() -> None:
    items = [
        _item("E1", "Alpha architecture detail.", slot_ids=["slot_1"]),
        _item("E2", "Beta loss formulation.", slot_ids=["slot_2"]),
        _item("E3", "Gamma dataset partition.", slot_ids=["slot_3"]),
    ]
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E1",
                    "slot_ids": ["slot_1"],
                    "statement": "Alpha architecture detail.",
                },
                {
                    "source_evidence_id": "E2",
                    "slot_ids": ["slot_2"],
                    "statement": "Beta loss formulation.",
                },
                {
                    "source_evidence_id": "E3",
                    "slot_ids": ["slot_3"],
                    "statement": "Gamma dataset partition.",
                },
            ]
        }
    )
    contract = _contract(
        _slot("slot_1", "Describe Alpha architecture."),
        _slot("slot_2", "Describe Beta loss."),
        _slot("slot_3", "Describe Gamma partition."),
    )

    result = await EvidenceExtractor(invoker).extract(
        contract,
        items,
        repairs_complete=True,
        question="Describe Alpha, Beta, and Gamma.",
    )

    assert len(invoker.calls) == 1
    assert len(result) == 3
    assert all(p.extractor_version == "v9-prose-curator-1" for p in result)
    assert all(p.prompt_version == "1" for p in result)
    assert all(p.validation_status == "quote_bound" for p in result)
    assert all(p.source.source_span_hash is not None for p in result)


@pytest.mark.asyncio
async def test_provider_failure_returns_zero_newly_qualified_packets() -> None:
    item = _item("E1", "Prose description.", slot_ids=["slot_1"])

    class _ErrorInvoker:
        async def invoke(self, **_kwargs: Any) -> Any:
            raise RuntimeError("Provider connection failed")

    extractor = EvidenceExtractor(_ErrorInvoker())
    contract = _contract(_slot("slot_1", "Describe prose."))

    result = await extractor.extract(
        contract,
        [item],
        repairs_complete=True,
        question="Describe prose?",
    )

    assert result == []


@pytest.mark.asyncio
async def test_provider_failure_is_distinct_from_valid_no_match() -> None:
    item = _item("E1", "Prose description.", slot_ids=["slot_1"])

    class _ErrorInvoker:
        async def invoke(self, **_kwargs: Any) -> Any:
            raise RuntimeError("Provider connection failed")

    extractor = EvidenceExtractor(_ErrorInvoker())
    outcome = await extractor.extract_with_outcome(
        _contract(_slot("slot_1", "Describe prose.")),
        [item],
        repairs_complete=True,
        question="Describe prose?",
    )

    assert outcome.packets == ()
    assert outcome.status == "provider_failed"
    assert outcome.failure_code == "provider_attempt_failed"
    assert outcome.provider_call_attempted is True
    assert outcome.provider_response_received is False


@pytest.mark.asyncio
async def test_malformed_provider_response_has_invalid_response_diagnostics() -> None:
    item = _item("E1", "Prose description.", slot_ids=["slot_1"])
    outcome = await EvidenceExtractor(_RecordingInvoker("not json")).extract_with_outcome(
        _contract(_slot("slot_1", "Describe prose.")),
        [item],
        repairs_complete=True,
        question="Describe prose?",
    )

    assert outcome.packets == ()
    assert outcome.status == "invalid_response"
    assert outcome.failure_code == "invalid_provider_response"
    assert outcome.provider_response_received is True


@pytest.mark.asyncio
async def test_content_block_provider_response_qualifies_source_bound_packet() -> None:
    statement = "The method uses a two-stage decoder for small lesions."
    response = AIMessage(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "packets": [
                            {
                                "source_evidence_id": "E1",
                                "slot_ids": ["S1"],
                                "statement": "a two-stage decoder",
                            }
                        ]
                    }
                ),
                "extras": {"signature": "provider-signature"},
            }
        ]
    )
    outcome = await EvidenceExtractor(
        _RecordingInvoker(response)
    ).extract_with_outcome(
        _contract(_slot("S1", "Describe the decoder architecture.")),
        [_item("E1", statement, slot_ids=["S1"])],
        repairs_complete=True,
        question="What decoder architecture is used?",
    )

    assert outcome.status == "provider_qualified"
    assert [packet.evidence_id for packet in outcome.packets] == ["curated:E1:S1"]
    assert outcome.packets[0].validation_status == "quote_bound"


@pytest.mark.asyncio
async def test_valid_empty_provider_response_is_no_match_not_failure() -> None:
    item = _item("E1", "Prose description.", slot_ids=["slot_1"])
    outcome = await EvidenceExtractor(
        _RecordingInvoker({"packets": []})
    ).extract_with_outcome(
        _contract(_slot("slot_1", "Describe prose.")),
        [item],
        repairs_complete=True,
        question="Describe prose?",
    )

    assert outcome.packets == ()
    assert outcome.status == "no_match"
    assert outcome.failure_code is None
    assert outcome.provider_response_received is True


@pytest.mark.asyncio
async def test_q5_positive_fixture_qualifies_miccss_css_prose_architecture() -> None:
    # Q5: nnMamba MICCSS CSS feature fusion description
    statement = (
        "In the MICCSS module, the CSS stage applies three flip branches "
        "along spatial dimensions before feeding into SiamSSM."
    )
    item = _item("E-nnmamba-1", statement, slot_ids=["miccss_css_fusion"])
    invoker = _RecordingInvoker(
        {
            "packets": [
                {
                    "source_evidence_id": "E-nnmamba-1",
                    "slot_ids": ["miccss_css_fusion"],
                    "statement": statement,
                }
            ]
        }
    )
    contract = _contract(
        _slot(
            "miccss_css_fusion",
            "Explain MICCSS CSS feature fusion and flip branches with SiamSSM.",
        )
    )

    result = await EvidenceExtractor(invoker).extract(
        contract,
        [item],
        repairs_complete=True,
        question="根據 nnMamba 架構描述，請重建 MICCSS 模塊中 CSS 階段的特徵融合流程",
    )

    assert len(result) == 1
    assert result[0].validation_status == "quote_bound"
    assert result[0].source.source_span_hash is not None
    assert result[0].slot_ids == ["miccss_css_fusion"]
    assert invoker.calls[0]["messages"][0]["role"] == "user"

    from langchain_core.messages import convert_to_messages
    from langchain_google_genai import ChatGoogleGenerativeAI

    adapter = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="not-used",
    )
    request = adapter._prepare_request(convert_to_messages(invoker.calls[0]["messages"]))
    assert len(request["contents"]) == 1


@pytest.mark.asyncio
async def test_q23_positive_fixture_qualifies_table_1_numeric_evidence() -> None:
    # Q23: SegFormer3D vs nnFormer Table 1 numeric calculation
    statement = "Table 1 | SegFormer3D Params 3.4M | GFLOPs 13.2 | nnFormer Params 115.8M | GFLOPs 171.6"
    item = _item(
        "E-segformer3d-1",
        statement,
        slot_ids=["table_1_metrics"],
        table_id="Table 1",
    )
    contract = _contract(
        RequiredSlot(
            slot_id="table_1_metrics",
            description="Extract Table 1 Params and GFLOPs.",
            locator_hints=["Table 1"],
        )
    )

    extractor = EvidenceExtractor()
    result = extractor.extract_deterministic(contract, [item])

    assert len(result) >= 2
    assert all(p.validation_status == "deterministic_valid" for p in result)
    assert all(p.source.source_span_hash is not None for p in result)
    assert all(p.extractor_version == "v9-deterministic-1" for p in result)


@pytest.mark.asyncio
async def test_q24_unresolved_fixture_without_table_3_evidence_remains_unresolved() -> None:
    # Q24: SegVol zoom-out-zoom-in ablation requires Table 3; candidate only has sliding window prose without Table 3
    statement = "The standard sliding window approach requires 45 seconds per volume."
    item = _item(
        "E-segvol-no-table3",
        statement,
        slot_ids=["zoom_out_zoom_in_ablation"],
        table_id="Table 2",  # Mismatched table locator!
    )
    contract = _contract(
        RequiredSlot(
            slot_id="zoom_out_zoom_in_ablation",
            description="Table 3 zoom-out-zoom-in ablation results comparing Dice and inference time.",
            locator_hints=["Table 3"],
        )
    )

    invoker = _RecordingInvoker({"packets": []})
    extractor = EvidenceExtractor(invoker)
    result = await extractor.extract(
        contract,
        [item],
        repairs_complete=True,
        question="根據 SegVol 的 zoom-out-zoom-in ablation，請列出 Table 3 的三組 Dice／時間",
    )

    # Since Table 3 is required by slot locator hints, but candidate has Table 2 (mismatched),
    # the candidate is not eligible for Table 3 slot, and result remains empty (unresolved).
    assert result == []
    assert len(invoker.calls) == 0


def _packet_for_test(
    evidence_id: str,
    statement: str,
    *,
    table_id: str | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        schema_version="1",
        evidence_id=evidence_id,
        task_id="task-1",
        round_id="round-1",
        query_id="query-1",
        slot_ids=["dice", "theorem"],
        statement=statement,
        support_type="direct",
        source=EvidenceSource(doc_id="doc-1", chunk_id="chunk-1"),
        scope=EvidenceScope(),
        locator=SourceLocator(table_id=table_id, section="Results"),
        validation_status="invalid",
    )
